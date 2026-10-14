"""Run via scripts/run_ui.py. UI delegates pixel I/O and inference to services."""
from __future__ import annotations

import hashlib
import json

import altair as alt
import pandas as pd
import streamlit as st

from pcb_lab.inference.reports import pareto_models
from pcb_lab.inference.sources import GalleryImageSource, UploadImageSource, URLImageSource, WebcamImageSource
from pcb_lab.ui.resources import get_benchmark, get_gallery, get_service
from pcb_lab.ui.visuals import CLASS_COLORS, overlay, png_bytes
from pcb_lab.ui.industrial_camera import camera_controls

REASONS = {"yolo_hit": "YOLO có box đạt ngưỡng đã khóa.",
           "ad_hit": "EfficientAD đạt ngưỡng bất thường; chưa xác định loại lỗi.",
           "both_hit": "YOLO và EfficientAD đều đạt ngưỡng.",
           "all_pass": "Các tầng đã chạy đều dưới ngưỡng đã khóa."}


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def gallery_controls(prefix, *, multiple=False):
    report_mode = st.checkbox("Xem gallery test sau đánh giá đã khóa", key=f"{prefix}_report")
    partitions = ["calibration", "fusion"] + (["test"] if report_mode else [])
    partition = st.selectbox("Partition", partitions, key=f"{prefix}_partition")
    samples = get_gallery(partition, report_mode)
    by_id = {s.sample_id: s for s in samples}
    if multiple:
        ids = st.multiselect("Ảnh mẫu (tối đa 20)", list(by_id), key=f"{prefix}_samples", max_selections=20)
        return [GalleryImageSource(by_id[sid]) for sid in ids], (partition, tuple(ids), report_mode)
    sid = st.selectbox("Ảnh mẫu", list(by_id), key=f"{prefix}_sample")
    return GalleryImageSource(by_id[sid]), (partition, sid, by_id[sid].sha256, report_mode)


def show_result(result, *, prefix="single"):
    if result.decision == "ERROR":
        st.error(f"ERROR — {result.error_reason}")
        st.download_button("Tải JSON lỗi", json_bytes(result.to_dict()), "inspection-error.json", "application/json", key=f"{prefix}_error_json")
        return
    cols = st.columns(4)
    cols[0].metric("Kết luận", result.decision)
    route = "Early-exit" if result.early_exit else "Routed" if result.mode == "CASCADE" else "OR · cả hai tầng" if result.mode == "OR" else "Một tầng"
    cols[1].metric("Tuyến xử lý", route)
    cols[2].metric("YOLO (ms)", f"{result.timing_ms['yolo']:.1f}" if result.yolo_executed else "Không chạy")
    cols[3].metric("AD (ms)", f"{result.timing_ms['efficientad']:.1f}" if result.efficientad_executed else "Không chạy")
    st.write(" → ".join(result.routing_trace))
    st.write(REASONS.get(result.reason, result.reason))
    if result.metadata.get("ground_truth_available"):
        truth = result.metadata["ground_truth"]
        st.info(f"Nhãn manifest: {truth} · {'Khớp' if truth == result.decision else 'Không khớp'} với dự đoán trên ảnh này.")
    else:
        st.info("Chưa có nhãn chuẩn để đánh giá đúng/sai.")
    if result.metadata.get("warning"):
        st.warning(result.metadata["warning"])
    st.caption(f"{result.image.shape[1]} × {result.image.shape[0]} px · Model {result.model_id} · "
               f"Core {result.latency_ms:.1f} ms · Nạp/tra cache {result.timing_ms['model_load']:.1f} ms · "
               f"Tổng {result.timing_ms['total']:.1f} ms (không gồm render)")
    controls = st.columns(3)
    boxes = controls[0].checkbox("Overlay box YOLO", True, key=f"{prefix}_boxes")
    heatmap = controls[1].checkbox("Overlay heatmap", True, key=f"{prefix}_heatmap", disabled=result.anomaly_map is None)
    opacity = controls[2].slider("Độ phủ heatmap", 0., 1., .45, key=f"{prefix}_opacity")
    gt = st.checkbox("Hiện box ground truth (trắng)", key=f"{prefix}_gt", disabled=not result.metadata.get("ground_truth_available"))
    annotated = overlay(result, boxes=boxes, heatmap=heatmap, opacity=opacity, ground_truth=gt)
    original, output = st.columns(2)
    original.image(result.image, caption="Ảnh gốc · RGB", width="stretch")
    output.image(annotated, caption="Kết quả · tọa độ ảnh gốc", width="stretch")
    st.caption(" · ".join(f"{name}: {color}" for name, color in CLASS_COLORS.items()))
    if result.anomaly_map is None:
        st.caption("EfficientAD: Không chạy — không có heatmap.")
    else:
        st.caption(f"Anomaly score: {result.anomaly_score:.8f} · Ngưỡng: {result.metadata['tau_ad']} · "
                   "Heatmap đổi màu theo min/max của ảnh để quan sát; màu không phải xác suất lỗi.")
    st.dataframe(pd.DataFrame(result.detections, columns=["class_name", "confidence", "xyxy_original"]), hide_index=True)
    if result.detections:
        with st.expander("Phóng to vùng lỗi"):
            index = st.selectbox("Box", range(len(result.detections)), key=f"{prefix}_zoom")
            x1, y1, x2, y2 = result.detections[index]["xyxy_original"]
            st.image(annotated.crop((max(0, int(x1) - 12), max(0, int(y1) - 12),
                                    min(annotated.width, int(x2) + 12), min(annotated.height, int(y2) + 12))), width=420)
    with st.expander("Thông số, score và provenance"):
        st.json(result.to_dict())
    st.download_button("Tải overlay PNG", png_bytes(annotated), "inspection-overlay.png", "image/png", key=f"{prefix}_png")
    st.download_button("Tải dự đoán JSON", json_bytes(result.to_dict()), "inspection.json", "application/json", key=f"{prefix}_json")


def single_screen(service, model_id):
    st.header("Kiểm thử đơn ảnh")
    st.caption("PNG/JPEG · tối đa 20 MiB, 4 triệu pixel · mô hình dành cho PCB trần.")
    kind = st.radio("Nguồn ảnh", ["Upload", "Gallery", "Webcam", "URL", "Camera công nghiệp"], horizontal=True, key="single_source")
    source, identity = None, None
    if kind == "Gallery":
        source, identity = gallery_controls("single")
    elif kind == "Camera công nghiệp":
        source, identity = camera_controls()
    elif kind == "URL":
        url = st.text_input("URL ảnh HTTP(S)")
        if url:
            source, identity = URLImageSource(url), url
    else:
        upload = st.camera_input("Chụp ảnh PCB") if kind == "Webcam" else st.file_uploader("Chọn ảnh", type=["png", "jpg", "jpeg"])
        if upload is not None:
            data = upload.getvalue()
            source = (WebcamImageSource if kind == "Webcam" else UploadImageSource)(data, upload.name)
            identity = hashlib.sha256(data).hexdigest()
    key = (model_id, service.registry.config_hash, kind, identity)
    if st.button("Chạy kiểm tra", disabled=source is None, type="primary"):
        with st.spinner("Đang nạp model đã đăng ký / chạy suy luận…"):
            st.session_state.single_result = (key, service.inspect(source, model_id))
    stored = st.session_state.get("single_result")
    if stored and stored[0] == key:
        show_result(stored[1])


def benchmark_screen():
    st.header("Benchmark đã khóa · H00 vs H01")
    try:
        report = get_benchmark()
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Chưa đánh giá hoặc báo cáo không hợp lệ: {exc}")
        return
    frozen = report.frozen
    runtime = frozen["runtime_specs"]
    groups = sorted({r["source_group"] for r in report.per_group})
    st.caption(f"DeepPCB · split {frozen['split_name']} · N={frozen['sample_count']} · "
               f"{runtime['device']} · seed {runtime['seed']} · batch {runtime['batch_size']} · "
               f"warmup {runtime['warmup_runs']} · khóa {frozen['created_at']}")
    delta = report.h00_h01()
    a, b, c, d = st.columns(4)
    a.metric("H01 giảm mean latency", f"{delta['latency_reduction_pct']:.1f}%")
    b.metric("Δ Recall H01 − H00", f"{delta['recall_delta'] * 100:.2f} điểm %")
    c.metric("Δ FRR H01 − H00", f"{delta['frr_delta'] * 100:.2f} điểm %")
    d.metric("Δ ECPB H01 − H00", f"{delta['ecpb_delta']:.4f}")
    summary = pd.DataFrame(report.summary)
    st.dataframe(summary[summary.model_id.isin(["H00", "H01"])], hide_index=True)
    st.caption("Latency lịch sử Bước 7 được tổng hợp từ thời gian từng tầng; cascade bỏ thời gian AD ở ảnh early-exit. "
               "Các số này không gồm decode/render hoặc nạp model và không thay thế phép đo E2E trực tiếp.")
    selected = st.multiselect("Model trong benchmark", summary.model_id.tolist(), default=summary.model_id.tolist())
    filtered = summary[summary.model_id.isin(selected)].copy()
    if not selected:
        st.info("Chọn ít nhất một model để xem biểu đồ.")
        return
    filtered["Pareto"] = filtered.model_id.isin(pareto_models(report.summary))
    st.dataframe(filtered, hide_index=True)
    st.caption("Recall, balanced accuracy và F1: cao hơn tốt hơn. FRR, ECPB và latency: thấp hơn tốt hơn. "
               "ECPB = (100·FN_sim + FP_sim) / 10.000, giả định 1% defect và 99% good.")
    plots = st.columns(3)
    for col, metric, title in zip(plots, ["defect_recall", "frr", "ecpb"], ["Defect Recall", "FRR", "ECPB"]):
        chart = alt.Chart(filtered).mark_circle(size=150).encode(
            x=alt.X("latency_mean_ms:Q", title="Mean latency (ms)"),
            y=alt.Y(f"{metric}:Q", title=title, scale=alt.Scale(zero=False)),
            color="model_id:N", shape="Pareto:N",
            tooltip=["model_id", "defect_recall", "frr", "ecpb", "latency_mean_ms", "latency_p95_ms", "Pareto"])
        col.altair_chart(chart, width="stretch")
    st.caption("Pareto xét đồng thời Recall ↑, FRR ↓, ECPB ↓, mean latency ↓ trên toàn bộ các model đã khóa.")
    st.subheader("Phân rã nhóm nguồn")
    group_frame = pd.DataFrame(report.per_group)
    group_frame = group_frame[group_frame.model_id.isin(selected)]
    st.dataframe(group_frame, hide_index=True)
    group_plot = group_frame.melt(id_vars=["model_id", "source_group"], value_vars=["defect_recall", "frr"], var_name="metric")
    st.altair_chart(alt.Chart(group_plot).mark_bar().encode(x="model_id:N", y="value:Q", color="source_group:N", xOffset="source_group:N", column="metric:N", tooltip=["model_id", "source_group", "metric", "value"]), width="stretch")
    st.warning(f"Chỉ có {len(groups)} nhóm nguồn ({', '.join(groups)}). Kết quả không chứng minh khả năng chuyển sang bo mạch bất kỳ hoặc PCB có linh kiện.")
    st.subheader("Sáu lớp khuyết tật · detection")
    classes = pd.DataFrame(report.per_class)
    classes = classes[classes.model_id.isin(selected)]
    st.dataframe(classes, hide_index=True)
    if not classes.empty:
        class_plot = classes.melt(id_vars=["model_id", "class_name"], value_vars=["ap_50", "class_recall"], var_name="metric")
        st.altair_chart(alt.Chart(class_plot).mark_bar().encode(x="class_name:N", y="value:Q", color="model_id:N", xOffset="model_id:N", row="metric:N", tooltip=["model_id", "class_name", "metric", "value"]), width="stretch")
    st.caption("Bảng lớp giữ nguyên phạm vi model trong CSV. EfficientAD không phân loại sáu lỗi; không có pixel metrics vì dataset không có mask chuẩn.")
    with st.expander("Cấu hình đã khóa và checkpoint"):
        st.json(frozen)
    for name, data in report.csv_files.items():
        st.download_button(f"Tải {name}", data, name, "text/csv")


def batch_screen(service, model_id):
    st.header("Batch Inspection")
    kind = st.radio("Nguồn batch", ["Upload", "Gallery"], horizontal=True)
    if kind == "Gallery":
        sources, identity = gallery_controls("batch", multiple=True)
    else:
        uploads = st.file_uploader("Chọn tối đa 20 ảnh", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        sources = [UploadImageSource(u.getvalue(), u.name) for u in uploads]
        identity = tuple((s.name, hashlib.sha256(s.data).hexdigest()) for s in sources)
    key = (model_id, service.registry.config_hash, kind, identity)
    if len(sources) > 20:
        st.warning("Mỗi batch hỗ trợ tối đa 20 ảnh.")
    if st.button("Chạy batch", disabled=not 1 <= len(sources) <= 20, type="primary"):
        progress = st.progress(0., text="Đang kiểm tra…")
        results = service.inspect_batch(sources, model_id, progress=lambda done, total: progress.progress(done / total, text=f"{done}/{total}"))
        st.session_state.batch_results = (key, results)
    stored = st.session_state.get("batch_results")
    if stored and stored[0] == key:
        results = stored[1]
        frame = pd.DataFrame([r.summary() for r in results])
        cols = st.columns(4)
        for col, status in zip(cols, ["GOOD", "DEFECT", "ERROR"]):
            col.metric(status, sum(r.decision == status for r in results))
        valid = [r for r in results if r.decision != "ERROR"]
        rate = sum(r.efficientad_executed for r in valid) / len(valid) if valid else None
        cols[3].metric("AD chạy / ảnh thành công", f"{rate:.1%}" if rate is not None else "N/A")
        st.dataframe(frame, hide_index=True)
        if valid and all(r.metadata.get("ground_truth_available") for r in valid):
            st.write("Confusion matrix của batch có nhãn (loại ảnh ERROR):")
            st.dataframe(pd.crosstab(frame.loc[frame.decision != "ERROR", "ground_truth"], frame.loc[frame.decision != "ERROR", "decision"]).reindex(index=["GOOD", "DEFECT"], columns=["GOOD", "DEFECT"], fill_value=0))
        else:
            st.caption("Batch không nhãn chỉ tổng hợp dự đoán; tên file không được dùng làm ground truth.")
        st.download_button("Tải batch CSV", frame.to_csv(index=False).encode("utf-8-sig"), "batch.csv", "text/csv")
        st.download_button("Tải batch JSON", json_bytes([r.to_dict() for r in results]), "batch.json", "application/json")
        selected = st.selectbox("Xem lại ảnh", range(len(results)), format_func=lambda i: f"{i + 1}. {results[i].sample_id}")
        show_result(results[selected], prefix="batch_review")


def main():
    st.set_page_config(page_title="PCB Lab · Inspection", page_icon="🔬", layout="wide")
    st.title("PCB Lab · Kiểm thử khuyết tật")
    st.caption("YOLO11 + EfficientAD · cấu hình đã khóa · ảnh PCB trần")
    page = st.sidebar.radio("Màn hình", ["Single Inspection", "Benchmark H00 vs H01", "Batch Inspection"])
    if page == "Benchmark H00 vs H01":
        benchmark_screen()
        return
    try:
        service = get_service()
        model_id = st.sidebar.selectbox("Mô hình đã đăng ký", list(service.registry.specs), index=4)
        spec = service.registry.get(model_id)
        st.sidebar.caption(f"{spec.mode} · τ YOLO: {spec.tau_yolo} · τ AD: {spec.tau_ad} · CPU")
        if page == "Single Inspection":
            single_screen(service, model_id)
        else:
            batch_screen(service, model_id)
    except (OSError, ValueError, KeyError, PermissionError) as exc:
        st.error(f"Không thể mở cấu hình/nguồn dữ liệu: {exc}")


if __name__ == "__main__":
    main()
