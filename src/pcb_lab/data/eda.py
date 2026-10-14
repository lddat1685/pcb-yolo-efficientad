"""Vietnamese profiles and deterministic train/calibration-only overlays."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from .manifest import (
    PARTITIONS, artifact_path, dataset_path, finite_number, load_dataset,
    output_path, parse_yolo, sha256_file,
)

DEVELOPMENT_PARTITIONS = ("train", "calibration")
COLORS = ("#ff5353", "#ffcd38", "#24dfed", "#d88aff", "#65ed76", "#ff97ce")


def distribution(values: list[float]) -> dict:
    """Linear quantiles at (n-1)*q, including well-defined singleton inputs."""
    values = sorted(values)
    if not values:
        return {"n": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None}

    def quantile(q):
        i = (len(values) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        return round(values[lo] + (values[hi] - values[lo]) * (i - lo), 6)

    return {"n": len(values), "min": round(values[0], 6), "p05": quantile(.05),
            "p50": quantile(.5), "p95": quantile(.95), "max": round(values[-1], 6)}


def _table(headers, rows) -> list[str]:
    def cell(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return str(v).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(cell, row)) + " |" for row in rows), ""]


def build_profile(report: dict, observations: list[dict], names: list[str], protocol: dict) -> tuple[str, dict, list[dict]]:
    """Aggregate audit-only counts for all splits; size/shortcut EDA uses dev."""
    lines = ["# Hồ sơ dữ liệu PCB — kiểm toán và EDA", "",
             f"Kết quả kiểm toán: **{report['overall']}**. Dataset: `{report['dataset_root']}`.", "",
             "Số ảnh lấy từ manifest; box/lớp lấy từ các dòng nhãn YOLO hợp lệ và được đối chiếu với manifest. "
             "Mẫu lỗi vẫn nằm trong tổng ảnh. Nhãn không đọc được không được tính như ảnh có 0 lỗi; "
             "xem `errors` và các check FAIL trong data_audit.json. Box không hợp lệ không vào thống kê kích thước.", "",
             "## Partition và nhóm nguồn", ""]
    lines += _table(["Partition", "Ảnh", "Good", "Defect", "Nhóm nguồn", "% good"], [
        [p, c["images"], c["good"], c["defect"], c["source_groups"],
         100 * c["good"] / c["images"] if c["images"] else None]
        for p, c in report["counts"]["by_partition"].items()])
    total = report["counts"]["total"]
    lines += [f"Tổng: {total['images']} ảnh; {total['good']} good; {total['defect']} defect.", "",
              "Giữ tên `fusion` của nguồn; vai trò dự kiến của dự án là `development_selection`. "
              "Không sửa protocol nguồn hoặc chia lại tập. `source_group` chưa phải design/bo vật lý đã xác minh.", ""]
    group_rows, groups = [], {}
    for p in PARTITIONS:
        for group in sorted({r["source_group"] for r in observations if r["split"] == p}):
            rows = [r for r in observations if r["split"] == p and r["source_group"] == group]
            good = sum(r["is_defect"] is False for r in rows)
            defect = sum(r["is_defect"] is True for r in rows)
            box_count = sum(len(r["boxes"]) for r in rows)
            group_rows.append([p, group, len(rows), good, defect, box_count])
            groups[f"{p}/{group}"] = {"images": len(rows), "good": good, "defect": defect, "boxes": box_count}
    lines += _table(["Partition", "Nhóm nguồn", "Ảnh", "Good", "Defect", "Box hợp lệ"], group_rows)
    lines += ["## Lớp và số lỗi mỗi ảnh", "",
              "Bảng test chỉ phục vụ đối chiếu số lượng annotation, không dùng chọn cấu hình/ngưỡng. "
              "Một ảnh có thể có nhiều lớp; tổng số ảnh theo lớp có thể vượt số ảnh defect.", ""]
    lines += _table(["Partition", "Lớp", "Số box", "Số ảnh chứa lớp"], [
        [p, name, report["class_box_counts"][p][name], report["class_image_counts"][p][name]]
        for p in PARTITIONS for name in names])
    histograms, histogram_rows = {}, []
    for p in PARTITIONS:
        rows = [r for r in observations if r["split"] == p]
        histogram = Counter(len(r["boxes"]) for r in rows if r["label_readable"])
        histograms[p] = {str(k): v for k, v in sorted(histogram.items())}
        histogram_rows += [[p, k, v] for k, v in sorted(histogram.items())]
    lines += _table(["Partition", "Box hợp lệ/ảnh có nhãn đọc được", "Số ảnh"], histogram_rows)
    lines += ["## Kích thước box trên train/calibration", "",
              "Đơn vị px tại ảnh đã decode, diện tích px². Phân vị tuyến tính tại chỉ số `(n−1)×q`. "
              "Không dùng box test/fusion cho EDA kích thước hay chọn ảnh minh họa.", ""]
    measurements = {name: [] for name in names}
    size_stats, size_rows = {}, []
    dev = [r for r in observations if r["split"] in DEVELOPMENT_PARTITIONS]
    for row in dev:
        if row["image_size"] is None:
            continue
        iw, ih = row["image_size"]
        for box in row["boxes"]:
            x1, y1, x2, y2 = box["xyxy_normalized"]
            w, h = round((x2 - x1) * iw, 6), round((y2 - y1) * ih, 6)
            measurements[names[box["class_id"]]].append({"sample_id": row["sample_id"], "partition": row["split"],
                "width": w, "height": h, "short_side": min(w, h), "area": round(w * h, 6)})
    for name in names:
        size_stats[name] = {}
        for metric in ("width", "height", "short_side", "area"):
            stats = distribution([v[metric] for v in measurements[name]])
            size_stats[name][metric] = stats
            size_rows.append([name, metric, stats["n"], stats["min"], stats["p05"], stats["p50"], stats["p95"], stats["max"]])
    lines += _table(["Lớp", "Đại lượng", "N box", "Min", "P5", "P50", "P95", "Max"], size_rows)
    small = {}
    for name in ("pin_hole", "mouse_bite"):
        rows = measurements.get(name, [])
        smallest = min(rows, key=lambda v: (v["area"], v["short_side"], v["sample_id"])) if rows else None
        short = size_stats.get(name, {}).get("short_side", distribution([]))
        fractions = {str(threshold): sum(r["short_side"] < threshold for r in rows) / len(rows) if rows else None
                     for threshold in (16, 32)}
        small[name] = {"n": len(rows), "smallest_by_area": smallest, "short_side": short,
                       "fraction_short_side_lt_px": fractions,
                       "resize_256_short_side": distribution([r["short_side"] * .4 for r in rows])}
        lines += [f"### {name}", ""]
        if smallest:
            lines += [f"Box nhỏ nhất theo diện tích: **{smallest['width']:g} × {smallest['height']:g} px** "
                      f"({smallest['area']:g} px²), `{smallest['sample_id']}` thuộc `{smallest['partition']}`.", ""]
        else:
            lines += ["Không có box hợp lệ để thống kê; các phân vị/tỷ lệ là N/A.", ""]
        lines += _table(["N box", "Cạnh ngắn min", "P5", "P50", "P95", "% cạnh <16 px", "% cạnh <32 px"], [
            [len(rows), short["min"], short["p05"], short["p50"], short["p95"],
             fractions["16"] * 100 if rows else None, fractions["32"] * 100 if rows else None]])
        resized = small[name]["resize_256_short_side"]
        lines += _table(["Resize 640→256", "Min cạnh", "P5", "P50", "P95"], [
            [name, resized["min"], resized["p05"], resized["p50"], resized["p95"]]])
    lines += ["Ngưỡng mô tả chọn trước: `min(w,h) < 16` và `< 32` px (bất đẳng thức nghiêm ngặt). "
              "Nếu resize ảnh 640→256, các cạnh tương ứng <6.4 và <12.8 px. Đây là thước đo rủi ro mất chi tiết, "
              "không phải ngưỡng suy luận và không chứng minh lỗi biến mất. Tile native 256 giữ thang pixel "
              "nhưng có thể cắt box ở biên; Step 2 phải kiểm tra ánh xạ/overlap, Step 1 chưa chạy TileManager/model.", "",
              "## Shortcut và giới hạn diễn giải", ""]
    revealing = sum(isinstance(r["image"], str) and
                    ("_good" in Path(r["image"]).stem or "_defect" in Path(r["image"]).stem) for r in dev)
    uniforms = sum(r["uniform"] is True and r["is_defect"] is False for r in dev)
    modes = dict(sorted(Counter(r["source_mode"] for r in dev).items()))
    normal = dict(sorted(Counter(r["normal_evidence"] for r in dev if r["is_defect"] is False).items()))
    transforms = dict(sorted(Counter(r["transform"] for r in dev).items()))
    lines += [f"- {revealing}/{len(dev)} tên ảnh train/calibration chứa `_good` hoặc `_defect`. "
              "Tên file/ID/split không được làm feature mô hình.",
              "- Pair template/test chia sẻ nội dung/layout; kiểm tra tách pair và nhóm trên mọi partition. "
              "Không thêm template bằng cách đoán tên hoặc suy số ảnh = 2×pair.",
              f"- Giữ nguyên {uniforms} good có nền đồng nhất trong train/calibration. "
              "Độ đồng nhất không phải lý do loại mẫu; nguồn đã xác nhận normal.",
              "- Dataset card mô tả good đã làm sạch và một phần lỗi bổ sung thủ công. "
              "Overlay chỉ kiểm tra tính hợp lý nhãn; không đủ xác nhận/loại trừ dấu vết tổng hợp hoặc shortcut học được.",
              "- Không có pixel mask chuẩn, không báo pixel AUROC/AUPRO hoặc biến bbox thành ground truth segmentation.",
              "- `pin_hole` khác `missing_hole` của PKU. Không gộp domain/schema.",
              "- Số crop không phải số bo độc lập. Bootstrap nội bộ nhóm không chứng minh tổng quát hóa sang domain mới.", ""]
    lines += _table(["Source mode train/calibration", "Số ảnh"], modes.items())
    lines += _table(["Bằng chứng good train/calibration", "Số ảnh"], normal.items())
    lines += _table(["Transform ghi trong manifest train/calibration", "Số ảnh"], transforms.items())
    warnings = []
    if revealing:
        warnings.append({"id": "shortcut_filename", "status": "WARN", "n_violations": revealing,
                         "detail": "Development filenames expose labels; never use paths/IDs as model features."})
    lines += ["## Kiểm tra và giới hạn lượt chạy", ""]
    lines += _table(["Check", "Trạng thái", "Số vi phạm"], [
        [c["id"], c["status"], c["n_violations"]] for c in sorted(report["checks"] + warnings, key=lambda c: c["id"])])
    lines += [f"Release: {report['release']['n_verified']}/{report['release']['n_locked']} tệp khớp SHA-256. "
              f"Số lỗi ghi nhận: {len(report['errors'])}. Chi tiết lỗi ở data_audit.json.", "",
              "Ảnh test chỉ decode/hash trong audit, không xuất/xem. Overlay được tạo riêng bằng CLI hoặc "
              "`render_overlays`; xem `overlay_errors.json` và `overlay_index.json` nếu đã render. "
              "`run_audit` tự nó không render overlay.", ""]
    summary = {"partitions": list(DEVELOPMENT_PARTITIONS), "by_source_group": groups,
               "boxes_per_image": histograms, "box_sizes": size_stats, "small_defects": small,
               "uniform_good_development": uniforms, "filename_label_hints": revealing,
               "source_modes_development": modes, "normal_evidence_development": normal,
               "transforms_development": transforms,
               "source_group_is_not_verified_board_or_design": protocol.get("source_group_is_not_verified_board_or_design")}
    return "\n".join(lines), summary, warnings


def render_overlays(dataset_root: Path, out_dir: Path,
                    partitions=("train", "calibration"), per_class: int = 5) -> list[Path]:
    """Render at most per_class images per class per allowed partition.

    Invalid partitions raise before metadata/image I/O. Sample failures are
    retained in overlay_errors.json; try later candidates instead of aborting.
    """
    partitions = tuple(partitions)
    if any(p not in DEVELOPMENT_PARTITIONS for p in partitions):
        raise ValueError("Overlays allow only train and calibration; test/fusion are forbidden")
    if type(per_class) is not int or per_class < 0:
        raise ValueError("per_class must be a nonnegative integer")
    root = Path(dataset_root).resolve()
    out_dir = output_path(root, Path(out_dir))
    data = load_dataset(root)
    errors, warnings, index, paths = [], [], [], []
    forbidden = set(data.holdout["samples"]["test"]) | set(data.holdout["samples"]["fusion"])
    # A reused canonical image path cannot bypass the split guard through a new ID.
    forbidden_images = set()
    for row in data.rows:
        if row.get("split") in ("test", "fusion") or (isinstance(row.get("sample_id"), str) and row["sample_id"] in forbidden):
            try:
                forbidden_images.add(dataset_path(root, row.get("image")))
            except (OSError, ValueError):
                # No open occurs; this malformed path is reported by the audit.
                errors.append({"sample_id": row.get("sample_id") if isinstance(row.get("sample_id"), str) else None,
                               "stage": "overlay_guard", "message": "Cannot resolve forbidden-partition image path"})
    for partition in DEVELOPMENT_PARTITIONS:
        if partition not in partitions:
            continue
        allowed = set(data.holdout["samples"][partition])
        for cls, name in enumerate(data.names):
            candidates = []
            for row in data.rows:
                if row.get("split") != partition:
                    continue
                sid = row.get("sample_id")
                if not isinstance(sid, str) or sid not in allowed or sid in forbidden:
                    continue
                boxes = row.get("boxes")
                if not isinstance(boxes, list):
                    continue
                matching = [b for b in boxes if isinstance(b, dict) and b.get("class_id") == cls
                            and isinstance(b.get("xyxy"), list) and len(b["xyxy"]) == 4
                            and all(finite_number(v) for v in b["xyxy"])]
                if not matching:
                    continue
                # First small boxes, then boxes closest to the image border, then ID.
                def rank(b):
                    x1, y1, x2, y2 = b["xyxy"]
                    w = row.get("width") if finite_number(row.get("width")) else 0
                    h = row.get("height") if finite_number(row.get("height")) else 0
                    return (min(x2 - x1, y2 - y1), min(x1, y1, w - x2, h - y2))
                candidates.append((min(rank(b) for b in matching), sid, row))
            made = 0
            for _, sid, row in sorted(candidates, key=lambda entry: (entry[0], entry[1])):
                if made >= per_class:
                    break
                try:
                    image_path = dataset_path(root, row.get("image"))
                    if image_path in forbidden_images:
                        raise ValueError("Image path also belongs to a forbidden partition")
                    label_path = dataset_path(root, row.get("label"))
                    if sha256_file(image_path) != row.get("sha256") or sha256_file(label_path) != row.get("label_sha256"):
                        raise ValueError("Image/label SHA-256 mismatch; overlay skipped")
                    boxes, issues = parse_yolo(label_path.read_text(encoding="utf-8-sig"), data.names)
                    if issues or not any(b["class_id"] == cls for b in boxes):
                        raise ValueError(f"Invalid label or selected class absent: {issues}")
                    with Image.open(image_path) as source:
                        source.load()
                        rgb = source.convert("RGB")
                        canvas = Image.new("RGB", (rgb.width, rgb.height + 40), "#111827")
                        canvas.paste(rgb, (0, 40))
                        draw = ImageDraw.Draw(canvas)
                        draw.text((8, 5), f"{partition} | {name} | {sid}", fill="white")
                        draw.text((8, 21), "Ground truth YOLO boxes | native pixel scale", fill="#cbd5e1")
                        for box in boxes:
                            x1, y1, x2, y2 = box["xyxy_normalized"]
                            xy = [x1 * rgb.width, y1 * rgb.height + 40, x2 * rgb.width, y2 * rgb.height + 40]
                            color = COLORS[box["class_id"] % len(COLORS)]
                            draw.rectangle(xy, outline=color, width=2)
                            caption = data.names[box["class_id"]]
                            tx = max(0, min(xy[0], rgb.width - draw.textlength(caption)))
                            ty = max(40, xy[1] - 12)
                            draw.rectangle(draw.textbbox((tx, ty), caption), fill="#111827")
                            draw.text((tx, ty), caption, fill=color)
                        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in sid)[:100]
                        suffix = hashlib.sha256(sid.encode()).hexdigest()[:8]
                        relative = f"overlays/{partition}/{name}/{safe}_{suffix}.png"
                        path = artifact_path(out_dir, relative)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        canvas.save(path, format="PNG")
                        canvas.close()
                        rgb.close()
                    paths.append(path)
                    index.append({"partition": partition, "class_name": name, "sample_id": sid, "path": relative})
                    made += 1
                except (OSError, UnicodeError, ValueError, Image.DecompressionBombError) as exc:
                    errors.append({"sample_id": sid, "stage": "overlay_render", "message": f"{partition}/{name}: {exc}"})
            if made < per_class:
                warnings.append({"sample_id": None, "stage": "overlay_coverage",
                                 "message": f"{partition}/{name}: requested {per_class}, rendered {made}"})
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (("overlay_errors.json", sorted(errors, key=lambda e: (e["sample_id"] or "", e["stage"], e["message"]))),
                              ("overlay_warnings.json", warnings),
                              ("overlay_index.json", sorted(index, key=lambda e: e["path"]))):
        artifact_path(out_dir, filename).write_text(json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    return sorted(paths)
