"""Snapshot camera controls; acquisition occurs only on an explicit button click."""
from PIL import Image
import streamlit as st

from pcb_lab.cameras.imv import IMVCamera
from pcb_lab.inference.industrial import IndustrialCameraSource, saved_camera_frame
from pcb_lab.ui.visuals import png_bytes


def camera_controls():
    st.info("Dahua/iRAYPLE A3504MG100 · GigE · chụp từng ảnh. Chạy ứng dụng trên máy nối camera. "
            "Cài MV Viewer + IMV SDK x64 rồi ngắt kết nối camera trong MV Viewer trước khi chụp ở đây.")
    st.caption("Hướng dẫn cài SDK, mạng và khắc phục lỗi: docs/CAMERA_DAHUA.md. "
               "Camera ở máy khác không tự xuất hiện qua webcam của trình duyệt.")
    mode = st.radio("Cách nhận ảnh camera", ["GigE trực tiếp", "Ảnh lưu từ MV Viewer"], key="industrial_mode")
    frame = None
    if mode == "Ảnh lưu từ MV Viewer":
        upload = st.file_uploader("Ảnh Mono8/RGB8 từ MV Viewer", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                                  key="industrial_file")
        if upload is not None:
            try:
                frame = saved_camera_frame(upload.getvalue(), upload.name)
            except Exception as exc:
                st.error(str(exc))
    else:
        if st.button("Quét camera GigE", key="industrial_scan"):
            st.session_state.pop("industrial_snapshot", None)
            st.session_state.industrial_devices = []
            try:
                with IMVCamera() as camera:
                    st.session_state.industrial_devices = camera.enumerate()
                    st.session_state.industrial_sdk = camera.version
            except Exception as exc:
                st.error(str(exc))
        devices = st.session_state.get("industrial_devices", [])
        if not devices:
            st.info("Nhấn Quét camera GigE. Nếu không thấy thiết bị, kiểm tra SDK, Ethernet và IP trong MV Viewer.")
            return None, None
        by_key = {device.key: device for device in devices}
        key = st.selectbox("Camera", list(by_key), format_func=lambda k: f"{by_key[k].model} · {by_key[k].serial} · {by_key[k].ip}",
                           key="industrial_device")
        st.caption(f"SDK {st.session_state.get('industrial_sdk', '')} · card mạng {by_key[key].interface_ip}")
        trigger = st.checkbox("Dùng software trigger", key="industrial_trigger")
        timeout = int(st.number_input("Thời gian chờ ảnh (ms)", 100, 10000, 3000, key="industrial_timeout"))
        manual = st.checkbox("Đặt exposure thủ công", key="industrial_manual")
        exposure = float(st.number_input("Exposure (µs)", 1., 1_000_000., 10000., key="industrial_exposure")) if manual else None
        settings = (key, trigger, timeout, exposure)
        if st.button("Chụp ảnh camera", key="industrial_capture"):
            st.session_state.pop("industrial_snapshot", None)
            try:
                with st.spinner("Đang chụp một ảnh…"), IMVCamera() as camera:
                    snapshot = camera.capture(key, software_trigger=trigger, timeout_ms=timeout, exposure_us=exposure)
                st.session_state.industrial_snapshot = (settings, snapshot)
            except Exception as exc:
                st.error(str(exc))
        stored = st.session_state.get("industrial_snapshot")
        if stored and stored[0] == settings:
            frame = stored[1]
    if frame is None:
        return None, None
    height, width = frame.pixels.shape[:2]
    st.image(frame.pixels, caption=f"Ảnh camera trước xử lý · {width} × {height} px", width="stretch")
    st.download_button("Tải ảnh camera gốc PNG", png_bytes(Image.fromarray(frame.pixels)),
                       "camera-original.png", "image/png", key="industrial_raw")
    roi = None
    if st.checkbox("Chọn ROI (tọa độ trên ảnh camera)", key="industrial_roi"):
        # Geometry-specific keys prevent obsolete bounds when switching sensor images.
        prefix = f"industrial_{width}_{height}"
        cols = st.columns(4)
        x = int(cols[0].number_input("X", 0, width - 1, 0, key=prefix + "x"))
        y = int(cols[1].number_input("Y", 0, height - 1, 0, key=prefix + "y"))
        w = int(cols[2].number_input("Rộng ROI", 1, width - x, min(640, width - x), key=prefix + f"w{x}"))
        h = int(cols[3].number_input("Cao ROI", 1, height - y, min(640, height - y), key=prefix + f"h{y}"))
        roi = (x, y, w, h)
    side = st.selectbox("Cạnh ảnh xử lý tối đa", [640, 1280, 1920, 4096], index=1, key="industrial_side")
    try:
        source = IndustrialCameraSource(frame, roi=roi, max_side=side)
    except ValueError as exc:
        st.error(str(exc))
        return None, None
    out_w, out_h = source.metadata["inspection_size"]
    st.caption(f"Ảnh đưa vào AI: {out_w} × {out_h} px. Box và heatmap dùng tọa độ ảnh này; "
               "JSON ghi ROI và tỷ lệ so với ảnh camera. Đổi ROI không chụp lại camera.")
    st.warning("Thu nhỏ toàn ảnh có thể mất lỗi nhỏ. Nên chọn ROI 640×640 và kiểm chứng bằng ảnh có nhãn "
               "trên camera thực tế; benchmark DeepPCB không chứng minh độ chính xác trên ảnh camera mới.")
    return source, source.identity
