"""One-shot GigE acquisition through the optional Dahua/iRAYPLE IMV SDK."""
from __future__ import annotations

import ctypes as C
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from threading import RLock

import numpy as np

from ._imv_types import DeviceList, Frame, U64, bind

MONO8 = 0x01080001
MAX_SENSOR_PIXELS = 16_000_000
MAX_SENSOR_SIDE = 8192
_SDK_LOCK = RLock()


class CameraError(RuntimeError):
    """An actionable acquisition failure, never a GOOD inspection result."""


def find_library() -> Path:
    """Explicit environment settings take priority over common install locations."""
    explicit = os.environ.get("MVSDK_LIBRARY")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise CameraError("MVSDK_LIBRARY không trỏ tới DLL tồn tại: " + str(path))
        return path
    roots = [Path(os.environ["MVSDK_ROOT"])] if os.environ.get("MVSDK_ROOT") else []
    if not roots:
        for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            if os.environ.get(variable):
                base = Path(os.environ[variable])
                roots.extend(base / vendor / app for vendor in ("HuarayTech", "Dahua", "iRAYPLE")
                             for app in ("MV Viewer", "MVviewer"))
    for root in roots:
        for relative in ("Runtime/x64/MVSDKmd.dll", "Development/Runtime/x64/MVSDKmd.dll",
                         "SDK/Runtime/x64/MVSDKmd.dll", "MVSDKmd.dll"):
            path = root / relative
            if path.is_file():
                return path.resolve()
    raise CameraError("Chưa tìm thấy SDK MV Viewer x64. Cài SDK trên máy nối camera; đặt "
                      "MVSDK_ROOT tới thư mục SDK hoặc MVSDK_LIBRARY tới Runtime/x64/MVSDKmd.dll. "
                      "Xem docs/CAMERA_DAHUA.md.")


@dataclass(frozen=True)
class CameraInfo:
    key: str
    model: str
    serial: str
    ip: str
    interface_ip: str


@dataclass
class CameraFrame:
    pixels: np.ndarray
    metadata: dict


def _decode(value):
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="replace")


class IMVCamera:
    """Each capture opens and closes the camera. No handle survives a UI rerun."""

    def __init__(self, *, _dll=None):
        self._dll_directory = None
        self.library = None
        if _dll is None:
            if os.name != "nt" or C.sizeof(C.c_void_p) != 8:
                raise CameraError("Kết nối IMV trực tiếp hiện hỗ trợ Windows/Python 64-bit. "
                                  "Máy khác có thể dùng ảnh PNG/BMP lưu từ MV Viewer.")
            self.library = find_library()
            try:
                self._dll_directory = os.add_dll_directory(str(self.library.parent))
                _dll = C.WinDLL(str(self.library))
                bind(_dll)
            except (OSError, AttributeError) as exc:
                self.close()
                raise CameraError("Không nạp được IMV SDK x64 và các DLL phụ thuộc. "
                                  "Cài MV Viewer/SDK đầy đủ; kiểm tra MVSDK_LIBRARY. " + str(exc)) from exc
        self.dll = _dll

    def close(self):
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _call(self, name, *args):
        code = getattr(self.dll, "IMV_" + name)(*args)
        if code != 0:
            hint = { -107: "IP camera và card mạng khác subnet.",
                     -111: "Camera có thể đang bị MV Viewer/phần mềm khác chiếm quyền.",
                     -116: "Hết thời gian chờ ảnh; kiểm tra trigger, Ethernet và exposure."}.get(code, "")
            raise CameraError(f"IMV_{name}: mã {code}. {hint}")

    @property
    def version(self):
        with _SDK_LOCK:
            return _decode(self.dll.IMV_GetVersion() or b"unknown")

    def enumerate(self) -> list[CameraInfo]:
        with _SDK_LOCK:
            devices = DeviceList()
            self._call("EnumDevices", C.byref(devices), 1)  # GigE only
            if devices.count > 100 or (devices.count and not devices.devices):
                raise CameraError("Danh sách thiết bị IMV không hợp lệ; kiểm tra phiên bản SDK.")
            return [CameraInfo(_decode(d.key), _decode(d.model), _decode(d.serial),
                               _decode(d.device.gige.ip), _decode(d.interface.gige.ip))
                    for d in devices.devices[:devices.count] if d.interface_type == 1]

    def capture(self, key: str, *, software_trigger=False, timeout_ms=3000,
                exposure_us: float | None = None) -> CameraFrame:
        if not key or "\0" in key or len(key.encode("utf-8")) >= 256:
            raise ValueError("Camera key không hợp lệ")
        if not isinstance(timeout_ms, int) or not 100 <= timeout_ms <= 10000:
            raise ValueError("Timeout phải trong khoảng 100–10000 ms")
        if exposure_us is not None and (not math.isfinite(exposure_us) or exposure_us <= 0):
            raise ValueError("Exposure phải là số dương hữu hạn (microsecond)")
        with _SDK_LOCK:
            cameras = {c.key: c for c in self.enumerate()}
            if key not in cameras:
                raise CameraError("Camera đã mất kết nối. Quét lại thiết bị GigE.")
            handle, frame = C.c_void_p(), Frame()
            opened = grabbing = acquired = False
            restore, errors = [], []
            result = None

            def enum(name, symbol):
                encoded = name.encode("ascii")
                old = U64()
                self._call("GetEnumFeatureValue", handle, encoded, C.byref(old))
                restore.append(("SetEnumFeatureValue", (handle, encoded, old.value)))
                self._call("SetEnumFeatureSymbol", handle, encoded, symbol.encode("ascii"))

            try:
                identifier = C.c_char_p(key.encode("utf-8"))
                self._call("CreateHandle", C.byref(handle), 1, C.cast(identifier, C.c_void_p))
                if not handle.value:
                    raise CameraError("SDK trả về handle rỗng")
                self._call("Open", handle)
                opened = True
                enum("AcquisitionMode", "Continuous")
                enum("PixelFormat", "Mono8")
                enum("TriggerSelector", "FrameStart")
                enum("TriggerMode", "Off")
                if software_trigger:
                    enum("TriggerSource", "Software")
                    enum("TriggerMode", "On")
                if exposure_us is not None:
                    enum("ExposureAuto", "Off")
                    old = C.c_double()
                    self._call("GetDoubleFeatureValue", handle, b"ExposureTime", C.byref(old))
                    restore.append(("SetDoubleFeatureValue", (handle, b"ExposureTime", old.value)))
                    self._call("SetDoubleFeatureValue", handle, b"ExposureTime", exposure_us)
                self._call("StartGrabbing", handle)
                grabbing = True
                if software_trigger:
                    self._call("ExecuteCommandFeature", handle, b"TriggerSoftware")
                self._call("GetFrame", handle, C.byref(frame), timeout_ms)
                acquired = True
                info = frame.info
                width, height = info.width, info.height
                stride = width + info.padding_x
                if (info.status != 0 or info.pixel_format != MONO8 or not frame.data or
                    width < 1 or height < 1 or width * height > MAX_SENSOR_PIXELS or
                    max(width, height) > MAX_SENSOR_SIDE or stride * height > info.size or
                    info.size > 64 * 1024 * 1024):
                    raise CameraError("Frame lỗi, không phải Mono8 hoặc kích thước/buffer không hợp lệ")
                # Copy while the SDK still owns a valid buffer; discard line padding.
                pixels = np.frombuffer(C.string_at(frame.data, stride * height), dtype=np.uint8)
                pixels = pixels.reshape(height, stride)[:, :width].copy()
                result = CameraFrame(pixels, dict(camera=asdict(cameras[key]), sdk_version=self.version,
                    block_id=int(info.block_id), device_timestamp=int(info.timestamp),
                    captured_at=datetime.now(timezone.utc).isoformat(), pixel_format="Mono8",
                    trigger="software" if software_trigger else "free_run", requested_exposure_us=exposure_us))
            except Exception as exc:
                errors.append(str(exc))
            finally:
                cleanup = []
                if acquired:
                    cleanup.append(("ReleaseFrame", (handle, C.byref(frame))))
                if grabbing:
                    cleanup.append(("StopGrabbing", (handle,)))
                cleanup.extend(reversed(restore))
                if opened:
                    cleanup.append(("Close", (handle,)))
                if handle.value:
                    cleanup.append(("DestroyHandle", (handle,)))
                for name, args in cleanup:
                    try:
                        self._call(name, *args)
                    except Exception as exc:
                        errors.append("Cleanup: " + str(exc))
            if errors:
                raise CameraError("; ".join(errors))
            return result
