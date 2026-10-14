"""Camera lifecycle, ABI and image geometry, without installing the vendor SDK."""
import ctypes as C
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from pcb_lab.cameras._imv_types import DeviceInfo, DeviceList, Frame, FrameInfo, GigEDevice, UsbDevice
from pcb_lab.cameras.imv import CameraError, CameraFrame, IMVCamera, MONO8, find_library
from pcb_lab.inference.industrial import IndustrialCameraSource, saved_camera_frame
from test_step8_ui import fake_service


class FakeSDK:
    def __init__(self, fail=None):
        self.fail, self.calls = fail, []
        self.devices = (DeviceInfo * 1)()
        device = self.devices[0]
        device.key, device.model, device.serial = b"Dahua:123", b"A3504MG100", b"123"
        device.interface_type = 1
        device.device.gige.ip = b"192.168.10.2"
        device.interface.gige.ip = b"192.168.10.1"
        self.buffer = (C.c_ubyte * 8)(1, 2, 3, 99, 4, 5, 6, 99)
        self.features = {b"AcquisitionMode": 0, b"PixelFormat": MONO8,
                         b"TriggerSelector": 0, b"TriggerMode": 1, b"TriggerSource": 2,
                         b"ExposureAuto": 2, b"ExposureTime": 15000.}
        self.original = self.features.copy()
        self.frame_overrides = {}

    def __getattr__(self, function):
        def call(*args):
            self.calls.append((function, args))
            if function == self.fail:
                return -116
            if function == "IMV_GetVersion":
                return b"2.4.1"
            if function == "IMV_EnumDevices":
                args[0]._obj.count, args[0]._obj.devices = 1, self.devices
            elif function == "IMV_CreateHandle":
                assert args[1] == 1
                assert C.string_at(args[2]) == b"Dahua:123"
                args[0]._obj.value = 123
            elif function in {"IMV_GetEnumFeatureValue", "IMV_GetDoubleFeatureValue"}:
                args[2]._obj.value = self.features[args[1]]
            elif function in {"IMV_SetEnumFeatureValue", "IMV_SetDoubleFeatureValue"}:
                self.features[args[1]] = args[2]
            elif function == "IMV_SetEnumFeatureSymbol":
                self.features[args[1]] = {b"Continuous": 2, b"Mono8": MONO8, b"FrameStart": 0,
                                          b"Off": 0, b"On": 1, b"Software": 7}[args[2]]
            elif function == "IMV_GetFrame":
                frame = args[1]._obj
                frame.data = C.cast(self.buffer, C.POINTER(C.c_ubyte))
                values = dict(width=3, height=2, size=8, pixel_format=MONO8, padding_x=1,
                              status=0, block_id=42, timestamp=123456)
                values.update(self.frame_overrides)
                for key, value in values.items():
                    setattr(frame.info, key, value)
            elif function == "IMV_ReleaseFrame":
                for index in range(8):
                    self.buffer[index] = 0  # invalidate vendor-owned memory immediately
            return 0
        return call


def test_native_x64_abi_matches_sdk_header():
    if C.sizeof(C.c_void_p) != 8:
        pytest.skip("x64 ABI")
    assert C.sizeof(GigEDevice) == 3092
    assert C.sizeof(UsbDevice) == 3096
    assert C.sizeof(DeviceInfo) == 10312
    assert DeviceInfo.device.offset == 3096
    assert DeviceInfo.interface.offset == 7752
    assert C.sizeof(DeviceList) == 16
    assert FrameInfo.timestamp.offset == 32
    assert C.sizeof(FrameInfo) == 136
    assert Frame.info.offset == 16
    assert C.sizeof(Frame) == 192


@pytest.mark.parametrize("software", [False, True])
def test_capture_copies_padded_buffer_restores_settings_and_releases(software):
    sdk = FakeSDK()
    with IMVCamera(_dll=sdk) as camera:
        assert camera.enumerate()[0].ip == "192.168.10.2"
        frame = camera.capture("Dahua:123", software_trigger=software, exposure_us=1000.)
    np.testing.assert_array_equal(frame.pixels, [[1, 2, 3], [4, 5, 6]])
    assert sdk.features == sdk.original
    names = [name for name, _ in sdk.calls]
    assert names.index("IMV_ReleaseFrame") < names.index("IMV_StopGrabbing")
    assert names[-2:] == ["IMV_Close", "IMV_DestroyHandle"]
    assert ("IMV_ExecuteCommandFeature" in names) == software
    assert frame.metadata["block_id"] == 42


@pytest.mark.parametrize("failure", ["IMV_Open", "IMV_SetEnumFeatureSymbol", "IMV_StartGrabbing",
                                     "IMV_ExecuteCommandFeature", "IMV_GetFrame", "IMV_ReleaseFrame",
                                     "IMV_StopGrabbing", "IMV_SetEnumFeatureValue", "IMV_Close"])
def test_failures_still_destroy_handle_and_attempt_remaining_cleanup(failure):
    sdk = FakeSDK(fail=failure)
    with pytest.raises(CameraError, match=failure):
        IMVCamera(_dll=sdk).capture("Dahua:123", software_trigger=True)
    names = [name for name, _ in sdk.calls]
    assert names[-1] == "IMV_DestroyHandle"
    if failure != "IMV_Open":
        assert "IMV_Close" in names
    if failure == "IMV_GetFrame":
        assert "IMV_StopGrabbing" in names
        assert "IMV_ReleaseFrame" not in names
        assert sdk.features == sdk.original


@pytest.mark.parametrize("overrides", [{"status": 1}, {"pixel_format": 0}, {"size": 2},
                                      {"width": 0}, {"width": 9000}, {"size": 100_000_000}])
def test_bad_frame_rejected_and_buffer_released(overrides):
    sdk = FakeSDK()
    sdk.frame_overrides = overrides
    with pytest.raises(CameraError, match="Frame"):
        IMVCamera(_dll=sdk).capture("Dahua:123")
    assert "IMV_ReleaseFrame" in [name for name, _ in sdk.calls]


def test_stale_device_and_invalid_timeout_do_not_open():
    sdk = FakeSDK()
    with pytest.raises(CameraError, match="mất kết nối"):
        IMVCamera(_dll=sdk).capture("missing")
    with pytest.raises(ValueError):
        IMVCamera(_dll=sdk).capture("Dahua:123", timeout_ms=-1)
    assert "IMV_Open" not in [name for name, _ in sdk.calls]


def test_explicit_library_and_missing_sdk_are_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("MVSDK_LIBRARY", str(tmp_path / "missing.dll"))
    with pytest.raises(CameraError, match="MVSDK_LIBRARY"):
        find_library()
    monkeypatch.delenv("MVSDK_LIBRARY")
    monkeypatch.setenv("MVSDK_ROOT", str(tmp_path))
    with pytest.raises(CameraError, match="Chưa tìm thấy SDK"):
        find_library()
    dll = tmp_path / "Runtime/x64/MVSDKmd.dll"
    dll.parent.mkdir(parents=True)
    dll.touch()
    assert find_library() == dll.resolve()


def test_full_5mp_and_roi_preserve_geometry_and_infer(fake_service):
    raw = np.arange(2592 * 1944, dtype=np.uint8).reshape(1944, 2592)
    frame = CameraFrame(raw, {"block_id": 42})
    full = IndustrialCameraSource(frame)
    assert full.get_frame()[1].shape == (960, 1280, 3)
    roi = IndustrialCameraSource(frame, roi=(100, 200, 640, 640))
    np.testing.assert_array_equal(roi.pixels[:, :, 0], raw[200:840, 100:740])
    assert roi.describe()["sensor_to_inspection_scale"] == [1., 1.]
    assert roi.identity != full.identity
    service, state = fake_service
    result = service.inspect(roi, "H01")
    assert result.decision == "DEFECT"
    assert result.metadata["roi_xywh"] == [100, 200, 640, 640]
    assert result.metadata["ground_truth_available"] is False
    bounded = IndustrialCameraSource(CameraFrame(np.zeros((4000, 4000), np.uint8), {}), max_side=4096)
    assert bounded.pixels.shape == (2000, 2000, 3)


def test_roi_outside_sensor_and_high_bit_depth_rejected():
    with pytest.raises(ValueError, match="ROI"):
        IndustrialCameraSource(CameraFrame(np.zeros((10, 10), np.uint8), {}), roi=(9, 9, 2, 2))
    buffer = BytesIO()
    Image.fromarray(np.zeros((10, 10), np.uint16)).save(buffer, "PNG")
    with pytest.raises(ValueError, match="Mono8"):
        saved_camera_frame(buffer.getvalue(), "16bit.png")


def test_saved_5mp_bmp_accepts_native_image_without_upload_limit():
    buffer = BytesIO()
    Image.new("L", (2592, 1944), 23).save(buffer, "BMP")
    frame = saved_camera_frame(buffer.getvalue(), "camera.bmp")
    assert frame.pixels.shape == (1944, 2592)
    assert IndustrialCameraSource(frame).pixels[0, 0].tolist() == [23, 23, 23]


def test_ui_missing_sdk_and_capture_reruns(fake_service, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from pcb_lab.ui import industrial_camera, resources
    from pcb_lab.inference.registry import REPO_ROOT
    service, state = fake_service
    monkeypatch.setattr(resources, "get_service", lambda: service)

    def missing():
        raise CameraError("SDK is missing")

    monkeypatch.setattr(industrial_camera, "IMVCamera", missing)
    app = AppTest.from_file(str(REPO_ROOT / "src/pcb_lab/ui/app.py"), default_timeout=30).run()
    app.radio(key="single_source").set_value("Camera công nghiệp").run()
    app.button(key="industrial_scan").click().run()
    assert not app.exception
    assert any("SDK is missing" in error.value for error in app.error)
    sdk = FakeSDK()
    monkeypatch.setattr(industrial_camera, "IMVCamera", lambda: IMVCamera(_dll=sdk))
    app.button(key="industrial_scan").click().run()
    app.button(key="industrial_capture").click().run()
    assert not app.exception
    inspect = next(button for button in app.button if button.label == "Chạy kiểm tra")
    inspect.click().run()
    assert app.session_state.single_result[1].decision != "ERROR"
    before = len(state["calls"])
    app.run()
    assert len(state["calls"]) == before
    assert [n for n, _ in sdk.calls].count("IMV_GetFrame") == 1
    sdk.fail = "IMV_GetFrame"
    app.button(key="industrial_capture").click().run()
    assert not app.exception
    assert "industrial_snapshot" not in app.session_state
    assert next(b for b in app.button if b.label == "Chạy kiểm tra").disabled
    assert not app.metric  # previous GOOD/DEFECT result must not remain visible
