# IMV GigE camera integration verification — 2026-10-05

Target: Dahua/iRAYPLE A3504MG100 on Ethernet. Development machine: Windows x64,
Python 3.11 CPU environment. **No MV Viewer installation and no physical camera
on this machine. Hardware acquisition remains unverified.**

Implemented snapshot acquisition through the IMV SDK ABI (official SDK 2.4.1
Build20231102 headers), GigE enumeration, optional software trigger/manual
exposure, bounded Mono8 buffer copy, parameter restoration and resource cleanup.
Streamlit supports capture, raw PNG download, ROI/resize and inference. Saved
Mono8/RGB8 camera files can also be processed without the SDK. The inference
service now checks artifact/checkpoint class labels without requiring dataset
files on the camera PC. Evaluation's default canonical-dataset guard is retained.

## Executed checks

| Check | Result |
|---|---|
| `pytest tests/test_industrial_camera.py -q` | 24 passed; simulated SDK, ABI, failure cleanup, padded buffers, image geometry and UI state |
| `pytest tests -q -ra --junitxml=reports/camera_integration/pytest.xml` | 461 passed, 24 skipped, 0 failed; 223.33 seconds |
| `scripts/verify_reproducibility.py` | 60 PASS, 0 FAIL, 0 MISSING |
| `scripts/verify_step8_parity.py` | 10 cases PASS; 15 fields exactly equal between CLI and UI per case |
| Real checkpoint smoke with unavailable `DATASET_ROOT` and dataset lookup forbidden | B01, B02, B03 and H01 successfully inferred on a synthetic Mono8 640×640 image |
| `scripts/camera_probe.py` and `run_camera_check.bat` | Correct missing-SDK message and exit code 1; no false camera success |
| `git diff --check` | Passed |

JUnit details: [pytest.xml](pytest.xml). The 24 skips consist of 9 missing augment
entry-point cases, 14 opt-in artifact cases without their environment variables,
and 1 missing teacher checkpoint case. Skips are not passes. Real checkpoint
smoke proves loading/inference only, not camera-domain detection accuracy.

All frozen operating points, protected model weights and benchmark artifacts
remain unchanged. Camera capture does not write firmware, IP configuration or
persistent UserSet. Acquisition errors are shown as errors and invalidate the
current snapshot; they do not produce a GOOD decision.

## Required hardware acceptance on the camera PC

Follow [CAMERA_DAHUA.md](../../docs/CAMERA_DAHUA.md): install the vendor x64 SDK,
verify an image in MV Viewer, disconnect its camera control, enumerate the exact
model/serial with `run_camera_check.bat`, and capture in the UI. Confirm free-run,
software trigger, exposure, unplug/timeout recovery, ROI coordinates and restored
settings on the real firmware. Collect labeled camera images before assessing
accuracy. No hardware acceptance claim is made by this report.
