"""Phase A — Spec-first fault injections for audit C02, M01, m02.

Ba test này PHẢI ĐỎ trên HEAD hiện tại (chưa fix) và XANH sau khi merge
remediation/impl. Mục đích: chống PASS giả khi export hỏng (C02),
bắt buộc đọc frozen (M01), và escape LaTeX (m02).

Thiết kế tương thích xuôi: tự phát hiện API HEAD vs API fixed (registry,
verify_parity, run_test_evaluation) và chọn đường mock phù hợp, nên cùng
một file chạy được trên cả hai trạng thái.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Test 1 — Chống PASS giả khi export lỗi (C02)
# Truyền mô hình export hỏng vào export_runtime.verify_parity,
# assert phải báo FAILED, cấm passed=True hoặc matches=1/1.
# ---------------------------------------------------------------------------

def _fake_registry_with_ad(tmp_path: Path):
    """Tạo ModelRegistry tối thiểu có ad backend để verify có ý nghĩa."""
    # Tái sử dụng logic fake_service của test_step8_ui để không phụ thuộc dataset thật.
    from pcb_lab.inference.registry import ModelRegistry
    from pcb_lab.inference.service import InferenceService, LoadedModel
    from pcb_lab.inference.tiling import TileManager
    from pcb_lab.models.yolo.adapter import Detection

    frozen_src = REPO / "artifacts/test/frozen_evaluation.json"
    frozen = json.loads(frozen_src.read_text(encoding="utf-8-sig"))
    target = tmp_path / "artifacts/test/frozen_evaluation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(frozen), encoding="utf-8")
    for entry in frozen["checkpoints"].values():
        ckpt = tmp_path / entry["path"]
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"fixture-ckpt")
        (ckpt.parent / "artifact.json").write_text("{}")

    state: dict = {"score": 0.9, "ad": 2.0, "loads": [], "calls": []}

    class Detector:
        tile_manager = TileManager()

        def __init__(self, asset):
            self.asset = asset

        def predict(self, pixels):
            name = self.asset.model_id
            state["calls"].append(name)
            if name.startswith("B03"):
                return SimpleNamespace(
                    anomaly_score=state["ad"],
                    anomaly_map=np.full(pixels.shape[:2], state["ad"], dtype=np.float32),
                    boxes=[],
                )
            return [Detection(0, "open_circuit", state["score"], (2.0, 3.0, 20.0, 30.0))]

    def loader(asset, device):
        state["loads"].append(asset.model_id)
        return LoadedModel(Detector(asset))

    registry = ModelRegistry(tmp_path)
    service = InferenceService(registry, model_loader=loader)
    return registry, service, state


def _load_verify():
    spec = importlib.util.spec_from_file_location("export_runtime", str(REPO / "scripts/export_runtime.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    return mod.verify_parity, mod


def test_export_broken_must_report_failed(tmp_path, monkeypatch):
    """C02: export hỏng (load hoặc inference) phải FAILED, không được passed=True 1/1."""
    verify, mod = _load_verify()
    sig = inspect.signature(verify)
    has_exported_param = "exported_ad_path" in sig.parameters

    # Dùng registry/service giả để không cần dataset/ckpt thật và không chạm đĩa thật.
    registry, service, state = _fake_registry_with_ad(tmp_path / "exp_case")

    # Sample tối thiểu — GalleryImageSource sẽ bị mock nên sample_id là đủ.
    # Với fixed API, Sample cần sample_id; với HEAD, profile_inspect dùng Sample.load_image
    # nên cũng cần mock GalleryImageSource.
    from io import BytesIO
    from PIL import Image
    from pcb_lab.inference.sources import UploadImageSource

    def png_bytes(size=(256, 256)):
        buf = BytesIO()
        Image.new("RGB", size, (128, 128, 128)).save(buf, "PNG")
        return buf.getvalue()

    dummy_samples = [SimpleNamespace(sample_id="synthetic_good", sample_id_="x")]

    # Monkeypatch GalleryImageSource -> UploadImageSource để bypass dataset.
    # HEAD dùng GalleryImageSource(sample) -> profile_inspect -> source.get_frame()
    # Fixed dùng GalleryImageSource(sample) -> service.inspect -> source.get_frame()
    monkeypatch.setitem(verify.__globals__, "GalleryImageSource",
                        lambda sample: UploadImageSource(png_bytes()))

    # Service patch: dồn mọi InferenceService(...) trong verify về service giả.
    # verify tạo InferenceService(registry) nội bộ; ta thay bằng service đã mock loader.
    from pcb_lab.inference.service import InferenceService as RealService

    def fake_service_factory(reg, **kw):
        # kw có thể chứa model_loader; nếu có thì tôn trọng, nếu không dùng loader giả.
        loader = kw.get("model_loader")
        if loader is not None:
            return RealService(reg, model_loader=loader, **{k: v for k, v in kw.items() if k != "model_loader"})
        return service

    monkeypatch.setitem(verify.__globals__, "InferenceService", fake_service_factory)

    # --- Trường hợp 1: load hỏng ---
    if has_exported_param:
        # Fixed API: truyền file hỏng trực tiếp.
        broken = tmp_path / "broken_export.pt"
        broken.write_bytes(b"not-a-valid-torchscript-archive")
        result = verify(dummy_samples, "B03", registry, None, None, n_samples=1, exported_ad_path=broken)
        parity = result["parity"]
        assert parity.get("passed") is False, f"broken load must not pass, got {parity}"
        # Cấm 1/1
        total = result.get("total", 1)
        assert parity.get("decision_matches", 0) != total or parity.get("decision_matches", 0) == 0, \
            f"broken load must not report matches=1/1, got {parity}"
        assert parity.get("status") == "FAILED", f"broken load status must be FAILED, got {parity}"
        assert parity.get("failed_samples", 1) >= 1 or parity.get("failures"), \
            f"broken load must record failures, got {parity}"
    else:
        # HEAD API: không có exported_ad_path, phải patch torch.jit.load để mô phỏng hỏng.
        # Trên HEAD bug sẽ trả passed=True 1/1 — test này sẽ ĐỎ đúng như kỳ vọng.
        with patch("torch.jit.load", side_effect=RuntimeError("injected invalid TorchScript archive")):
            # HEAD also loads from REPO/artifacts/runtime/... ; patch covers it.
            result = verify(dummy_samples, "B03", registry, None, None, n_samples=1)
        parity = result["parity"]
        assert parity.get("passed") is False, \
            f"C02 BUG: broken TorchScript load returned passed={parity.get('passed')} matches={parity.get('decision_matches')} — expected FAILED"
        assert parity.get("decision_matches", 0) == 0, \
            f"C02 BUG: broken load must not report 1/1 matches, got {parity}"

    # --- Trường hợp 2: inference exception (nếu có EfficientAdEngine) ---
    # Trên HEAD, verify bắt exception rồi reset delta=0 và copy decision -> PASS giả.
    # Trên fixed, nó phải FAILED.
    if has_exported_param:
        # Fixed: tạo target hợp lệ nhưng patch engine để infer lỗi.
        valid_target = tmp_path / "valid_target.pt"

        class Target(torch.nn.Module):
            def forward(self, x):
                return x[:, 0] * 0

        torch.jit.trace(Target(), torch.zeros(1, 3, 256, 256), check_trace=False).save(str(valid_target))
        # Patch EfficientAdEngine.infer_tiles trong module verify để ném lỗi khi pipeline chạy.
        orig_engine = verify.__globals__.get("EfficientAdEngine")
        if orig_engine is not None:
            with patch.object(orig_engine, "infer_tiles", side_effect=RuntimeError("injected exported inference error")):
                result2 = verify(dummy_samples, "B03", registry, None, None, n_samples=1, exported_ad_path=valid_target)
            parity2 = result2["parity"]
            assert parity2.get("passed") is False, f"inference exception must not pass, got {parity2}"
            assert parity2.get("status") == "FAILED", f"inference exception status must be FAILED, got {parity2}"
    else:
        # HEAD: patch EfficientAdEngine.infer_tiles nếu tồn tại, hoặc patch profile_inspect path.
        # HEAD's per-sample try/except cũng swallow lỗi và trả PASS giả, nên assert sẽ ĐỎ.
        try:
            from pcb_lab.models.efficientad.adapter import EfficientAdEngine
            has_engine = True
        except Exception:
            has_engine = False
        if has_engine and "EfficientAdEngine" in verify.__globals__:
            with patch.object(verify.__globals__["EfficientAdEngine"], "infer_tiles",
                              side_effect=RuntimeError("injected exported inference error")):
                # Cần đảm bảo _ts_model is not None để đi vào nhánh AD -> cần file TS hợp lệ tồn tại.
                # Tạo file tạm và patch REPO path? Đơn giản: không patch load, để nó load file thật (B03 có thật)
                # nhưng inference sẽ ném.
                result2 = verify(dummy_samples, "B03", registry, None, None, n_samples=1)
            parity2 = result2["parity"]
            # Trên HEAD bug, parity2 vẫn passed=True -> test ĐỎ.
            assert parity2.get("passed") is False, \
                f"C02 BUG: inference exception returned passed={parity2.get('passed')} — expected FAILED"


# ---------------------------------------------------------------------------
# Test 2 — Bắt buộc đọc từ Frozen Config (M01)
# Truyền synthetic frozen có tau_yolo=0.99 vào evaluator, assert evaluator
# tôn trọng 0.99, không dùng ngưỡng mặc định 0.14.
# ---------------------------------------------------------------------------

def test_evaluator_must_use_frozen_tau_not_default(tmp_path, monkeypatch):
    """M01: frozen tau_yolo=0.99 với score 0.5 phải cho 0 DEFECT B01 (HEAD bug cho 440)."""
    import pcb_lab.evaluation.test_eval as ev

    sig = inspect.signature(ev.run_test_evaluation)
    is_fixed = "output_dir" in sig.parameters or "timing_rounds" in sig.parameters

    base_frozen = json.loads((REPO / "artifacts/test/frozen_evaluation.json").read_text(encoding="utf-8-sig"))

    if is_fixed:
        # ---- Fixed API: dùng fake_service + tmp dataset + Registry ----
        from pcb_lab.inference.service import InferenceService

        frozen = copy.deepcopy(base_frozen)
        frozen["operating_points"]["B01"]["tau_yolo"] = 0.99

        # Tạo fake repo/dataset với 440 ids và checkpoint giả
        case_root = tmp_path / "frozen_case"
        case_root.mkdir()
        for entry in frozen["checkpoints"].values():
            p = case_root / entry["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fixture-ckpt")
            entry["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            (p.parent / "artifact.json").write_text("{}")
        ids = [f"sample_{i}_{'defect' if i % 2 else 'good'}" for i in range(440)]
        frozen["test_split_sha256"] = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()

        ds_root = case_root / "dataset"
        holdout = ds_root / "benchmarks/deeppcb/splits/holdout.json"
        holdout.parent.mkdir(parents=True, exist_ok=True)
        holdout.write_text(json.dumps({"samples": {"test": ids}, "groups": {}, "pair_ids": {}}, ensure_ascii=False))

        frozen_path = case_root / "frozen.json"
        frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

        from pcb_lab.inference.registry import ModelRegistry
        from pcb_lab.inference.sources import UploadImageSource
        from pcb_lab.inference.tiling import TileManager
        from pcb_lab.models.yolo.adapter import Detection

        # Tái sử dụng make_fake logic: cần InferenceService với loader trả score 0.5
        # Dùng trực tiếp fake_service helper bằng cách gọi fixture function
        # Thay vì import fixture, tự tạo service:
        state: dict = {"score": 0.5, "ad": 0.1, "calls": []}

        class Det:
            tile_manager = TileManager()

            def __init__(self, asset):
                self.asset = asset

            def describe(self):
                return {"infer_params": {"tile_size": 256, "stride": 224, "tau_pixel": 0.5, "min_area": 10, "nms_iou": 0.3}}

            def predict(self, pixels):
                name = self.asset.model_id
                state["calls"].append(name)
                if name.startswith("B03"):
                    return SimpleNamespace(anomaly_score=state["ad"],
                                           anomaly_map=np.zeros(pixels.shape[:2], dtype=np.float32),
                                           boxes=[])
                return [Detection(0, "open_circuit", state["score"], (2.0, 3.0, 20.0, 30.0))]

        def loader(asset, device):
            from pcb_lab.inference.service import LoadedModel
            return LoadedModel(Det(asset))

        # Patch ManifestDataset để trả 440 sample giả
        samples = [SimpleNamespace(sample_id=sid,
                                   source_group="group44000" if i < 200 else "group92000",
                                   is_defect=sid.endswith("defect"), boxes=[])
                   for i, sid in enumerate(ids)]

        monkeypatch.setattr("pcb_lab.data.samples.ManifestDataset", lambda *a, **kw: samples)

        class Src(UploadImageSource):
            def __init__(self, sample):
                self.sample = sample

            def get_frame(self):
                return self.sample.sample_id, np.zeros((32, 32, 3), dtype=np.uint8)

            def describe(self):
                return {"split": "test", "ground_truth": "DEFECT" if self.sample.is_defect else "GOOD"}

        monkeypatch.setattr("pcb_lab.inference.sources.GalleryImageSource", Src)

        # Patch InferenceService để dùng loader giả
        monkeypatch.setattr("pcb_lab.inference.service.InferenceService",
                            lambda registry, **kw: InferenceService(registry, model_loader=loader, **kw))

        # Import lại ev sau patch để đảm bảo nó dùng patched service? Không cần, run_test_evaluation
        # đã import InferenceService bên trong hàm nên patch trên module service sẽ có hiệu lực.
        out = case_root / "evaluation"
        result = ev.run_test_evaluation(frozen_path, ds_root, case_root,
                                        warmup_runs=1, timing_rounds=1, output_dir=out)
        assert result["predictions_count"] == 440 * 7, "fixed evaluator must persist 3080 rows (7 models × 440)"
        rows = [json.loads(line) for line in (out / "predictions_test.jsonl").read_text(encoding="utf-8").splitlines()]
        b01_defects = sum(1 for r in rows if r["model_id"] == "B01" and r["decision"] == "DEFECT")
        assert b01_defects == 0, \
            f"M01 BUG: B01 tau 0.99 với score 0.5 phải 0 DEFECT, got {b01_defects} (đang dùng ngưỡng mặc định 0.14?)"
        # H01 giữ tau_yolo=0.14 riêng (không đổi) nên với score 0.5 vẫn early-exit 440/440.
        # Điều này chứng minh evaluator đọc ĐÚNG từng ngưỡng frozen theo model_id, không trộn lẫn.
        h01_early = sum(1 for r in rows if r["model_id"] == "H01" and r.get("early_exit"))
        assert h01_early == 440, f"H01 phải early_exit 440/440 (tau 0.14), got {h01_early}"
        assert all(r.get("anomaly_score") is None for r in rows if r["model_id"] == "H01"), \
            "H01 early-exit không được có anomaly_score"

    else:
        # ---- HEAD API: patch YoloDetector/EfficientAdDetector/Image.open ----
        fake_root = tmp_path / "synthetic_evaluation"
        (fake_root / "artifacts/test").mkdir(parents=True, exist_ok=True)
        frozen = copy.deepcopy(base_frozen)
        frozen["operating_points"]["B01"]["tau_yolo"] = 0.99
        frozen_path = fake_root / "frozen.json"
        frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

        # Chuẩn bị holdout/manifest giả để HEAD đọc được (dù sẽ patch loader nên không cần ảnh thật,
        # nhưng HEAD vẫn mở holdout.json và samples.jsonl trước khi patch loader).
        # Tạo dataset giả tối thiểu.
        ds_root = tmp_path / "head_ds"
        for sub in ["benchmarks/deeppcb/splits", "benchmarks/deeppcb/manifests"]:
            (ds_root / sub).mkdir(parents=True, exist_ok=True)
        ids = [f"sample_{i}_{'defect' if i % 2 else 'good'}" for i in range(440)]
        # HEAD holdout: samples.test = ids
        (ds_root / "benchmarks/deeppcb/splits/holdout.json").write_text(
            json.dumps({"samples": {"test": ids}}, ensure_ascii=False), encoding="utf-8")
        # HEAD manifest: mỗi dòng có sample_id, image, boxes, is_defect, source_group
        man_path = ds_root / "benchmarks/deeppcb/manifests/samples.jsonl"
        with man_path.open("w", encoding="utf-8") as f:
            for sid in ids:
                row = {
                    "sample_id": sid,
                    "split": "test",
                    "image": "images/dummy.png",
                    "boxes": [],
                    "is_defect": sid.endswith("defect"),
                    "source_group": "group44000",
                }
                f.write(json.dumps(row) + "\n")

        # Tạo checkpoint giả để head không fail khi resolve path? Thực tế head không kiểm tra file,
        # chỉ truyền parent vào from_artifact nên patch sẽ bỏ qua. Nhưng vẫn cần artifact.json?
        for key, ent in frozen["checkpoints"].items():
            p = fake_root / ent["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fake-ckpt")
            (p.parent / "artifact.json").write_text("{}")

        det = SimpleNamespace(class_id=0, class_name="open_circuit", confidence=0.5,
                              xyxy_original=(1, 1, 10, 10))
        model = SimpleNamespace(predict=lambda image: [det])
        fake_image = SimpleNamespace(convert=lambda mode: object())

        import contextlib
        import io

        with patch.object(ev.YoloDetector, "from_artifact", return_value=model), \
             patch.object(ev.EfficientAdDetector, "from_artifact", return_value=object()), \
             patch.object(ev, "_run_efficientad_fast", return_value=(0.0, [])), \
             patch.object(ev.Image, "open", return_value=fake_image), \
             contextlib.redirect_stdout(io.StringIO()):
            ev.run_test_evaluation(frozen_path, dataset_root=ds_root, repo_root=fake_root, warmup_runs=0)

        pred_path = fake_root / "artifacts/test/predictions_test.jsonl"
        assert pred_path.is_file(), "HEAD evaluator must write predictions_test.jsonl"
        rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()]
        b01_defects = sum(1 for r in rows if r["model_id"] == "B01" and r["decision"] == "DEFECT")
        # Trên HEAD bug: dùng OPERATING_POINTS 0.14 -> 0.5 >= 0.14 -> 440 DEFECT
        # Kỳ vọng sau fix: 0 DEFECT
        assert b01_defects == 0, \
            f"M01 BUG: B01 synthetic tau 0.99 + score 0.5 phải 0 DEFECT, got {b01_defects} — evaluator đang bỏ qua frozen (dùng 0.14)"


# ---------------------------------------------------------------------------
# Test 3 — LaTeX Escape (m02)
# Kiểm tra các tệp .tex trong exports/thesis/ không chứa "_" chưa escape.
# ---------------------------------------------------------------------------

def test_thesis_tex_no_unescaped_underscore():
    """m02: H01_sec phải viết H01\\_sec trong mọi .tex (text mode tabular)."""
    thesis_dir = REPO / "exports/thesis"
    assert thesis_dir.is_dir(), f"thesis dir missing: {thesis_dir}"
    tex_files = sorted(thesis_dir.glob("*.tex"))
    assert tex_files, "no .tex files found in exports/thesis"

    unescaped_re = re.compile(r"(?<!\\)_")
    failures: list[str] = []

    for path in tex_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for idx, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if not stripped or stripped.startswith("%"):
                continue
            # Bỏ comment inline: phần sau % không phải text mode.
            content = line.split("%", 1)[0]
            # Bỏ qua dòng chứa \verb hoặc $...$ nếu có (không cần cho repo hiện tại nhưng an toàn)
            # Đơn giản: nếu dòng chứa _, kiểm tra từng occurrence không có \ trước.
            if unescaped_re.search(content):
                # Cho phép các dòng chỉ chứa comment LaTeX escape? Không.
                # Ghi nhận lỗi, show context.
                failures.append(f"{path.name}:{idx}: unescaped '_' in: {line.strip()!r}")

    assert not failures, "m02 LaTeX escape failed — found unescaped '_' in thesis .tex:\n" + "\n".join(failures)

    # Thêm kiểm p95: overall_comparison phải có cột p95 sau fix (không bắt buộc đỏ, nhưng đảm bảo bản fix đủ cột)
    # Trên HEAD cũ thiếu p95; after fix có 11 cột với p95. Không fail test nếu thiếu, chỉ cảnh báo cho Phase B.
    # Nhưng để test spec-first phát hiện thiếu, ta chỉ check escape; p95 sẽ được verify ở Phase B bằng file diff.
