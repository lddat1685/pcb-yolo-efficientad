"""Spec-first YOLO view tests; Step 3 implementation is imported only when run."""

import errno
from pathlib import Path

import pytest

from helpers.step3_spec import (CLASSES, assert_view, dataset_snapshot, field, records, sha256)


@pytest.mark.parametrize("partitions", [("fusion",), ("test",), ("train", "test"),
                                        ("calibration", "fusion")])
def test_view_rejects_forbidden_partitions(synthetic_dataset_root, tmp_path, partitions):
    # Protect plan §1.2/§6 and contract view.py: no fusion/test training or validation.
    from pcb_lab.models.yolo.view import build_yolo_view
    out = tmp_path / "forbidden"
    before = dataset_snapshot(synthetic_dataset_root)
    try:
        with pytest.raises(PermissionError):
            build_yolo_view(synthetic_dataset_root, out, partitions=partitions)
        assert not out.exists() or not any(out.iterdir())
    finally:
        assert dataset_snapshot(synthetic_dataset_root) == before


def test_view_schema_labels_and_dataset_unchanged(synthetic_dataset_root, tmp_path):
    # Protect plan §1.3/§11 and contract invariants 1-2: classes, labels, no source writes/cache.
    from pcb_lab.models.yolo.view import build_yolo_view
    before = dataset_snapshot(synthetic_dataset_root)
    try:
        view = build_yolo_view(synthetic_dataset_root, tmp_path / "view")
        assert_view(synthetic_dataset_root, tmp_path / "view", view)
    finally:
        assert dataset_snapshot(synthetic_dataset_root) == before


def test_view_copy_and_hardlink_have_identical_content(synthetic_dataset_root, tmp_path):
    # Protect contract view.py: stable signature, explicit copy, bytes equal to hardlinks.
    from pcb_lab.models.yolo.view import build_yolo_view
    before = dataset_snapshot(synthetic_dataset_root)
    try:
        views = []
        for mode in ("hardlink", "copy"):
            out = tmp_path / mode
            view = build_yolo_view(synthetic_dataset_root, out, link=mode)
            assert field(view, "link_mode") == mode  # same-volume synthetic fixture
            assert_view(synthetic_dataset_root, out, view)
            payload = {p.relative_to(out).as_posix(): sha256(p)
                       for folder in ("images", "labels") for p in (out / folder).rglob("*")
                       if p.is_file()}
            views.append((view, payload))
        assert views[0][1] == views[1][1]
        assert field(views[0][0], "view_signature") == field(views[1][0], "view_signature")
        again = build_yolo_view(synthetic_dataset_root, tmp_path / "again")
        assert field(again, "view_signature") == field(views[0][0], "view_signature")
    finally:
        assert dataset_snapshot(synthetic_dataset_root) == before


def test_view_limit_selects_first_ids_and_rebuild_removes_stale(synthetic_dataset_root, tmp_path):
    # Protect contract view limit/reuse: exact first n good/m defect; no stale full-view images.
    from pcb_lab.models.yolo.view import build_yolo_view
    out = tmp_path / "view"
    before = dataset_snapshot(synthetic_dataset_root)
    try:
        build_yolo_view(synthetic_dataset_root, out)
        view = build_yolo_view(synthetic_dataset_root, out, limit={"good": 1, "defect": 1})
        rows = records(synthetic_dataset_root)
        for part in ("train", "calibration"):
            selected = [sorted((r for r in rows if r["split"] == part and r["is_defect"] == defect),
                               key=lambda r: r["sample_id"])[0] for defect in (False, True)]
            assert {sha256(p) for p in (out / "images" / part).iterdir()} == {r["sha256"] for r in selected}
            assert len(list((out / "labels" / part).iterdir())) == 2
            assert field(view, "counts")[part] == {"images": 2, "good": 1, "defect": 1}
    finally:
        assert dataset_snapshot(synthetic_dataset_root) == before


def test_view_cross_volume_error_falls_back_to_copy(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect contract view.py: simulated EXDEV tests fallback without requiring a second drive.
    import os
    from pcb_lab.models.yolo.view import build_yolo_view
    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "simulated cross-volume hardlink")
    monkeypatch.setattr(os, "link", cross_device)
    view = build_yolo_view(synthetic_dataset_root, tmp_path / "view")
    assert field(view, "link_mode") == "copy"
    assert_view(synthetic_dataset_root, tmp_path / "view", view)


def test_view_signature_changes_when_one_fixture_image_changes(tmp_path):
    # Protect plan §11 provenance and contract view_signature: include image content, not only IDs.
    from helpers.dataset_builder import build_clean_fixture, make_png_with_text
    from pcb_lab.models.yolo.view import build_yolo_view
    # Separate synthetic roots avoid writing through hardlinks to the first view.
    first = build_clean_fixture(tmp_path / "dataset-one")
    second = build_clean_fixture(tmp_path / "dataset-two")
    selected = next(row for row in second.records if row["split"] == "train")
    second.overwrite_record_image(selected["sample_id"], make_png_with_text(text="changed-pixel-fixture"))
    from helpers.dataset_builder import pixel_sha_of_png
    selected["pixel_sha256"] = pixel_sha_of_png(second.root / selected["image"])
    second.write_manifest()
    second.write_release()
    a = build_yolo_view(first.root, tmp_path / "view-one", link="copy")
    b = build_yolo_view(second.root, tmp_path / "view-two", link="copy")
    assert field(a, "view_signature") != field(b, "view_signature")


@pytest.mark.dataset
def test_real_view_counts_boxes_isolation_and_read_only(real_dataset_root, tmp_path):
    # Protect plan §1.2-1.3 and contract invariant 1; held-out hashes come from metadata only.
    from pcb_lab.models.yolo.view import build_yolo_view
    before = dataset_snapshot(real_dataset_root)
    try:
        view = build_yolo_view(real_dataset_root, tmp_path / "view")
        assert_view(real_dataset_root, tmp_path / "view", view)
        assert field(view, "counts") == {
            "train": {"images": 1792, "good": 895, "defect": 897},
            "calibration": {"images": 460, "good": 230, "defect": 230}}
        assert field(view, "box_counts")["train"] == dict(zip(CLASSES, [1216, 973, 1228, 981, 884, 865]))
        again = build_yolo_view(real_dataset_root, tmp_path / "again", link="copy")
        assert_view(real_dataset_root, tmp_path / "again", again)
        assert field(again, "view_signature") == field(view, "view_signature")
    finally:
        assert dataset_snapshot(real_dataset_root) == before
