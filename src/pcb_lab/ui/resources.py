"""Streamlit resource caches keyed by provenance, shared across images/models."""
import streamlit as st

from pcb_lab.data.manifest import METADATA_PATHS, resolve_dataset_root
from pcb_lab.inference.registry import ModelRegistry, REPO_ROOT, file_stamp
from pcb_lab.inference.reports import load_benchmark
from pcb_lab.inference.service import InferenceService, load_model


@st.cache_resource(show_spinner=False)
def cached_model(asset, device, fingerprint):
    return load_model(asset, device)


def model_loader(asset, device):
    return cached_model(asset, device, asset.fingerprint())


@st.cache_resource(show_spinner=False)
def _service(repo_root, freeze_stamp, device):
    return InferenceService(ModelRegistry(repo_root), device=device, model_loader=model_loader)


def get_service(repo_root=REPO_ROOT, device="cpu"):
    return _service(str(repo_root), file_stamp(repo_root / "artifacts/test/frozen_evaluation.json"), device)


@st.cache_data(show_spinner=False)
def _gallery(repo_root, dataset_root, partition, report_mode, stamps):
    return ModelRegistry(repo_root).gallery(partition, dataset_root=dataset_root, report_mode=report_mode)


def get_gallery(partition, report_mode=False, *, repo_root=REPO_ROOT, dataset_root=None):
    root = resolve_dataset_root(dataset_root)
    stamps = tuple(file_stamp(root / p) for p in METADATA_PATHS.values())
    stamps += (file_stamp(repo_root / "artifacts/test/frozen_evaluation.json"),)
    if report_mode:
        stamps += report_stamps(repo_root)
    return _gallery(str(repo_root), str(root), partition, report_mode, stamps)


def report_stamps(repo_root):
    return tuple(file_stamp(repo_root / "artifacts/test" / name) for name in (
        "frozen_evaluation.json", "benchmark_summary.csv", "benchmark_per_group.csv", "benchmark_per_class.csv"))


@st.cache_data(show_spinner=False)
def _benchmark(repo_root, stamps):
    return load_benchmark(repo_root)


def get_benchmark(repo_root=REPO_ROOT):
    return _benchmark(str(repo_root), report_stamps(repo_root))
