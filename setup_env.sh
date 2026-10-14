#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "$(uname -s)" == Darwin && "$(uname -m)" != arm64 ]]; then
    echo 'This pinned PyTorch demo requires Apple Silicon with an ARM64 Python on macOS.' >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    if [[ -n "${PCB_PYTHON:-}" ]]; then
        "$PCB_PYTHON" -m venv .venv
    else
        candidate=""
        for version in python3.11 python3.12 python3; do
            if command -v "$version" >/dev/null 2>&1 && "$version" -c 'import sys; sys.exit(not ((3,11) <= sys.version_info[:2] < (3,13)))'; then
                candidate="$version"
                break
            fi
        done
        [[ -n "$candidate" ]] || { echo 'Install Python 3.11/3.12 with venv support first.' >&2; exit 1; }
        "$candidate" -m venv .venv
    fi
fi
source .venv/bin/activate
python -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), "Use Python 3.11 or 3.12; rename an incompatible .venv before setup."'
python -m ensurepip --upgrade
python -m pip install --upgrade pip
if [[ "$(uname -s)" == Darwin ]]; then
    # macOS wheels are published on PyPI, not the Linux/Windows CPU index.
    python -m pip install torch==2.6.0 torchvision==0.21.0
else
    python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
python scripts/verify_reproducibility.py
echo 'Setup complete. Run bash run_ui.sh to open the demo.'
