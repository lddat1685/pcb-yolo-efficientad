#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ -x .venv/bin/python ]] || { echo 'Run bash setup_env.sh first.' >&2; exit 1; }
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Streamlit opens the default browser in desktop sessions.
exec python -m streamlit run src/pcb_lab/ui/app.py --server.port 8501 --server.address localhost --server.headless false --browser.gatherUsageStats false
