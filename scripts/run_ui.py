"""Start the internal Streamlit UI on localhost using this Python environment."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    from streamlit.web import cli
    extra = sys.argv[1:]
    sys.argv = ["streamlit", "run", str(REPO / "src/pcb_lab/ui/app.py"),
                "--server.address=127.0.0.1", "--server.headless=true", "--browser.gatherUsageStats=false", *extra]
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
