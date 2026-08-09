"""Run the source checkout's deterministic Phase 2 demo without installing."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase2_demo import main  # noqa: E402


if __name__ == "__main__":
    main()
