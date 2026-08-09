"""Run the deterministic Phase 4 selective-search benchmark."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase4_demo import main  # noqa: E402


if __name__ == "__main__":
    main()
