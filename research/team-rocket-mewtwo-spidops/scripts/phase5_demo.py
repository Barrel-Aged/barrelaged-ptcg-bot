"""Run the deterministic Phase 5 search-guided learning benchmark."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase5_demo import main


if __name__ == "__main__":
    main()
