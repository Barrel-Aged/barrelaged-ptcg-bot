"""Run the source checkout's deterministic demo without installing it."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.demo import main  # noqa: E402


if __name__ == "__main__":
    main()
