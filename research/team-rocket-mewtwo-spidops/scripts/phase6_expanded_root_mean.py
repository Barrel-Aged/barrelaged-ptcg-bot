"""Run one explicit, sequential Phase 6AK artifact stage."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_expanded_root_mean import main  # noqa: E402


if __name__ == "__main__":
    main()
