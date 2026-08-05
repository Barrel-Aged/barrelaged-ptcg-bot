"""Run the preregistered Phase 6AO selective-search recovery."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.integrated_gate_training import train_phase6ao_gate  # noqa: E402


def main() -> None:
    integrated = PROJECT_ROOT / "artifacts" / "integrated_controller"
    base = integrated / "integrated_planner.json"
    preserved = integrated / "integrated_planner_phase6an.json"
    preserved.write_bytes(base.read_bytes())
    report = train_phase6ao_gate(
        base_model_path=preserved,
        development_path=PROJECT_ROOT / "artifacts" / "phase6_expected_value_surfaces" / "expected_value_surface_dataset.json",
        calibration_path=PROJECT_ROOT / "artifacts" / "phase6_high_precision_confirmation" / "calibration_panel.json",
        validation_path=PROJECT_ROOT / "artifacts" / "phase6_high_precision_confirmation" / "validation_panel.json",
        output_path=base,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
