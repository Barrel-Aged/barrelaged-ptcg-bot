"""Run the locked Phase 6AP monotone-gate recovery."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.integrated_monotone_gate import train_phase6ap_gate  # noqa: E402


def main() -> None:
    integrated = PROJECT_ROOT / "artifacts" / "integrated_controller"
    rejected = integrated / "integrated_planner_phase6ao_rejected.json"
    rejected.write_bytes((integrated / "integrated_planner.json").read_bytes())
    report = train_phase6ap_gate(
        base_model_path=integrated / "integrated_planner_phase6an.json",
        development_path=PROJECT_ROOT / "artifacts" / "phase6_expected_value_surfaces" / "expected_value_surface_dataset.json",
        additional_development_path=PROJECT_ROOT / "artifacts" / "phase6_high_precision_confirmation" / "calibration_panel.json",
        selection_path=PROJECT_ROOT / "artifacts" / "phase6_high_precision_confirmation" / "validation_panel.json",
        validation_path=PROJECT_ROOT / "artifacts" / "phase6_public_range_counterfactual_advantages" / "validation_panel.json",
        output_path=integrated / "integrated_planner.json",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
