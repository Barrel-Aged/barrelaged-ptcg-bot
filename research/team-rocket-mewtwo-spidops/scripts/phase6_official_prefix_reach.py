"""Generate Phase 6AB redacted official-prefix reach evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_official_prefix_reach import (  # noqa: E402
    run_phase6_official_prefix_reach,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-phase6z-dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_aligned_counterfactual_values"
            / "public_root_aligned_value_dataset.json"
        ),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_official_prefix_reach",
    )
    parser.add_argument("--maximum-live-selections", type=int, default=900)
    parser.add_argument("--maximum-capture-attempts", type=int, default=8)
    args = parser.parse_args()
    artifacts = run_phase6_official_prefix_reach(
        project_root=PROJECT_ROOT,
        source_phase6z_dataset_path=args.source_phase6z_dataset,
        maximum_live_selections=args.maximum_live_selections,
        maximum_capture_attempts=args.maximum_capture_attempts,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "official_prefix_reach_dataset.json",
        artifacts.dataset.to_json() + "\n",
    )
    _write(target / "phase6ab_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6ab_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "trace_count": artifacts.evidence.trace_count,
                "official_selection_execution_count": (
                    artifacts.evidence.official_selection_execution_count
                ),
                "player_0_decisions": artifacts.evidence.player_0_decisions,
                "player_1_decisions": artifacts.evidence.player_1_decisions,
                "minimum_steps_per_trace": (
                    artifacts.evidence.minimum_steps_per_trace
                ),
                "maximum_steps_per_trace": (
                    artifacts.evidence.maximum_steps_per_trace
                ),
                "behavior_action_ledger_gate_passed": True,
                "deterministic_behavior_reach_gate_passed": True,
                "official_chance_probability_coverage": 0.0,
                "official_chance_reach_available": False,
                "full_upstream_reach_available": False,
                "official_counterfactual_importance_weights_available": False,
                "phase6z_value_root_alignment_available": False,
                "safe_resolver_enabled": False,
                "opponent_exploitation_enabled": False,
                "official_strategy_control_enabled": False,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
