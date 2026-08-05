#!/usr/bin/env python3
"""Run the single preregistered N13 calibrated norm2 adapter path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_n12_counterfactual_margin_adapter as trainer  # noqa: E402


PROTOCOL = "N13_CALIBRATED_NORM2_ADAPTER_PROTOCOL.md"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n13_calibrated_norm2_adapter/training_v1"
LEARNING_RATE = 5.0e-5


def locked_args() -> argparse.Namespace:
    return argparse.Namespace(
        candidate=trainer.BASE,
        corpus=trainer.CORPUS,
        audit=trainer.AUDIT,
        replays=trainer.REPLAYS,
        parity_report=trainer.PARITY,
        output=OUTPUT,
        steps=400,
        anchor_batch_size=128,
        learning_rate=LEARNING_RATE,
        correction_margin=0.10,
        retention_cap=0.25,
        softplus_temperature=0.25,
        n6_kl_coef=1.0,
        n9_kl_coef=1.0,
        n6_retention_coef=0.25,
        n9_retention_coef=0.25,
        displacement_coef=0.01,
        gradient_cap=0.25,
        seed=812101,
        threads=4,
    )


def validate_n13(args: argparse.Namespace) -> None:
    expected = locked_args()
    for key, value in vars(expected).items():
        if getattr(args, key) != value:
            raise ValueError(f"N13 locked argument changed: {key}")


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N13 accepts no command-line overrides")
    args = locked_args()
    trainer.PROTOCOL = PROTOCOL
    trainer.PREDECESSOR_PROTOCOL = "N12_COUNTERFACTUAL_MARGIN_RESULT.md"
    trainer.parse_args = lambda: args
    trainer.validate_args = validate_n13
    result = trainer.main()
    report_path = OUTPUT / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema"] = "rocket-n13-calibrated-norm2-training/v1"
    report["calibration"] = {
        "source": "N12_COUNTERFACTUAL_MARGIN_RESULT.md",
        "single_path": True,
        "learning_rate_multiplier_over_n12": 100.0,
        "invalid_n12_v2_partial_read": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n13_report": str(report_path), "admitted": report["admitted"]}))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
