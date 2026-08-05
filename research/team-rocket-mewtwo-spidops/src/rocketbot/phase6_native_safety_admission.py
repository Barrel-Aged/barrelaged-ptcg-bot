"""Phase 6Q durable fail-closed native safety-admission evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .native_plan_blueprint import NativePlanBlueprintArtifact, NativePlanOutcomeDataset
from .native_plan_safety_admission import (
    NativePlanSafetyAdmission,
    evaluate_native_plan_safety_admission,
)
from .phase6_native_plan_iterations import Phase6NativePlanIterationsEvidence


PHASE6_NATIVE_SAFETY_ADMISSION_SCHEMA_VERSION = 1


def _fingerprint(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Phase6NativeSafetyAdmissionEvidence:
    benchmark_id: str
    status: str
    admission: NativePlanSafetyAdmission
    counterfactual_bounds_built: bool
    resolver_executed: bool
    exploitation_executed: bool
    official_strategy_control_enabled: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6Q evidence identity is invalid")
        if self.admission.admitted:
            raise ValueError("checked Phase 6Q evidence is expected to reject admission")
        if any(
            (
                self.counterfactual_bounds_built,
                self.resolver_executed,
                self.exploitation_executed,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("rejected Phase 6Q evidence cannot execute capabilities")
        if dict(self.claims).get("phase6_complete") != "false":
            raise ValueError("Phase 6Q cannot mark Phase 6 complete")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_SAFETY_ADMISSION_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "admission": self.admission.to_dict(),
            "execution": {
                "counterfactual_bounds_built": self.counterfactual_bounds_built,
                "resolver_executed": self.resolver_executed,
                "exploitation_executed": self.exploitation_executed,
                "official_strategy_control_enabled": (
                    self.official_strategy_control_enabled
                ),
            },
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        rows = "\n".join(
            f"| {item.check_id} | {'pass' if item.passed else 'FAIL'} | "
            f"{item.observed} | {item.required} | {item.reason} |"
            for item in self.admission.checks
        )
        return f"""# Phase 6Q native safety admission

Evidence fingerprint: `{self.fingerprint}`  
Status: **{self.status}**

The Phase 6P candidate is rejected before counterfactual bounds are built.
No resolver, exploitation overlay, or official strategy control is executed.

| Check | Result | Observed | Required | Reason |
|---|---|---|---|---|
{rows}
"""


@dataclass(frozen=True, slots=True)
class Phase6NativeSafetyAdmissionArtifacts:
    admission: NativePlanSafetyAdmission
    evidence: Phase6NativeSafetyAdmissionEvidence


def run_phase6_native_safety_admission(
    *, project_root: str | Path | None = None
) -> Phase6NativeSafetyAdmissionArtifacts:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    directory = root / "artifacts" / "phase6_native_plan_iterations"
    dataset = NativePlanOutcomeDataset.from_dict(
        json.loads(
            (directory / "iterative_native_plan_dataset.json").read_text(
                encoding="utf-8"
            )
        )
    )
    model = NativePlanBlueprintArtifact.from_dict(
        json.loads(
            (directory / "iterative_native_plan_blueprint.json").read_text(
                encoding="utf-8"
            )
        )
    )
    iteration_evidence = Phase6NativePlanIterationsEvidence.from_dict(
        json.loads((directory / "phase6p_evidence.json").read_text(encoding="utf-8"))
    )
    admission = evaluate_native_plan_safety_admission(
        dataset, model, iteration_evidence
    )
    claims = (
        ("safety_admission", "rejected_as_required"),
        ("counterfactual_value_bounds", "not_built"),
        ("safe_resolver", "not_executed"),
        ("opponent_exploitation", "not_executed"),
        ("official_strategy_control", "disabled"),
        ("pokemon_strength", "not_claimed"),
        ("kaggle_submission", "not_attempted"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6NativeSafetyAdmissionEvidence(
        benchmark_id="phase6q-native-safety-admission-v1",
        status="phase6q_native_safety_admission_rejected_as_required",
        admission=admission,
        counterfactual_bounds_built=False,
        resolver_executed=False,
        exploitation_executed=False,
        official_strategy_control_enabled=False,
        claims=claims,
    )
    return Phase6NativeSafetyAdmissionArtifacts(admission, evidence)


def main() -> None:
    print(run_phase6_native_safety_admission().evidence.to_json())


if __name__ == "__main__":
    main()
