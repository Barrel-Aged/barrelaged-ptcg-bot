"""Deterministic Phase 6H official-replay trajectory-substrate evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .competition import KAGGLE_WATCHTOWER_SUBSTITUTION, resolve_official_deck
from .official_evaluation import OFFICIAL_PANEL_DECKS, rocket_compatibility_profile
from .official_trajectory import (
    OfficialTrajectoryDataset,
    PublicRangeCandidate,
    build_official_trajectory_dataset,
)
from .public_replay import PublicReplayDataset


PHASE6_TRAJECTORY_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class Phase6TrajectoryEvidence:
    benchmark_id: str
    status: str
    source_public_replay_fingerprint: str
    trajectory_dataset_fingerprint: str
    range_updater_fingerprint: str
    matches: int
    episodes: int
    steps: int
    behavior_labeled_steps: int
    behavior_missing_steps: int
    observed_transition_links: int
    terminal_transition_links: int
    candidate_ranges: int
    maximum_complete_action_count: int
    public_feature_width: int
    private_feature_width: int
    action_feature_width: int
    eligible_future_reveal_links: int
    paired_public_frames: int
    public_state_mismatches: int
    public_history_mismatches: int
    public_range_nll: float
    uniform_range_nll: float
    public_range_brier: float
    uniform_range_brier: float
    public_range_top1_accuracy: float
    split_step_counts: tuple[tuple[str, int], ...]
    source_official_engine_executed: bool
    dataset_boundary_kind: str
    dataset_official_engine_executed: bool
    branchable_transition_game: bool
    serialization_roundtrip_exact: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6H benchmark identity/status is required")
        for field_name in (
            "source_public_replay_fingerprint",
            "trajectory_dataset_fingerprint",
            "range_updater_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        for field_name in (
            "matches",
            "episodes",
            "steps",
            "behavior_labeled_steps",
            "behavior_missing_steps",
            "observed_transition_links",
            "terminal_transition_links",
            "candidate_ranges",
            "maximum_complete_action_count",
            "public_feature_width",
            "private_feature_width",
            "action_feature_width",
            "eligible_future_reveal_links",
            "paired_public_frames",
            "public_state_mismatches",
            "public_history_mismatches",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        for field_name in (
            "public_range_nll",
            "uniform_range_nll",
            "public_range_brier",
            "uniform_range_brier",
            "public_range_top1_accuracy",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        if not 0 <= self.public_range_top1_accuracy <= 1:
            raise ValueError("range top-1 accuracy must be in [0, 1]")
        if self.dataset_boundary_kind != "abstract_information_set_game":
            raise ValueError("Phase 6H dataset boundary must remain abstract")
        if not self.source_official_engine_executed:
            raise ValueError("Phase 6H source must be genuine official replay")
        if self.dataset_official_engine_executed or self.branchable_transition_game:
            raise ValueError("logged dataset cannot claim official branching execution")
        if not self.serialization_roundtrip_exact:
            raise ValueError("Phase 6H dataset serialization must round-trip exactly")
        if len(dict(self.claims)) != len(self.claims):
            raise ValueError("Phase 6H claim names must be unique")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_TRAJECTORY_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "source_public_replay_fingerprint": self.source_public_replay_fingerprint,
            "trajectory_dataset_fingerprint": self.trajectory_dataset_fingerprint,
            "range_updater_fingerprint": self.range_updater_fingerprint,
            "matches": self.matches,
            "episodes": self.episodes,
            "steps": self.steps,
            "behavior_labeled_steps": self.behavior_labeled_steps,
            "behavior_missing_steps": self.behavior_missing_steps,
            "observed_transition_links": self.observed_transition_links,
            "terminal_transition_links": self.terminal_transition_links,
            "candidate_ranges": self.candidate_ranges,
            "maximum_complete_action_count": self.maximum_complete_action_count,
            "public_feature_width": self.public_feature_width,
            "private_feature_width": self.private_feature_width,
            "action_feature_width": self.action_feature_width,
            "eligible_future_reveal_links": self.eligible_future_reveal_links,
            "paired_public_frames": self.paired_public_frames,
            "public_state_mismatches": self.public_state_mismatches,
            "public_history_mismatches": self.public_history_mismatches,
            "public_range_nll": self.public_range_nll,
            "uniform_range_nll": self.uniform_range_nll,
            "public_range_brier": self.public_range_brier,
            "uniform_range_brier": self.uniform_range_brier,
            "public_range_top1_accuracy": self.public_range_top1_accuracy,
            "split_step_counts": dict(self.split_step_counts),
            "source_official_engine_executed": self.source_official_engine_executed,
            "dataset_boundary_kind": self.dataset_boundary_kind,
            "dataset_official_engine_executed": self.dataset_official_engine_executed,
            "branchable_transition_game": self.branchable_transition_game,
            "serialization_roundtrip_exact": self.serialization_roundtrip_exact,
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        claim_rows = "\n".join(
            f"| {name} | {value} |" for name, value in self.claims
        )
        return f"""# Phase 6H logged official-trajectory substrate

Benchmark: `{self.benchmark_id}`  
Evidence fingerprint: `{self.fingerprint}`  
Status: **{self.status}**

The source transitions genuinely occurred in the official engine. The derived
dataset is an offline, non-branchable information-set trajectory abstraction;
it is not official execution, a trained blueprint, or a safe resolver.

| Metric | Value |
|---|---:|
| Source matches / isolated episodes | {self.matches} / {self.episodes} |
| Acting-observer decision steps | {self.steps} |
| Contract-valid behavior labels | {self.behavior_labeled_steps} |
| Missing behavior labels | {self.behavior_missing_steps} |
| Observed / terminal transition links | {self.observed_transition_links} / {self.terminal_transition_links} |
| Complete range candidates | {self.candidate_ranges} |
| Maximum complete ordered actions | {self.maximum_complete_action_count} |
| Public / private / action feature widths | {self.public_feature_width} / {self.private_feature_width} / {self.action_feature_width} |
| Causal later-reveal supervision links | {self.eligible_future_reveal_links} |
| Paired common-public frames | {self.paired_public_frames} |
| Public state/history mismatches | {self.public_state_mismatches} / {self.public_history_mismatches} |
| Public-only range NLL (uniform) | {self.public_range_nll:.6f} ({self.uniform_range_nll:.6f}) |
| Public-only range Brier (uniform) | {self.public_range_brier:.6f} ({self.uniform_range_brier:.6f}) |
| Public-only range top-1 | {self.public_range_top1_accuracy:.3%} |

Range metrics describe deterministic inference over the six known local deck
profiles on this small replay corpus. They are not held-out population
calibration or evidence of playing strength.

## Claim boundary

| Claim | Status |
|---|---|
{claim_rows}
"""


@dataclass(frozen=True, slots=True)
class Phase6TrajectoryArtifacts:
    dataset: OfficialTrajectoryDataset
    evidence: Phase6TrajectoryEvidence


def _load_public_replays(path: Path) -> PublicReplayDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset = PublicReplayDataset.from_dict(raw)
    recorded = raw.get("fingerprint")
    if recorded is not None and str(recorded) != dataset.fingerprint:
        raise ValueError("public replay artifact fingerprint is stale or corrupt")
    if not dataset.validated:
        raise ValueError("public replay artifact failed its leakage audit")
    return dataset


def _range_candidates() -> tuple[PublicRangeCandidate, ...]:
    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    return tuple(
        PublicRangeCandidate(
            candidate_id=profile.archetype,
            archetype_id=profile.archetype,
            style_id="unknown",
            card_ids=profile.card_ids,
            deck_fingerprint=profile.fingerprint,
            prior=1.0,
            source=profile.provenance,
        )
        for profile in OFFICIAL_PANEL_DECKS + (target,)
    )


def run_phase6_official_trajectory(
    *,
    project_root: str | Path | None = None,
) -> Phase6TrajectoryArtifacts:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    public_replays = _load_public_replays(
        root
        / "artifacts"
        / "phase6_game_theory_foundation"
        / "public_replay_dataset.json"
    )
    dataset = build_official_trajectory_dataset(public_replays, _range_candidates())
    restored = OfficialTrajectoryDataset.from_dict(json.loads(dataset.to_json()))
    roundtrip = restored.fingerprint == dataset.fingerprint
    summary = dataset.summary
    audit = dataset.leakage_audit
    split_counts = tuple(
        sorted(
            (str(name), int(count))
            for name, count in summary["split_step_counts"].items()
        )
    )
    gates = (
        summary["steps"] == 610,
        summary["behavior_labeled_steps"] == summary["steps"],
        summary["behavior_missing_steps"] == 0,
        summary["observed_transition_links"] == summary["steps"],
        summary["terminal_transition_links"] == summary["episodes"],
        audit.passed,
        audit.paired_public_frames == 616,
        summary["public_range_nll"] < summary["uniform_range_nll"],
        summary["public_range_brier"] < summary["uniform_range_brier"],
        {name for name, _ in split_counts} == {"train", "calibration", "validation"},
        roundtrip,
    )
    status = (
        "phase6h_logged_trajectory_substrate_validated_partial"
        if all(gates)
        else "failed"
    )
    claims = (
        ("official_replay_source", "genuine_native_execution"),
        ("selected_alignment", "corrected_next_frame_semantics_610_of_610"),
        ("common_public_history", "zero_cross_perspective_mismatches"),
        ("private_identity_persistence", "fingerprints_only_zero_raw_ids"),
        ("future_reveal_feature_input", "zero_labels_supervision_only"),
        ("logged_information_set_trajectory", "validated_offline_substrate"),
        ("public_range_updater", "deterministic_uncalibrated_non_neural_baseline"),
        ("branchable_full_game_transition_model", "not_implemented"),
        ("full_game_neural_blueprint", "not_trained"),
        ("official_safe_resolve_pipeline", "not_deployed"),
        ("official_exploitation_overlay", "disabled"),
        ("kaggle_submission", "not_attempted"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6TrajectoryEvidence(
        benchmark_id="phase6h-official-public-trajectory-v1",
        status=status,
        source_public_replay_fingerprint=public_replays.fingerprint,
        trajectory_dataset_fingerprint=dataset.fingerprint,
        range_updater_fingerprint=dataset.range_updater_fingerprint,
        matches=int(summary["matches"]),
        episodes=int(summary["episodes"]),
        steps=int(summary["steps"]),
        behavior_labeled_steps=int(summary["behavior_labeled_steps"]),
        behavior_missing_steps=int(summary["behavior_missing_steps"]),
        observed_transition_links=int(summary["observed_transition_links"]),
        terminal_transition_links=int(summary["terminal_transition_links"]),
        candidate_ranges=int(summary["candidate_ranges"]),
        maximum_complete_action_count=int(summary["maximum_complete_action_count"]),
        public_feature_width=int(summary["public_feature_width"]),
        private_feature_width=int(summary["private_feature_width"]),
        action_feature_width=int(summary["action_feature_width"]),
        eligible_future_reveal_links=int(summary["eligible_future_reveal_links"]),
        paired_public_frames=audit.paired_public_frames,
        public_state_mismatches=audit.public_state_mismatches,
        public_history_mismatches=audit.public_history_mismatches,
        public_range_nll=float(summary["public_range_nll"]),
        uniform_range_nll=float(summary["uniform_range_nll"]),
        public_range_brier=float(summary["public_range_brier"]),
        uniform_range_brier=float(summary["uniform_range_brier"]),
        public_range_top1_accuracy=float(summary["public_range_top1_accuracy"]),
        split_step_counts=split_counts,
        source_official_engine_executed=dataset.source_official_engine_executed,
        dataset_boundary_kind=dataset.provenance.kind.value,
        dataset_official_engine_executed=dataset.provenance.official_engine_executed,
        branchable_transition_game=dataset.branchable_transition_game,
        serialization_roundtrip_exact=roundtrip,
        claims=claims,
    )
    if evidence.status == "failed":
        raise RuntimeError("Phase 6H trajectory substrate failed a declared gate")
    return Phase6TrajectoryArtifacts(dataset=dataset, evidence=evidence)


def main() -> None:
    print(run_phase6_official_trajectory().evidence.to_json())


if __name__ == "__main__":
    main()
