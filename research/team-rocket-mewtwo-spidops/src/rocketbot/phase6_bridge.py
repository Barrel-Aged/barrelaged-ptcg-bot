"""Live official-engine shadow bridge audit for Phase 6G."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from .competition import (
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    AblationMode,
    CompetitionAgentConfig,
    Phase6CompetitionAgent,
    resolve_official_deck,
)
from .official_bridge import (
    BridgeAuditedCompetitionAgent,
    OfficialBridgeDecisionAudit,
    OfficialDecisionBridge,
)
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    OfficialMatchRunner,
    PriorityBaseline,
    rocket_compatibility_profile,
)


PHASE6_BRIDGE_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_runtime_fingerprint(runtime_directory: Path) -> str:
    files = (
        runtime_directory / "cg" / "game.py",
        runtime_directory / "cg" / "api.py",
        runtime_directory / "deck.csv",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"official runtime fingerprint inputs missing: {missing}")
    return _fingerprint(
        {
            "schema": PHASE6_BRIDGE_SCHEMA_VERSION,
            "files": {path.name: _file_sha256(path) for path in files},
        }
    )


@dataclass(frozen=True, slots=True)
class OfficialBridgeMatchEvidence:
    match_id: str
    opponent_label: str
    opponent_style: str
    target_seat: int
    native_result: int
    native_result_reason: int
    native_selections: int
    target_decisions: int
    illegal_actions: int
    runner_fallbacks: int
    policy_fallbacks: int
    soft_timeouts: int
    engine_error: str | None
    bridge_errors: int
    exact_selection_roundtrips: int
    maximum_option_count: int
    maximum_complete_action_count: int

    def __post_init__(self) -> None:
        if not self.match_id or not self.opponent_label or not self.opponent_style:
            raise ValueError("bridge match identity must be non-empty")
        if self.target_seat not in (0, 1):
            raise ValueError("bridge target seat must be zero or one")
        for field_name in (
            "native_selections",
            "target_decisions",
            "illegal_actions",
            "runner_fallbacks",
            "policy_fallbacks",
            "soft_timeouts",
            "bridge_errors",
            "exact_selection_roundtrips",
            "maximum_option_count",
            "maximum_complete_action_count",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True, slots=True)
class Phase6BridgeEvidence:
    benchmark_id: str
    status: str
    runtime_fingerprint: str
    bridge_fingerprint: str
    compatibility_deck_fingerprint: str
    matches: tuple[OfficialBridgeMatchEvidence, ...]
    official_engine_executed: bool
    target_decisions: int
    bridge_errors: int
    exact_selection_roundtrips: int
    all_abstract_actions_contract_valid: bool
    hidden_canary_invariant_decisions: int
    own_private_sensitivity_checks: int
    own_private_sensitivity_passes: int
    unknown_option_fields: tuple[str, ...]
    maximum_option_count: int
    maximum_complete_action_count: int
    mean_bridge_latency_ms: float
    maximum_bridge_latency_ms: float
    native_illegal_actions: int
    native_fallbacks: int
    native_soft_timeouts: int
    shadow_action_changes: int
    decisions_fingerprint: str
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6G bridge benchmark identity/status is required")
        for field_name in (
            "runtime_fingerprint",
            "bridge_fingerprint",
            "compatibility_deck_fingerprint",
            "decisions_fingerprint",
        ):
            value = getattr(self, field_name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if not self.official_engine_executed:
            raise ValueError("Phase 6G live bridge evidence must execute the official engine")
        if self.target_decisions <= 0 or not self.matches:
            raise ValueError("Phase 6G bridge evidence requires native decisions")
        if self.exact_selection_roundtrips > self.target_decisions:
            raise ValueError("roundtrip count cannot exceed decisions")
        if self.own_private_sensitivity_passes > self.own_private_sensitivity_checks:
            raise ValueError("private sensitivity passes cannot exceed checks")
        if min(self.mean_bridge_latency_ms, self.maximum_bridge_latency_ms) < 0:
            raise ValueError("bridge latency cannot be negative")
        if len(dict(self.claims)) != len(self.claims):
            raise ValueError("Phase 6G claim names must be unique")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_BRIDGE_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "runtime_fingerprint": self.runtime_fingerprint,
            "bridge_fingerprint": self.bridge_fingerprint,
            "compatibility_deck_fingerprint": self.compatibility_deck_fingerprint,
            "matches": [asdict(item) for item in self.matches],
            "official_engine_executed": self.official_engine_executed,
            "target_decisions": self.target_decisions,
            "bridge_errors": self.bridge_errors,
            "exact_selection_roundtrips": self.exact_selection_roundtrips,
            "all_abstract_actions_contract_valid": self.all_abstract_actions_contract_valid,
            "hidden_canary_invariant_decisions": self.hidden_canary_invariant_decisions,
            "own_private_sensitivity_checks": self.own_private_sensitivity_checks,
            "own_private_sensitivity_passes": self.own_private_sensitivity_passes,
            "unknown_option_fields": list(self.unknown_option_fields),
            "maximum_option_count": self.maximum_option_count,
            "maximum_complete_action_count": self.maximum_complete_action_count,
            "mean_bridge_latency_ms": self.mean_bridge_latency_ms,
            "maximum_bridge_latency_ms": self.maximum_bridge_latency_ms,
            "native_illegal_actions": self.native_illegal_actions,
            "native_fallbacks": self.native_fallbacks,
            "native_soft_timeouts": self.native_soft_timeouts,
            "shadow_action_changes": self.shadow_action_changes,
            "decisions_fingerprint": self.decisions_fingerprint,
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        match_rows = "\n".join(
            f"| {item.match_id} | {item.opponent_label} | {item.target_seat} | "
            f"{item.target_decisions} | {item.bridge_errors} | "
            f"{item.illegal_actions + item.runner_fallbacks + item.policy_fallbacks} |"
            for item in self.matches
        )
        claim_rows = "\n".join(
            f"| {name} | {value} |" for name, value in self.claims
        )
        return f"""# Phase 6G official shadow-bridge evidence

Benchmark: `{self.benchmark_id}`  
Evidence fingerprint: `{self.fingerprint}`  
Status: **{self.status}**

These matches genuinely execute the staged official `cg` engine with the
explicit Watchtower compatibility deck. The bridge runs in shadow mode after
the preserved baseline chooses; it returns no action to the engine and records
whether the exact ordered native selection can be represented and lifted.

| Match | Opponent | Seat | Target decisions | Bridge errors | Native safety events |
|---|---|---:|---:|---:|---:|
{match_rows}

| Metric | Value |
|---|---:|
| Target decisions | {self.target_decisions} |
| Exact selection roundtrips | {self.exact_selection_roundtrips} |
| Bridge errors | {self.bridge_errors} |
| Hidden-canary invariant decisions | {self.hidden_canary_invariant_decisions} |
| Own-private sensitivity | {self.own_private_sensitivity_passes}/{self.own_private_sensitivity_checks} |
| Unknown option fields | {len(self.unknown_option_fields)} |
| Maximum native options | {self.maximum_option_count} |
| Maximum complete ordered actions | {self.maximum_complete_action_count} |
| Mean bridge audit latency | {self.mean_bridge_latency_ms:.3f} ms |
| Maximum bridge audit latency | {self.maximum_bridge_latency_ms:.3f} ms |
| Native illegal/fallback/timeout events | {self.native_illegal_actions + self.native_fallbacks + self.native_soft_timeouts} |
| Shadow action changes | {self.shadow_action_changes} |

## Claim boundary

| Claim | Status |
|---|---|
{claim_rows}
"""


@dataclass(frozen=True, slots=True)
class Phase6BridgeArtifacts:
    evidence: Phase6BridgeEvidence
    decisions: tuple[OfficialBridgeDecisionAudit, ...]


def _decision_payload(record: OfficialBridgeDecisionAudit) -> dict[str, Any]:
    # Latency is excluded from the content fingerprint because it is a machine-
    # timing diagnostic, while all structural bridge fields remain covered.
    payload = asdict(record)
    payload.pop("bridge_latency_ms")
    return payload


def run_phase6_official_bridge_audit(
    *,
    project_root: Path | None = None,
    runtime_directory: Path | None = None,
    match_limit: int | None = None,
) -> Phase6BridgeArtifacts:
    root = project_root or Path(__file__).resolve().parents[2]
    runtime = runtime_directory or (
        root / "work" / "official_kaggle" / "sample_submission"
    )
    runtime = runtime.resolve()
    runtime_fingerprint = official_runtime_fingerprint(runtime)
    bridge = OfficialDecisionBridge(
        ruleset_fingerprint=runtime_fingerprint,
        official_engine_executed=True,
    )
    deck = resolve_official_deck(
        substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
        fail_on_unresolved=True,
    )
    target_profile = rocket_compatibility_profile(deck)
    opponent_profiles: tuple[KaggleDeckProfile, ...] = (
        *OFFICIAL_PANEL_DECKS,
        target_profile,
    )
    panel = [
        (profile, seat)
        for profile in opponent_profiles
        for seat in (0, 1)
    ]
    if match_limit is not None:
        if match_limit <= 0:
            raise ValueError("match_limit must be positive")
        panel = panel[:match_limit]
    styles = ("tempo", "attack_rush", "control")
    runner = OfficialMatchRunner(runtime)
    decisions: list[OfficialBridgeDecisionAudit] = []
    matches: list[OfficialBridgeMatchEvidence] = []
    for index, (opponent_profile, seat) in enumerate(panel):
        style = styles[index % len(styles)]
        match_id = f"phase6g-bridge-{index:02d}-{opponent_profile.archetype}-s{seat}"
        delegate = Phase6CompetitionAgent(
            deck,
            config=CompetitionAgentConfig(ablation=AblationMode.POLICY_ONLY),
        )
        audited = BridgeAuditedCompetitionAgent(
            delegate,
            bridge,
            match_id=match_id,
        )
        execution = runner.run_match(
            match_id=match_id,
            target_agent=audited,
            target_deck=target_profile,
            opponent_agent=PriorityBaseline(opponent_profile.card_ids, style=style),
            opponent_deck=opponent_profile,
            opponent_style=style,
            target_seat=seat,
            capture_replay=False,
        )
        records = tuple(audited.audit_records)
        decisions.extend(records)
        matches.append(
            OfficialBridgeMatchEvidence(
                match_id=match_id,
                opponent_label=opponent_profile.label,
                opponent_style=style,
                target_seat=seat,
                native_result=execution.record.winner,
                native_result_reason=-1,
                native_selections=execution.record.selections,
                target_decisions=len(records),
                illegal_actions=execution.record.illegal_action_count,
                runner_fallbacks=execution.record.runner_fallback_count,
                policy_fallbacks=int(execution.agent_summary.get("fallback_count", 0)),
                soft_timeouts=execution.record.soft_timeout_count,
                engine_error=execution.record.engine_error,
                bridge_errors=sum(record.error is not None for record in records),
                exact_selection_roundtrips=sum(
                    record.selection_roundtrip_exact for record in records
                ),
                maximum_option_count=max(
                    (record.option_count for record in records), default=0
                ),
                maximum_complete_action_count=max(
                    (record.complete_action_count for record in records), default=0
                ),
            )
        )
    target_decisions = len(decisions)
    bridge_errors = sum(record.error is not None for record in decisions)
    roundtrips = sum(record.selection_roundtrip_exact for record in decisions)
    sensitivity_records = [
        record
        for record in decisions
        if record.own_private_sensitivity is not None
    ]
    unknown_fields = tuple(
        sorted(
            {
                field_name
                for record in decisions
                for field_name in record.unknown_option_fields
            }
        )
    )
    native_illegal = sum(item.illegal_actions for item in matches)
    native_fallbacks = sum(
        item.runner_fallbacks + item.policy_fallbacks for item in matches
    )
    native_timeouts = sum(item.soft_timeouts for item in matches)
    if bridge_errors or roundtrips != target_decisions:
        raise RuntimeError("Phase 6G bridge failed exact shadow selection lifting")
    if not all(record.every_abstract_action_contract_valid for record in decisions):
        raise RuntimeError("Phase 6G bridge emitted an invalid abstract action")
    if not all(record.hidden_canary_invariant for record in decisions):
        raise RuntimeError("Phase 6G bridge hidden-state canary changed features")
    if not all(record.own_private_sensitivity for record in sensitivity_records):
        raise RuntimeError("Phase 6G bridge failed own-private sensitivity")
    if unknown_fields:
        raise RuntimeError(f"Phase 6G bridge saw unknown option fields: {unknown_fields}")
    if native_illegal or native_fallbacks or native_timeouts:
        raise RuntimeError(
            "Phase 6G native safety event occurred: "
            f"illegal={native_illegal}, fallbacks={native_fallbacks}, "
            f"soft_timeouts={native_timeouts}"
        )
    decision_fingerprint = _fingerprint(
        [_decision_payload(record) for record in decisions]
    )
    claims = (
        ("official_engine_execution", "validated_live_shadow_panel"),
        ("agent_visible_information_sets", "validated_one_step_boundary"),
        ("complete_ordered_action_lifting", "validated_all_observed_decisions"),
        ("hidden_state_canary", "zero_feature_changes"),
        ("shadow_policy_transparency", "exact_delegate_selection_unchanged"),
        ("full_game_transition_abstraction", "not_implemented"),
        ("full_game_neural_blueprint", "not_trained"),
        ("official_safe_resolve_pipeline", "not_deployed"),
        ("official_exploitation_overlay", "disabled"),
        ("matched_compute_neural_ablations", "not_run_no_full_game_model"),
        ("exact_naic_deck", "blocked_prism_tower_missing"),
        ("kaggle_submission", "not_attempted"),
        ("phase6_complete", "false"),
    )
    latencies = [record.bridge_latency_ms for record in decisions]
    evidence = Phase6BridgeEvidence(
        benchmark_id=f"phase6g-official-shadow-bridge-{len(matches)}-match-v1",
        status="phase6g_official_shadow_bridge_validated_partial",
        runtime_fingerprint=runtime_fingerprint,
        bridge_fingerprint=bridge.fingerprint,
        compatibility_deck_fingerprint=deck.fingerprint,
        matches=tuple(matches),
        official_engine_executed=True,
        target_decisions=target_decisions,
        bridge_errors=bridge_errors,
        exact_selection_roundtrips=roundtrips,
        all_abstract_actions_contract_valid=True,
        hidden_canary_invariant_decisions=sum(
            record.hidden_canary_invariant for record in decisions
        ),
        own_private_sensitivity_checks=len(sensitivity_records),
        own_private_sensitivity_passes=sum(
            bool(record.own_private_sensitivity) for record in sensitivity_records
        ),
        unknown_option_fields=unknown_fields,
        maximum_option_count=max(record.option_count for record in decisions),
        maximum_complete_action_count=max(
            record.complete_action_count for record in decisions
        ),
        mean_bridge_latency_ms=fmean(latencies),
        maximum_bridge_latency_ms=max(latencies),
        native_illegal_actions=native_illegal,
        native_fallbacks=native_fallbacks,
        native_soft_timeouts=native_timeouts,
        shadow_action_changes=0,
        decisions_fingerprint=decision_fingerprint,
        claims=claims,
    )
    return Phase6BridgeArtifacts(evidence=evidence, decisions=tuple(decisions))


def main() -> None:
    print(run_phase6_official_bridge_audit().evidence.to_json())


if __name__ == "__main__":
    main()
