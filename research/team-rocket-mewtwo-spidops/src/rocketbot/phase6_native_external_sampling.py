"""Phase 6M: bounded native endgame external-sampling collection.

This module wires the Phase 6L traversal to genuine licensed native-search
descendants.  To keep the branching finite, it starts from late states found
on a policy spine under one already selected hidden completion.  Consequently
the records are valid endgame traversal samples for the supplied abstract
game, but they are not full-game Deep CFR iterations or equilibrium evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

from .competition import KAGGLE_WATCHTOWER_SUBSTITUTION, resolve_official_deck
from .external_sampling import (
    AdvantageReservoir,
    ExternalSamplingBudgetExceeded,
    ExternalSamplingConfig,
    ExternalSamplingTraversal,
    StrategyReservoir,
)
from .game_theory import AbstractAction, InformationSetActor
from .native_blueprint_features import (
    NATIVE_ACTION_FEATURE_NAMES,
    NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT,
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
    NativeBlueprintFeatureBridge,
)
from .native_regret_blueprint import (
    NativeRegretBlueprintArtifact,
    NativeRegretBlueprintProvider,
    restore_native_regret_blueprint,
)
from .official_evaluation import OFFICIAL_PANEL_DECKS, KaggleDeckProfile, rocket_compatibility_profile
from .official_search_game import OfficialNativeSearchGame, OfficialSearchGameState
from .phase6_native_game import _completions, _runtime_basics


PHASE6_NATIVE_EXTERNAL_SAMPLING_SCHEMA_VERSION = 1
PHASE6_NATIVE_EXTERNAL_SAMPLING_SEED = 6_020


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _seed(*items: Any) -> int:
    return int.from_bytes(sha256(_canonical_json(items).encode("utf-8")).digest()[:8], "big")


_FORBIDDEN_RAW_PRIVATE_KEYS = frozenset(
    {
        "your_deck",
        "your_prize",
        "opponent_deck",
        "opponent_prize",
        "opponent_hand",
        "opponent_active",
        "search_begin_input",
        "observation_json",
        "search_id",
        "hypothesis_fingerprint",
    }
)


def _raw_private_key_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(
            int(str(key) in _FORBIDDEN_RAW_PRIVATE_KEYS)
            + _raw_private_key_count(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_raw_private_key_count(item) for item in value)
    return 0


@dataclass(frozen=True, slots=True)
class NativeEndgameTraversalEvidence:
    candidate_id: str
    traverser: int
    iteration: int
    root_depth: int
    root_legal_actions: int
    attempted_roots: int
    transitions: int
    terminal_leaves: int
    maximum_depth_reached: int
    advantage_samples: int
    strategy_samples: int
    nonzero_advantage_actions: int
    maximum_absolute_advantage: float
    source_official_engine_executed: bool
    hidden_completion_prior_official: bool
    selected_completion_truth_policy_input: bool
    complete_within_budget: bool

    def __post_init__(self) -> None:
        if not self.candidate_id or self.traverser not in (0, 1):
            raise ValueError("native endgame traversal identity is invalid")
        if min(
            self.iteration,
            self.root_depth,
            self.root_legal_actions,
            self.attempted_roots,
            self.transitions,
            self.terminal_leaves,
            self.maximum_depth_reached,
            self.advantage_samples,
        ) <= 0:
            raise ValueError("native endgame traversal accounting must be positive")
        if self.strategy_samples < 0 or self.nonzero_advantage_actions < 0:
            raise ValueError("native endgame sample counts cannot be negative")
        if not math.isfinite(self.maximum_absolute_advantage) or self.maximum_absolute_advantage < 0:
            raise ValueError("native endgame maximum advantage is invalid")
        if not self.source_official_engine_executed:
            raise ValueError("native endgame traversal requires official source execution")
        if self.hidden_completion_prior_official or self.selected_completion_truth_policy_input:
            raise ValueError("native endgame abstraction cannot promote hidden truth")
        if not self.complete_within_budget:
            raise ValueError("partial external-sampling traversals cannot become evidence")


@dataclass(slots=True)
class NativeExternalSamplingDataset:
    game_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    source_blueprint_fingerprint: str
    source_blueprint_game_fingerprint: str
    feature_schema_fingerprint: str
    candidate_ids: tuple[str, ...]
    traversals: tuple[NativeEndgameTraversalEvidence, ...]
    player_0_advantages: AdvantageReservoir
    player_1_advantages: AdvantageReservoir
    average_strategy: StrategyReservoir
    source_official_engine_executed: bool = True
    hidden_completion_prior_official: bool = False
    fixed_completion_endgame_roots: bool = True
    selected_completion_truth_policy_input: bool = False
    full_game_deep_cfr_iterations: bool = False
    equilibrium_convergence_validated: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "game_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
            "source_blueprint_fingerprint",
            "source_blueprint_game_fingerprint",
            "feature_schema_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if self.feature_schema_fingerprint != NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT:
            raise ValueError("native external-sampling feature schema mismatch")
        if not self.candidate_ids or len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("native external-sampling candidates must be unique")
        if len(self.traversals) != len(self.candidate_ids):
            raise ValueError("native external-sampling requires one closing traversal per candidate")
        observed_candidates = [item.candidate_id for item in self.traversals]
        if set(observed_candidates) != set(self.candidate_ids) or len(observed_candidates) != len(
            set(observed_candidates)
        ):
            raise ValueError("native external-sampling candidate panel is incomplete")
        if not self.source_official_engine_executed:
            raise ValueError("native external-sampling source must be official execution")
        if any(
            (
                self.hidden_completion_prior_official,
                self.selected_completion_truth_policy_input,
                self.full_game_deep_cfr_iterations,
                self.equilibrium_convergence_validated,
            )
        ):
            raise ValueError("native endgame dataset cannot promote unvalidated claims")
        if not self.fixed_completion_endgame_roots:
            raise ValueError("native endgame dataset must disclose fixed-completion roots")
        if self.raw_private_card_ids_persisted != 0:
            raise ValueError("native endgame dataset cannot persist raw private cards")
        for player, reservoir in (
            (0, self.player_0_advantages),
            (1, self.player_1_advantages),
        ):
            if reservoir.seen < len(reservoir.samples):
                raise ValueError("native advantage reservoir accounting is invalid")
            if any(sample.traverser != player for sample in reservoir.samples):
                raise ValueError("native advantage sample is in the wrong player reservoir")
            for sample in reservoir.samples:
                if sample.game_fingerprint != self.game_fingerprint:
                    raise ValueError("native advantage sample game mismatch")
                if tuple(name for name, _ in sample.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
                    raise ValueError("native advantage public features are incomplete")
                if tuple(name for name, _ in sample.private_features) != NATIVE_PRIVATE_FEATURE_NAMES:
                    raise ValueError("native advantage private features are incomplete")
                for _, features in sample.action_features:
                    if tuple(name for name, _ in features) != NATIVE_ACTION_FEATURE_NAMES:
                        raise ValueError("native advantage action features are incomplete")
                if not sample.source_official_engine_executed:
                    raise ValueError("native advantage lost official source provenance")
                if sample.hidden_completion_prior_official:
                    raise ValueError("native advantage promoted the supplied prior")
        if any(
            sample.game_fingerprint != self.game_fingerprint
            or not sample.source_official_engine_executed
            or sample.hidden_completion_prior_official
            for sample in self.average_strategy.samples
        ):
            raise ValueError("native strategy reservoir provenance is inconsistent")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_EXTERNAL_SAMPLING_SCHEMA_VERSION,
            "game_fingerprint": self.game_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_blueprint_fingerprint": self.source_blueprint_fingerprint,
            "source_blueprint_game_fingerprint": (
                self.source_blueprint_game_fingerprint
            ),
            "source_blueprint_game_matches_live_game": (
                self.source_blueprint_game_fingerprint == self.game_fingerprint
            ),
            "feature_schema_fingerprint": self.feature_schema_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "traversals": [asdict(item) for item in self.traversals],
            "reservoirs": {
                "player_0_advantages": self.player_0_advantages.to_dict(),
                "player_1_advantages": self.player_1_advantages.to_dict(),
                "average_strategy": self.average_strategy.to_dict(),
            },
            "provenance": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "fixed_completion_endgame_roots": self.fixed_completion_endgame_roots,
                "selected_completion_truth_policy_input": (
                    self.selected_completion_truth_policy_input
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
            "claims": {
                "external_sampling_semantics": True,
                "complete_endgame_traversals": True,
                "full_game_deep_cfr_iterations": self.full_game_deep_cfr_iterations,
                "equilibrium_convergence_validated": (
                    self.equilibrium_convergence_validated
                ),
                "deployable_blueprint": False,
                "safe_resolver": False,
            },
        }
        if _raw_private_key_count(result) != 0:
            raise ValueError("native endgame serialization contains forbidden private keys")
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeExternalSamplingDataset":
        if _raw_private_key_count(raw) != 0:
            raise ValueError("native endgame artifact contains forbidden private keys")
        provenance = raw["provenance"]
        claims = raw["claims"]
        reservoirs = raw["reservoirs"]
        result = cls(
            game_fingerprint=str(raw["game_fingerprint"]),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            source_blueprint_fingerprint=str(raw["source_blueprint_fingerprint"]),
            source_blueprint_game_fingerprint=str(
                raw["source_blueprint_game_fingerprint"]
            ),
            feature_schema_fingerprint=str(raw["feature_schema_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            traversals=tuple(
                NativeEndgameTraversalEvidence(**item) for item in raw["traversals"]
            ),
            player_0_advantages=AdvantageReservoir.from_dict(
                reservoirs["player_0_advantages"]
            ),
            player_1_advantages=AdvantageReservoir.from_dict(
                reservoirs["player_1_advantages"]
            ),
            average_strategy=StrategyReservoir.from_dict(
                reservoirs["average_strategy"]
            ),
            source_official_engine_executed=bool(
                provenance["source_official_engine_executed"]
            ),
            hidden_completion_prior_official=bool(
                provenance["hidden_completion_prior_official"]
            ),
            fixed_completion_endgame_roots=bool(
                provenance["fixed_completion_endgame_roots"]
            ),
            selected_completion_truth_policy_input=bool(
                provenance["selected_completion_truth_policy_input"]
            ),
            full_game_deep_cfr_iterations=bool(
                claims["full_game_deep_cfr_iterations"]
            ),
            equilibrium_convergence_validated=bool(
                claims["equilibrium_convergence_validated"]
            ),
            raw_private_card_ids_persisted=int(
                provenance["raw_private_card_ids_persisted"]
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("native external-sampling dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6NativeExternalSamplingEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    game_fingerprint: str
    runtime_fingerprint: str
    source_blueprint_fingerprint: str
    candidate_count: int
    completed_traversals: int
    player_0_traversals: int
    player_1_traversals: int
    native_transition_count: int
    external_sampling_transitions: int
    advantage_samples_seen: int
    strategy_samples_seen: int
    retained_advantage_samples: int
    retained_strategy_samples: int
    nonzero_advantage_actions: int
    maximum_absolute_advantage: float
    raw_private_card_ids_persisted: int
    source_official_engine_executed: bool
    hidden_completion_prior_official: bool
    full_game_deep_cfr_iterations: bool
    equilibrium_convergence_validated: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6M evidence identity/status is required")
        for field_name in (
            "dataset_fingerprint",
            "game_fingerprint",
            "runtime_fingerprint",
            "source_blueprint_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if min(
            self.candidate_count,
            self.completed_traversals,
            self.native_transition_count,
            self.external_sampling_transitions,
            self.advantage_samples_seen,
            self.retained_advantage_samples,
        ) <= 0:
            raise ValueError("Phase 6M evidence accounting must be positive")
        if self.strategy_samples_seen < 0 or self.retained_strategy_samples < 0:
            raise ValueError("Phase 6M strategy sample counts cannot be negative")
        if self.player_0_traversals < 0 or self.player_1_traversals < 0:
            raise ValueError("Phase 6M player traversal counts cannot be negative")
        if self.completed_traversals != self.candidate_count:
            raise ValueError("Phase 6M did not complete one traversal per candidate")
        if self.player_0_traversals + self.player_1_traversals != self.candidate_count:
            raise ValueError("Phase 6M player traversal accounting is inconsistent")
        if self.nonzero_advantage_actions < 0 or not math.isfinite(
            self.maximum_absolute_advantage
        ):
            raise ValueError("Phase 6M advantage diagnostics are invalid")
        if self.raw_private_card_ids_persisted != 0:
            raise ValueError("Phase 6M persisted raw private cards")
        if not self.source_official_engine_executed:
            raise ValueError("Phase 6M requires genuine native transitions")
        if self.hidden_completion_prior_official or self.full_game_deep_cfr_iterations or self.equilibrium_convergence_validated:
            raise ValueError("Phase 6M cannot promote abstract endgame collection")
        if dict(self.claims).get("phase6_complete") != "false":
            raise ValueError("Phase 6M cannot mark Phase 6 complete")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_EXTERNAL_SAMPLING_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "dataset_fingerprint": self.dataset_fingerprint,
            "game_fingerprint": self.game_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "source_blueprint_fingerprint": self.source_blueprint_fingerprint,
            "candidate_count": self.candidate_count,
            "traversals": {
                "completed": self.completed_traversals,
                "player_0": self.player_0_traversals,
                "player_1": self.player_1_traversals,
                "native_total_transitions": self.native_transition_count,
                "external_sampling_transitions": self.external_sampling_transitions,
            },
            "memory": {
                "advantage_samples_seen": self.advantage_samples_seen,
                "strategy_samples_seen": self.strategy_samples_seen,
                "retained_advantage_samples": self.retained_advantage_samples,
                "retained_strategy_samples": self.retained_strategy_samples,
                "nonzero_advantage_actions": self.nonzero_advantage_actions,
                "maximum_absolute_advantage": self.maximum_absolute_advantage,
            },
            "provenance": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        claims = "\n".join(f"| {name} | {value} |" for name, value in self.claims)
        return f"""# Phase 6M native endgame external-sampling wiring

Evidence fingerprint: `{self.fingerprint}`  
Dataset fingerprint: `{self.dataset_fingerprint}`  
Status: **{self.status}**

All descendant transitions in these completed endgame traversals ran through
the licensed native search engine. The hidden-completion prior, choice of one
fixed particle per endgame root, feature mapping, source Phase 6K behavior
policy, reservoirs, and MCCFR interpretation are research abstractions.

| Check | Result |
|---|---:|
| Candidates / closing-player traversals | {self.candidate_count} / {self.completed_traversals} |
| Player 0 / player 1 traversals | {self.player_0_traversals} / {self.player_1_traversals} |
| External-sampling transitions | {self.external_sampling_transitions} |
| Total native transitions | {self.native_transition_count} |
| Advantage samples seen / retained | {self.advantage_samples_seen} / {self.retained_advantage_samples} |
| Strategy samples seen / retained | {self.strategy_samples_seen} / {self.retained_strategy_samples} |
| Nonzero advantage actions | {self.nonzero_advantage_actions} |
| Maximum absolute advantage | {self.maximum_absolute_advantage:.6f} |
| Raw private card IDs persisted | {self.raw_private_card_ids_persisted} |

These are late, fixed-particle endgame traversals. They validate native feature,
legality, branch, provenance, and reservoir wiring; they are not full-game Deep
CFR iterations, a convergence result, counterfactual safety bounds, or a
deployable policy.

## Claim boundary

| Claim | Status |
|---|---|
{claims}
"""


@dataclass(frozen=True, slots=True)
class Phase6NativeExternalSamplingArtifacts:
    dataset: NativeExternalSamplingDataset
    evidence: Phase6NativeExternalSamplingEvidence


def _raw_action(
    game: OfficialNativeSearchGame,
    state: OfficialSearchGameState,
    enriched: AbstractAction,
) -> AbstractAction:
    matches = [
        action
        for action in game.legal_actions(state)
        if action.action_id == enriched.action_id
        and action.official_option_indices == enriched.official_option_indices
    ]
    if len(matches) != 1:
        raise RuntimeError("native external-sampling action did not map exactly once")
    return matches[0]


def _policy_spine(
    game: OfficialNativeSearchGame,
    state: OfficialSearchGameState,
    *,
    bridge: NativeBlueprintFeatureBridge,
    provider: NativeRegretBlueprintProvider,
    maximum_depth: int,
    weakened_player: int | None = None,
) -> tuple[tuple[OfficialSearchGameState, ...], tuple[OfficialSearchGameState, ...]]:
    if weakened_player not in (None, 0, 1):
        raise ValueError("weakened player must be 0, 1, or None")
    decisions: list[list[OfficialSearchGameState]] = [[], []]
    for _ in range(maximum_depth):
        if state.terminal:
            return tuple(decisions[0]), tuple(decisions[1])
        if state.actor is InformationSetActor.CHANCE:
            outcomes = game.chance_outcomes(state)
            state = game.transition(
                state,
                None,
                chance_outcome=min(outcomes, key=lambda item: item.outcome_id),
            )
            continue
        player = game.player_to_act(state)
        assert player is not None
        decisions[player].append(state)
        view = bridge.view(game, state)
        distribution = provider.strategy(view.information_set, view.public_belief)
        chosen = (
            min(distribution.probabilities, key=lambda item: (item[1], item[0]))[0]
            if player == weakened_player
            else distribution.argmax
        )
        enriched = next(
            action
            for action in view.information_set.legal_actions
            if action.action_id == chosen
        )
        state = game.transition(state, _raw_action(game, state, enriched))
    raise RuntimeError("native external-sampling policy spine exceeded depth cap")


def _late_traversal(
    game: OfficialNativeSearchGame,
    roots: Sequence[OfficialSearchGameState],
    *,
    traverser: int,
    iteration: int,
    candidate_id: str,
    bridge: NativeBlueprintFeatureBridge,
    provider: NativeRegretBlueprintProvider,
    maximum_root_actions: int,
    maximum_transitions: int,
    maximum_attempts: int,
    seed: int,
):
    attempted = 0
    last_error: Exception | None = None
    for root in reversed(tuple(roots)):
        actions = game.legal_actions(root)
        if not 1 <= len(actions) <= maximum_root_actions:
            continue
        attempted += 1
        try:
            result = ExternalSamplingTraversal(
                ExternalSamplingConfig(
                    maximum_transitions=maximum_transitions,
                    maximum_depth=4_096,
                )
            ).run(
                game,
                provider.strategy,
                traverser=traverser,
                iteration=iteration,
                rng=random.Random(_seed(seed, candidate_id, traverser, root.depth)),
                root_state=root,
                information_view=lambda boundary, state, player: (
                    bridge.view(boundary, state).information_set,
                    bridge.view(boundary, state).public_belief,
                ),
                action_adapter=lambda boundary, state, action: _raw_action(
                    boundary, state, action
                ),
            )
            if not result.advantage_samples:
                raise RuntimeError("completed native traversal produced no advantage sample")
            return result, root, attempted
        except ExternalSamplingBudgetExceeded as exc:
            last_error = exc
            if attempted >= maximum_attempts:
                break
    raise RuntimeError(
        f"no bounded native endgame root completed for player {traverser} "
        f"after {attempted} attempts"
    ) from last_error


def _terminal_closing_root(
    game: OfficialNativeSearchGame,
    roots_by_player: tuple[
        tuple[OfficialSearchGameState, ...],
        tuple[OfficialSearchGameState, ...],
    ],
    *,
    maximum_root_actions: int,
) -> tuple[int, OfficialSearchGameState]:
    """Find the latest multi-action node whose every child is terminal.

    This deliberately tiny boundary is what makes the native evidence a wiring
    test rather than a misleading claim of scalable full-game traversal.
    """

    candidates = sorted(
        (
            (state.depth, player, state)
            for player in (0, 1)
            for state in roots_by_player[player]
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    for _, player, state in candidates:
        actions = game.legal_actions(state)
        if not 1 <= len(actions) <= maximum_root_actions:
            continue
        children = tuple(game.transition(state, action) for action in actions)
        if all(child.terminal for child in children):
            return player, state
    raise RuntimeError("policy spine has no bounded immediate-terminal closing root")


def run_phase6_native_external_sampling(
    *,
    project_root: str | Path | None = None,
    source_blueprint_path: str | Path | None = None,
    reservoir_capacity: int = 1_024,
    maximum_root_actions: int = 32,
    maximum_transitions: int = 1_000,
    maximum_attempts: int = 3,
    maximum_spine_depth: int = 2_000,
    seed: int = PHASE6_NATIVE_EXTERNAL_SAMPLING_SEED,
) -> Phase6NativeExternalSamplingArtifacts:
    if min(
        reservoir_capacity,
        maximum_root_actions,
        maximum_transitions,
        maximum_attempts,
        maximum_spine_depth,
    ) <= 0:
        raise ValueError("Phase 6M collection limits must be positive")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    blueprint_path = Path(
        source_blueprint_path
        or root / "artifacts" / "phase6_native_blueprint" / "native_regret_blueprint.json"
    ).resolve()
    artifact = NativeRegretBlueprintArtifact.from_dict(
        json.loads(blueprint_path.read_text(encoding="utf-8"))
    )
    provider = restore_native_regret_blueprint(artifact)
    bridge = NativeBlueprintFeatureBridge()
    path_text = str(runtime)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    from cg.game import battle_finish, battle_start  # type: ignore[import-not-found]

    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    candidates: tuple[KaggleDeckProfile, ...] = OFFICIAL_PANEL_DECKS + (target,)
    observation, start = battle_start(
        list(target.card_ids), list(OFFICIAL_PANEL_DECKS[0].card_ids)
    )
    if observation is None:
        raise RuntimeError(
            f"official engine rejected Phase 6M decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    try:
        completions = _completions(
            observation,
            target=target,
            candidates=candidates,
            basic_card_ids=_runtime_basics(runtime),
        )
        with OfficialNativeSearchGame(
            observation,
            completions,
            runtime_directory=runtime,
        ) as game:
            advantages = (
                AdvantageReservoir(reservoir_capacity, seed=_seed(seed, "p0")),
                AdvantageReservoir(reservoir_capacity, seed=_seed(seed, "p1")),
            )
            strategies = StrategyReservoir(
                reservoir_capacity, seed=_seed(seed, "strategy")
            )
            traversal_evidence = []
            external_transitions = 0
            chance_outcomes = game.chance_outcomes(game.root_state())
            traversal_index = 0
            for candidate_index, (candidate, outcome) in enumerate(
                zip(candidates, chance_outcomes, strict=True)
            ):
                scenario_root = game.transition(
                    game.root_state(), None, chance_outcome=outcome
                )
                for mode, weakened_player in (
                    ("balanced", None),
                    ("p0_weakened", 0),
                ):
                    roots_by_player = _policy_spine(
                        game,
                        scenario_root,
                        bridge=bridge,
                        provider=provider,
                        maximum_depth=maximum_spine_depth,
                        weakened_player=weakened_player,
                    )
                    try:
                        traverser, terminal_root = _terminal_closing_root(
                            game,
                            roots_by_player,
                            maximum_root_actions=maximum_root_actions,
                        )
                    except RuntimeError:
                        # Skipping is safer than turning a capped partial traversal
                        # into a biased advantage sample. The final panel remains
                        # explicit in ``candidate_ids`` and the evidence counts.
                        continue
                    traversal_index += 1
                    scenario_id = f"{candidate.archetype}:{mode}"
                    result, selected_root, attempted = _late_traversal(
                        game,
                        (terminal_root,),
                        traverser=traverser,
                        iteration=traversal_index,
                        candidate_id=scenario_id,
                        bridge=bridge,
                        provider=provider,
                        maximum_root_actions=maximum_root_actions,
                        maximum_transitions=maximum_transitions,
                        maximum_attempts=maximum_attempts,
                        seed=seed,
                    )
                    external_transitions += result.transitions
                    for sample in result.advantage_samples:
                        advantages[traverser].add(sample)
                    for sample in result.strategy_samples:
                        strategies.add(sample)
                    all_advantages = [
                        abs(value)
                        for sample in result.advantage_samples
                        for _, value in sample.instantaneous_advantages
                    ]
                    traversal_evidence.append(
                        NativeEndgameTraversalEvidence(
                            candidate_id=scenario_id,
                            traverser=traverser,
                            iteration=traversal_index,
                            root_depth=selected_root.depth,
                            root_legal_actions=len(game.legal_actions(selected_root)),
                            attempted_roots=attempted,
                            transitions=result.transitions,
                            terminal_leaves=result.terminal_leaves,
                            maximum_depth_reached=result.maximum_depth_reached,
                            advantage_samples=len(result.advantage_samples),
                            strategy_samples=len(result.strategy_samples),
                            nonzero_advantage_actions=sum(
                                value > 1e-12 for value in all_advantages
                            ),
                            maximum_absolute_advantage=max(
                                all_advantages, default=0.0
                            ),
                            source_official_engine_executed=(
                                result.source_official_engine_executed
                            ),
                            hidden_completion_prior_official=(
                                result.hidden_completion_prior_official
                            ),
                            selected_completion_truth_policy_input=False,
                            complete_within_budget=result.complete,
                        ),
                    )
            dataset = NativeExternalSamplingDataset(
                game_fingerprint=game.fingerprint,
                runtime_fingerprint=game.ruleset_fingerprint,
                deck_fingerprint=target.fingerprint,
                source_blueprint_fingerprint=artifact.fingerprint,
                source_blueprint_game_fingerprint=artifact.game_fingerprint,
                feature_schema_fingerprint=bridge.fingerprint,
                candidate_ids=tuple(
                    item.candidate_id for item in traversal_evidence
                ),
                traversals=tuple(traversal_evidence),
                player_0_advantages=advantages[0],
                player_1_advantages=advantages[1],
                average_strategy=strategies,
            )
            serialized = dataset.to_dict()
            restored = NativeExternalSamplingDataset.from_dict(
                json.loads(json.dumps(serialized))
            )
            if restored.fingerprint != dataset.fingerprint:
                raise RuntimeError("native external-sampling roundtrip drifted")
            all_evidence_advantages = sum(
                item.nonzero_advantage_actions for item in traversal_evidence
            )
            max_advantage = max(
                item.maximum_absolute_advantage for item in traversal_evidence
            )
            claims = (
                ("native_descendant_transitions", "genuine_official_search_execution"),
                (
                    "external_sampling",
                    "complete_immediate-terminal_endgame_traversals; capped scenarios skipped",
                ),
                ("advantage_memory", "separate_player_uniform_reservoirs"),
                ("player_balanced_native_panel", "not_required_for_wiring_probe"),
                (
                    "strategy_memory",
                    "separate_uniform_reservoir_interface; native samples only when encountered",
                ),
                ("hidden_completion_prior", "research_abstraction_not_official"),
                ("endgame_root", "fixed_hidden_completion_policy_features_redacted"),
                ("source_phase6k_policy", "behavior_only_not_final_blueprint"),
                ("full_game_deep_cfr", "not_run"),
                ("equilibrium_convergence", "not_validated"),
                ("counterfactual_value_bounds", "not_calibrated_full_game"),
                ("safe_resolver", "not_deployed_official"),
                ("pokemon_strength", "not_claimed"),
                ("kaggle_submission", "not_attempted"),
                ("phase6_complete", "false"),
            )
            gates = (
                len(traversal_evidence) >= 2,
                all(item.complete_within_budget for item in traversal_evidence),
                advantages[0].seen + advantages[1].seen > 0,
                _raw_private_key_count(serialized) == 0,
                restored.fingerprint == dataset.fingerprint,
            )
            status = (
                "phase6m_native_endgame_external_sampling_validated_partial"
                if all(gates)
                else "failed"
            )
            evidence = Phase6NativeExternalSamplingEvidence(
                benchmark_id="phase6m-native-endgame-external-sampling-v1",
                status=status,
                dataset_fingerprint=dataset.fingerprint,
                game_fingerprint=game.fingerprint,
                runtime_fingerprint=game.ruleset_fingerprint,
                source_blueprint_fingerprint=artifact.fingerprint,
                candidate_count=len(traversal_evidence),
                completed_traversals=len(traversal_evidence),
                player_0_traversals=sum(
                    item.traverser == 0 for item in traversal_evidence
                ),
                player_1_traversals=sum(
                    item.traverser == 1 for item in traversal_evidence
                ),
                native_transition_count=game.transition_count,
                external_sampling_transitions=external_transitions,
                advantage_samples_seen=advantages[0].seen + advantages[1].seen,
                strategy_samples_seen=strategies.seen,
                retained_advantage_samples=(
                    len(advantages[0].samples) + len(advantages[1].samples)
                ),
                retained_strategy_samples=len(strategies.samples),
                nonzero_advantage_actions=all_evidence_advantages,
                maximum_absolute_advantage=max_advantage,
                raw_private_card_ids_persisted=_raw_private_key_count(serialized),
                source_official_engine_executed=True,
                hidden_completion_prior_official=False,
                full_game_deep_cfr_iterations=False,
                equilibrium_convergence_validated=False,
                claims=claims,
            )
            if evidence.status == "failed":
                raise RuntimeError("Phase 6M native external sampling failed a gate")
            return Phase6NativeExternalSamplingArtifacts(dataset, evidence)
    finally:
        battle_finish()


def main() -> None:
    print(run_phase6_native_external_sampling().evidence.to_json())


if __name__ == "__main__":
    main()
