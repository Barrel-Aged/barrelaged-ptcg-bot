"""Phase 6J evidence for the branchable official native-search game boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from .competition import KAGGLE_WATCHTOWER_SUBSTITUTION, resolve_official_deck
from .game_theory import AbstractAction, ChanceOutcome, InformationSetActor
from .official_bridge import (
    official_private_observation_payload,
    sanitize_official_public_state,
)
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    PriorityBaseline,
    rocket_compatibility_profile,
)
from .official_search_game import (
    HiddenZoneCompletion,
    OfficialNativeSearchGame,
    build_hidden_zone_completion,
)


PHASE6_NATIVE_GAME_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class NativeScenarioTraversalEvidence:
    candidate_id: str
    completion_fingerprint: str
    public_evidence_adjustments: int
    terminal_reached: bool
    terminal_depth: int
    player_0_decisions: int
    player_1_decisions: int
    explicit_coin_chance_nodes: int
    maximum_complete_action_count: int
    distinct_information_sets: int
    terminal_utility_player_0: float
    terminal_zero_sum_exact: bool
    midgame_branch_verified: bool

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("native scenario candidate ID is required")
        _validate_digest(self.completion_fingerprint, "completion_fingerprint")
        for field_name in (
            "public_evidence_adjustments",
            "terminal_depth",
            "player_0_decisions",
            "player_1_decisions",
            "explicit_coin_chance_nodes",
            "maximum_complete_action_count",
            "distinct_information_sets",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if not math.isfinite(self.terminal_utility_player_0):
            raise ValueError("terminal utility must be finite")


@dataclass(frozen=True, slots=True)
class Phase6NativeGameEvidence:
    benchmark_id: str
    status: str
    runtime_fingerprint: str
    game_fingerprint: str
    compatibility_deck_fingerprint: str
    source_official_engine_executed: bool
    game_boundary_kind: str
    game_official_engine_executed: bool
    hidden_completion_prior_official: bool
    branchable_native_transitions: bool
    completion_count: int
    completion_probability_sum: float
    raw_private_card_ids_persisted: int
    root_public_observation_equivalence: bool
    root_private_observation_equivalence: bool
    root_information_set_invariant_across_completions: bool
    root_public_belief_invariant_across_completions: bool
    selected_completion_truth_enters_policy_payload: bool
    root_legal_action_count: int
    root_distinct_alternative_children: int
    repeated_root_transition_deterministic: bool
    tampered_action_rejected: bool
    tampered_chance_outcome_rejected: bool
    all_scenarios_terminal: bool
    both_players_traversed_in_every_scenario: bool
    native_transition_count: int
    native_state_count: int
    scenarios: tuple[NativeScenarioTraversalEvidence, ...]
    completion_manifests: tuple[Mapping[str, Any], ...]
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6J benchmark identity/status is required")
        for field_name in (
            "runtime_fingerprint",
            "game_fingerprint",
            "compatibility_deck_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if self.game_boundary_kind != "abstract_information_set_game":
            raise ValueError("native search game boundary must remain abstract")
        if not self.source_official_engine_executed or self.game_official_engine_executed:
            raise ValueError("native source/abstract game provenance flags disagree")
        if self.hidden_completion_prior_official:
            raise ValueError("supplied hidden completion prior cannot claim official status")
        if self.completion_count <= 0 or len(self.scenarios) != self.completion_count:
            raise ValueError("Phase 6J completion/scenario counts disagree")
        if not math.isclose(self.completion_probability_sum, 1.0, abs_tol=1e-12):
            raise ValueError("Phase 6J completion probabilities must sum to one")
        for field_name in (
            "raw_private_card_ids_persisted",
            "root_legal_action_count",
            "root_distinct_alternative_children",
            "native_transition_count",
            "native_state_count",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if len(self.completion_manifests) != self.completion_count:
            raise ValueError("Phase 6J completion manifest count mismatch")
        if any("your_deck" in manifest or "opponent_deck" in manifest for manifest in self.completion_manifests):
            raise ValueError("completion manifests cannot persist raw private zones")
        if len(dict(self.claims)) != len(self.claims):
            raise ValueError("Phase 6J claim names must be unique")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_GAME_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "runtime_fingerprint": self.runtime_fingerprint,
            "game_fingerprint": self.game_fingerprint,
            "compatibility_deck_fingerprint": self.compatibility_deck_fingerprint,
            "provenance": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "game_boundary_kind": self.game_boundary_kind,
                "game_official_engine_executed": self.game_official_engine_executed,
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "branchable_native_transitions": self.branchable_native_transitions,
            },
            "completion_count": self.completion_count,
            "completion_probability_sum": self.completion_probability_sum,
            "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            "root_checks": {
                "public_observation_equivalence": self.root_public_observation_equivalence,
                "private_observation_equivalence": self.root_private_observation_equivalence,
                "information_set_invariant_across_completions": self.root_information_set_invariant_across_completions,
                "public_belief_invariant_across_completions": self.root_public_belief_invariant_across_completions,
                "selected_completion_truth_enters_policy_payload": self.selected_completion_truth_enters_policy_payload,
                "legal_action_count": self.root_legal_action_count,
                "distinct_alternative_children": self.root_distinct_alternative_children,
                "repeated_transition_deterministic": self.repeated_root_transition_deterministic,
                "tampered_action_rejected": self.tampered_action_rejected,
                "tampered_chance_outcome_rejected": self.tampered_chance_outcome_rejected,
            },
            "traversal": {
                "all_scenarios_terminal": self.all_scenarios_terminal,
                "both_players_traversed_in_every_scenario": self.both_players_traversed_in_every_scenario,
                "native_transition_count": self.native_transition_count,
                "native_state_count": self.native_state_count,
                "scenarios": [asdict(item) for item in self.scenarios],
            },
            "completion_manifests": [dict(item) for item in self.completion_manifests],
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        scenario_rows = "\n".join(
            (
                f"| {item.candidate_id} | {item.terminal_depth} | "
                f"{item.player_0_decisions} / {item.player_1_decisions} | "
                f"{item.explicit_coin_chance_nodes} | "
                f"{item.maximum_complete_action_count} | "
                f"{item.distinct_information_sets} | "
                f"{item.terminal_utility_player_0:+.1f} | "
                f"{str(item.midgame_branch_verified).lower()} |"
            )
            for item in self.scenarios
        )
        claim_rows = "\n".join(
            f"| {name} | {value} |" for name, value in self.claims
        )
        return f"""# Phase 6J branchable native-search game boundary

Benchmark: `{self.benchmark_id}`  
Evidence fingerprint: `{self.fingerprint}`  
Runtime fingerprint: `{self.runtime_fingerprint}`  
Game fingerprint: `{self.game_fingerprint}`  
Status: **{self.status}**

The descendants below are genuine transitions executed by the licensed native
search engine. The root distribution over hidden-zone completions is supplied
by the research adapter and is not an official belief model. Consequently, the
game correctly retains abstract provenance and does not claim official-engine
execution as a policy/game object.

| Hidden opponent candidate | Terminal depth | P0 / P1 decisions | Coin chance nodes | Max complete actions | Distinct information sets | P0 utility | Midgame branching |
|---|---:|---:|---:|---:|---:|---:|---|
{scenario_rows}

## Boundary checks

| Check | Value |
|---|---:|
| Hidden completion particles | {self.completion_count} |
| Completion probability sum | {self.completion_probability_sum:.12f} |
| Root public/private native equivalence | {str(self.root_public_observation_equivalence).lower()} / {str(self.root_private_observation_equivalence).lower()} |
| Root information-set/belief invariant across truth particles | {str(self.root_information_set_invariant_across_completions).lower()} / {str(self.root_public_belief_invariant_across_completions).lower()} |
| Selected completion enters policy payload | {str(self.selected_completion_truth_enters_policy_payload).lower()} |
| Root legal actions / distinct alternative children | {self.root_legal_action_count} / {self.root_distinct_alternative_children} |
| Repeated root transition deterministic | {str(self.repeated_root_transition_deterministic).lower()} |
| Tampered action/chance rejected | {str(self.tampered_action_rejected).lower()} / {str(self.tampered_chance_outcome_rejected).lower()} |
| All six native paths terminal | {str(self.all_scenarios_terminal).lower()} |
| Both players traversed in every path | {str(self.both_players_traversed_in_every_scenario).lower()} |
| Native transitions / states created | {self.native_transition_count} / {self.native_state_count} |
| Raw private card IDs persisted | {self.raw_private_card_ids_persisted} |

These paths use deterministic priority policies only to exercise the full-game
boundary. Their wins/losses are not a blueprint evaluation or strength claim.
The live root comes from an unseeded native battle, so this artifact is not
claimed byte-deterministic across independent runs.

## Claim boundary

| Claim | Status |
|---|---|
{claim_rows}
"""


@dataclass(slots=True)
class Phase6NativeGameArtifacts:
    evidence: Phase6NativeGameEvidence


def _runtime_basics(runtime_directory: Path) -> tuple[int, ...]:
    path_text = str(runtime_directory)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    from cg.api import all_card_data  # type: ignore[import-not-found]

    return tuple(card.cardId for card in all_card_data() if card.basic)


def _completions(
    observation: Mapping[str, Any],
    *,
    target: KaggleDeckProfile,
    candidates: tuple[KaggleDeckProfile, ...],
    basic_card_ids: tuple[int, ...],
) -> tuple[HiddenZoneCompletion, ...]:
    probability = 1.0 / len(candidates)
    return tuple(
        build_hidden_zone_completion(
            observation,
            hypothesis_id=profile.archetype,
            probability=probability,
            your_candidate_id=target.archetype,
            opponent_candidate_id=profile.archetype,
            your_archetype_id=target.archetype,
            opponent_archetype_id=profile.archetype,
            your_style_id="unknown",
            opponent_style_id="unknown",
            your_card_ids=target.card_ids,
            opponent_card_ids=profile.card_ids,
            your_deck_fingerprint=target.fingerprint,
            opponent_deck_fingerprint=profile.fingerprint,
            basic_card_ids=basic_card_ids,
            manual_coin=True,
            source=(
                "Phase 6J equal-weight local candidate prior; native hidden-zone "
                f"completion derived from {profile.provenance}"
            ),
        )
        for profile in candidates
    )


def _matching_action(
    actions: tuple[AbstractAction, ...],
    selection: tuple[int, ...],
) -> AbstractAction:
    matches = [item for item in actions if item.official_option_indices == selection]
    if len(matches) != 1:
        raise RuntimeError("baseline native selection did not lift exactly once")
    return matches[0]


def _traverse_scenario(
    game: OfficialNativeSearchGame,
    chance_outcome: ChanceOutcome,
    *,
    target: KaggleDeckProfile,
    opponent: KaggleDeckProfile,
    max_depth: int = 2_000,
) -> NativeScenarioTraversalEvidence:
    state = game.transition(game.root_state(), None, chance_outcome=chance_outcome)
    agents = (
        PriorityBaseline(target.card_ids, style="tempo"),
        PriorityBaseline(opponent.card_ids, style="tempo"),
    )
    decisions = [0, 0]
    coin_nodes = 0
    maximum_actions = 0
    information_sets: set[str] = set()
    midgame_branch_verified = False
    for _ in range(max_depth):
        if state.terminal:
            break
        if state.actor is InformationSetActor.CHANCE:
            outcomes = game.chance_outcomes(state)
            if len(outcomes) != 2:
                raise RuntimeError("non-root native chance node is not a coin")
            coin_nodes += 1
            state = game.transition(state, None, chance_outcome=min(outcomes, key=lambda item: item.outcome_id))
            continue
        player = state.player_index
        assert player is not None
        information_set = game.information_set(state, player=player)
        belief = game.public_belief(state, player=player)
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise RuntimeError("native information set/public belief mismatch")
        information_sets.add(information_set.info_set_id)
        actions = information_set.legal_actions
        maximum_actions = max(maximum_actions, len(actions))
        if (
            not midgame_branch_verified
            and state.depth >= 10
            and 2 <= len(actions) <= 16
        ):
            left = game.transition(state, actions[0])
            right = game.transition(state, actions[1])
            repeated = game.transition(state, actions[0])
            midgame_branch_verified = (
                left.state_id != right.state_id
                and left.state_id == repeated.state_id
                and left.observation == repeated.observation
            )
        selection = tuple(agents[player].choose(state.observation))
        state = game.transition(state, _matching_action(actions, selection))
        decisions[player] += 1
    if not state.terminal:
        raise RuntimeError("native search traversal exceeded the full-game depth cap")
    utility_0 = game.terminal_utility(state, 0)
    utility_1 = game.terminal_utility(state, 1)
    completion = game.completion(opponent.archetype)
    return NativeScenarioTraversalEvidence(
        candidate_id=opponent.archetype,
        completion_fingerprint=completion.fingerprint,
        public_evidence_adjustments=completion.public_evidence_adjustments,
        terminal_reached=True,
        terminal_depth=state.depth,
        player_0_decisions=decisions[0],
        player_1_decisions=decisions[1],
        explicit_coin_chance_nodes=coin_nodes,
        maximum_complete_action_count=maximum_actions,
        distinct_information_sets=len(information_sets),
        terminal_utility_player_0=utility_0,
        terminal_zero_sum_exact=math.isclose(utility_0, -utility_1, abs_tol=1e-12),
        midgame_branch_verified=midgame_branch_verified,
    )


def run_phase6_native_search_game(
    *,
    project_root: str | Path | None = None,
) -> Phase6NativeGameArtifacts:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
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
    candidates = OFFICIAL_PANEL_DECKS + (target,)
    basics = _runtime_basics(runtime)
    observation, start = battle_start(
        list(target.card_ids),
        list(OFFICIAL_PANEL_DECKS[0].card_ids),
    )
    if observation is None:
        raise RuntimeError(
            f"official engine rejected Phase 6J root decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    try:
        completions = _completions(
            observation,
            target=target,
            candidates=candidates,
            basic_card_ids=basics,
        )
        with OfficialNativeSearchGame(
            observation,
            completions,
            runtime_directory=runtime,
        ) as game:
            chance_root = game.root_state()
            chance_outcomes = game.chance_outcomes(chance_root)
            scenario_roots = [
                game.transition(chance_root, None, chance_outcome=outcome)
                for outcome in chance_outcomes
            ]
            public_equivalence = all(
                sanitize_official_public_state(state.observation)
                == sanitize_official_public_state(observation)
                for state in scenario_roots
            )
            private_equivalence = all(
                official_private_observation_payload(state.observation)
                == official_private_observation_payload(observation)
                for state in scenario_roots
            )
            root_beliefs = [
                game.public_belief(state, player=int(state.player_index))
                for state in scenario_roots
            ]
            root_information_sets = [
                game.information_set(state, player=int(state.player_index))
                for state in scenario_roots
            ]
            belief_invariant = len({item.fingerprint for item in root_beliefs}) == 1
            information_set_invariant = (
                len({item.info_set_id for item in root_information_sets}) == 1
            )
            selected_truth_enters_policy = not (
                belief_invariant and information_set_invariant
            )
            root_actions = game.legal_actions(scenario_roots[0])
            children = [
                game.transition(scenario_roots[0], action) for action in root_actions
            ]
            repeated = game.transition(scenario_roots[0], root_actions[0])
            repeated_deterministic = (
                repeated.state_id == children[0].state_id
                and repeated.observation == children[0].observation
            )
            tampered_action_rejected = False
            try:
                game.transition(
                    scenario_roots[0],
                    AbstractAction(
                        action_id=root_actions[0].action_id,
                        label="tampered",
                        official_option_indices=root_actions[0].official_option_indices,
                        features=root_actions[0].features,
                    ),
                )
            except ValueError:
                tampered_action_rejected = True
            tampered_chance_rejected = False
            try:
                game.transition(
                    chance_root,
                    None,
                    chance_outcome=ChanceOutcome(
                        outcome_id=chance_outcomes[0].outcome_id,
                        probability=chance_outcomes[0].probability,
                        public_label="tampered",
                    ),
                )
            except ValueError:
                tampered_chance_rejected = True

            scenarios = tuple(
                _traverse_scenario(
                    game,
                    outcome,
                    target=target,
                    opponent=profile,
                )
                for outcome, profile in zip(chance_outcomes, candidates, strict=True)
            )
            manifests = tuple(item.manifest for item in completions)
            raw_private_ids_persisted = sum(
                int(
                    any(
                        key in manifest
                        for key in (
                            "your_deck",
                            "your_prize",
                            "opponent_deck",
                            "opponent_prize",
                            "opponent_hand",
                            "opponent_active",
                        )
                    )
                )
                for manifest in manifests
            )
            all_terminal = all(item.terminal_reached for item in scenarios)
            both_players = all(
                item.player_0_decisions > 0 and item.player_1_decisions > 0
                for item in scenarios
            )
            all_midgame_branching = all(
                item.midgame_branch_verified for item in scenarios
            )
            gates = (
                game.source_official_engine_executed,
                game.provenance.kind.value == "abstract_information_set_game",
                not game.provenance.official_engine_executed,
                game.branchable_native_transitions,
                not game.hidden_completion_prior_official,
                len(chance_outcomes) == 6,
                math.isclose(sum(item.probability for item in chance_outcomes), 1.0),
                public_equivalence,
                private_equivalence,
                belief_invariant,
                information_set_invariant,
                not selected_truth_enters_policy,
                len(children) == len(root_actions) >= 2,
                len({item.state_id for item in children}) == len(children),
                repeated_deterministic,
                tampered_action_rejected,
                tampered_chance_rejected,
                raw_private_ids_persisted == 0,
                all_terminal,
                both_players,
                all(item.terminal_zero_sum_exact for item in scenarios),
                all_midgame_branching,
            )
            status = (
                "phase6j_branchable_native_search_game_validated_partial"
                if all(gates)
                else "failed"
            )
            claims = (
                ("native_descendant_transitions", "genuine_official_search_execution"),
                ("hidden_completion_prior", "supplied_equal_weight_research_abstraction"),
                ("game_boundary", "abstract_information_set_game"),
                ("full_game_reachability", "six_of_six_priority_paths_terminal"),
                ("counterfactual_action_branching", "validated_root_and_midgame"),
                ("selected_hidden_truth_policy_input", "none_at_root_invariant"),
                ("raw_private_zone_persistence", "fingerprints_and_counts_only"),
                ("equilibrium_blueprint", "not_trained"),
                ("counterfactual_value_bounds", "not_calibrated_full_game"),
                ("safe_resolver", "not_deployed_official"),
                ("opponent_exploitation", "disabled_official"),
                ("priority_path_results", "boundary_exercise_not_strength_evidence"),
                ("kaggle_submission", "not_attempted"),
                ("phase6_complete", "false"),
            )
            evidence = Phase6NativeGameEvidence(
                benchmark_id="phase6j-native-search-information-set-game-v1",
                status=status,
                runtime_fingerprint=game.ruleset_fingerprint,
                game_fingerprint=game.fingerprint,
                compatibility_deck_fingerprint=target.fingerprint,
                source_official_engine_executed=game.source_official_engine_executed,
                game_boundary_kind=game.provenance.kind.value,
                game_official_engine_executed=game.provenance.official_engine_executed,
                hidden_completion_prior_official=game.hidden_completion_prior_official,
                branchable_native_transitions=game.branchable_native_transitions,
                completion_count=len(completions),
                completion_probability_sum=sum(
                    item.probability for item in chance_outcomes
                ),
                raw_private_card_ids_persisted=raw_private_ids_persisted,
                root_public_observation_equivalence=public_equivalence,
                root_private_observation_equivalence=private_equivalence,
                root_information_set_invariant_across_completions=information_set_invariant,
                root_public_belief_invariant_across_completions=belief_invariant,
                selected_completion_truth_enters_policy_payload=selected_truth_enters_policy,
                root_legal_action_count=len(root_actions),
                root_distinct_alternative_children=len(
                    {item.state_id for item in children}
                ),
                repeated_root_transition_deterministic=repeated_deterministic,
                tampered_action_rejected=tampered_action_rejected,
                tampered_chance_outcome_rejected=tampered_chance_rejected,
                all_scenarios_terminal=all_terminal,
                both_players_traversed_in_every_scenario=both_players,
                native_transition_count=game.transition_count,
                native_state_count=game.created_state_count,
                scenarios=scenarios,
                completion_manifests=manifests,
                claims=claims,
            )
            if evidence.status == "failed":
                raise RuntimeError("Phase 6J native search game failed a declared gate")
            return Phase6NativeGameArtifacts(evidence=evidence)
    finally:
        battle_finish()


def main() -> None:
    print(run_phase6_native_search_game().evidence.to_json())


if __name__ == "__main__":
    main()
