from __future__ import annotations

import json
from pathlib import Path
import unittest

from rocketbot.native_plan_blueprint import (
    NativePlanBlueprintArtifact,
    NativePlanOutcomeDataset,
    native_plan_sample_view,
    restore_native_plan_blueprint,
)
from rocketbot.native_plan_game import NATIVE_PLAN_ACTIONS
from rocketbot.phase6_native_plan_iterations import (
    Phase6NativePlanIterationsEvidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativePlanIterationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = PROJECT_ROOT / "artifacts" / "phase6_native_plan_iterations"
        cls.dataset = NativePlanOutcomeDataset.from_dict(
            json.loads(
                (cls.directory / "iterative_native_plan_dataset.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cls.artifact = NativePlanBlueprintArtifact.from_dict(
            json.loads(
                (cls.directory / "iterative_native_plan_blueprint.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        raw_evidence = json.loads(
            (cls.directory / "phase6p_evidence.json").read_text(encoding="utf-8")
        )
        cls.evidence = Phase6NativePlanIterationsEvidence.from_dict(raw_evidence)

    def test_rounds_update_all_three_models_and_accumulate_memories(self) -> None:
        evidence = self.evidence
        self.assertEqual(
            evidence.status,
            "phase6p_iterative_native_plan_candidate_validated_partial",
        )
        self.assertEqual(len(evidence.rounds), 3)
        previous = (0, 0, 0)
        for round_item in evidence.rounds:
            current = (
                round_item.cumulative_player_0_samples,
                round_item.cumulative_player_1_samples,
                round_item.cumulative_strategy_samples,
            )
            self.assertTrue(all(after > before for before, after in zip(previous, current)))
            self.assertLess(
                round_item.final_training_loss,
                round_item.initial_training_loss,
            )
            self.assertGreater(round_item.player_0_mean_strategy_drift, 0.0)
            self.assertGreater(round_item.player_1_mean_strategy_drift, 0.0)
            self.assertGreater(round_item.average_plan_mean_strategy_drift, 0.0)
            self.assertGreater(round_item.nonzero_regret_actions_added, 0)
            previous = current
        self.assertEqual(
            evidence.final_dataset_fingerprint,
            self.dataset.fingerprint,
        )
        self.assertEqual(evidence.final_model_fingerprint, self.artifact.fingerprint)
        self.assertEqual(evidence.raw_private_card_ids_persisted, 0)
        self.assertGreater(evidence.raw_regret_p95_absolute, 1e9)
        self.assertGreater(
            evidence.raw_regret_maximum_absolute,
            evidence.raw_regret_p95_absolute,
        )
        self.assertEqual(evidence.maximum_restore_drift, 0.0)

    def test_sampling_strategies_are_iteration_indexed_and_model_is_not_safe(self) -> None:
        samples = (
            self.dataset.player_0_regrets.samples
            + self.dataset.player_1_regrets.samples
        )
        self.assertEqual({sample.iteration for sample in samples}, {1, 2, 3})
        strategy_vectors = {
            tuple(round(value, 10) for _, value in sample.current_strategy)
            for sample in samples
        }
        self.assertGreater(len(strategy_vectors), 3)
        provider = restore_native_plan_blueprint(self.artifact)
        information_set, belief = native_plan_sample_view(samples[0])
        support = {action.action_id for action in NATIVE_PLAN_ACTIONS}
        self.assertEqual(set(provider.strategy(information_set, belief).as_dict()), support)
        for player in (0, 1):
            self.assertEqual(
                set(
                    provider.current_strategy(
                        information_set, belief, player=player
                    ).as_dict()
                ),
                support,
            )
        self.assertFalse(self.artifact.equilibrium_convergence_validated)
        self.assertFalse(self.artifact.calibrated_counterfactual_values)
        self.assertFalse(self.artifact.safe_resolver_compatible)
        self.assertFalse(self.artifact.deployable_blueprint)
        with self.assertRaises(RuntimeError):
            provider.counterfactual_values(information_set, belief, player=1)

    def test_evidence_roundtrip_preserves_strict_claim_boundary(self) -> None:
        restored = Phase6NativePlanIterationsEvidence.from_dict(
            json.loads(self.evidence.to_json())
        )
        self.assertEqual(restored.fingerprint, self.evidence.fingerprint)
        claims = dict(restored.claims)
        self.assertEqual(
            claims["native_descendant_transitions"],
            "genuine_official_search_execution",
        )
        self.assertEqual(claims["turn_plan_game"], "research_abstraction")
        self.assertEqual(claims["equilibrium_convergence"], "not_validated")
        self.assertEqual(claims["counterfactual_value_bounds"], "not_calibrated")
        self.assertEqual(claims["safe_resolver"], "not_enabled")
        self.assertEqual(claims["phase6_complete"], "false")


if __name__ == "__main__":
    unittest.main()
