from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import unittest

from rocketbot import (
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BATCH_SEEDS,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_COVERAGE,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_RANK,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_FROZEN_RADIUS,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_MEAN_WIDTH,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_RANGE_ECE,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_TARGET_SE,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_CANDIDATE_TOP1,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_RANGE_TOP1,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_SLICE_COVERAGE,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_RETURNS_PER_ACTION,
    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_VALIDATION_COVERAGE,
    HypothesisActionAdvantageBound,
    Phase6PublicRangeCounterfactualEvidence,
    PublicRangeCounterfactualAdvantageBounds,
    PublicRangeCounterfactualEvaluation,
    PublicRangeCounterfactualPanel,
    PublicRangeCounterfactualReference,
    counterfactual_advantage_bounds_for_root,
)
from rocketbot.phase6_branch_conditioned_values import (
    build_branch_value_examples_from_roots,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "phase6_public_range_counterfactual_advantages"
)


class PublicRangeCounterfactualAdvantageArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(name: str) -> dict:
            return json.loads(
                (ARTIFACT_DIRECTORY / name).read_text(encoding="utf-8")
            )

        cls.reference_raw = raw("counterfactual_advantage_reference.json")
        cls.panel_raw = raw("validation_panel.json")
        cls.evaluation_raw = raw("counterfactual_advantage_evaluation.json")
        cls.evidence_raw = raw("phase6am_evidence.json")
        cls.reference = PublicRangeCounterfactualReference.from_dict(
            cls.reference_raw
        )
        cls.panel = PublicRangeCounterfactualPanel.from_dict(cls.panel_raw)
        cls.evaluation = PublicRangeCounterfactualEvaluation.from_dict(
            cls.evaluation_raw
        )
        cls.evidence = Phase6PublicRangeCounterfactualEvidence.from_dict(
            cls.evidence_raw
        )

    def test_reference_was_frozen_before_collection(self) -> None:
        reference = self.reference
        self.assertEqual(len(reference.development_root_ids), 360)
        self.assertEqual(len(set(reference.development_root_ids)), 360)
        self.assertEqual(reference.calibration_score_count, 360)
        self.assertEqual(
            reference.calibration_coverage,
            PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_COVERAGE,
        )
        self.assertEqual(
            reference.calibration_rank,
            PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_RANK,
        )
        self.assertAlmostEqual(
            reference.advantage_radius,
            PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_FROZEN_RADIUS,
        )
        self.assertTrue(reference.prior_roots_declared_development)
        self.assertEqual(reference.validation_roots_collected_before_freeze, 0)
        self.assertFalse(reference.absolute_official_chance_reach_available)
        self.assertFalse(reference.safe_resolver_enabled)
        self.assertFalse(reference.official_strategy_control_enabled)

    def test_panel_is_complete_fresh_and_officially_accounted(self) -> None:
        panel = self.panel
        self.assertEqual(panel.source_reference_fingerprint, self.reference.fingerprint)
        self.assertEqual(panel.match_namespace, "phase6am-validation")
        self.assertEqual(panel.official_match_count, 36)
        self.assertEqual(panel.branch_observation_count, 720)
        self.assertEqual(panel.terminal_rollout_count, 5_760)
        self.assertGreaterEqual(panel.native_transition_count, 5_760)
        self.assertGreater(panel.live_selection_count, 0)
        self.assertEqual(
            {
                (
                    root.true_candidate_id_supervision_only,
                    root.true_intent_id_supervision_only,
                )
                for root in panel.roots
            },
            {
                (candidate, intent)
                for candidate in panel.candidate_ids
                for intent in panel.intent_ids
            },
        )
        self.assertFalse(
            {root.root_id for root in panel.roots}
            & set(self.reference.development_root_ids)
        )
        self.assertTrue(
            all(root.match_id.startswith("phase6am-validation-") for root in panel.roots)
        )

    def test_public_range_and_counterfactual_target_shapes(self) -> None:
        for root in self.panel.roots:
            self.assertEqual(root.batch_seeds, PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BATCH_SEEDS)
            self.assertEqual(len(root.candidate_summaries), 6)
            self.assertTrue(
                all(item.particle_count == 32 for item in root.candidate_summaries)
            )
            hypotheses = root.public_belief.opponent_range.hypotheses
            self.assertEqual(len(hypotheses), 6)
            self.assertAlmostEqual(sum(item.weight for item in hypotheses), 1.0)
            self.assertFalse(root.candidate_truth_selected_for_range_input)
            self.assertEqual(root.raw_private_card_ids_persisted, 0)
            self.assertEqual(
                root.common_chance_ratio.counterfactual_likelihood_ratio,
                1.0,
            )
            self.assertFalse(
                root.common_chance_ratio.absolute_chance_reach_available
            )
            self.assertEqual(len(root.batches), 2)
            targets = root.targets(self.reference.advantage_radius)
            self.assertEqual(len(targets), 5)
            examples = build_branch_value_examples_from_roots(
                root.batches,
                self.panel.candidate_ids,
                expected_count=2,
            )
            for action_index, target in enumerate(targets):
                returns = tuple(
                    value
                    for example in examples
                    for value in example.actions[action_index].terminal_returns
                )
                self.assertEqual(
                    len(returns),
                    PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_RETURNS_PER_ACTION,
                )
                probabilities = dict(root.batches[0].value_root.blueprint_probabilities)
                baseline = tuple(
                    sum(
                        probabilities[action.action_id]
                        * example.actions[index].terminal_returns[return_index]
                        for index, action in enumerate(example.actions)
                    )
                    for example in examples
                    for return_index in range(len(example.actions[0].terminal_returns))
                )
                advantages = tuple(
                    value - baseline_value
                    for value, baseline_value in zip(returns, baseline, strict=True)
                )
                self.assertAlmostEqual(
                    target.target_standard_error,
                    statistics.stdev(advantages) / math.sqrt(len(advantages)),
                )

    def test_bound_provider_calibrates_only_the_selected_range_hypothesis(self) -> None:
        for root in self.panel.roots:
            bounds = counterfactual_advantage_bounds_for_root(root, self.reference)
            self.assertEqual(
                PublicRangeCounterfactualAdvantageBounds.from_dict(bounds.to_dict()),
                bounds,
            )
            self.assertEqual(len(bounds.bounds), 30)
            calibrated = [item for item in bounds.bounds if item.calibrated]
            unsupported = [item for item in bounds.bounds if not item.calibrated]
            self.assertEqual(len(calibrated), 5)
            self.assertEqual(
                {item.hypothesis_id for item in calibrated},
                {f"official-range:{root.top_candidate_id}"},
            )
            self.assertTrue(
                all(
                    (item.lower, item.point_center, item.upper) == (-2.0, 0.0, 2.0)
                    for item in unsupported
                )
            )
            self.assertFalse(bounds.absolute_chance_reach_available)
            self.assertFalse(bounds.safe_resolver_enabled)

    def test_roundtrip_provenance_and_no_private_zone_persistence(self) -> None:
        for artifact, raw in (
            (self.reference, self.reference_raw),
            (self.panel, self.panel_raw),
            (self.evaluation, self.evaluation_raw),
            (self.evidence, self.evidence_raw),
        ):
            self.assertEqual(artifact.to_dict(), raw)
        self.assertEqual(_raw_private_key_count(self.panel_raw), 0)
        for raw_root in self.panel_raw["roots"]:
            claims = raw_root["claims"]
            self.assertTrue(claims["one_complete_official_match_executed"])
            self.assertTrue(claims["retained_snapshot_replayed_per_batch"])
            self.assertTrue(claims["descendant_transitions_official_engine_executed"])
            self.assertFalse(claims["range_model_official_engine_executed"])
            self.assertFalse(claims["absolute_official_chance_reach_available"])

    def test_gate_flags_are_exact_and_downstream_is_fail_closed(self) -> None:
        evaluation = self.evaluation
        self.assertEqual(
            evaluation.precision_gate_passed,
            evaluation.maximum_target_standard_error
            <= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_TARGET_SE,
        )
        self.assertEqual(
            evaluation.coverage_gate_passed,
            evaluation.overall_coverage
            >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_VALIDATION_COVERAGE,
        )
        self.assertEqual(
            evaluation.slice_gate_passed,
            evaluation.minimum_slice_coverage
            >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_SLICE_COVERAGE,
        )
        self.assertEqual(
            evaluation.width_gate_passed,
            evaluation.mean_interval_width
            <= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_MEAN_WIDTH,
        )
        self.assertEqual(
            evaluation.bootstrap_gate_passed,
            evaluation.coverage_bootstrap_lower_bound > 0,
        )
        self.assertEqual(
            evaluation.range_gate_passed,
            evaluation.candidate_top1_accuracy
            >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_RANGE_TOP1
            and evaluation.candidate_ece
            <= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_RANGE_ECE
            and evaluation.minimum_candidate_top1_accuracy
            >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_CANDIDATE_TOP1,
        )
        expected_admission = all(
            (
                evaluation.precision_gate_passed,
                evaluation.coverage_gate_passed,
                evaluation.slice_gate_passed,
                evaluation.width_gate_passed,
                evaluation.bootstrap_gate_passed,
                evaluation.range_gate_passed,
                evaluation.reach_gate_passed,
                evaluation.collection_holdout_gate_passed,
            )
        )
        self.assertEqual(
            evaluation.normalized_counterfactual_advantage_bounds_admitted,
            expected_admission,
        )
        self.assertFalse(evaluation.absolute_official_counterfactual_values_available)
        self.assertFalse(evaluation.equilibrium_convergence_validated)
        self.assertFalse(evaluation.safe_resolver_enabled)
        self.assertFalse(evaluation.opponent_exploitation_enabled)
        self.assertFalse(evaluation.official_strategy_control_enabled)
        self.assertEqual(self.evidence.evaluation_fingerprint, evaluation.fingerprint)

    def test_frozen_outcome_rejects_only_the_precision_gate(self) -> None:
        evaluation = self.evaluation
        self.assertFalse(evaluation.precision_gate_passed)
        self.assertTrue(evaluation.coverage_gate_passed)
        self.assertTrue(evaluation.slice_gate_passed)
        self.assertTrue(evaluation.width_gate_passed)
        self.assertTrue(evaluation.bootstrap_gate_passed)
        self.assertTrue(evaluation.range_gate_passed)
        self.assertTrue(evaluation.reach_gate_passed)
        self.assertTrue(evaluation.collection_holdout_gate_passed)
        self.assertFalse(
            evaluation.normalized_counterfactual_advantage_bounds_admitted
        )
        self.assertAlmostEqual(evaluation.overall_coverage, 0.95)
        self.assertAlmostEqual(evaluation.mean_interval_width, 1.110039238550672)
        self.assertAlmostEqual(
            evaluation.minimum_slice_coverage,
            5.0 / 6.0,
        )
        self.assertAlmostEqual(
            evaluation.maximum_target_standard_error,
            0.20168842355216715,
        )
        self.assertAlmostEqual(evaluation.candidate_top1_accuracy, 1.0)
        self.assertAlmostEqual(
            evaluation.candidate_ece,
            2.975360839596597e-06,
        )
        self.assertAlmostEqual(
            evaluation.coverage_bootstrap_lower_bound,
            0.022222222222222254,
        )
        self.assertEqual(
            self.evidence.status,
            "normalized_counterfactual_advantage_bounds_rejected_fail_closed",
        )

    def test_capability_metric_and_provenance_tampering_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.reference, validation_roots_collected_before_freeze=1)
        with self.assertRaises(ValueError):
            replace(
                self.panel.roots[0],
                candidate_truth_selected_for_range_input=True,
            )
        with self.assertRaises(ValueError):
            replace(
                self.evaluation,
                precision_gate_passed=not self.evaluation.precision_gate_passed,
            )
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)
        with self.assertRaises(ValueError):
            HypothesisActionAdvantageBound(
                hypothesis_id="unsupported",
                action_id="setup",
                lower=-1.0,
                upper=1.0,
                point_center=0.0,
                calibrated=False,
            )


if __name__ == "__main__":
    unittest.main()
