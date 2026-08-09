import math
import unittest

import torch
from torch.nn import functional as F

from rocketbot.features import StateTensorizer
from rocketbot.model import Phase2AgentConfig, Phase2BeliefAwareAgent
from rocketbot.observation import ActionHistoryTensorizer
from rocketbot.phase3_demo import _combo_belief, _combo_observation
from rocketbot.phase4_demo import _set_head_values
from rocketbot.plans import TeamRocketPlanProposer
from rocketbot.search_gate import (
    SEARCH_GATE_FEATURE_NAMES,
    CalibratedSearchGate,
    PolicySignals,
    SearchComputeBank,
    SearchGateBatch,
    SearchGateCalibrator,
    SearchGateDataset,
    SearchGateDecision,
    SearchGateFeatureExtractor,
    SearchGateFeatures,
    SearchGateNetwork,
    SearchGateSupervision,
    search_gate_loss,
)


def _gate_context():
    torch.manual_seed(41)
    observation = _combo_observation()
    belief = _combo_belief(observation)
    config = Phase2AgentConfig(
        state_embedding_dim=24,
        plan_dim=12,
        strategic_hidden_dim=20,
        meta_hidden_dim=16,
        candidate_feature_dim=16,
        policy_hidden_dim=20,
        history_hidden_dim=16,
        belief_embedding_dim=16,
    )
    agent = Phase2BeliefAwareAgent(config)
    state_features = StateTensorizer()(
        [observation.public_state], perspective=observation.observer
    )
    history, mask = ActionHistoryTensorizer()([observation], max_events=8)
    with torch.no_grad():
        evaluation = agent(state_features, history, mask)
    _set_head_values(evaluation, (-0.55, 0.15, -0.10, 0.05, -0.60, 0.72))
    return observation, belief, evaluation


class SearchGatePrimitiveTests(unittest.TestCase):
    def test_feature_extractor_emits_normalized_phase4_signals(self) -> None:
        observation, belief, evaluation = _gate_context()
        features = SearchGateFeatureExtractor(TeamRocketPlanProposer()).extract(
            observation,
            evaluation,
            belief,
            PolicySignals(
                confidence=0.20,
                entropy=math.log(4.0) * 0.95,
                legal_action_count=4,
            ),
            SearchComputeBank(capacity=4.0, available=2.0),
        )

        self.assertEqual(len(features.values), len(SEARCH_GATE_FEATURE_NAMES))
        self.assertEqual(features.to_tensor().shape, (1, len(SEARCH_GATE_FEATURE_NAMES)))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in features.values))
        self.assertGreater(features.value("policy_uncertainty"), 0.75)
        self.assertGreater(features.value("head_disagreement"), 0.35)
        self.assertAlmostEqual(features.value("budget_fraction"), 0.5)
        self.assertEqual(SearchGateFeatures.from_dict(features.to_dict()), features)

    def test_compute_bank_is_persistent_and_enforces_budget(self) -> None:
        bank = SearchComputeBank(
            capacity=3.0,
            available=2.0,
            refill_per_turn=0.5,
        )
        spent = bank.spend(1.25)
        self.assertAlmostEqual(spent.available, 0.75)
        self.assertAlmostEqual(spent.spent_total, 1.25)
        self.assertAlmostEqual(spent.refill(2).available, 1.75)
        self.assertEqual(SearchComputeBank.from_dict(spent.to_dict()), spent)
        with self.assertRaises(ValueError):
            spent.spend(1.0)

    def test_dataset_roundtrip_loss_and_both_learned_heads(self) -> None:
        low = SearchGateFeatures(tuple(0.1 for _ in SEARCH_GATE_FEATURE_NAMES))
        high = SearchGateFeatures(tuple(0.9 for _ in SEARCH_GATE_FEATURE_NAMES))
        examples = (
            SearchGateSupervision(low, realized_search_gain=0.01, search_cost=0.08),
            SearchGateSupervision(high, realized_search_gain=0.30, search_cost=0.08),
        )
        dataset = SearchGateDataset(examples)
        self.assertEqual(SearchGateDataset.from_json(dataset.to_json()), dataset)

        batch = SearchGateBatch.from_examples(examples)
        network = SearchGateNetwork(hidden_dim=12)
        loss = search_gate_loss(network(batch.features), batch)
        loss.total.backward()

        self.assertGreater(loss.classification.detach().item(), 0.0)
        self.assertGreaterEqual(loss.gain_regression.detach().item(), 0.0)
        self.assertIsNotNone(network.search_logit.weight.grad)
        self.assertIsNotNone(network.gain_head.weight.grad)
        self.assertGreater(network.search_logit.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(network.gain_head.weight.grad.abs().sum().item(), 0.0)

    def test_calibration_is_heldout_and_utility_aware(self) -> None:
        logits = torch.tensor((-0.6, -0.2, 0.2, 0.6))
        gains = torch.tensor((0.00, 0.02, 0.16, 0.30))
        costs = torch.full((4,), 0.08)
        calibrator = SearchGateCalibrator.fit(
            logits,
            gains,
            costs,
            candidate_temperatures=(0.5, 1.0, 2.0),
            candidate_thresholds=(0.25, 0.5, 0.75),
        )
        labels = gains.gt(costs).float()
        uncalibrated_bce = F.binary_cross_entropy_with_logits(logits, labels)
        calibrated_bce = F.binary_cross_entropy_with_logits(
            logits / calibrator.temperature,
            labels,
        )
        self.assertLessEqual(float(calibrated_bce), float(uncalibrated_bce) + 1e-7)
        self.assertEqual(
            SearchGateCalibrator.from_json(calibrator.to_json()),
            calibrator,
        )

        never_profitable = SearchGateCalibrator.fit(
            torch.full((3,), 8.0),
            torch.zeros(3),
            torch.full((3,), 0.08),
            candidate_temperatures=(1.0,),
            candidate_thresholds=(0.5,),
        )
        self.assertEqual(never_profitable.threshold, 1.0)
        self.assertFalse(
            never_profitable.probabilities(torch.full((3,), 8.0))
            .gt(never_profitable.threshold)
            .any()
        )

    def test_positive_gate_can_be_blocked_by_compute_budget(self) -> None:
        observation, belief, evaluation = _gate_context()
        network = SearchGateNetwork(hidden_dim=12)
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.search_logit.bias.fill_(8.0)
            network.gain_head.bias.fill_(0.25)
        gate = CalibratedSearchGate(
            network,
            SearchGateFeatureExtractor(TeamRocketPlanProposer()),
            SearchGateCalibrator(temperature=1.0, threshold=0.5),
        )
        policy = PolicySignals(confidence=0.2, entropy=1.0, legal_action_count=4)

        affordable = gate.decide(
            observation,
            evaluation,
            belief,
            policy,
            SearchComputeBank(capacity=2.0, available=2.0),
            search_cost=1.0,
        )
        exhausted = gate.decide(
            observation,
            evaluation,
            belief,
            policy,
            SearchComputeBank(capacity=2.0, available=0.0),
            search_cost=1.0,
        )

        self.assertTrue(affordable.should_search)
        self.assertAlmostEqual(affordable.bank_after.available, 1.0)
        self.assertTrue(exhausted.wanted_search)
        self.assertFalse(exhausted.should_search)
        self.assertTrue(exhausted.budget_limited)
        self.assertIn("compute_budget_exhausted", exhausted.reasons)

        forged = affordable.to_dict()
        forged["bank_after"]["available"] = 2.0
        with self.assertRaises(ValueError):
            SearchGateDecision.from_dict(forged)


if __name__ == "__main__":
    unittest.main()
