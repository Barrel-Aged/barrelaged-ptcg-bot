import unittest

import torch

from rocketbot.policy import (
    ActionComponent,
    ActionPrefix,
    ActionToken,
    CandidateBatch,
    NeuralAutoregressiveTacticalPolicy,
    TacticalCandidate,
)


class TacticalPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(55)
        self.policy = NeuralAutoregressiveTacticalPolicy(
            state_embedding_dim=12,
            plan_dim=7,
            candidate_feature_dim=8,
            hidden_dim=16,
        )

    def test_candidate_scoring_mask_and_recurrent_advance(self) -> None:
        state = torch.randn(2, 12)
        plan = torch.randn(2, 7)
        hidden = self.policy.initial_hidden(state, plan)
        candidates = CandidateBatch(
            features=torch.randn(2, 4, 8),
            legal_mask=torch.tensor(
                [[True, False, True, True], [False, True, True, False]],
                dtype=torch.bool,
            ),
        )
        logits = self.policy.score_next(hidden, candidates)
        self.assertEqual(tuple(hidden.shape), (2, 16))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertTrue(torch.isneginf(logits[0, 1]).item())
        self.assertTrue(torch.isneginf(logits[1, 0]).item())
        advanced = self.policy.advance(hidden, candidates.features[:, 2])
        self.assertEqual(tuple(advanced.shape), (2, 16))
        self.assertFalse(torch.equal(advanced, hidden))

    def test_candidate_batch_requires_one_legal_choice(self) -> None:
        candidates = CandidateBatch(
            features=torch.zeros(1, 2, 8),
            legal_mask=torch.zeros(1, 2, dtype=torch.bool),
        )
        with self.assertRaises(ValueError):
            self.policy.score_next(torch.zeros(1, 16), candidates)

    def test_semantic_prefix_and_candidate_adapter_contract(self) -> None:
        action = ActionToken(ActionComponent.ACTION_KIND, "attack", "Attack")
        end = ActionToken(ActionComponent.END, "end", "End action")
        prefix = ActionPrefix().append(action).append(end)
        self.assertTrue(prefix.is_complete)
        self.assertEqual(ActionPrefix.from_json(prefix.to_json()), prefix)
        with self.assertRaises(ValueError):
            prefix.append(action)

        batch = CandidateBatch.from_single_step(
            (
                TacticalCandidate(action, (0.0,) * 8),
                TacticalCandidate(end, (1.0,) * 8, legal=False),
            )
        )
        self.assertEqual(tuple(batch.features.shape), (1, 2, 8))
        self.assertEqual(batch.legal_mask.tolist(), [[True, False]])


if __name__ == "__main__":
    unittest.main()
