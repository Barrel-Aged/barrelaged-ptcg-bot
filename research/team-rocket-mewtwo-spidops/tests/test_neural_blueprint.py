import json
import unittest

from rocketbot.deck import load_default_deck
from rocketbot.exact_game import (
    DuelCard,
    RocketPrizeDuel,
    analytic_equilibrium_profile,
)
from rocketbot.neural_blueprint import (
    DUEL_BLUEPRINT_FEATURE_NAMES,
    DuelInformationEncoder,
    NeuralBlueprintArtifact,
    provider_profile,
    regret_matching_targets_from_profile,
    restore_neural_duel_blueprint,
    train_neural_duel_blueprint,
)


class NeuralBlueprintTests(unittest.TestCase):
    def test_encoder_training_and_exact_checkpoint_roundtrip(self):
        game = RocketPrizeDuel()
        teacher = analytic_equilibrium_profile(game)
        state = next(
            item
            for item in game.all_states()
            if item.cards[0] is DuelCard.SPIDOPS and item.history == ()
        )
        information_set = game.information_set(state, player=0)
        belief = game.public_belief(state, player=0)
        encoder = DuelInformationEncoder()
        self.assertEqual(
            len(encoder.encode(information_set, belief)),
            len(DUEL_BLUEPRINT_FEATURE_NAMES),
        )
        self.assertEqual(sum(encoder.legal_action_mask(information_set)), 2)

        run = train_neural_duel_blueprint(
            game,
            teacher,
            regret_matching_targets_from_profile(game, teacher),
            deck_fingerprint=load_default_deck().fingerprint,
            strategy_mode="policy",
            epochs=500,
        )
        self.assertTrue(run.provider.checkpoint.trained)
        self.assertLess(run.report.metrics.nash_conv, 0.001)
        self.assertLess(run.report.conditional_value_rmse, 0.002)

        encoded = json.loads(json.dumps(run.artifact.to_dict()))
        restored_artifact = NeuralBlueprintArtifact.from_dict(encoded)
        restored = restore_neural_duel_blueprint(game, restored_artifact)
        restored_profile = provider_profile(game, restored, label=run.profile.label)
        for info in game.information_sets():
            representative = game.states_for_information_set(info.info_set_id)[0]
            actor = representative.acting_player
            perspective = game.public_belief(representative, player=actor)
            self.assertEqual(
                restored.strategy(info, perspective),
                run.provider.strategy(info, perspective),
            )
        self.assertEqual(restored_profile.strategies, run.profile.strategies)
        self.assertEqual(restored.checkpoint.model_fingerprint, run.artifact.fingerprint)

        encoded["seed"] += 1
        with self.assertRaises(ValueError):
            NeuralBlueprintArtifact.from_dict(encoded)


if __name__ == "__main__":
    unittest.main()
