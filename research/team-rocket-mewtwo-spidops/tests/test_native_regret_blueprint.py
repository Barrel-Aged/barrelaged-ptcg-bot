from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import unittest

from rocketbot.game_theory import AbstractAction
from rocketbot.native_blueprint_features import (
    NATIVE_ACTION_FEATURE_NAMES,
    NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT,
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
)
from rocketbot.native_regret_blueprint import (
    NativeRegretBlueprintArtifact,
    NativeRegretDataset,
    NativeRegretTrainingSample,
    native_regret_sample_view,
    restore_native_regret_blueprint,
    train_native_regret_blueprint,
)
from rocketbot.trajectory_neural import TrajectoryBootstrapArtifact


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


class NativeRegretBlueprintTests(unittest.TestCase):
    def _bootstrap(self) -> TrajectoryBootstrapArtifact:
        path = (
            PROJECT_ROOT
            / "artifacts"
            / "phase6_trajectory_neural"
            / "logged_trajectory_bootstrap.json"
        )
        return TrajectoryBootstrapArtifact.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def _sample(self, split: str, index: int) -> NativeRegretTrainingSample:
        actions = tuple(
            AbstractAction(
                action_id=f"native:{action_index}",
                label=f"select [{action_index}]",
                official_option_indices=(action_index,),
                features=tuple((name, float(action_index)) for name in NATIVE_ACTION_FEATURE_NAMES),
            )
            for action_index in range(2)
        )
        target = ((actions[0].action_id, 0.9), (actions[1].action_id, 0.1))
        behavior = ((actions[0].action_id, 0.5), (actions[1].action_id, 0.5))
        return NativeRegretTrainingSample(
            round_index=1,
            source_candidate_id_evaluation_only=f"candidate_{index}",
            player=index % 2,
            depth=10 + index,
            split=split,
            information_set_id=f"synthetic-info-{index}",
            public_belief_fingerprint=_digest(f"public-{index}"),
            private_observation_fingerprint=_digest(f"private-{index}"),
            public_features=tuple((name, float(index)) for name in NATIVE_PUBLIC_FEATURE_NAMES),
            private_features=tuple((name, float(index)) for name in NATIVE_PRIVATE_FEATURE_NAMES),
            actions=actions,
            behavior_strategy=behavior,
            action_values=((actions[0].action_id, 1.0), (actions[1].action_id, -1.0)),
            instantaneous_regrets=((actions[0].action_id, 1.0), (actions[1].action_id, -1.0)),
            cumulative_regrets=((actions[0].action_id, 1.0), (actions[1].action_id, -1.0)),
            target_strategy=target,
            baseline_value=0.0,
            rollouts_per_action=1,
            native_transition_count=4,
        )

    def test_phase6k_candidate_is_restorable_and_cannot_promote_safety(self) -> None:
        bootstrap = self._bootstrap()
        self.assertEqual(tuple(bootstrap.public_normalizer.names), NATIVE_PUBLIC_FEATURE_NAMES)
        self.assertEqual(tuple(bootstrap.private_normalizer.names), NATIVE_PRIVATE_FEATURE_NAMES)
        self.assertEqual(tuple(bootstrap.action_normalizer.names), NATIVE_ACTION_FEATURE_NAMES)
        samples = tuple(
            self._sample(split, index)
            for index, split in enumerate(("train", "calibration", "validation"))
        )
        dataset = NativeRegretDataset(
            game_fingerprint=_digest("game"),
            runtime_fingerprint=_digest("runtime"),
            deck_fingerprint=_digest("deck"),
            source_bootstrap_fingerprint=bootstrap.fingerprint,
            feature_schema_fingerprint=NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT,
            sampling_seed=7,
            training_rounds=1,
            samples=samples,
        )
        restored_dataset = NativeRegretDataset.from_dict(json.loads(dataset.to_json()))
        self.assertEqual(restored_dataset.fingerprint, dataset.fingerprint)

        run = train_native_regret_blueprint(
            dataset,
            bootstrap,
            seed=7,
            epochs=4,
            learning_rate=0.001,
        )
        self.assertTrue(run.artifact.equilibrium_objective_trained)
        self.assertFalse(run.artifact.equilibrium_convergence_validated)
        self.assertFalse(run.artifact.deployable_blueprint)
        self.assertFalse(run.artifact.calibrated_counterfactual_values)
        self.assertFalse(run.artifact.safe_resolver_compatible)
        self.assertTrue(run.report.gradient_parameter_names)
        self.assertTrue(run.report.initialized_parameter_names)
        self.assertEqual(
            tuple(metric.split for metric in run.report.split_metrics),
            ("train", "calibration", "validation"),
        )

        artifact = NativeRegretBlueprintArtifact.from_dict(
            json.loads(run.artifact.to_json())
        )
        restored = restore_native_regret_blueprint(artifact)
        information_set, belief = native_regret_sample_view(samples[0])
        before = run.provider.predict(information_set, belief)
        after = restored.predict(information_set, belief)
        self.assertEqual(before, after)
        self.assertEqual(
            set(before.strategy.as_dict()),
            {action.action_id for action in information_set.legal_actions},
        )
        self.assertFalse(restored.claim_boundary["equilibrium_convergence_validated"])
        self.assertFalse(restored.claim_boundary["safe_resolving"])
        with self.assertRaises(RuntimeError):
            restored.counterfactual_values(information_set, belief, player=0)

        with self.assertRaises(ValueError):
            replace(run.artifact, equilibrium_convergence_validated=True)

    def test_phase6k_live_artifact_keeps_truth_out_of_policy_features(self) -> None:
        directory = PROJECT_ROOT / "artifacts" / "phase6_native_blueprint"
        evidence = json.loads((directory / "phase6k_evidence.json").read_text(encoding="utf-8"))
        dataset = NativeRegretDataset.from_dict(
            json.loads((directory / "native_regret_dataset.json").read_text(encoding="utf-8"))
        )
        artifact = NativeRegretBlueprintArtifact.from_dict(
            json.loads((directory / "native_regret_blueprint.json").read_text(encoding="utf-8"))
        )
        self.assertEqual(
            evidence["status"],
            "phase6k_native_regret_blueprint_candidate_validated_partial",
        )
        self.assertEqual(evidence["dataset_fingerprint"], dataset.fingerprint)
        self.assertEqual(evidence["model_fingerprint"], artifact.fingerprint)
        self.assertEqual(artifact.dataset_fingerprint, dataset.fingerprint)
        self.assertTrue(evidence["provenance"]["source_official_engine_executed"])
        self.assertFalse(evidence["provenance"]["model_official_engine_executed"])
        self.assertFalse(evidence["checks"]["selected_hidden_truth_in_policy_features"])
        self.assertEqual(evidence["checks"]["raw_private_card_ids_persisted"], 0)
        self.assertEqual({sample.player for sample in dataset.samples}, {0, 1})
        self.assertEqual(
            {sample.source_candidate_id_evaluation_only for sample in dataset.samples},
            {"abomasnow", "alakazam", "crustle", "dragapult", "rocket_mewtwo", "starmie"},
        )
        split_by_information_set = {}
        for sample in dataset.samples:
            previous = split_by_information_set.setdefault(
                sample.information_set_id,
                sample.split,
            )
            self.assertEqual(previous, sample.split)
            policy_payload = {
                "public": sample.public_features,
                "private": sample.private_features,
                "actions": [action.features for action in sample.actions],
            }
            self.assertNotIn(
                sample.source_candidate_id_evaluation_only,
                json.dumps(policy_payload, sort_keys=True),
            )
        provider = restore_native_regret_blueprint(artifact)
        information_set, belief = native_regret_sample_view(dataset.samples[0])
        prediction = provider.predict(information_set, belief)
        self.assertEqual(
            set(prediction.strategy.as_dict()),
            {action.action_id for action in information_set.legal_actions},
        )
        self.assertFalse(provider.claim_boundary["deployable_blueprint"])


if __name__ == "__main__":
    unittest.main()
