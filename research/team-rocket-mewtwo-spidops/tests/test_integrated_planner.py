import json
from pathlib import Path
import unittest

from rocketbot.competition import (
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    SelectionContract,
    resolve_official_deck,
)
from rocketbot.integrated_planner import (
    ARCHETYPE_NAMES,
    INTENT_NAMES,
    PLAN_NAMES,
    RESOURCE_NAMES,
    STRATEGIC_AXES,
    FrozenStrategicPlanner,
    IntegratedBeliefPlannerAgent,
    PlannerMemory,
    planner_feature_items,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = PROJECT_ROOT / "artifacts" / "integrated_controller" / "integrated_planner.json"


def observation():
    return {
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}, {"type": 13, "attackId": 608}],
            "deck": None,
            "contextCard": None,
            "effect": None,
        },
        "logs": [],
        "current": {
            "turn": 7,
            "turnActionCount": 3,
            "yourIndex": 0,
            "firstPlayer": 0,
            "result": -1,
            "stadium": [{"id": 1257}],
            "looking": None,
            "players": [
                {
                    "active": [{"id": 431, "hp": 230, "maxHp": 280, "energies": [5, 5], "energyCards": [{"id": 15}], "tools": [], "preEvolution": []}],
                    "bench": [{"id": 401, "hp": 130, "maxHp": 130, "energies": [1], "energyCards": [{"id": 1}], "tools": [], "preEvolution": [{"id": 400}]}],
                    "deckCount": 31,
                    "discard": [{"id": 1220}],
                    "prize": [None] * 4,
                    "handCount": 5,
                    "hand": [{"id": 1134}, {"id": 1}],
                },
                {
                    "active": [{"id": 741, "hp": 90, "maxHp": 90, "energies": [5], "energyCards": [{"id": 5}], "tools": [], "preEvolution": []}],
                    "bench": [],
                    "deckCount": 33,
                    "discard": [{"id": 742}],
                    "prize": [None] * 3,
                    "handCount": 4,
                    "hand": None,
                },
            ],
        },
    }


class IntegratedPlannerTests(unittest.TestCase):
    def setUp(self):
        self.deck = resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )

    def test_frozen_joint_heads_are_shaped_normalized_and_admitted(self):
        raw = json.loads(MODEL.read_text(encoding="utf-8"))
        planner = FrozenStrategicPlanner(raw)
        features = planner_feature_items(observation(), PlannerMemory())
        result = planner.evaluate(features, [0.2] * 5, allow_safe_resolve=True)
        self.assertEqual(len(result.archetype_probabilities), len(ARCHETYPE_NAMES))
        self.assertEqual(len(result.intent_probabilities), len(INTENT_NAMES))
        self.assertEqual(len(result.resource_probabilities), len(RESOURCE_NAMES))
        self.assertEqual(len(result.final_plan_probabilities), len(PLAN_NAMES))
        self.assertEqual(len(result.strategic_values), len(STRATEGIC_AXES))
        self.assertEqual(len(result.strategic_weights), len(STRATEGIC_AXES))
        self.assertAlmostEqual(sum(result.archetype_probabilities), 1.0)
        self.assertAlmostEqual(sum(result.intent_probabilities), 1.0)
        self.assertAlmostEqual(sum(result.final_plan_probabilities), 1.0)
        self.assertAlmostEqual(sum(result.strategic_weights), 1.0)
        self.assertTrue(result.bounds_admitted)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in result.resource_probabilities))

    def test_live_action_is_neural_legal_persistent_and_serializable(self):
        agent = IntegratedBeliefPlannerAgent(self.deck, MODEL)
        choice = agent.choose(observation())
        self.assertEqual(SelectionContract.from_observation(observation()).validate(choice), tuple(choice))
        self.assertEqual(agent.memory.decision_index, 1)
        self.assertEqual(len(agent.memory.telemetry), 1)
        summary = agent.summary()
        self.assertEqual(summary["strategy_provider"], "frozen_public_belief_neural_blueprint")
        self.assertFalse(summary["heuristic_final_controller"])
        self.assertFalse(summary["sampled_rollout_final_controller"])
        self.assertIn(summary["telemetry"][-1]["strategic_objective"], STRATEGIC_AXES)
        json.dumps(summary)

    def test_artifact_fingerprint_rejects_weight_tampering(self):
        raw = json.loads(MODEL.read_text(encoding="utf-8"))
        raw["layers"]["plan_value_head"]["bias"][0] += 0.01
        with self.assertRaises(ValueError):
            FrozenStrategicPlanner(raw)


if __name__ == "__main__":
    unittest.main()
