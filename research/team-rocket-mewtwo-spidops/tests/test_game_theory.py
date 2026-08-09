from hashlib import sha256
import json
import unittest

from rocketbot.game_theory import (
    AbstractAction,
    ActionDistribution,
    BaselineConstrainedExploitationOverlay,
    BlueprintFallbackResolver,
    CounterfactualValueBounds,
    ExploitationConstraint,
    GameBoundaryKind,
    GameBoundaryProvenance,
    GameTheoreticPlanningRequest,
    HypothesisValueBound,
    InformationSet,
    InformationSetActor,
    OpponentModelSignal,
    OpponentRange,
    PublicBeliefState,
    RangeHypothesis,
    ResolveBudget,
    SafeResolveRequest,
    SafePlanningPipeline,
    SafetyCertificate,
    UniformBlueprintStub,
)


def digest(label):
    return sha256(label.encode()).hexdigest()


def fixture():
    opponent_range = OpponentRange.normalized(
        (
            RangeHypothesis(
                hypothesis_id="setup-hand",
                weight=3.0,
                hidden_resource_counts=(("search", 1), ("energy", 2)),
                intent_id="setup",
                style_id="rational",
                source="public action posterior",
            ),
            RangeHypothesis(
                hypothesis_id="disruption-hand",
                weight=1.0,
                hidden_resource_counts=(("gust", 1),),
                intent_id="disruption",
                style_id="deceptive",
                source="public action posterior",
            ),
        )
    )
    belief = PublicBeliefState(
        public_state_fingerprint=digest("state"),
        public_history_fingerprint=digest("history"),
        turn=5,
        acting_player=InformationSetActor.SELF,
        opponent_range=opponent_range,
        public_features=(("prize_margin", 0.25), ("board_pressure", -0.1)),
        updater_fingerprint=digest("updater"),
    )
    actions = (
        AbstractAction("setup", "Preserve setup", (0,), (("setup", 1.0),)),
        AbstractAction("attack", "Take prize", (1,), (("tempo", 1.0),)),
    )
    info_set = InformationSet(
        info_set_id="turn5-main",
        public_belief_fingerprint=belief.fingerprint,
        actor=InformationSetActor.SELF,
        legal_actions=actions,
        depth=0,
    )
    return opponent_range, belief, info_set


class _TinyAbstractGame:
    def __init__(self, info_set):
        self._info_set = info_set
        self._provenance = GameBoundaryProvenance(
            name="tiny abstract unit game",
            kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
            schema_version=1,
            ruleset_fingerprint=digest("tiny-rules"),
            official_engine_executed=False,
            fidelity_notes="Unit-test abstraction only.",
        )

    @property
    def provenance(self):
        return self._provenance

    @property
    def fingerprint(self):
        return digest("tiny-game")

    def information_set(self, belief):
        return self._info_set

    def transition(self, information_set, action, *, chance_outcome=None):
        raise NotImplementedError

    def chance_outcomes(self, information_set):
        return ()

    def terminal_utility(self, information_set, player):
        raise ValueError("not terminal")


class GameTheoryPivotTests(unittest.TestCase):
    def test_public_belief_range_is_normalized_serializable_and_truth_free(self):
        opponent_range, belief, _ = fixture()
        self.assertAlmostEqual(opponent_range.hypotheses[0].weight, 0.75)
        self.assertAlmostEqual(opponent_range.entropy, 0.5623351446)
        self.assertGreater(opponent_range.effective_sample_size, 1.0)
        restored = PublicBeliefState.from_dict(belief.to_dict())
        self.assertEqual(restored, belief)
        encoded = json.dumps(belief.to_dict(), sort_keys=True)
        self.assertNotIn("true_hidden", encoded)
        self.assertEqual(len(belief.fingerprint), 64)
        restored_range = OpponentRange.from_dict(opponent_range.to_dict())
        self.assertEqual(restored_range.to_dict(), opponent_range.to_dict())
        self.assertEqual(
            PublicBeliefState.from_dict(belief.to_dict()).fingerprint,
            belief.fingerprint,
        )

    def test_abstract_boundary_cannot_claim_official_execution(self):
        with self.assertRaises(ValueError):
            GameBoundaryProvenance(
                name="bad abstraction",
                kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
                schema_version=1,
                ruleset_fingerprint=digest("bad"),
                official_engine_executed=True,
                fidelity_notes="invalid claim",
            )

    def test_information_set_boundary_distinguishes_chance_and_player_nodes(self):
        _, belief, _ = fixture()
        chance = InformationSet(
            info_set_id="chance-draw",
            public_belief_fingerprint=belief.fingerprint,
            actor=InformationSetActor.CHANCE,
            legal_actions=(),
            depth=1,
        )
        self.assertEqual(chance.actor, InformationSetActor.CHANCE)
        with self.assertRaises(ValueError):
            InformationSet(
                info_set_id="bad-player-node",
                public_belief_fingerprint=belief.fingerprint,
                actor=InformationSetActor.SELF,
                legal_actions=(),
                depth=1,
            )

    def test_uniform_blueprint_and_fallback_resolver_are_explicit_stubs(self):
        opponent_range, belief, info_set = fixture()
        game = _TinyAbstractGame(info_set)
        blueprint = UniformBlueprintStub(
            game_fingerprint=game.fingerprint,
            deck_fingerprint=digest("deck"),
        )
        baseline = blueprint.strategy(info_set, belief)
        self.assertFalse(blueprint.checkpoint.trained)
        self.assertEqual(baseline.as_dict(), {"setup": 0.5, "attack": 0.5})
        bounds = CounterfactualValueBounds(
            opponent_player=1,
            bounds=tuple(
                HypothesisValueBound(item.hypothesis_id, -0.1, 0.1, 0.0)
                for item in opponent_range.hypotheses
            ),
            confidence=0.95,
            baseline_strategy_fingerprint=baseline.fingerprint,
            source="unit-test interval stub",
        )
        pipeline = SafePlanningPipeline(
            blueprint=blueprint,
            resolver=BlueprintFallbackResolver(),
        )
        decision = pipeline.decide(
            game,
            GameTheoreticPlanningRequest(
                information_set=info_set,
                belief=belief,
                resolve_budget=ResolveBudget(10, 100, 5.0),
                opponent_value_bounds=bounds,
            ),
        )
        self.assertEqual(decision.final_strategy, baseline)
        self.assertTrue(decision.resolved.fallback_to_blueprint)
        self.assertTrue(decision.resolved.certificate.safe)
        self.assertFalse(decision.official_engine_executed)
        self.assertEqual(
            decision.final_strategy_safety_status,
            "inherits_resolve_certificate",
        )
        self.assertEqual(decision.planner_status, "abstract_interface_stub_untrained")

    def test_resolve_bounds_are_bound_to_the_exact_baseline_strategy(self):
        opponent_range, belief, info_set = fixture()
        baseline = ActionDistribution.normalized({"setup": 0.5, "attack": 0.5})
        mismatched_bounds = CounterfactualValueBounds(
            opponent_player=1,
            bounds=tuple(
                HypothesisValueBound(item.hypothesis_id, -0.1, 0.1, 0.0)
                for item in opponent_range.hypotheses
            ),
            confidence=0.95,
            baseline_strategy_fingerprint=digest("different-baseline"),
            source="unit mismatch",
        )
        with self.assertRaises(ValueError):
            SafeResolveRequest(
                information_set=info_set,
                belief=belief,
                baseline_strategy=baseline,
                opponent_value_bounds=mismatched_bounds,
                budget=ResolveBudget(10, 100, 5.0),
            )

    def test_exploitation_overlay_stays_within_safe_baseline_constraints(self):
        _, belief, info_set = fixture()
        baseline = ActionDistribution.normalized({"setup": 0.5, "attack": 0.5})
        exploit = ActionDistribution.normalized({"setup": 0.05, "attack": 0.95})
        certificate = SafetyCertificate(
            safe=True,
            max_counterfactual_value_violation=0.0,
            tolerance=0.0,
            confidence=0.95,
            bound_fingerprint=digest("bounds"),
            method="unit safe bound",
        )
        overlay = BaselineConstrainedExploitationOverlay(
            ExploitationConstraint(
                max_kl_from_safe_strategy=0.01,
                max_blend=0.5,
                min_baseline_mass_fraction=0.8,
                min_model_confidence=0.6,
            )
        )
        result = overlay.apply(
            safe_strategy=baseline,
            safety_certificate=certificate,
            opponent_model=OpponentModelSignal(
                exploitative_strategy=exploit,
                confidence=0.9,
                effective_samples=100,
                calibration_error=0.05,
                model_fingerprint=digest("opponent-model"),
            ),
        )
        result.strategy.validate_support(info_set)
        self.assertTrue(result.used_opponent_model)
        self.assertLessEqual(result.blend, 0.2 + 1e-9)
        self.assertLessEqual(result.kl_from_safe_strategy, 0.01 + 1e-9)
        self.assertGreaterEqual(result.strategy.probability("setup"), 0.4 - 1e-9)

        unsafe = SafetyCertificate(
            safe=False,
            max_counterfactual_value_violation=0.1,
            tolerance=0.0,
            confidence=0.95,
            bound_fingerprint=digest("unsafe-bounds"),
            method="unit unsafe bound",
        )
        fallback = overlay.apply(
            safe_strategy=baseline,
            safety_certificate=unsafe,
            opponent_model=OpponentModelSignal(
                exploitative_strategy=exploit,
                confidence=1.0,
                effective_samples=100,
                calibration_error=0.0,
                model_fingerprint=digest("opponent-model"),
            ),
        )
        self.assertFalse(fallback.used_opponent_model)
        self.assertEqual(fallback.strategy, baseline)


if __name__ == "__main__":
    unittest.main()
