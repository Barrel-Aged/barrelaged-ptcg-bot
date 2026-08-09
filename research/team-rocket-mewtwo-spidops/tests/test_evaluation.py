import json
import unittest

from rocketbot.evaluation import (
    ExternalEvidence,
    OfficialMatchRecord,
    Phase6EvaluationReport,
    wilson_interval,
)


def record(match_id, *, ablation, archetype, style, winner, seat=0):
    return OfficialMatchRecord(
        match_id=match_id,
        agent_label="rocket",
        ablation=ablation,
        opponent_label=archetype,
        opponent_archetype=archetype,
        opponent_style=style,
        agent_seat=seat,
        winner=winner,
        turns=10,
        selections=30,
        decision_count=15,
        runner_fallback_count=0,
        illegal_action_count=0,
        soft_timeout_count=0,
        mean_latency_ms=1.0,
        max_latency_ms=3.0,
        action_counts=(("attack", 3),),
        lethal_opportunities=2,
        lethal_conversions=2,
        first_attack_turn=3,
        first_prize_turn=5,
        own_prizes_remaining=2,
        opponent_prizes_remaining=4,
        mean_own_board=4.0,
        mean_opponent_board=3.0,
        mean_own_hp=500.0,
        mean_opponent_hp=400.0,
        mean_own_hand=5.0,
        mean_option_count=6.0,
        strategic_values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        strategic_weights=(1 / 6,) * 6,
        strategic_aggregate=0.35,
        belief_nll=0.2,
        belief_brier=0.1,
        search_invocations=2,
        search_changes=1,
        mean_search_bonus=0.25,
    )


class EvaluationTests(unittest.TestCase):
    def test_wilson_interval_and_multidimensional_report(self) -> None:
        low, high = wilson_interval(1.5, 2)
        self.assertLess(low, 0.75)
        self.assertGreater(high, 0.75)
        matches = (
            record("m1", ablation="policy_only", archetype="abomasnow", style="tempo", winner=0),
            record("m2", ablation="full", archetype="starmie", style="control", winner=2),
        )
        external = ExternalEvidence(
            simulator_source="official",
            simulator_file_date="2026-07-15",
            card_pool_unique_ids=1267,
            card_pool_sha256="a" * 64,
            exact_deck_playable=False,
            incompatibility="Prism Tower absent",
            prior_related_submission=None,
            prior_related_public_score=None,
            submission_status="blocked_exact_deck_card_pool",
            leaderboard_observed_at="2026-07-26",
        )
        report = Phase6EvaluationReport.build(
            benchmark_id="unit",
            benchmark_seed=1,
            deck_fingerprint="b" * 64,
            environment_fingerprint="c" * 64,
            matches=matches,
            external=external,
        )
        self.assertEqual(report.tactical.decisions, 30)
        self.assertEqual(report.tactical.illegal_action_rate, 0.0)
        self.assertEqual(len(report.by_ablation), 2)
        self.assertEqual(dict(report.claims)["superhuman_strength"], "not_established")
        self.assertIn("Prism Tower", report.to_markdown())
        json.loads(report.to_json())
        self.assertEqual(len(report.fingerprint), 64)


if __name__ == "__main__":
    unittest.main()

