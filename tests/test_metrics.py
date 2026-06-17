"""Unit tests for the eval metrics (Elo, win-rate, latency aggregation)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.eval.metrics import Standings, expected_score, update_elo  # noqa: E402


def test_expected_score_symmetry():
    assert abs(expected_score(1000, 1000) - 0.5) < 1e-9
    assert expected_score(1200, 1000) > 0.5 > expected_score(1000, 1200)


def test_update_elo_conserves_points_and_directions():
    a, b = update_elo(1000, 1000, score_a=1.0)
    assert a > 1000 > b
    assert abs((a - 1000) + (b - 1000)) < 1e-9   # zero-sum
    a2, b2 = update_elo(1000, 1000, score_a=0.5)
    assert abs(a2 - 1000) < 1e-9 and abs(b2 - 1000) < 1e-9   # draw between equals = no change


def test_standings_tallies_winrate_and_latency():
    s = Standings()
    s.record_game("hero", "opp", "hero", latencies_a=[0.1, 0.3], latencies_b=[0.05])
    s.record_game("hero", "opp", None)
    s.record_game("hero", "opp", "opp")
    rec = dict(s.table())["hero"]
    assert (rec.wins, rec.draws, rec.losses) == (1, 1, 1)
    assert abs(rec.win_rate - 1 / 3) < 1e-9
    assert rec.max_latency_s == 0.3
    assert "hero" in s.render() and "elo" in s.render()
