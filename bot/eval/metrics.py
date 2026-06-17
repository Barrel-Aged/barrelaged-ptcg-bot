"""Evaluation metrics: win-rate, Elo, and per-move latency aggregation.

Self-contained — no dependency on the daily strategy analyzer or its snapshots. Latency matters
because the engine enforces a ~600 s/episode compute bank (RULES_AND_REQUIREMENTS §9).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    """score_a: 1 win / 0.5 draw / 0 loss. Returns updated (a, b)."""
    exp_a = expected_score(rating_a, rating_b)
    delta = k * (score_a - exp_a)
    return rating_a + delta, rating_b - delta


@dataclass
class Record:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo: float = 1000.0
    move_latencies_s: list[float] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def max_latency_s(self) -> float:
        return max(self.move_latencies_s) if self.move_latencies_s else 0.0

    @property
    def mean_latency_s(self) -> float:
        return sum(self.move_latencies_s) / len(self.move_latencies_s) if self.move_latencies_s else 0.0


class Standings:
    """Accumulates results + latency across games and reports a ranked table."""

    def __init__(self):
        self._rec: dict[str, Record] = {}

    def _r(self, name: str) -> Record:
        return self._rec.setdefault(name, Record())

    def record_game(self, name_a: str, name_b: str, winner: str | None,
                    latencies_a: list[float] | None = None, latencies_b: list[float] | None = None):
        ra, rb = self._r(name_a), self._r(name_b)
        if winner is None:
            ra.draws += 1; rb.draws += 1; score_a = 0.5
        elif winner == name_a:
            ra.wins += 1; rb.losses += 1; score_a = 1.0
        else:
            ra.losses += 1; rb.wins += 1; score_a = 0.0
        ra.elo, rb.elo = update_elo(ra.elo, rb.elo, score_a)
        if latencies_a:
            ra.move_latencies_s.extend(latencies_a)
        if latencies_b:
            rb.move_latencies_s.extend(latencies_b)

    def table(self) -> list[tuple[str, Record]]:
        return sorted(self._rec.items(), key=lambda kv: kv[1].elo, reverse=True)

    def render(self) -> str:
        rows = [f"{'#':<3}{'name':<22}{'W':>4}{'L':>4}{'D':>4}{'win%':>7}{'elo':>7}{'maxlat':>9}"]
        for i, (name, r) in enumerate(self.table(), 1):
            rows.append(f"{i:<3}{name:<22}{r.wins:>4}{r.losses:>4}{r.draws:>4}"
                        f"{r.win_rate*100:>6.1f}%{r.elo:>7.0f}{r.max_latency_s:>8.3f}s")
        return "\n".join(rows)
