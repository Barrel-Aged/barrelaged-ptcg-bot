"""Gauntlet: play the bot against opponents piloting the prechosen archetype decks.

Self-contained, built on the curated decks (`bot/eval/gauntlet/*.csv`) + whatever policy the bot
uses — independent of the daily strategy analyzer. Sequential here (cg holds one global battle per
process); `parallel.py` runs many of these across processes.

A "competitor" is a (name, deck_ids, policy_factory) triple. The default policy is whatever the
bot currently ships; swap in HeuristicPolicy / MctsPolicy as phases land.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

from bot.agent import Agent
from bot.policies.base import Policy
from bot.policies.default_safe import DefaultSafePolicy

GAUNTLET_DIR = os.path.join(os.path.dirname(__file__), "gauntlet")


def load_gauntlet_decks() -> dict[str, list[int]]:
    decks = {}
    for fn in sorted(os.listdir(GAUNTLET_DIR)):
        if fn.endswith(".csv"):
            with open(os.path.join(GAUNTLET_DIR, fn)) as f:
                decks[fn[:-4]] = [int(l) for l in f.read().splitlines() if l.strip()]
    return decks


@dataclass
class Competitor:
    name: str
    deck: list[int]
    policy_factory: Callable[[], Policy] = DefaultSafePolicy


def _agent_for(c: Competitor) -> Agent:
    a = Agent(policy=c.policy_factory())
    a._deck = c.deck  # pilot this competitor's prechosen deck
    return a


def play_game(a: Competitor, b: Competitor) -> tuple[str | None, list[float], list[float]]:
    """Play one cg game. Returns (winner_name|None, latencies_a, latencies_b)."""
    from cg.game import battle_finish, battle_select, battle_start

    agents = [_agent_for(a), _agent_for(b)]
    names = [a.name, b.name]
    lat: list[list[float]] = [[], []]

    obs, start = battle_start(a.deck, b.deck)
    if start.errorPlayer >= 0:
        battle_finish()
        return names[1 - start.errorPlayer], [], []  # illegal deck loses
    while obs["current"]["result"] < 0:
        yi = obs["current"]["yourIndex"]
        t0 = time.monotonic()
        sel = agents[yi].act(obs)
        lat[yi].append(time.monotonic() - t0)
        obs = battle_select(sel)
    result = obs["current"]["result"]
    battle_finish()
    winner = None if result == 2 else names[result]
    return winner, lat[0], lat[1]


def run_gauntlet(hero: Competitor, opponents: list[Competitor], games_each: int = 10):
    """Play `hero` vs each opponent `games_each` times; return a Standings."""
    from bot.eval.metrics import Standings

    standings = Standings()
    for opp in opponents:
        for g in range(games_each):
            # alternate who is player 0 to cancel first-player bias
            first, second = (hero, opp) if g % 2 == 0 else (opp, hero)
            winner, lat0, lat1 = play_game(first, second)
            la = lat0 if first.name == hero.name else lat1
            lb = lat1 if first.name == hero.name else lat0
            standings.record_game(hero.name, opp.name, winner, la, lb)
    return standings
