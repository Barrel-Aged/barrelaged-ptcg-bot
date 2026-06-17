"""Parallel gauntlet: run many cg games concurrently across processes.

`cg` keeps one global battle per process, so concurrency is by PROCESS, never thread. We use a
'spawn' pool so each worker imports `cg` fresh (clean native state). This powers fast head-to-head
eval now and feeds RL self-play later.
"""
from __future__ import annotations

import multiprocessing as mp

from bot.eval.gauntlet import Competitor, play_game
from bot.eval.metrics import Standings


def _play_task(args):
    hero, opp, swap = args
    a, b = (opp, hero) if swap else (hero, opp)
    winner, lat0, lat1 = play_game(a, b)
    hero_lat = lat1 if swap else lat0
    opp_lat = lat0 if swap else lat1
    return opp.name, winner, hero_lat, opp_lat


def run_gauntlet_parallel(hero: Competitor, opponents: list[Competitor],
                          games_each: int = 10, workers: int | None = None) -> Standings:
    workers = workers or max(1, mp.cpu_count() - 2)
    tasks = [(hero, opp, (g % 2 == 1)) for opp in opponents for g in range(games_each)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.map(_play_task, tasks)
    standings = Standings()
    for opp_name, winner, hero_lat, opp_lat in results:
        standings.record_game(hero.name, opp_name, winner, hero_lat, opp_lat)
    return standings
