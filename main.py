"""Kaggle submission entrypoint — Pokémon TCG AI Battle Challenge (cabt/`cg` engine).

The engine calls `agent(obs_dict)` each decision point. Keep this file at the top level of the
submission .tar.gz alongside `deck.csv`; the `bot/` and `cg/` packages ship beside it.

Competition agent = MctsPolicy (flat IS-MC over the Search API, deck-ID determinization, heuristic
rollouts). Falls back to the heuristic / always-legal policy if the engine or MCTS can't initialize.
"""
import os
import sys

# Make `bot` and the bundled `cg` importable. The eval harness exec()s this file (so `__file__` is
# undefined — guard it) and runs with CWD = the agent dir (/kaggle_simulations/agent/).
_paths = [os.getcwd(), "/kaggle_simulations/agent"]
try:
    _paths.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
for _p in _paths:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from bot.agent import Agent  # noqa: E402
from bot.engine_io import load_deck  # noqa: E402


def _build_agent() -> Agent:
    try:
        from bot.policies.mcts import MctsPolicy
        deck = load_deck()
        return Agent(policy=MctsPolicy(our_deck=deck))
    except Exception:
        return Agent()   # HeuristicPolicy (or DefaultSafe) — always legal


_AGENT = _build_agent()


def agent(obs_dict: dict) -> list:
    return _AGENT.act(obs_dict)
