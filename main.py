"""Kaggle submission entrypoint — Pokémon TCG AI Battle Challenge (cabt/`cg` engine).

The engine calls `agent(obs_dict)` each decision point. Keep this file at the top level of the
submission .tar.gz alongside `deck.csv`; the `bot/` package ships beside it.
"""
import os
import sys

# Make the `bot` package importable whether run from CWD or /kaggle_simulations/agent/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.agent import Agent  # noqa: E402

_AGENT = Agent()


def agent(obs_dict: dict) -> list:
    return _AGENT.act(obs_dict)
