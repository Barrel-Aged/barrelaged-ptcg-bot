# barrelaged-ptcg-bot

My competition agent for the Pokémon TCG AI Battle Challenge (cabt/`cg` engine). Built per the plan
in `~/ptcg-abc/plan/` (PROJECT_PLAN / ARCHITECTURE / RULES_AND_REQUIREMENTS / SELECTCONTEXT_MATRIX).

## Layout
```
main.py            # Kaggle entrypoint: agent(obs_dict) -> list[int]
deck.csv           # 60 card IDs (currently dragapult-ex, engine-verified legal)
bot/
  agent.py         # wires policy stack to the engine contract; guarantees legal, non-raising moves
  engine_io.py     # deck load, obs access, always-legal safe_selection
  budget.py        # BudgetManager — the ~600s/episode compute bank
  policies/        # base.py + default_safe.py (→ heuristic.py, mcts.py, rl.py)
  deck_id/         # opponent archetype identifier (Phase 2)
  eval/            # gauntlet + metrics (Phase-2 onward)
models/            # RL checkpoints (weights git-ignored)
tests/             # pytest guardrails (legality / no-raise / deck step)
```

## Current state
**Sprint 2 scaffold.** Ships `DefaultSafePolicy` — always-legal, never-times-out, plays badly on
purpose. Real heuristics arrive in Sprint 3 (Phase 1). The architecture (swappable policy, budget
bank, deck step) is in place and test-gated.

## Develop
```bash
export PYTHONPATH=/path/to/sample_submission   # cg engine, from the Simulation bundle
pip install pytest
pytest -q                                      # guardrails must stay green (Definition of Done)
./build.sh                                     # -> submission.tar.gz (main.py + deck.csv + bot/)
```

## Submission notes (from RULES_AND_REQUIREMENTS)
- `cg` is provided by the Environment — **not** bundled.
- **No internet at eval** — the bot is fully self-contained.
- Time is a **~600s per-episode bank** (not per-move) — `BudgetManager` spends it across the game.
