# Team Rocket's Mewtwo ex / Spidops strategic-planning research

This repository is the durable research record and project-authored source for
an ambitious Pokemon TCG agent: structured public game state, persistent plan
latents, six strategic value heads, state-conditioned value aggregation,
autoregressive tactical actions, opponent-plan and range abstractions,
selective search, search-guided learning, and a game-theoretic planning pivot.

Development was paused on 2026-08-05 after submitting one final internal-arena
iteration, `HelbrosRocketSpidopsArticunoV7`. That bot uses the strongest frozen
zero-Wobbuffet checkpoint retained by the experiments, the previously admitted
early-Spidops and exact-lethal guards, and a narrow public-information Articuno
bench-slot reservation. Its arena package SHA-256 is
`8858324f993656691caa59700b968b88390e3edce94068df8dfb5cf28e455289`.

No claim of superhuman strength, an 80% arena win rate, exploitability bounds,
or equilibrium convergence is made. The final arena estimate was pending when
development paused.

## Start here

- [`methodology/PROJECT_PLAN.md`](methodology/PROJECT_PLAN.md) is the full
  multi-phase plan, including the 18 research ideas, phases 1-6, target
  architecture, starting deck, and intended sequence.
- [`methodology/PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md`](methodology/PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md)
  records the public-belief, information-set, blueprint, safe re-solving, and
  constrained-exploitation target.
- [`methodology/PLAN_COMPLETION_MATRIX.md`](methodology/PLAN_COMPLETION_MATRIX.md)
  separates implemented interfaces and experiments from future work.
- [`methodology/STRENGTH_DIAGNOSIS_AND_RECOVERY.md`](methodology/STRENGTH_DIAGNOSIS_AND_RECOVERY.md)
  explains why the integrated research scaffold and the live heuristic/BC bot
  are different artifacts.
- [`methodology/PAUSED_DECK_REWORK_REVISIT_NOTE.md`](methodology/PAUSED_DECK_REWORK_REVISIT_NOTE.md)
  is the detailed continuation checkpoint.
- [`arena_bot/README.md`](arena_bot/README.md) documents the final submitted
  wrapper, deck, guards, hashes, and runtime boundary.

## Repository map

- `src/rocketbot/`: all project-authored strategic substrate, belief/range,
  plan-search, selective-search, learning, official-adapter, telemetry,
  replay, and game-theoretic interface code.
- `tests/`: project test suite and deterministic research checks.
- `scripts/`: experiment, audit, training, packaging, and analysis programs.
- `docs/`: supporting completion audits and design documents.
- `methodology/`: the complete chronological protocols, locks, reports, and
  rejection records from the project root.
- `arena_bot/`: final project-authored arena wrapper and exact deck identity.

## Core scaffold smoke

Python 3.11+ and PyTorch 2+ are required.

```bash
python -m pip install -e .
rocketbot-smoke
```

The full test collection includes experiments that require local official
engine artifacts and replay corpora. Those dependencies are intentionally not
published here; run focused core tests in an environment that lacks them.

## Important boundaries

The repository contains all source authored in this project, but it does not
redistribute the official competition engine, pulled opponent bundles, private
arena telemetry/replays, credentials, third-party packaged inference runtime,
or the 17 MB frozen model checkpoint. The final wrapper therefore documents
but does not independently recreate the submitted archive. See
[`PUBLICATION_BOUNDARY.md`](PUBLICATION_BOUNDARY.md) for the exact exclusions
and hashes.

Pokemon and related names are trademarks of their respective owners. This is
an independent research project and is not affiliated with or endorsed by The
Pokemon Company, Nintendo, Creatures, or Game Freak.

