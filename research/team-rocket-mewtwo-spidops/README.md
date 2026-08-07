# Team Rocket's Mewtwo ex / Spidops strategic-planning research

This repository is the durable research record and project-authored source for
an ambitious Pokemon TCG agent: structured public game state, persistent plan
latents, six strategic value heads, state-conditioned value aggregation,
autoregressive tactical actions, opponent-plan and range abstractions,
selective search, search-guided learning, and a game-theoretic planning pivot.

Development was paused on 2026-08-06 after submitting one post-V7 internal-
arena iteration, `HelbrosRocketSpidopsBC2RecoveryV8`. The final bot restores
the exact ScaleBCV2/V5 deck and neural checkpoint, retains the admitted early-
Spidops and exact-lethal guards, reserves the fifth bench slot for Articuno in
publicly identified effect-attack matchups, and redirects only selected
Wobbuffet Headbutt Bounce actions when a ready Spidops retreat is offered. Its
arena package SHA-256 is
`876ac11a1878267251844fad2fcae11fcad96f32a8f5b403ab16521e2c335e12`.

No claim of superhuman strength, an 80% arena win rate, exploitability bounds,
or equilibrium convergence is made. The first read-only arena snapshot was
8-3 and is far too small to be a mature strength estimate.

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
- [`methodology/BC2_RECOVERY_V8_SUBMISSION_AND_PAUSE.md`](methodology/BC2_RECOVERY_V8_SUBMISSION_AND_PAUSE.md)
  gives the final diagnosis, rejected zero-Wobbuffet arm, verification, hashes,
  submission acknowledgement, and pause boundary.
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
