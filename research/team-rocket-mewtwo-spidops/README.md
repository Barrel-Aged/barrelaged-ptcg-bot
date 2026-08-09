# Team Rocket's Mewtwo ex / Spidops strategic-planning research

This repository is the durable research record and project-authored source for
an ambitious Pokemon TCG agent: structured public game state, persistent plan
latents, six strategic value heads, state-conditioned value aggregation,
autoregressive tactical actions, opponent-plan and range abstractions,
selective search, search-guided learning, and a game-theoretic planning pivot.

The original Team Rocket deck track was paused on 2026-08-06 after the
Articuno bench-reservation and narrow Wobbuffet-exit iteration
`HelbrosRocketSpidopsBC2RecoveryV8`. A later ladder-strength investigation used
the same research infrastructure to build one final Mega Lucario candidate,
`HelbrosMegaLucarioV123PublicBCSafeHybrid`, from public leader trajectories.
V123 combines an admitted neural behavior clone with the live deterministic
Lucario baseline on low-confidence decisions. It was accepted by the internal
arena on 2026-08-08; no direct Kaggle submission was made.

No claim of superhuman strength, 1000 Elo, public-leader equivalence,
exploitability bounds, or equilibrium convergence is made. Small local panels
and initial arena snapshots are admission/health evidence, not mature strength
estimates.

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
- [`methodology/MAJKEL_LUCARIO_PUBLIC_BC_V3_SAFE_HYBRID_PROTOCOL.md`](methodology/MAJKEL_LUCARIO_PUBLIC_BC_V3_SAFE_HYBRID_PROTOCOL.md)
  is the preregistration for the final confidence-gated composition.
- [`methodology/MAJKEL_LUCARIO_PUBLIC_BC_V3_SAFE_HYBRID_RESULT.md`](methodology/MAJKEL_LUCARIO_PUBLIC_BC_V3_SAFE_HYBRID_RESULT.md)
  records its frozen calibration, official-engine gates, packaging, and arena
  boundary.
- [`arena_bot/README.md`](arena_bot/README.md) documents the final submitted
  Team Rocket wrapper, deck, guards, hashes, and runtime boundary.
- [`competition_bot_v123/README.md`](competition_bot_v123/README.md) documents
  the final Lucario composition and exact publication boundary.

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
- `competition_bot_v123/`: final authored Lucario entrypoint, hybrid router,
  exact deck, calibration summary, and runtime/model hashes.

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
arena telemetry/replays, credentials, copied baseline/inference runtimes, or
binary frozen model checkpoints. The final wrappers therefore document but do
not independently recreate the submitted archives. See
[`PUBLICATION_BOUNDARY.md`](PUBLICATION_BOUNDARY.md) for the exact exclusions
and hashes.

Pokemon and related names are trademarks of their respective owners. This is
an independent research project and is not affiliated with or endorsed by The
Pokemon Company, Nintendo, Creatures, or Game Freak.
