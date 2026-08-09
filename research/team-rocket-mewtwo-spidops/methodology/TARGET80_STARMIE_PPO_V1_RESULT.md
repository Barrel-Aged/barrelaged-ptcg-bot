# Target-80 Starmie PPO V1 result

Status: **closed after independent confirmation failure; no submission authority**  
Recorded: 2026-08-05

## Motivation

The first 19 rated games for `HelbrosRocketSpidopsWobbFreeV6` were 10-9, but
the exact live split was 0-4 against the two Starmie training pins. The larger
V5 archive was also only 1-3 against Starmie, so this was a persistent matchup
blocker rather than evidence that Wobbuffet removal alone caused a regression.

## Harness improvement

`scripts/train_rank1_second_seat_ppo.py` now supports:

- `--seat first|second`, while retaining `second` as its default;
- repeated `--opponent` arguments for deterministic round-robin multi-opponent
  rollouts and evaluations;
- code-based opponent bundles without a top-level `model.pt` in the final
  provenance report.

All source matches and evaluations used the bundled official engine. The
trainable policy saw only its normal public observation and own private zones.
The Wobbuffet-free deck and frozen neural trunk were preserved; only the action
and value heads were trainable. These changes improve the research harness but
do not establish policy strength.

## Frozen substrate

- control:
  `work/target80_candidates/flg_wobbfree_scale_control_v1`;
- deck SHA-256:
  `e2f02e3d29a37c3c126daf92c920693a6d25ec00943e2ee725422dbf27604b73`;
- base model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- Wobbuffet count: 0;
- seat: Rocket first;
- opponents:
  `SparkStarmieMistFlatV20_20260801_extracted` and
  `SparkStarmieFroslassCFV13_extracted`.

## Branch results

### MistFlat-only PPO

Trainer evaluation improved from 12/40 to 17/40. A fresh MistFlat panel was
directionally positive, 21/50 versus 15/50 control. It failed transfer to the
other live Starmie identity: 13/50 versus 14/50 control against Froslass.

Artifacts:

- `work/target80_training/starmie_mist_first_ppo_v1/`;
- `work/target80_starmie_ppo_v1_development/mist_first_100.manifest.json`;
- `work/target80_starmie_ppo_v1_development/froslass_first_100.manifest.json`.

### Joint two-Starmie PPO

Balanced trainer evaluation improved only from 25/80 to 28/80. Fresh panels
were split and rejected:

- MistFlat: candidate 18/50, control 14/50;
- Froslass: candidate 15/50 with one draw, control 22/50 with four draws.

The branch traded away the harder variant rather than learning a robust shared
strategy.

Artifacts:

- `work/target80_training/starmie_multi_first_ppo_v1/`;
- `work/target80_starmie_multi_ppo_v1_development/mist_first_100.manifest.json`;
- `work/target80_starmie_multi_ppo_v1_development/froslass_first_100.manifest.json`.

### Froslass-only PPO

Trainer evaluation appeared strongest, improving from 14/40 with two draws to
21/40 with one draw. The larger independent panel overturned it:

- Froslass: candidate 33/100 with three draws, control 39/100 with two draws;
- MistFlat cross-check: candidate 15/50, control 14/50.

The selected checkpoint therefore overfit its bounded trainer evaluation.

Artifacts:

- `work/target80_training/starmie_froslass_first_ppo_v1/`;
- `work/target80_starmie_froslass_ppo_v1_development/froslass_first_200.manifest.json`;
- `work/target80_starmie_froslass_ppo_v1_development/mist_first_100.manifest.json`.

## Decision

Close all three terminal-reward Starmie PPO checkpoints. Do not combine them
with V6, package them, submit them, or cite their trainer-best scores as
strength. Sparse end-of-game reward and small checkpoint evaluations did not
produce stable policy improvement.

The next aligned route is paired counterfactual supervision at identical
official roots, using common-random continuations and a game-disjoint holdout.
Any successor must improve absolute matchup strength, preserve dormant-route
parity, and plausibly support the requested 80% arena target before submission.
