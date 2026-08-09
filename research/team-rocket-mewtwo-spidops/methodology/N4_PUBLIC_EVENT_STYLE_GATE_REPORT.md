# N4 Public-Event Opponent-Style Gate Report

Status: **rejected at calibration; policy validation not run**

Protocol: `rocket-public-event-style-gate/v1`

## Purpose

N4 tested whether explicit, public opponent actions and board events could safely identify the two frozen arena opponents well enough to activate the N2 residual only against the apparent favorable Recovery V4 matchup. The feature representation used 644 public/own-visible values: 512 hashed opponent-event fields, 128 opponent-board hashes, and four public scalars. It did not use the opponent's hidden hand, deck order, or prizes.

## Data and execution

- Training: 400 official-engine games, 200 per opponent.
- Calibration: 200 official-engine games, 100 per opponent.
- Reserved validation: 400 official-engine games, 200 per opponent.
- Every collection game put the Rocket policy in the second seat and reported zero exception fallbacks.
- Frozen Rocket deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Frozen base model SHA-256: `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`.

## Calibration result

Training loss decreased from 0.7126 to 0.6189, but this did not produce a usable early-game classifier. Temperature calibration selected 4.6226. Final-step calibration accuracy was only 55%. Under the locked safety requirement of no more than 1% false activation on V110, the only admissible threshold was 1.0; that threshold activated on 0/100 V110 games and 0/100 Recovery V4 games. The classifier therefore provided no actionable separation.

The calibration gate failed before any policy-strength validation. N4 is rejected and must not be used to select or submit a bot.

## Reserved validation handling

The reserved validation collectors completed, but the `.pt` payloads remain unopened and unanalyzed because calibration had already failed. Their immutable hashes are recorded only to preserve the audit trail:

- `validation_v110.pt`: `a6bfff3228af073f28defbfe69a33cd94ce747ae2985cc1ca13271edd1333260`
- `validation_recovery.pt`: `3f2f78b0bc9d432223bae14102b3e9d60c044465cf0f8f28a9e8cec86600762f`

Classifier artifact SHA-256: `1f54153cd7e82cdb71d31504ad9551dd5c5edbe68e0d67415d6bbbd5b00fa4b2`.

## Decision

Opponent-style gating is not the limiting capability. The stronger next experiment is N5: directly fine-tune the Rocket relational blueprint under near-greedy on-policy play, then compare only final locked checkpoints in fresh interleaved development games.
