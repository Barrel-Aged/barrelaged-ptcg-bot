# Target-80 native Rocket Rush V1R protocol

Status: **completed; exact route rejected by strength and attacker-chain gates**  
Locked: 2026-08-05 (America/Denver)

## Hypothesis

The selected exact-Hancang model reached 136-73-1 in its locked broad
evaluation but only 12/30 and 11/30 against the two Starmie opponents. Fresh
official decision traces exposed a repeatable sequential miss: the policy can
establish Spidops on global turn 3 and create Charging Up fuel by retreating,
yet it often attaches to Mewtwo, evolves the wrong Tarountula, discards
Spidops to Ultra Ball, or ends with a one-energy Spidops.

V1R tests the complete public sequence rather than another isolated action:

1. prefer Tarountula as opening Active when the opening hand permits;
2. use Proton, directly or through Transceiver, to establish two Tarountula
   lines and at least three Team Rocket Pokemon;
3. search/reserve Spidops and attach Grass to the line or retreat pivot;
4. on the second own turn, evolve the correct target, attach, retreat or use
   Giovanni, use Charging Up, and select Rocket Rush; and
5. search, evolve, charge, and attach toward a second Spidops before spending
   subsequent energy elsewhere.

The motivation is not merely earlier damage. A first-seat turn-3 Rocket Rush
can remove Staryu before Mega Starmie evolves, while the second line is needed
to preserve the favorable single-prize trade if Starmie survives. This is a
public-state mechanism probe and prospective trajectory teacher. Its
deterministic wrapper is not eligible to become the final strategic policy.

## Development history excluded from admission

The one-game V1 smoke, four-game V1 smoke, and four-game corrected V1R smoke
under `work/target80_native_rocket_rush_smoke_*` were open implementation
debugging. V1 went 0/4 after failing to bank a Giovanni and to search the
second line. Corrected V1R went 2/4; two games used turn-3 Rocket Rush, while
the two wins were the games that reached three Spidops. These nine games are
excluded from every gate and may not be pooled with the locked panel.

## Frozen arms

Both arms use the exact same 60 cards and selected exact-deck checkpoint:

- deck SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`;
- model SHA-256:
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`;
- Wobbuffet (`432`) count: zero.

Control: `work/target80_candidates/hancang_elitebc_iter3_control_v1`

- semantic tree:
  `abbe2a41d279c3227d4d6e7436ce89745608e025ee597e2283f6cbe2e270a24b`;
- main:
  `3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341`.

Candidate: `work/target80_candidates/hancang_native_rocket_rush_route_v1r`

- semantic tree:
  `5dad6f831a99edfe57ce2b2d74eb222d4caa6ec4f5211cbb6c2b07b665b7be98`;
- main:
  `9639148fa063e796589c4259f9c6ae28d73b09a7555d513d57657738b23cc76b`.

## Frozen opponents and collector

| Opponent | Semantic tree | Deck |
|---|---|---|
| `SparkStarmieMistFlatV20_20260801_extracted` | `25690bea08bc73dcd93207b1e3ed77428ed09f1a15d6137251f2e06b6bd7dd79` | `64ef6e2c0f5fadc12f96426aa80a07e1402501ed00d1778a0badd25d171e4901` |
| `SparkStarmieFroslassCFV13_extracted` | `c693adec6062422241f2b52caf6beba6518913c2c1ed66581b369eba15b92f76` | `60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61` |

Collector:
`scripts/collect_target80_deck_matched_bc_v1_panel.py`

- SHA-256:
  `b5076f3e1f5765f437ec38fe13498c94b109fbac086d6372b377b907d59a0410`;
- genuine official `cg` legality and transitions;
- both packaged runtimes execute in isolated official-evaluator workers;
- candidate is always engine seat 0, matching the internal-arena convention;
- randomized interleaving of 12 candidate and 12 control games per opponent;
- no hidden opponent hand, visualizer truth, or realized hidden allocation;
- no output or manifest may exist before launch;
- neither result file may be opened or summarized until both collectors end.

Locked outputs and seeds:

- MistFlat: seed `806401`,
  `work/target80_native_rocket_rush_v1r/mist_route_vs_control.jsonl`;
- Froslass: seed `806402`,
  `work/target80_native_rocket_rush_v1r/fros_route_vs_control.jsonl`.

## Admission gates

All gates are required:

1. Exactly 48 terminal games, 12 per arm/opponent cell, unchanged semantic
   identities, and zero engine, worker, legality, timeout, or fallback errors.
2. Candidate at least 18 wins in 24 games (75%).
3. Candidate at least six more wins than the deck-identical control.
4. Candidate at least eight wins in each 12-game opponent family.
5. Opening-window Rocket Rush (turn 3 for this first-seat panel) in at least
   12 candidate games, including at least eight attacks directly into Staryu.
6. At least two Spidops observed in 18 candidate games, at least three Spidops
   in 10, and at least two simultaneously ready Spidops in eight.
7. Candidate first-Rocket-Rush mean no later than turn 5 among games where it
   attacks, with Rocket Rush occurring in at least 20 candidate games.

Failure closes this exact deck/model/wrapper and forbids a learned successor
from its traces. A mechanism pass authorizes only a separately frozen learned
policy experiment in which the wrapper actions become supervision or a
public-state plan target. It does not authorize packaging, arena submission,
Kaggle submission, or an 80% claim.

## Claim boundary

Games, legality, actions, transitions, prizes, and terminal outcomes are
genuine official-engine execution. The named opening plan, its deterministic
wrapper, and the admission telemetry are research abstractions. V1R is not
CFR, public-belief search, safe re-solving, or exploitability evidence. The
heuristic wrapper will not be submitted even if it passes.

## Result

Both sealed collectors completed before either result was analyzed. All 48
games terminated in the genuine official engine with exact cell counts,
unchanged bundle identities, and zero engine, worker, legality, timeout, or
fallback errors.

| Opponent | Candidate | Deck-identical control |
|---|---:|---:|
| MistFlat Starmie | 2-10 | 5-7 |
| Froslass Starmie | 2-8-2 | 3-9 |
| Total | **4-18-2** | **8-16** |

The sequential route executed its opening mechanism but did not convert it
into wins:

- Rocket Rush occurred in 22/24 candidate games, with mean first use on turn
  4.18 among those games;
- 15/24 used opening-window turn-3 Rocket Rush and 14 attacks directly targeted
  Staryu;
- only 11/24 ever showed two Spidops, 2/24 showed three, and 2/24 showed two
  simultaneously ready Spidops; and
- the candidate made 58 Rocket Rush attacks in total yet won only four games.

Thus gates 1, 5, and 7 passed, but the decisive gates failed: 4 candidate wins
versus the required 18, four fewer wins rather than six more than control,
2/12 rather than 8/12 in each family, and every multi-Spidops/ready-chain
floor. Early Staryu pressure by itself is not the missing strategic mechanism.
The exact deck/model/wrapper is closed, its traces are not authorized as
learning supervision, and no package or submission was created.

Evidence:

- `work/target80_native_rocket_rush_v1r/mist_route_vs_control.jsonl`, SHA-256
  `1bffa193d53006248c567f2693564f1bd17fec92851512fea6d9c0d78db2cf4d`;
- `work/target80_native_rocket_rush_v1r/fros_route_vs_control.jsonl`, SHA-256
  `d6249a8f5d4bc5bbf4b220e7b424989d4fc6762bc4386e71f975da1ae802580a`;
- Mist manifest SHA-256
  `8f75e3749fed597d3818728c341603eab3dd20b409a13142c9f30379976bd045`;
- Fros manifest SHA-256
  `85f66a1991497752e4a6a3d59b92c7dbaa8e224482d209f8465a289ec0cc1bd`.
