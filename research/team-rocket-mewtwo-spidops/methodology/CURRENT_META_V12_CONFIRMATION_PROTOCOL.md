# Current-meta V12 confirmation protocol

Frozen: 2026-08-07, before V12 confirmation outcomes.

## Decision context

V10 epoch 4 is the strongest coherent post-V8 candidate:

| Cell | V5 | V10 epoch 4 |
|---|---:|---:|
| generic Dragapult | 17/24 | 21/24 |
| current Marnie | 10/24 | 13/24 |
| Alakazam threat proxy | 24/24 | 24/24 |
| flat Starmie | 3/24 | 5/24 |
| aggregate | 54/96 | 63/96 |

It failed V10's preregistered absolute 8/24 flat-Starmie floor and remains
ineligible under that equal-cell protocol. V11's prospective Maximum Belt
repair also failed at 6/24. Neither threshold is changed.

The user objective, however, is a new internal-arena strength estimate. In the
immutable 2,453-game V8 pull, current scheduling was 1,412 Marnie, 473 Rocket,
380 Alakazam, 183 Soldier, and only two Starmie games. An equal-cell Starmie
absolute floor therefore answers a robustness question but materially
overweights the observed live frequency. V12 is a new, disclosed deployment
question: does the already frozen V10 epoch-4 checkpoint reproduce a broad
positive result on a fresh panel that includes the current high-frequency
families, a direct V5 mirror, and a Starmie tail-risk cell?

## Immutable candidate

- package source: `dragapult_v49_awr_epoch4_v10`;
- exact V5 deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- V5 early-Spidops and exact-lethal Bangle wrapper unchanged;
- no V8 Articuno/Wobbuffet guards and no V11 Maximum Belt;
- model is the already sealed V49 retention-AWR epoch 4; no retraining,
  interpolation, or checkpoint search is allowed.

## Fresh confirmation

Run 24 new side-swapped official-engine games against each of:

1. `soldier_drag_marl_v4_proxy`;
2. `AshV178MarnieFullBest_20260801_extracted`;
3. immutable V5 itself;
4. `violet_alakazam_threat_v1_proxy`; and
5. `SparkStarmieMistFlatV20_20260801_extracted`.

Total: 120 games. None may reuse a development output file.

Advance only with zero agent/engine/legality failures and all of:

- at least 65/120 wins overall;
- Dragapult at least 17/24;
- Marnie at least 11/24;
- V5 direct at least 12/24;
- Alakazam at least 20/24; and
- Starmie at least 4/24.

These floors require a positive broad result without pretending the known
Starmie weakness is solved. Failure closes V12. Passing authorizes exactly one
immutable package audit and one internal-arena submission. It does not
authorize Kaggle submission or a verified live-V49 strength claim.
