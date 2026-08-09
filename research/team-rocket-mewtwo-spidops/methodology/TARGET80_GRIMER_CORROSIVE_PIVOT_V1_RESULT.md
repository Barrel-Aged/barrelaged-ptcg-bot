# Target-80 Grimer corrosive pivot V1 result

Recorded: 2026-08-05

Status: **rejected; exact two-Grimer package closed**

## Frozen result

All 36 official-engine games were terminal, with zero draws, crashes, illegal
selections, or timeouts.

| Arm | W-L vs `AshV178MarnieFullBest` |
|---|---:|
| `GrimerCorrosivePivotV1` | 2-10 |
| deck-identical `GrimerCorrosiveControlV1` | 2-10 |
| immutable V5 | 7-5 |

V1 required at least 9-3 and a two-win lead over each control. It missed every
outcome gate: it tied the deck-identical control and trailed immutable V5 by
five wins.

## Mechanism audit

The intended card interaction was engine-real and frequent, but it did not
convert to match wins:

- `Corrosive Sludge` was used on ten distinct game-turns across six games,
  exceeding the four-game activation requirement.
- In five of those games, Grimmsnarl used `Shadow Bullet` to knock out the
  attacking Grimer and was then discarded by the Corrosive Sludge end-turn
  effect. Its attached Basic Darkness Energy was discarded with it: four,
  two, five, two, and two Energy respectively.
- The sixth activation game ended after repeated Froslass damage-counter
  placement knocked out Grimer, without the planned Grimmsnarl exchange.
- Every game containing a Corrosive Sludge activation was a candidate loss.
  This is a descriptive trace fact, not a causal matchup estimate.

The trace therefore rejects the original strategic premise, not merely its
implementation. Trading a one-Prize Grimer for the opponent's Grimmsnarl and
attached Energy occurred reliably, but the two-Grimer list lost too much setup
quality and board continuity to exploit the exchange. The deck-identical
control's 2-10 result also shows that this exact deck substitution itself was
materially harmful on the frozen seeds.

## Decision

Close the exact two-Grimer list and its public-state Corrosive pivot overlay.
Do not broaden, package, submit, or pool these games with another identity.
The result does not by itself exclude every possible singleton Grimer list,
but another Grimer attempt would require a materially new plan rather than
seed-level tuning of this failed package.

The live V5 package remains unchanged.

## Durable evidence

- protocol: `TARGET80_GRIMER_CORROSIVE_PIVOT_V1_PROTOCOL.md`;
- candidate: `work/target80_candidates/grimer_corrosive_pivot_v1/`;
- deck-identical control:
  `work/target80_candidates/grimer_corrosive_control_v1/`;
- smoke: `work/target80_grimer_corrosive_smoke_v1.jsonl`;
- frozen panel and traces:
  `work/target80_grimer_corrosive_explore_v1/`.

Evidence boundary: all card interactions, legality, and outcomes above are
from official-engine execution. This was a narrow public-state tactical
overlay, not the final game-theoretic planner or a hidden-information model.
