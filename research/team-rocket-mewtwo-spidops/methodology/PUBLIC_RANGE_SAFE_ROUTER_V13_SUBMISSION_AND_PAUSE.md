# Public-range safe router V13 submission and pause

Completed: 2026-08-07  
Status: **promoted in the internal arena; development paused**

## Outcome

Exactly one post-V8 identity was submitted:

- arena identity: `HelbrosRocketSpidopsSafeRouterV13`;
- arena acknowledgement: accepted, 33,836,166 bytes, queued for promotion on
  the next worker job;
- archive: `work/HelbrosRocketSpidopsSafeRouterV13.tar.gz`;
- archive SHA-256:
  `01520d506e3e4b9c58b55f69d92da18f185c9b0138708f5b81c94b23c706f8c5`;
- deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- main SHA-256:
  `3b85d0becaec72e6f7da8ddd17b7ab63ab06aea5c0cf30a6366a66866731efa5`;
- learned model SHA-256:
  `164bf4a0408c89dd9fbe531bfb40acb5eb6e099a85a33769505e5cb1f1c4048f`;
- safe V5 model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.

The submission is internal-arena only. No Kaggle upload or submission was
made. The acceptance acknowledgement proves delivery, not mature strength;
future V13 arena games must remain separate from V5 and V8 results.

The first read-only post-promotion snapshot was 10-5 at rating 791.1. Fifteen
games verify live scheduling and runtime health; they are not a stable strength
estimate. No monitoring loop was started.

## What changed

V13 is a public-information safe-strategy router, not another single-action
heuristic. It carries two synchronized autoregressive neural agents:

1. the byte-exact V5 model is the default safe blueprint;
2. a retained V49-adapted neural model is the exploitation overlay; and
3. the overlay latches only after active, Bench, or discard cards publicly
   identify Dragapult, Marnie, or Starmie.

Both neural agents consume every official observation from game start. Before
the public range activates, V13 returns the exact V5 action. After activation,
it returns the exact learned action. Opponent hand, deck, prizes, bot name,
seed, and replay metadata cannot activate the overlay. Unrecognized matchups,
including Rocket mirrors and Alakazam, remain on V5.

The exact V5 deck, official-engine adapter, legality path, early Poke Pad
Spidops guard, and exact-lethal Brave Bangle guard are unchanged. V13 does not
use sampled-rollout search and is not represented as an equilibrium solver.

## Learned-policy provenance

The adapted checkpoint was trained only after a strict behavior-parity audit
against archived V8-versus-Soldier V49 games. Of 183 games, the 77 games with
any V5/V8 behavioral mismatch were excluded completely. The remaining 106
games reproduced V5 exactly and were frozen into 83 training, 14 validation,
and 9 test games.

Only the final Transformer block and action head were updated for four epochs
with retention-weighted advantage learning: learning rate `1e-6`, KL weight
`1.0`, temperature `0.5`, and seed `20260807`. The selected epoch-4 checkpoint
reached 93.91% test top-1 retention with test KL 0.0205. These are imitation
retention measurements, not win-rate claims.

## Selection path

- V9's policy-only Dragapult intervention regressed from 17/24 to 13/24; its
  third-Articuno structural arm reached 18/24 but missed its required gain.
- V10 epoch 4 improved the four-cell development aggregate from V5's 54/96 to
  63/96, but failed the original absolute Starmie gate.
- V11's Maximum Belt composite reached only 6/24 in its Starmie screen and was
  closed.
- V12 reproduced a strong 70/120 current-meta panel but missed its direct-V5
  floor by one game, 11/24 versus the required 12/24.
- V13 was preregistered to preserve V5 in unidentified matchups while exposing
  only the already validated learned ranges.

No failed candidate was submitted.

## V13 verification

The official-observation parity audit replayed ten archived games and made 780
decisions:

- 780/780 returned actions were legal;
- 85 pre-activation decisions matched V5 exactly;
- 695 post-activation decisions matched the adapted checkpoint exactly; and
- zero routing mismatches occurred.

The fresh, side-swapped official-engine confirmation was:

| Matchup | V13 wins | Gate |
|---|---:|---:|
| Dragapult/disruption proxy | 20/24 | at least 17 |
| current Marnie | 13/24 | at least 11 |
| direct V5 | 13/24 | at least 12 |
| Alakazam threat proxy | 23/24 | at least 20 |
| flat Starmie | 5/24 | at least 4 |
| **Overall** | **74/120** | **at least 65** |

All 120 games ended normally with no engine or timeout result. The result
cleared every preregistered matchup and aggregate floor.

Additional release checks:

- 11 focused deterministic tests passed;
- the exact 60-card deck and both model hashes passed;
- archive: 26 runtime files, zero unsafe paths, zero links, and zero
  source/extracted hash mismatches;
- the independently extracted archive returned the exact deck; and
- four fresh extracted-archive official-engine games were 4/4 terminal.

## Claim boundary

The 74/120 panel is a controlled local confirmation against five fixed
opponents. It supports submission and provides evidence of a broad improvement
over the V5 development control, but it is not a live-arena estimate, a Kaggle
result, proof of equilibrium play, or proof of superhuman strength. A stable
strength estimate requires a mature V13-only arena sample under the live
opponent schedule.

## Pause boundary

Development is paused after this accepted submission. Do not start another
training run, deck mutation, local evaluation, arena identity, monitor,
publication job, or Kaggle submission without a new user request. The internal
arena may promote and schedule the immutable V13 identity while development
remains paused.
