# Target-80 own-turn plan milestone V3 protocol

Status: **prospectively locked controllable-line pilot; no deployment authority**  
Locked: 2026-08-05, after one excluded smoke and before either V3 shard.

## Hypothesis

V2 found four stable positive `engine_resilience` lines across both seats, but
failed overall stability after including an opponent turn: particle-half
stability was 50% and coin-stream stability 41.7%. V3 tests whether complete
Rocket turn plans themselves are stable when measured before the opponent acts.

The plan vectors, support floor, root selector, six strategic axes, and
state-conditioned meta weights are unchanged from V2. Only the evaluation
boundary changes. V3 stops at the first official observation belonging to the
opponent after Rocket's current turn completes. A terminal result reached
inside the Rocket turn still has value `4 * utility` and dominates every
dense nonterminal score.

This boundary measures a controllable strategic line, not full-game strength.
Passing can authorize plan-selector data collection only; it cannot authorize
deployment.

## Frozen substrate and plans

The exact 60-card Rocket package remains Wobbuffet-free:

- tree:
  `30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f`;
- deck:
  `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`;
- model:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- main:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`.

The Marnie opponent tree is
`49bb3353ab42fded18e32e38f154ad1a001d60c5852383045ff4f437be36efe1`;
the search runtime tree is
`2fab4c1c945dc97269f147b6237d37cb653228b6792263f28df05945e0744b61`.

The frozen plans are:

- `baseline`: exact guarded neural policy;
- `engine_resilience`: Spidops engine 1.25, second attacker 1.25, Charging
  Up 0.75, bench discipline 0.50, fuel preservation 0.50, attack tempo 0.25;
- `tempo_conversion`: attack tempo 1.25, fuel preservation 0.75, second
  attacker 0.50, tool assignment 0.50, bench discipline 0.50.

All plan logit biases are clipped to 1.5. The packaged Early-Pad and lethal
Brave-Bangle guards retain priority. Each retained root requires at least two
distinct plan actions, and every non-baseline plan root action requires frozen
policy probability at least 0.01.

## Locked execution

Root selection is unchanged from V2: one highest-priority critical MAIN root
per source game, turns 2-8, with up to eight candidate roots inspected.

| Shard | Rocket start order | Roots | Seed | Output |
|---|---|---:|---:|---|
| first | starts first | 8 | 858101 | `work/target80_own_turn_plan_milestone_v3/pilot_first.jsonl` |
| second | starts second | 8 | 858201 | `work/target80_own_turn_plan_milestone_v3/pilot_second.jsonl` |

Every root uses eight public-consistent hidden-zone particles and two shared
coin streams: 16 matched sets and 48 plan-line continuations. Fixed limits are
150 branch decisions, 180 seconds per root, 240 source attempts, 2,000 live
decisions, and one Torch thread.

Both shards must seal before rankings are read. Any changed hash, incomplete
root, illegal/fallback action, guarded-baseline mismatch, opponent action inside
the scoring horizon, hidden-hand/visualizer read, raw hidden allocation, or
aggregate-reproduction failure invalidates the pilot.

## Frozen gates

A stable non-baseline plan requires the same non-baseline winner in the full
sample, both four-particle halves, and both coin streams; full dense gain at
least 0.05; and nonnegative gain in every half and stream.

V3 passes only if:

1. all 16 roots and 256 matched sets are integrity-complete;
2. at least 12/16 roots have stable particle-half rankings;
3. at least 12/16 roots have stable coin-stream rankings;
4. at least four roots are stable non-baseline plan changes;
5. each seat contributes at least one stable change;
6. stable changes have mean dense gain at least 0.05;
7. at least three criticality categories are represented.

Passing authorizes only a larger game-disjoint own-turn plan corpus and a
separately locked selector. Any selector must later beat the exact guarded
baseline in full official-engine games across both seats and broad opponents,
then pass untouched confirmation. Failure closes V3 without threshold or
vector retuning.

## Claim boundary

All source and branch transitions and all legality are genuine official-engine
execution. Hidden particles, persistent plan vectors, the six strategic axes,
and their state-conditioned aggregation are research abstractions. V3 neither
models the opponent response nor proves equilibrium safety. Offline branch
search is teacher data, not the final deployed strategy.

## Frozen implementation

- collector:
  `4c2afa3f85d279b63ac431702fae90fbb2b2ba9e277349529b55fd162cd35860`;
- analyzer:
  `61b465f00bbbabad442f960513d2385ec76a37eb035b8932f11a6ce0e42cfb57`;
- frozen V2 dependency:
  `8e153f1baceb240ac522275aea869e170ce9bfdd2c0ad25ae4bb4118c068da17`.

The one-root smoke at
`work/target80_own_turn_plan_milestone_v3_smoke/` is excluded from all
pilot, training, development, confirmation, and strength claims.
