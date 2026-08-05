# Target-80 paired milestone-root V2 protocol

Status: **prospectively locked plan-level pilot; no deployment authority**  
Locked: 2026-08-05, after excluded integration smokes and before either
authoritative V2 shard is collected.

## Why V2 is materially different

Paired critical-root V1 completed every official-engine branch correctly, but
full-game terminal values tied too often: only 1/16 roots had a stable
non-baseline improvement, and particle-half ranking stability was 68.75%.

V2 changes both the intervention and the target:

- the intervention is a complete persistent strategic line, not merely a
  different first card action;
- each line remains active through the rest of the turn, the opponent response,
  and the start of Rocket's next turn;
- the value target is a bounded six-axis public strategic milestone at that
  fixed horizon, with any terminal result reached inside the horizon overriding
  the dense score.

This is an offline information-set teacher pilot. It is not a deployed sampled
rollout policy and does not reopen or retune V1.

## Frozen identities

The exact Wobbuffet-free Rocket package, Marnie opponent, and official
search-capable runtime are unchanged from
`TARGET80_PAIRED_CRITICAL_ROOT_V1_PROTOCOL.md`:

- Rocket tree:
  `30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f`;
- Rocket deck:
  `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`;
- Rocket model:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- Rocket main:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`;
- Marnie tree:
  `49bb3353ab42fded18e32e38f154ad1a001d60c5852383045ff4f437be36efe1`;
- runtime tree:
  `2fab4c1c945dc97269f147b6237d37cb653228b6792263f28df05945e0744b61`.

Deck integrity remains exactly 60 cards and zero Wobbuffet. The packaged
Early-Pad and lethal-Bangle guards have priority over every plan in source and
branch play.

## Frozen strategic plans

All plan weights are persistent public logit biases, clipped to an absolute
maximum of 1.5. They rerank only official legal options and use no hidden
opponent cards.

1. `baseline`: the exact guarded frozen neural policy, with no plan bias.
2. `engine_resilience`:
   `spidops_engine=1.25`, `second_attacker=1.25`,
   `charging_fuel=0.75`, `bench_discipline=0.50`,
   `fuel_preservation=0.50`, and `attack_tempo=0.25`.
3. `tempo_conversion`:
   `attack_tempo=1.25`, `fuel_preservation=0.75`,
   `second_attacker=0.50`, `tool_assignment=0.50`, and
   `bench_discipline=0.50`.

These are bounded plan proposals for paired comparison, not hand-authored
claims of superiority. A retained root must make at least two distinct root
actions across the three plans. Each non-baseline plan's root action must have
at least 0.01 probability under the frozen attack-safe policy.

## Root selection and horizon

Source games are ordinary official-engine games under the exact guarded
baseline. Capture up to eight distinct-turn MAIN roots in turns 2 through 8,
then retain at most one root per game.

Eligibility requires low policy margin, attack/setup tradeoff, Mewtwo
activation opportunity, or a missing second attacker from turn three onward.
The deterministic priority weights attack/setup tradeoff 8, Mewtwo activation
6, missing second attacker 4, sub-three-Spidops setup 3, and low policy margin
2. It subtracts `0.15 * abs(turn - 5)` and the policy margin capped at 3.
Exact ties prefer proximity to turn five, later source step, then higher
observation fingerprint.

For each plan, execute its root action and preserve that plan until the first
observation where Rocket again acts at least two alternating turns after the
root. If the game terminates sooner, record terminal utility times 4; this
strictly dominates every nonterminal dense score.

## Six public strategic targets

Every nonterminal horizon records six values bounded to [-1, 1]:

- `setup`: Rocket board density, Spidops depth, Mewtwo access, and ready
  attacker count;
- `tempo`: ready and active-ready attackers plus public active damage
  pressure;
- `resource_efficiency`: attached Energy, Energy on ready attackers, and
  realized attacker readiness;
- `prize_control`: public prize differential and active damage pressure;
- `board_control`: own board depth, ready attackers, and opponent board
  occupancy;
- `optionality`: open bench capacity, public hand/deck counts, and a ready
  Mewtwo-with-backup condition.

One state-conditioned weight vector is frozen at the root and applied to all
three branches. Prize-threshold roots emphasize prize/tempo; Mewtwo activation
roots emphasize setup/resource efficiency; missing-backup roots emphasize
setup/tempo/resource efficiency; otherwise the balanced mixture is used. Every
weight vector must sum to exactly one, and the analyzer independently
recomputes every aggregate from serialized axes.

## Locked collection

| Shard | Rocket start order | Roots | Seed | Output |
|---|---|---:|---:|---|
| first | starts first | 6 | 857101 | `work/target80_paired_milestone_root_v2/pilot_first.jsonl` |
| second | starts second | 6 | 857201 | `work/target80_paired_milestone_root_v2/pilot_second.jsonl` |

Each root uses eight public-consistent hidden-zone particles and two shared coin
streams: 16 matched sets and 48 plan continuations. Fixed parameters are plan
support 0.01, turn window 2-8, horizon two alternating turns, maximum 250
branch decisions, 300 seconds per root, 200 source attempts, 2,000 source-game
decisions, and one Torch thread.

Both shards must seal before any V2 plan ranking or value is read. Any changed
hash, missing root, incomplete set, illegal action, fallback exception,
guarded-baseline mismatch, visualizer/realized-hand read, raw hidden allocation,
or aggregate-reproduction error invalidates the pilot.

## Locked admission gates

The analyzer reproduces full, particle-half, and coin-stream plan rankings.
A stable non-baseline plan requires the same non-baseline winner in the full
sample, both four-particle halves, and both coin streams; a full paired dense
gain of at least 0.05; and nonnegative gain in every half and stream.

V2 passes only if:

1. all 12 roots and all 192 matched sets are integrity-complete;
2. at least 9/12 roots have stable particle-half rankings;
3. at least 9/12 roots have stable coin-stream rankings;
4. at least three roots are stable non-baseline plan changes;
5. each seat contributes at least one stable change;
6. stable changes have mean paired dense gain at least 0.05;
7. at least three criticality categories are represented.

Passing authorizes only a larger, game-disjoint plan-root corpus and a separately
locked plan selector. It does not authorize official development, confirmation,
packaging, submission, or a strength claim. Failure closes this exact V2
pilot; thresholds and plan vectors may not be retuned from its outcomes.

## Genuine execution versus abstraction

Source games and all descendant transitions are genuine official `cg` engine
execution with official legality. Hidden-zone particles, the three plan vectors,
the six strategic heads, and their meta-weighting are research abstractions.
Eight particles do not constitute an official posterior, equilibrium solution,
or counterfactual-value safety certificate. Sampled branches remain offline
teacher data and are explicitly not the final strategic approach.

## Frozen implementation

- V2 collector:
  `8e153f1baceb240ac522275aea869e170ce9bfdd2c0ad25ae4bb4118c068da17`;
- V2 analyzer:
  `4b458e5c14c261c83c6b41f66f3391d58a5c1ab9cdfc3dd94e85a9eb4c3f2775`;
- persistent-plan runtime:
  `f6c3e2056eb143b6ad4bf537095aeb5b78703b0c9f4ffd789587279e8e1b85af`;
- frozen V1 dependency:
  `4f6d26ff7aef0b79341e589c79321a883f1201a6a4b2faf8d6fe4539e92d94dd`.

The earlier single-action V2 smoke and the final one-root plan-level smoke in
`work/target80_paired_milestone_root_v2_smoke/` are excluded from all pilot,
training, development, confirmation, and strength claims.
