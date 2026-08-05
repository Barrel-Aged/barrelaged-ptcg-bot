# Target-80 randomized-support advantage V1 protocol

Status: **prospective and frozen before smoke or corpus collection**  
Frozen: 2026-08-05  
Scope: development data and policy research only; no submission authority

## Question

Can fresh official-engine trajectories with explicitly randomized, logged
action support identify useful action advantages for a Wobbuffet-free Team
Rocket Mewtwo/Spidops policy where the deterministic N24 corpus could not?

This experiment addresses a specific evidence gap. N24 observed only actions
selected by one deterministic policy. Its terminal labels could predict some
states, but they did not establish how plausible alternatives at the same
public state would perform. V1 therefore changes the data-generating policy;
it does not add another hand-written tactical rule or reuse trajectory-outcome
AUC as a proxy for action quality.

## Frozen Rocket substrate

Bundle:
`work/target80_candidates/mimikyu_maxbelt_neural_control_v1/`

- tree SHA-256:
  `30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f`
- `deck.csv` SHA-256:
  `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`
- `model.pt` SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`
- `main.py` SHA-256:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`

The deck has exactly 60 cards, zero Team Rocket's Wobbuffet, one Team
Rocket's Mimikyu, one Maximum Belt, and one Brave Bangle. It is the clean
neural-control version of the low-risk Mimikyu/Maximum-Belt substrate. The
rejected hand-authored `mimikyu_maxbelt_preserve_v2` policy is not the actor.
This substrate is not claimed to be stronger than V5; it is being used to
learn on the exact Wobbuffet-free identity that any resulting policy would
pilot.

## Frozen opponents

| Cell | Bundle | Tree SHA-256 | Deck SHA-256 | Main SHA-256 |
|---|---|---|---|---|
| Marnie | `work/arena_bundles/AshV178MarnieFullBest_20260801_extracted/` | `49bb3353ab42fded18e32e38f154ad1a001d60c5852383045ff4f437be36efe1` | `92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d` | `0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b` |
| Starmie | `work/arena_bundles/SparkStarmieFroslassCFV13_extracted/` | `fec4a911a29f18522cdc2dcd018831370760ee1d9452131f6ed4420b73c06a1e` | `60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61` | `3d570642a9294bca0abea9587786b4a9059a42d1c777c462631e131cde24e87e` |
| Alakazam | `work/arena_bundles_20260804/VioletAlakazamMKV93_extracted/` | `d24e5e225504383d01226d6c3a5d83ce7a388f2bfac4749b896bdc5a0e4daf9e` | `4e56a20249fad14097d06aaee2a241f355b90ef9bef19307f95b47798d0a2a3d` | `c324c943d2dfe62d28c9a404e264e6d25eabeea296e37f9575ee1231a76bbec4` |

Marnie is the largest live weakness, Starmie is a sparse but severe failure
cell, and Alakazam guards against overfitting away a current strength. These
three opponents are training coverage, not the later broad confirmation panel.

## Official-engine and information boundary

- Every game must execute through the bundled official `cg` engine and finish
  with result 0, 1, or 2.
- The opponent runs through the arena evaluator worker in its own process.
- The Rocket actor receives only the ordinary acting-player observation. On
  every recorded Rocket decision, its own hand must be present when public
  setup permits it and the opposing player's `hand` field must be `null`.
- Public deck identities may be used as matchup priors. Opponent private cards,
  engine internals, future observations, and terminal outcomes are never
  inputs to the behavior policy.
- This Windows engine build does not expose `BattleStartSeeded`. The recorded
  seed controls Python, NumPy, Torch, and exploratory choices, but must not be
  represented as control of the native engine's shuffle. The corpus is
  independently sampled, not a paired common-random-number experiment.
- No fallback action is permitted. Exceptions, timeouts, illegal selections,
  or hidden-information violations fail the shard.

## Frozen behavior policy

The reference policy is the exact frozen `main.py` behavior: its early Poké
Pad-to-missing-Spidops guard, lethal-only Brave Bangle guard, and otherwise the
greedy v10 policy with the attack-safe rule that suppresses END whenever an
ATTACK option is legal. Guarded decisions remain deterministic with propensity
1 and are labeled with their reference source; exploration applies only after
both packaged guards decline to act.

At every Rocket selection:

1. Evaluate the two packaged public-information guards. If either fires, apply
   its exact action, log propensity 1, and do not randomize that decision.
2. Otherwise extract the exact deployed public feature representation with the normal
   persistent `Belief` state.
3. For a one-choice selection (`maxCount == 1`), compute attack-safe reference
   logits and their softmax probabilities over finite legal options.
4. Let the reference argmax be `a0`. Alternatives are finite legal options
   other than `a0` whose reference probability is at least **0.03**.
5. If alternatives exist, choose `a0` with probability **0.85**. With
   probability **0.15**, choose only among the supported alternatives,
   proportionally to their reference probabilities.
6. If no supported alternative exists, choose `a0` with probability 1.
7. Zero-choice and multi-choice selections retain the exact greedy deployment
   behavior. Multi-choice randomization is deferred because its conditional
   autoregressive propensities require a separate audited schema.
8. Initial first-player selection (`select.type == 9`) is never randomized.

Each Rocket decision records the public observation, selected indices,
reference logits/probabilities, full behavior probabilities, chosen-action
propensity, reference argmax, supported alternatives, exploration flag,
selection type, and whether the attack-safe guard fired. All probabilities
are recorded before `battle_select`.

## Smoke and implementation gates

Smoke data are permanently excluded from training, validation, test, and
strength claims. The locked smoke is two Marnie games from each Rocket seat:

- seat 0 behavior seed `850101`;
- seat 1 behavior seed `850102`.

Before full collection:

- all four games must terminate legally;
- all recorded chosen actions must match the applied action;
- behavior probabilities must be finite, nonnegative, sum to 1 within `1e-6`,
  and assign positive mass equal to the recorded chosen propensity;
- every supported alternative must have reference probability at least 0.03;
- END must have zero support whenever ATTACK is legal;
- at least one eligible supported-alternative decision and at least one actual
  exploratory action must occur across the smoke;
- every Rocket row must pass the hidden-hand boundary audit;
- manifest hashes and replay hashes must reproduce when independently audited.

If a smoke gate fails, V1 stops. Any change to epsilon, support threshold,
eligible selection types, hashes, or counts requires a prospective V2. Smoke
win/loss results have no admission role.

## Locked corpus

Each row below is a separate immutable shard. Rocket plays both seats. No
outcomes may be inspected or summarized until all 18 shards are terminal,
hashed, and complete.

| Split | Games per opponent/seat | Total | Behavior seeds |
|---|---:|---:|---|
| train | 120 | 720 | Marnie `851101/851102`; Starmie `851201/851202`; Alakazam `851301/851302` |
| validation | 40 | 240 | Marnie `852101/852102`; Starmie `852201/852202`; Alakazam `852301/852302` |
| test | 40 | 240 | Marnie `853101/853102`; Starmie `853201/853202`; Alakazam `853301/853302` |

Total: **1,200 games**. Split membership is assigned by shard before any game
is played, so no trajectory or adjacent decision can cross splits. Draws are
retained as neutral outcomes and never counted as wins.

## Corpus admission gates

The complete corpus is admitted only if:

- all 1,200 games and all 18 manifests are present and hash-valid;
- no game, filename, behavior seed, or replay digest is duplicated;
- no illegal action, actor fallback, timeout, truncated replay, or nonterminal
  game exists;
- the information-boundary audit passes every Rocket decision;
- every logged propensity passes the smoke numerical checks;
- each opponent/seat/split cell contains exactly its locked game count;
- at least 500 eligible supported-alternative decisions and 60 realized
  exploratory actions exist overall, with realized interventions in every
  opponent and both seats.

These are data-quality gates, not outcome gates. Overall or per-cell win rates
must not be used to edit the corpus or behavior policy.

## Frozen learning direction

After corpus admission, freeze the v10 relational trunk and fit a conservative
action-value/advantage overlay on the logged one-choice decisions. Training may
use train only. Validation selects regularization, uncertainty margin, and the
maximum action-change rate. Test remains untouched until one configuration is
frozen.

The learner must use logged behavior propensities, twin or ensemble action
values, explicit baseline-action anchoring, and a support mask. It may change a
decision only when all conservative heads prefer the same supported action by
a frozen lower-confidence margin. It must fail closed to the reference policy
on missing features, unsupported actions, disagreement, or runtime errors.

Minimum offline requirements before official-engine development:

- action-specific held-out discrimination exceeds the reference-logit-only
  baseline, not merely state-level outcome AUC;
- validation Brier score and calibration are reported by opponent and seat;
- effective sample size after propensity weighting is reported and remains at
  least 25% of the raw supported-decision count in every training opponent;
- learned changes occur on at least 0.5% but no more than 12% of supported
  validation choices;
- both conservative heads agree on every admitted change;
- serialization reload reproduces all frozen test logits and choices exactly.

Failure closes this exact learning run. It does not authorize hand-tuning on
test outcomes.

## Later strength gates and submission rule

Offline success grants only a new, preregistered official-engine development
panel. It does not grant packaging or arena submission. A candidate must then
beat the exact Wobbuffet-free same-deck neural control in side-swapped Marnie,
both Starmie variants, Alakazam, Rocket mirror, and a resource-wall cell; show
no seat or matchup collapse; and pass a fresh untouched confirmation panel.

At most one immutable successor may be submitted. The active goal is proven
only by that single identity recording at least 160 wins in at least 200 rated
internal-arena games (80% raw wins; draws are non-wins). Results may not be
pooled across V5, controls, candidates, or retries. No Kaggle submission is
authorized by this protocol.
