# Pairwise causal recovery protocol

Status: frozen after the first causal-Q gate rejected and before any fresh
pairwise validation or test match was generated.

## Prior result and restart boundary

The 300-game causal-Q collection produced 598 complete roots and 2,392 terminal
official branches without a legality or engine error. Its regression candidate
was rejected: validation admitted no intervention and untouched paired utility
did not improve. All 300 games are now development data. Their former split is
not reused as evidence.

The likely failure is target dilution. Most shortlisted actions tie at terminal
utility, so mean-squared regression is minimized by predicting approximately
zero even though the smaller set of non-tied action pairs contains useful
ordering information. This restart changes the learning target and creates new
evidence; it does not reinterpret the rejected result.

## Frozen fresh evidence

Six new 30-game second-seat collections are generated against two independent
top Marnie implementations: three threshold-selection collections and three
separately launched untouched-admission collections. Every root uses the official visualizer's transient
realized hidden state, the licensed native branch API, two terminal rollouts per
candidate, and an explicitly shared fair manual-coin sequence across candidates
within each rollout. Raw hidden cards are not serialized or supplied to a
policy. Shuffles and other opaque native chance remain engine-controlled.

The three `pairwise_fresh_*` files are threshold-selection evidence. The three
`pairwise_test_*` files are untouched admission evidence. Each contains exactly
games 1-30 from an independent engine process.

All 300 prior games are training-only. No fresh games are used to fit the model
or normalize features. A root must have every candidate complete both rollouts.

## Pairwise model

For each non-tied alternative-versus-blueprint pair, the target is whether the
alternative's mean terminal utility is higher. Ties are omitted from BCE but
remain in gate evaluation. The input contains the frozen blueprint's contextual
global token, alternative-minus-blueprint option token, their elementwise token
product, and blueprint-logit difference. A layer-normalized MLP maps this to one
preference logit. The final layer starts at zero.

Class-balanced BCE is optimized with AdamW at `3e-4`, weight decay `1e-4`, seed
`615501`, root batches of 32, at most 80 epochs, and patience 12. The checkpoint
with minimum selection-pair BCE is frozen. No architecture or seed sweep is
performed.

The selection threshold is chosen from predicted probabilities. It must
intervene on at least five and at most 35% of selection roots, achieve at least
65% precision among interventions, and have positive paired mean utility. If no
threshold qualifies, the model is rejected without opening the test result for
promotion.

## Untouched admission and deployment boundary

Admission requires the separately launched `pairwise_test_*` 30-game files:

- at least eight interventions;
- intervention rate no greater than 35%;
- intervention precision at least 65%;
- positive paired mean utility; and
- a game-clustered 95% bootstrap lower bound strictly above zero.

Passing admits only a candidate for fresh complete-match evaluation. Runtime
scope remains fail-closed: the first single-choice main-phase root per turn,
through turn 8, only while Rocket is second into publicly identified Marnie,
and only among the frozen blueprint's semantic top-four actions. All other
actions come from the unchanged blueprint. Official match and arena gates stay
authoritative; this empirical exploit overlay is not an equilibrium or safe-
resolving certificate and is not the project's final strategic architecture.
