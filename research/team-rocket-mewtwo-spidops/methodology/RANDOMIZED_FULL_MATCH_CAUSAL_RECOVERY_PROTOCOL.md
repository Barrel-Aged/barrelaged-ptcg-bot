# Randomized Full-Match Causal Recovery Protocol

Status: frozen before collection. This protocol replaces the rejected
live-truth branch-value, pairwise, empirical-rule, and single-rule recovery
experiments. Those results remain part of the audit trail and are not reused as
test evidence here.

## Question and claim boundary

Can one randomized early action improve the qualifying Team Rocket
Mewtwo/Spidops policy when it goes second into the two strongest public Marnie
policies?

Every trajectory in this experiment is a complete execution of the supplied
official local `cg` engine. The only intervention is a uniformly randomized
legal action at one public decision. No visualizer truth, hidden-hand access,
`search_begin` branch, simulated descendant, or heuristic outcome label is
used. Consequently, treatment assignment is causal for the distribution of
eligible roots reached by the frozen blueprint and opponent. It is not an
equilibrium certificate, a hidden-information solution, an arena rating, or a
claim about every opponent.

The learned component is an opponent-exploitation overlay constrained by the
frozen neural blueprint. It does not replace the public-belief / safe-resolving
target architecture in `PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md`.

## Frozen objects

- Team Rocket blueprint:
  `artifacts/team_rocket_recovery/candidates/scale_value_calibrated_v1`
- Blueprint model SHA-256:
  `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`
- Qualifying Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`
- Opponent A: `AshMarnieGeneralV110_extracted`, model SHA-256
  `552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b`
- Opponent B: `HelbrosMarnieGeneralRecoveryV4_extracted`, model SHA-256
  `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c`
- Both opponent deck files have SHA-256
  `8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b`.
- The official engine and legality adapter are loaded from the already audited
  packaged runtime. The randomized collector does not modify them.

## Population and randomized intervention

The Team Rocket deck is seat 1. The official engine must report seat 0 as the
first player, making Team Rocket genuinely second. An accepted game has a
turn-2 Team Rocket main-selection root with exactly one required choice and at
least two legal options.

At that first eligible root:

1. Run the frozen greedy blueprint to obtain its action and update its public
   belief tracker normally.
2. Form a shortlist of at most four legal, semantically distinct actions using
   the frozen blueprint logits. The blueprint action is position 0.
3. Assign one shortlist position uniformly with a seeded PRNG. The propensity
   is exactly `1 / shortlist_size` and is serialized.
4. Execute that legal action in the official engine and use the frozen
   blueprint for every later Team Rocket decision.
5. Serialize the redacted public observation, shortlist, assignment,
   propensity, and terminal win/loss/draw. Never serialize engine truth.

There is at most one randomized root per accepted game. Games without an
eligible root are reported in the manifest but are not outcome-selected.

## Frozen data split and scale

Collection uses independent processes and seeds for each opponent:

- training: 500 accepted games per opponent (1,000 total),
- validation: 200 per opponent (400 total),
- untouched test: 200 per opponent (400 total).

The training and validation files are development data. The test files must not
be read by a training, feature-selection, regularization-selection, threshold-
selection, or debugging command. They may be opened once only after a complete
candidate artifact and its manifest have been frozen from training and
validation.

If a collection process fails, it may be restarted only into a new file with a
new documented seed. Partial files cannot be silently concatenated. Duplicate
game identifiers are rejected by the trainer.

## Frozen learner family

The neural blueprint is frozen. Its contextual global and option tokens are
projected by principal components fit on training data only. A low-capacity
ridge outcome model predicts terminal utility from:

- a projected global token and the blueprint state value,
- a projected contextual option token,
- a projected option-minus-blueprint-option token,
- within-shortlist blueprint-logit features and shortlist position.

Regularization is chosen from the fixed grid `{0.1, 1, 10, 100, 1000}` using
validation data only. A seven-member training-game bootstrap ensemble supplies
epistemic dispersion. No test outcome may influence the projection, ridge
penalty, ensemble, or intervention threshold.

At inference the overlay compares every alternative with shortlist position 0.
It may change the action only when the best alternative's ensemble lower
confidence advantage clears a validation-selected margin. The fixed confidence
multipliers are `{0, 0.5, 1.0}`; margin candidates are `{0, 0.02, 0.05, 0.10}`
plus validation positive-advantage quantiles at
`{0.1, 0.2, ..., 0.9}`. The selected policy must intervene on between 5% and
60% of validation roots.

## Randomized validation and test estimands

For a deterministic candidate policy `pi`, evaluation uses both:

- inverse-propensity scoring (IPS), and
- a doubly robust (DR) estimate using the training-only outcome ensemble.

The primary estimand is candidate-minus-blueprint terminal utility. Confidence
intervals use a stratified game bootstrap that resamples independently within
each opponent. Validation selects the ridge penalty and intervention gate by
the lower 95% DR bound, subject to:

- combined DR point improvement above zero,
- combined 95% DR lower bound above zero,
- at least 20 validation assignments matching the candidate policy,
- no opponent-specific DR point estimate below `-0.05`, and
- the intervention-rate bounds above.

The untouched randomized test gate requires:

- combined DR point improvement above zero,
- combined 95% DR lower bound above zero,
- combined IPS point improvement above zero,
- at least 30 test assignments matching the candidate policy,
- each opponent-specific DR point estimate above `-0.03`, and
- no schema, legality, traceback, propensity, split, or hash failure.

Failure rejects the artifact. The threshold is not retuned after test opening.

## Fresh official full-match admission

Passing randomized test evidence is necessary but not sufficient. Package the
overlay and run fresh genuine-second-seat official panels, at least 120 games
against each frozen opponent. Admission requires:

- zero engine/agent errors, tracebacks, or illegal fallbacks,
- at least 10 actual interventions overall,
- at least 54 wins of 120 (45%) against each opponent,
- at least 116 wins of 240 (48.33%) combined, and
- a combined win rate not below a concurrently collected 240-game frozen-
  blueprint control panel.

Only then may the candidate receive a non-rated arena evaluation. A rated arena
submission is allowed only if that evaluation is not weaker than the current
live Team Rocket bot. Arena strength and rank are reported as observed results,
never inferred from local games.

## Reproducibility and stopping

The collector writes one JSON row per accepted game and an atomic manifest with
file hashes, object hashes, counts, parameters, action-position balance, win
counts, and the explicit claim boundary. Collection count is fixed in advance;
there is no outcome-based early stopping. All rejected artifacts remain
archived and are never packaged as the live candidate.
