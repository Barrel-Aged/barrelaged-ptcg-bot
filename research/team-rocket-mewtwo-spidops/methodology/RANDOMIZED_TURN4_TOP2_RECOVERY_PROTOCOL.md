# Randomized Turn-4 Top-2 Recovery Protocol

Status: frozen before collection. This is a new experiment following the
validation rejection of `RANDOMIZED_FULL_MATCH_CAUSAL_RECOVERY_PROTOCOL.md`.
Its sealed test remains unopened and is not reused here.

## Motivation and claim boundary

The turn-2 experiment established that blueprint ranks 3 and 4 are usually
harmful, while ranks 1 and 2 are close. It found no validation-certified
conditional improvement at the opening root. This experiment moves the same
causal design to Team Rocket's turn 4, after setup choices and opponent public
actions have created more strategically distinct states, and spends all
assignment probability on the blueprint's top two semantically distinct legal
actions.

Every outcome is from a complete official local-engine match. No visualizer
truth, hidden card, search branch, or simulated descendant is accessed. The
experiment estimates an exploitative public-policy improvement against the two
frozen Marnie opponents; it is not an equilibrium, safety, rating, or universal
strength claim.

## Frozen objects and intervention

The Team Rocket blueprint, qualifying deck, opponents, and hashes are exactly
those frozen in `RANDOMIZED_FULL_MATCH_CAUSAL_RECOVERY_PROTOCOL.md`.

Team Rocket is seat 1 and must be declared genuinely second by the official
observation. At the first eligible Team Rocket turn-4 main-selection root:

1. obtain the frozen greedy blueprint action,
2. form the two highest-logit semantically distinct legal actions, with the
   blueprint action in position 0,
3. assign position 0 or 1 uniformly with propensity `0.5`,
4. execute that action, then use the unchanged blueprint for the rest of the
   complete official match, and
5. serialize only the public observation, legal shortlist, assignment,
   propensity, and terminal outcome.

There is exactly one randomized intervention opportunity per accepted game.
Games ending before an eligible turn-4 root are counted in the manifest but
not outcome-selected; the estimand is explicitly conditional on reaching such
a root under the frozen blueprint.

## Frozen data split

Independent files and seeds are used for each opponent:

- training: 1,000 accepted games per opponent (2,000 total),
- validation: 250 per opponent (500 total),
- untouched test: 250 per opponent (500 total).

The test files may be collected concurrently but cannot be read until a model,
regularization choice, confidence multiplier, and intervention margin have
been frozen from training and validation. Partial or duplicate files are
rejected.

## Frozen learner and selection

Use the same frozen-token PCA, low-capacity ridge ensemble, fixed penalty grid,
confidence grid, margin grid, stratified bootstrap, IPS estimator, and DR
estimator defined in the turn-2 protocol. The only configuration changes are:

- expected root turn is 4,
- shortlist size is exactly 2, and
- seven bootstrap ensemble members are retained.

Validation admission requires:

- intervention rate between 5% and 60%,
- at least 100 logged assignments matching the proposed deterministic policy,
- combined DR point improvement above zero,
- combined 95% DR lower bound above zero,
- combined IPS point improvement above zero, and
- neither opponent-specific DR point estimate below `-0.04`.

If no configuration passes, reject the experiment and do not open its test.

## Untouched test and full-match admission

The frozen test gate requires combined DR and IPS point improvements above
zero, a combined 95% DR lower bound above zero, at least 100 policy-matching
assignments, and neither opponent-specific DR estimate below `-0.03`.

A test pass remains insufficient for deployment. Build the bounded turn-4
overlay and collect fresh candidate and frozen-blueprint control panels of at
least 120 genuine-second-seat games per opponent. Admission requires zero
errors/fallbacks, at least 20 interventions, at least 54/120 wins against each
opponent, at least 116/240 combined wins, and no combined deficit versus the
concurrent blueprint control. Only a passing artifact may receive a non-rated
arena evaluation; a rated submission additionally requires that evaluation not
to be weaker than the live Team Rocket bot.

