# Team Rocket's Mewtwo ex / Spidops strategic planner

This repository contains the six-phase research scaffold and the continuing
strength campaign for a belief-aware Team Rocket's Mewtwo ex / Spidops bot.
The plan produced state, belief, value-head, planning, official-engine,
telemetry, replay, packaging, and evaluation infrastructure. It did not by
itself train a superhuman strategy. The early integrated heuristic/search
controller was a research artifact, not the final intended game-theoretic
approach.

The strongest live Rocket recovery bot is currently
`HelbrosRocketSpidopsScaleBCV2`: active at rank 23 and rating 983.2 after 3,978
rated games (2,210–1,766–2) in the 2026-07-29 snapshot. It is a useful generic
relational neural policy but not a completed realization of the belief-aware
planner. N5 terminal PPO failed its locked gate. N6 produced replicated paired
rollout diagnostics but no student because persistent belief features could
not be reconstructed losslessly. The active N6R correction recollects exact
actor-visible feature snapshots before distillation. See
`STRENGTH_DIAGNOSIS_AND_RECOVERY.md`, `N6_BELIEF_ROLLOUT_RESULT.md`, and
`N6R_EXACT_FEATURE_BELIEF_ROLLOUT_PROTOCOL.md` for current evidence and claim
boundaries. Older phase bullets below are retained as research history.

The checked-in deck is the user-supplied transcription of Juho Kallama's
7th-place NAIC 2026 Team Rocket box list. It is validated as a 60-card deck with
15 Pokemon, 34 Trainers, and 11 Energy.

## Implemented capabilities

- A versioned, validated, fingerprinted deck configuration.
- Immutable game-state, action, turn-record, and match-trace models with stable
  JSON serialization and SHA-256 state digests.
- A deterministic state tensorizer with generic board/resource features and
  archetype-specific Team Rocket board features.
- A serializable latent plan snapshot and recurrent plan memory that persists
  strategic context between turns.
- Independent value heads for setup, tempo, resource efficiency, prize control,
  board control, and optionality.
- A state-conditioned meta-network that learns how to combine those values.
- A simulator-agnostic autoregressive tactical-policy contract plus a reference
  candidate-scoring implementation.
- Standard-library tests and a deterministic end-to-end smoke demo.
- A no-leakage `AgentObservation` that preserves public counts/history while
  removing private deck, prize, and opponent-hand identities.
- A neural opponent model over six strategic intents and ten coarse private
  resource categories, trained from public trajectory prefixes and later
  reveals.
- Held-out temperature calibration and plan/resource NLL, Brier, accuracy, and
  expected-calibration-error metrics.
- A multi-hypothesis inverse-planning beam built from minimum enabling
  resources, with rational, imperfect, deceptive, and unmodeled-play support.
- Bayesian reweighting from observed action sequences and later card-category
  reveals, plus serializable neural/symbolic belief fusion.
- Explicit belief conditioning of persistent plan memory, all strategic values,
  meta-value aggregation, and autoregressive tactical decoding.
- A deterministic Phase 2 contract demo with a held-out decision probe showing
  that inferred beliefs improve tactical choice accuracy over an uninformed
  context on the synthetic fixture.
- A Team Rocket-specific high-level proposer that returns a few complete,
  normalized, strategically distinct Mewtwo, Spidops, disruption, defensive,
  and Clefairy lines.
- Full retained-posterior search over rational, imperfect, deceptive, and
  unmodeled hidden-resource scenarios, with observed-behavior likelihood updates.
- Branch-local autoregressive tactical execution for every plan/scenario pair,
  retaining candidate probabilities, action prefixes, entropy, resource costs,
  and strategic rollout outcomes.
- Risk-aware plan selection using all six value heads, learned meta-weights,
  terminal estimates, priors, objective achievement, policy confidence,
  scenario variance, and resource cost.
- A canonical rich result containing plans, priors, tactical rollouts, scenario
  beliefs, inferred intent, values, uncertainty, objectives, and search costs.
- A deterministic Phase 3 benchmark that searches four plans across all 19
  retained scenarios and demonstrates a 729x estimated raw-branch reduction.
- A 24-signal search-criticality extractor covering policy confidence, value-head
  disagreement, opponent-belief entropy, public tactical thresholds, plan
  ambiguity, optionality tension, turn context, and available compute.
- A learned two-head search gate trained on matched policy/search outcomes to
  predict both whether search pays for itself and its expected value gain.
- Held-out temperature calibration and a utility-aware invocation threshold,
  with accuracy, precision, recall, Brier, ECE, invocation concentration, and
  realized-versus-oracle net-utility metrics.
- An immutable, serializable compute bank that distinguishes wanting search from
  being able to afford it and records budget before and after every decision.
- A selective policy wrapper that returns the routine tactical decision unchanged
  when the gate declines, and calls the full Phase 3 planner only for approved
  pivotal states.
- A deterministic Phase 4 benchmark that invokes search on 98.2% of profitable
  pivotal fixtures, 0% of routine fixtures, and retains 98.2% of oracle net
  search utility on the held-out synthetic test split.
- An environment-neutral self-play runner that invokes Phase 4 selective search,
  threads the compute bank through an episode, independently validates searched
  decisions, and admits or rejects every searched root.
- Fingerprinted teacher records containing the public observation, routine
  alternative, complete search artifact, outcome evidence, modal legal tactical
  sequence, plan/belief/value/uncertainty targets, and reproducibility metadata.
- Separate compute-budget and value-equivalent search costs so accounting units
  cannot silently contaminate search-utility labels.
- A same-root counterfactual auditor that limits training to competitive plans
  with identical posterior support, normalized branches, complete rollouts,
  finite labels, and bounded uncertainty.
- Multi-task distillation into the real autoregressive tactical policy,
  persistent plan memory, opponent beliefs, six strategic values, meta-value,
  a learned search-uncertainty head, and a cheap high-level plan policy.
- Episode-disjoint held-out evaluation plus checkpoint manifests binding student
  weights to dataset, deck, environment, teacher, search configuration, seed,
  and optimization history fingerprints.
- Post-distillation paired revalidation that removes expert search only when its
  remaining value no longer pays for computation, while retaining search on
  belief-critical states.
- A deterministic Phase 5 benchmark with four admitted teachers, eight audited
  counterfactuals, 95.0% held-out loss reduction, full named-gradient coverage,
  perfect held-out plan ranking, and exactly one retained hard-state search.
- A self-contained standard-library integration-baseline policy over the official Kaggle
  observation/selection contract, with persistent strategic plans, six-axis
  diagnostics, an archetype posterior, bounded native look-ahead, legal
  failsafes, and per-decision telemetry.
- Fail-closed official deck resolution. The exact list resolves 58/60 slots
  because the current TEF–POR engine does not contain Prism Tower CRI 80; an
  explicit Watchtower compatibility profile is labelled and fingerprinted as
  non-exact.
- A 144-match official-engine benchmark spanning four ablations, six
  archetypes, three adversarial styles, and both seats: 7,272 target decisions,
  zero illegal actions/fallbacks/timeouts, 80.6% local panel score, belief and
  search diagnostics, and six queued native replay reviews.
- A deterministic baseline archive builder that refuses the compatibility
  deck without an explicit flag and includes manifest hashes plus the licensed
  multi-platform competition runtime.
- Serializable public-belief and opponent-range schemas, a provenance-enforced
  abstract information-set game boundary, neural blueprint provider contracts,
  counterfactual-value bound and safe re-solving contracts, and a separate
  baseline-constrained opponent-exploitation overlay. The generic full-game
  blueprint remains untrained and its generic fallback resolver is a no-op;
  exact concrete implementations are restricted to the validation microgame.
- A whitelist-only official truth-replay converter producing 12 isolated player
  perspectives, 1,232 public frames, 610 causally aligned acting-observer
  decisions, 923 redacted hidden event identities, and 320 supervision-only
  later reveals with no unknown event types or detected structural leakage
  violations. Schema 2 corrects the native visualizer's next-frame `selected`
  semantics and excludes behavior targets from feature fingerprints.
- An exactly enumerable 55-state imperfect-information validation game with an
  analytic equilibrium, pure best-response enumeration, NashConv/exploitability
  metrics, deterministic tabular CFR, and a per-hypothesis bounded local
  resolver. This microgame is explicitly abstract and non-official.
- Two deterministic neural-blueprint candidates over that exact microgame: a
  public-belief policy/value model and a supervised regret-matching model. The
  selected policy checkpoint has exact NashConv `0.00004637`, restores with zero
  output drift, and is explicitly not a full-game or official-engine result.
- Calibrated opponent counterfactual-value intervals for the selected microgame
  blueprint, with 100% coverage on 12 calibration-heldout targets, explicit
  abstraction/OOD/interruption widening, fail-closed gates, and zero false-safe
  certificates across six exactly audited resolve roots.
- An all-or-blueprint exact resolver that completes a fixed 102-mixture grid or
  returns the neural blueprint exactly. Across nine endgame and three earlier
  roots it has strict per-hypothesis non-regression; iteration, node, deadline,
  and OOD interruption probes expose no partial strategy.
- A separately trained synthetic microgame opponent model plus robust exploit
  overlay. Six roots enable bounded exploitation; every final strategy is
  certified against all 64 pure opponent continuations, while low-confidence or
  wrong-seat models return the Phase 6E safe strategy exactly.
- A production-disabled one-step official observation/action bridge validated
  in shadow across 12 native matches and 566 target decisions. Every observed
  ordered selection lifts and round-trips exactly, all public hidden-state
  canaries remain invariant, own-private sensitivity passes 557/557 checks,
  and the wrapper changes zero actions. This is not a full-game transition game
  or a deployed neural/safe/exploit pipeline.
- A deterministic Phase 6H offline trajectory corpus with 610/610 valid
  behavior labels, complete ordered action sets, 610 observed continuation
  links, 12 terminal returns, 8,794 causal later-reveal supervision links, and
  zero paired public state/history mismatches. Its public-only six-deck range
  updater is an uncalibrated non-neural baseline; the corpus is not branchable.
- A deterministic Phase 6I multi-task neural bootstrap over that corpus with
  train-only feature normalization, match-disjoint train/calibration/validation
  partitions, variable-cardinality legal-action scoring, public-only range
  prediction, outcome/next-public auxiliary heads, and exact checkpoint
  restoration across all 610 decisions. Its action target is logged heuristic
  behavior; it is not an equilibrium blueprint or deployable policy.
- A production-disabled Phase 6J branchable full-game boundary over the
  licensed native search API. Six simultaneous hidden-completion roots preserve
  identical policy-facing root information sets, alternative/repeated root and
  midgame branches execute correctly, explicit coin chance is represented, and
  all six exercised paths reach zero-sum native terminal states. Native
  descendants are genuine engine execution; the supplied hidden prior remains
  an uncalibrated abstraction.
- A production-disabled Phase 6K full-game neural candidate initialized from
  Phase 6I and trained on terminal-valued Phase 6J action branches. Two rounds
  cover 24 information sets, both players, all six completion candidates, and
  15,452 native branch transitions. Loss falls, legal support and restoration
  are exact, but held-out deviation diagnostics are mixed. This is sampled
  regret-matching wiring, not Deep CFR convergence, safe values, deployment, or
  a strength gain.
- A deterministic Phase 6L external-sampling MCCFR reference implementation.
  Exhaustive enumeration verifies every conditional advantage estimate to
  floating-point tolerance on the exact non-official microgame; a 20,000-
  iteration tabular run reaches NashConv `0.00461750`. This validates the
  reach-corrected estimator and separate strategy/advantage memories, not a
  full-game neural equilibrium.
- A production-disabled Phase 6M native endgame wiring probe. Five immediate-
  terminal roots exercise both players, semantic public/private/action
  features, fail-closed budgets, redacted serialization, and separate
  reservoirs over 45 genuine native descendant transitions. Every sampled
  advantage is zero because all legal actions at those closing prompts have
  the same terminal utility; the probe validates integration only and is not
  useful training or performance evidence.
- A deterministic Phase 6N outcome-sampling MCCFR reference. Exhaustive
  enumeration over 60 sampled terminal histories validates an arbitrary
  nonzero pre-action baseline correction to `4.16e-17` maximum error. The
  20,000-iteration exact-microgame reference reaches NashConv `0.03111612`;
  this validates the higher-variance estimator, not native convergence.
- A production-disabled Phase 6O native turn-plan candidate. Five persistent
  plan choices are executed through a fixed legal tactical policy over 5,569
  genuine native descendant transitions. Both-player outcome-regret memories,
  a separate average-strategy memory, and three restorable neural networks are
  nonempty. The raw targets are extremely heavy-tailed and transformed for
  training, so this is nontrivial training wiring—not an equilibrium, safety,
  deployment, or strength result.
- A production-disabled Phase 6P collect–fit–update loop. Three rounds use the
  previous regret heads to generate the next both-player trajectories and keep
  average-strategy training separate. The loop is functional, but its raw
  regret p95 reaches `5.04e11` and maximum `1.24e16`; it therefore fails the
  variance gate for counterfactual calibration and remains non-deployable.
- A deterministic Phase 6Q fail-closed safety admission. It binds the Phase 6P
  dataset/model/evidence fingerprints and rejects 13 of 16 necessary gates,
  including variance, independent value calibration, held-out particles,
  belief calibration, and a full-game regret proxy. It builds no fake wide
  bounds and executes no resolver or exploitation overlay.
- A production-disabled Phase 6R depth-limited variance probe. An exact
  non-official audit preserves 18 pre-cutoff regret expectations to
  `5.55e-17`. The live six-step horizon executes 1,474 genuine native tactical
  descendants across 36 cutoff trajectories and lowers the observed raw-regret
  p95 / maximum to `17.90` / `134.92`. Its Phase 6K leaves remain confidence-
  zero, full-range, and uncalibrated, so this is not a value, safety,
  convergence, deployment, or strength result.
- A production-disabled Phase 6S held-out leaf diagnostic. Eighteen disjoint
  within-zone order particles supply 54 genuine terminal continuations over
  6,240 native transitions. Validation coverage is 100% only because every
  interval clips to the full `[-1, 1]` range; informative fraction is zero and
  maximum rollout standard error is `0.667`. The static public range matches
  uniform NLL exactly. Leaf, belief, and counterfactual calibration gates all
  remain false.
- A production-disabled Phase 6T allocation/range diagnostic. Thirty-six
  public-consistent hidden-allocation particles produce 216 public decision
  rows over 274 genuine native descendants. On the 72 held-out rows, the
  Phase 6I neural range is worse than uniform (`1.8109` versus `1.7918` NLL)
  and far worse than the causal revealed-card updater (`0.8959` NLL). The
  neural allocation gate and public-belief calibration remain false; the
  causal updater becomes the next belief baseline, not a safety certificate.
- A production-disabled Phase 6U low-noise leaf diagnostic. Twenty-four
  allocation particles branch into 240 terminal continuations and 29,690
  genuine native transitions. Ten rollouts reduce maximum target standard
  error to `0.333`, but calibration selects the zero baseline over the Phase 6I
  and Phase 6K value heads; the interval remains full-width with zero
  informative fraction. Sampling precision passes while model bias, value
  calibration, and every safety/deployment gate remain unresolved.
- A production-disabled Phase 6V purpose-trained public leaf. A ridge model is
  trained only on Phase 6U variants 0-1 with leave-one-candidate-out
  hyperparameter selection, then calibrated and validated on six plus six rows
  from a new 24-particle native run. It improves fresh validation RMSE from the
  best reference's `0.489898` to `0.446013`, but its mean 90% interval width is
  `1.997117` on a width-2 utility range. The empirical gate therefore remains
  false, and no counterfactual, resolver, control, or strength claim follows.
- A production-disabled Phase 6W collection-heldout audit. Three complete
  24-particle collections are assigned wholly to training, conformal
  calibration, and untouched validation. The retrained public leaf achieves
  `0.595137` validation RMSE versus zero's `0.493288`, while its interval still
  averages `1.721623` wide. This rejects Phase 6V's small-split improvement as
  non-generalizing and leaves every downstream capability disabled.
- A production-disabled Phase 6X opponent-deviation substrate. Twelve supplied
  hidden-allocation hypotheses branch all five opponent turn plans into 600
  terminal continuations and 73,538 genuine native transitions. All validation
  surfaces have nontrivial action sensitivity, with mean span `0.633333` and
  maximum rollout SE `0.333333`. These are sampled-reach conditional action
  values, not reach-weighted counterfactual values; safety and control remain
  disabled.
- A production-disabled Phase 6Y counterfactual-reach ledger. An exhaustive
  24-row microgame audit matches direct counterfactual reach/value sums at zero
  error and exercises weights up to `2.0`. A fresh native collection attaches
  redacted reach records to 12 surfaces, 600 continuations, and 70,741 genuine
  transitions. All native weights are `1.0` because every macro root is already
  the opponent's first decision; all 12 public information sets are unique.
  Reach bookkeeping passes, while native counterfactual calibration, resolving,
  and control remain disabled.
- A production-disabled Phase 6Z public-root-aligned value experiment. Eighteen
  later live roots are split wholly across train, calibration, and validation;
  three zero-substitution hidden allocations per root branch all five plans
  into 810 terminal continuations and 70,931 genuine native transitions. A
  hypothesis-conditioned ridge model and per-action conformal intervals are fit
  without crossing root splits. It is rejected at `75.56%` validation coverage
  and `0.847985` RMSE versus zero's `0.771402`. The acting player's observed
  hand is fixed within each root and the hidden prior is supplied, so the
  outputs remain conditional values and every
  counterfactual, resolver, exploitation, and control gate stays closed.
- A production-disabled Phase 6AA public-archetype range diagnostic. A
  train-only six-prototype provider selects temperature/shrinkage on six
  calibration roots and reaches validation NLL `0.800538` versus uniform
  `1.791759`, Brier `0.378643` versus `0.833333`, and top-1 `5/6`. The held-out
  archetype-signal gate passes, but ECE is `0.476073`, each holdout has only six
  roots, and hidden hands/resources/intents are unmodeled. Official range,
  counterfactual, resolver, exploitation, and control gates remain false.
- A production-disabled Phase 6AB official-prefix reach ledger. Eighteen fresh
  live battles reach player 1's second MAIN turn and preserve continuous,
  redacted public-state/history chains across 512 genuine `battle_select`
  executions. Declared deterministic behavior and sampling factors are exactly
  one, but the live adapter exposes no shuffle/draw/Prize/coin likelihoods.
  Chance reach, full sampling reach, counterfactual weights, Phase 6Z value-root
  alignment, resolving, exploitation, and control remain unavailable.
- A production-disabled Phase 6AC joint official truth/value substrate. Eighteen
  new live roots bind 568 redacted prefix selections, one transient realized
  official hidden-state sample, exact native root/next-draw replay parity, and
  all five plan values at the same root. The run executes 270 terminal
  continuations and 29,460 genuine native descendants. A common opaque chance
  law cancels symbolically in the on-policy likelihood ratio, but absolute
  chance reach remains unknown and every information set has only one truth
  sample; value calibration, safety, exploitation, and control stay disabled.
- A production-disabled Phase 6AD observer-oriented official range provider.
  The checked collection contains 180 fresh roots, 5,178 genuine prefix
  selections, and 34,560 public-consistent allocation particles across six
  declared opponent decks. Train/calibration/validation each contain 60 roots.
  Candidate and hidden hand/Prize/deck hash-bucket calibration pass the
  preregistered in-scope gates on untouched validation roots. Opponent intent,
  full official belief, counterfactual values, safe resolving, exploitation,
  official control, and performance claims remain false.
- A production-disabled Phase 6AE seat-decoupled joint range provider. The
  checked collection crosses six exact candidate decks with six controlled
  player-1 intent proxies over 324 fresh official roots while player 0 stays on
  one fixed policy. Its 108-root untouched validation split reaches `0.879630`
  declared-proxy top-1 accuracy, `0.042259` intent ECE, and `0.042195` joint
  ECE. This admits only the declared-policy proxy scope: human latent intent,
  full official belief, counterfactual values, safe resolving, exploitation,
  official control, submission, and strength claims remain false.
- A production-disabled Phase 6AF implementation-heldout joint range provider.
  Two opponent-policy implementations train the intent model, a third selects
  calibration, and an untouched fourth evaluates transfer. The 432 fresh roots
  reach `0.826389` validation top-1, `0.090055` intent ECE, `0.091196` joint
  ECE, and at least `0.583333` top-1 in every intent-family slice. This is still
  controlled policy-family evidence, not human latent intent or authorization
  for values, resolving, exploitation, official control, or submission.
- A production-disabled Phase 6AG full-game neural point-value leaf. Its frozen
  288-match panel uses two training implementations, a split calibration
  implementation, and an untouched fourth implementation. Validation episode
  MSE `0.938795` beats zero `1.000000` and the causal board heuristic
  `1.019981`. Ninety-percent progress-bucket intervals fail coverage and width
  gates, so counterfactual values, resolving, exploitation, official control,
  submission, and strength claims remain false.
- A production-disabled Phase 6AH repeated root/action expected-value
  experiment. It completes 180 new official matches and 14,400 terminal native
  continuations across 360 supplied hidden allocations and 900 five-plan
  targets. Repetition lowers maximum target SE to `0.258199`, and validation
  point MSE `0.704402` beats zero `0.812674` and heuristic `0.895724`; however,
  action-contrast MSE `0.028905` loses to zero-contrast `0.026840`. Coverage
  `0.872222`, mean width `1.826374`, and minimum slice coverage `0.733333` also
  fail. No conditional value model, counterfactual bound, resolver,
  exploitation, official control, submission, or strength claim is admitted.
- A production-disabled Phase 6AI causal branch-conditioned value experiment.
  It adds immediate public-only native child deltas for every supplied
  allocation/plan branch across 180 fresh roots and 14,400 continuations. The
  grouped neural model improves point MSE but fails the frozen action-contrast,
  bootstrap, and interval gates. Its causal branch heuristic provides a useful
  diagnostic ordering, but no learned action-sensitive value is admitted.
- A production-disabled Phase 6AJ semiparametric confirmation. Its model and
  interval radius are frozen before a 36-root, implementation-heldout panel is
  opened. Hybrid point MSE `0.582840` beats zero `0.764236`, root heuristic
  `0.891487`, and standalone branch heuristic `0.853716`; contrast MSE
  `0.022868` beats zero `0.025278`, with a positive `0.000753` bootstrap lower
  bound. The action-sensitive point signal is admitted, but mean interval width
  `1.839791` fails the `1.75` gate. Counterfactual values, resolving,
  exploitation, official control, submission, and strength claims stay false.
- A production-disabled Phase 6AK expanded-root-mean confirmation. It declares
  all 216 Phase 6AI/6AJ roots as development data, freezes a strongly
  regularized ridge state mean, freezes bounds on 36 new calibration roots,
  and then opens 36 distinct validation roots. Validation MSE `0.398773` beats
  the frozen Phase 6AJ hybrid `0.503581` and every nonlearned reference;
  contrast and bootstrap gates also pass. Overall coverage `0.916667` and mean
  width `1.676068` pass, but the worst intent-slice coverage is only `0.666667`
  against the `0.75` floor. This validates a scoped point diagnostic, not a
  conditional expected-mean or counterfactual-value provider. Resolving,
  exploitation, official control, submission, and strength claims stay false.
- A production-disabled Phase 6AL high-precision confirmation. It freezes the
  exact Phase 6AK mean, then collects 36 calibration and 36 validation roots
  with two replay batches and 32 terminal returns per action. Point MSE
  `0.430678` beats all frozen references, but the contrast-bootstrap lower
  bound is `-0.000512`, coverage `0.888889` misses `0.90`, and worst-slice
  coverage remains `0.666667`. The preregistered protocol therefore rejects
  both action-sensitive point and conditional expected-mean admission; every
  counterfactual, resolver, exploitation, control, submission, and strength
  gate remains closed.
- A production-disabled Phase 6AM public-range counterfactual-advantage audit.
  It freezes a baseline-relative branch center and `0.555019619275336`
  conformal radius from the complete Phase 6AI-6AL development ledger before
  collecting 36 new official roots. The held-out six-candidate public range is
  top-1 correct on all roots; coverage `0.95`, worst-slice coverage `0.833333`,
  width `1.110039`, and the bootstrap gate pass. Maximum target SE `0.201688`
  narrowly exceeds the frozen `0.20` ceiling, so the complete 6-hypothesis x
  5-action normalized bound interface is rejected and remains inspection-only.
  Absolute values, equilibrium, resolving, exploitation, official control,
  submission, and strength claims stay false.

The durable multi-phase roadmap is in [PROJECT_PLAN.md](PROJECT_PLAN.md). Module
contracts and future integration seams are in
[docs/PHASE1_ARCHITECTURE.md](docs/PHASE1_ARCHITECTURE.md),
[docs/PHASE2_ARCHITECTURE.md](docs/PHASE2_ARCHITECTURE.md),
[docs/PHASE3_ARCHITECTURE.md](docs/PHASE3_ARCHITECTURE.md),
[docs/PHASE4_ARCHITECTURE.md](docs/PHASE4_ARCHITECTURE.md),
[docs/PHASE5_ARCHITECTURE.md](docs/PHASE5_ARCHITECTURE.md), and
[docs/PHASE6_ARCHITECTURE.md](docs/PHASE6_ARCHITECTURE.md). Requirement-level
evidence is recorded in [docs/PHASE2_COMPLETION_AUDIT.md](docs/PHASE2_COMPLETION_AUDIT.md),
[docs/PHASE3_COMPLETION_AUDIT.md](docs/PHASE3_COMPLETION_AUDIT.md),
[docs/PHASE4_COMPLETION_AUDIT.md](docs/PHASE4_COMPLETION_AUDIT.md), and
[docs/PHASE5_COMPLETION_AUDIT.md](docs/PHASE5_COMPLETION_AUDIT.md), and
[docs/PHASE6_COMPLETION_AUDIT.md](docs/PHASE6_COMPLETION_AUDIT.md). The durable
game-theoretic target and sequencing are in
[PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md](PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md).
Phase 6A-6B implementation details are in
[docs/PHASE6A_B_GAME_THEORY_FOUNDATION.md](docs/PHASE6A_B_GAME_THEORY_FOUNDATION.md).
Phase 6C neural microgame evidence is in
[docs/PHASE6C_NEURAL_BLUEPRINT.md](docs/PHASE6C_NEURAL_BLUEPRINT.md).
Phase 6D interval calibration is in
[docs/PHASE6D_COUNTERFACTUAL_BOUNDS.md](docs/PHASE6D_COUNTERFACTUAL_BOUNDS.md).
Phase 6E resolver evidence is in
[docs/PHASE6E_SAFE_RESOLVING.md](docs/PHASE6E_SAFE_RESOLVING.md).
Phase 6F exploitation evidence is in
[docs/PHASE6F_CERTIFIED_EXPLOITATION.md](docs/PHASE6F_CERTIFIED_EXPLOITATION.md).
Phase 6G one-step official bridge evidence is in
[docs/PHASE6G_OFFICIAL_SHADOW_BRIDGE.md](docs/PHASE6G_OFFICIAL_SHADOW_BRIDGE.md).
Phase 6H corrected trajectory evidence is in
[docs/PHASE6H_OFFICIAL_TRAJECTORY_SUBSTRATE.md](docs/PHASE6H_OFFICIAL_TRAJECTORY_SUBSTRATE.md).
Phase 6I logged neural-bootstrap evidence is in
[docs/PHASE6I_LOGGED_NEURAL_BOOTSTRAP.md](docs/PHASE6I_LOGGED_NEURAL_BOOTSTRAP.md).
Phase 6J branchable native-search evidence is in
[docs/PHASE6J_BRANCHABLE_NATIVE_SEARCH_GAME.md](docs/PHASE6J_BRANCHABLE_NATIVE_SEARCH_GAME.md).
Phase 6K sampled-regret blueprint-candidate evidence is in
[docs/PHASE6K_NATIVE_REGRET_BLUEPRINT.md](docs/PHASE6K_NATIVE_REGRET_BLUEPRINT.md).
Phase 6L reach-corrected estimator evidence is in
[docs/PHASE6L_REACH_CORRECTED_MCCFR.md](docs/PHASE6L_REACH_CORRECTED_MCCFR.md).
Phase 6M native endgame wiring evidence is in
[docs/PHASE6M_NATIVE_EXTERNAL_SAMPLING.md](docs/PHASE6M_NATIVE_EXTERNAL_SAMPLING.md).
Phase 6N outcome-sampling reference evidence is in
[docs/PHASE6N_OUTCOME_SAMPLING_MCCFR.md](docs/PHASE6N_OUTCOME_SAMPLING_MCCFR.md).
Phase 6O native turn-plan evidence is in
[docs/PHASE6O_NATIVE_TURN_PLAN_BLUEPRINT.md](docs/PHASE6O_NATIVE_TURN_PLAN_BLUEPRINT.md).
Phase 6P iterative training and failed variance-gate evidence is in
[docs/PHASE6P_ITERATIVE_NATIVE_PLAN_MCCFR.md](docs/PHASE6P_ITERATIVE_NATIVE_PLAN_MCCFR.md).
Phase 6Q fail-closed safety admission is in
[docs/PHASE6Q_NATIVE_SAFETY_ADMISSION.md](docs/PHASE6Q_NATIVE_SAFETY_ADMISSION.md).
Phase 6R depth-limited variance evidence is in
[docs/PHASE6R_DEPTH_LIMITED_NATIVE_PLAN.md](docs/PHASE6R_DEPTH_LIMITED_NATIVE_PLAN.md).
Phase 6S native leaf-continuation evidence is in
[docs/PHASE6S_NATIVE_LEAF_CALIBRATION.md](docs/PHASE6S_NATIVE_LEAF_CALIBRATION.md).
Phase 6T hidden-allocation/public-range evidence is in
[docs/PHASE6T_NATIVE_ALLOCATION_RANGE.md](docs/PHASE6T_NATIVE_ALLOCATION_RANGE.md).
Phase 6U low-noise allocation-heldout leaf evidence is in
[docs/PHASE6U_LOW_NOISE_NATIVE_LEAF.md](docs/PHASE6U_LOW_NOISE_NATIVE_LEAF.md).
Phase 6V purpose-trained public-leaf evidence is in
[docs/PHASE6V_PUBLIC_NATIVE_LEAF.md](docs/PHASE6V_PUBLIC_NATIVE_LEAF.md).
Phase 6W collection-heldout leaf evidence is in
[docs/PHASE6W_COLLECTION_HELDOUT_LEAF.md](docs/PHASE6W_COLLECTION_HELDOUT_LEAF.md).
Phase 6X opponent-hypothesis deviation-value evidence is in
[docs/PHASE6X_OPPONENT_DEVIATION_VALUES.md](docs/PHASE6X_OPPONENT_DEVIATION_VALUES.md).
Phase 6Y counterfactual-reach evidence is in
[docs/PHASE6Y_COUNTERFACTUAL_REACH_LEDGER.md](docs/PHASE6Y_COUNTERFACTUAL_REACH_LEDGER.md).
Phase 6Z public-root-aligned value evidence is in
[docs/PHASE6Z_PUBLIC_ROOT_ALIGNED_VALUES.md](docs/PHASE6Z_PUBLIC_ROOT_ALIGNED_VALUES.md).
Phase 6AA aligned public-archetype range evidence is in
[docs/PHASE6AA_ALIGNED_PUBLIC_RANGE.md](docs/PHASE6AA_ALIGNED_PUBLIC_RANGE.md).
Phase 6AB redacted official-prefix reach evidence is in
[docs/PHASE6AB_OFFICIAL_PREFIX_REACH.md](docs/PHASE6AB_OFFICIAL_PREFIX_REACH.md).
Phase 6AC same-root official truth/value evidence is in
[docs/PHASE6AC_JOINT_OFFICIAL_VALUE_SAMPLES.md](docs/PHASE6AC_JOINT_OFFICIAL_VALUE_SAMPLES.md).
Phase 6AD observer-oriented official hidden-card range evidence is in
[docs/PHASE6AD_OFFICIAL_HIDDEN_CARD_RANGE.md](docs/PHASE6AD_OFFICIAL_HIDDEN_CARD_RANGE.md).
Phase 6AE seat-decoupled official intent-proxy evidence is in
[docs/PHASE6AE_OFFICIAL_INTENT_PROXY.md](docs/PHASE6AE_OFFICIAL_INTENT_PROXY.md).
Phase 6AF policy-implementation-heldout evidence is in
[docs/PHASE6AF_POLICY_IMPLEMENTATION_GENERALIZATION.md](docs/PHASE6AF_POLICY_IMPLEMENTATION_GENERALIZATION.md).
Phase 6AG official full-game value-leaf evidence is in
[docs/PHASE6AG_OFFICIAL_VALUE_LEAF.md](docs/PHASE6AG_OFFICIAL_VALUE_LEAF.md).
Phase 6AH repeated expected-value evidence is in
[docs/PHASE6AH_EXPECTED_VALUE_SURFACES.md](docs/PHASE6AH_EXPECTED_VALUE_SURFACES.md).
Phase 6AI causal branch-conditioned evidence is in
[docs/PHASE6AI_CAUSAL_BRANCH_VALUES.md](docs/PHASE6AI_CAUSAL_BRANCH_VALUES.md).
Phase 6AJ semiparametric confirmation evidence is in
[docs/PHASE6AJ_SEMIPARAMETRIC_BRANCH_HYBRID.md](docs/PHASE6AJ_SEMIPARAMETRIC_BRANCH_HYBRID.md).
Phase 6AK expanded-root-mean evidence is in
[docs/PHASE6AK_EXPANDED_ROOT_MEAN.md](docs/PHASE6AK_EXPANDED_ROOT_MEAN.md).
Phase 6AL high-precision confirmation evidence is in
[docs/PHASE6AL_HIGH_PRECISION_CONFIRMATION.md](docs/PHASE6AL_HIGH_PRECISION_CONFIRMATION.md).
Phase 6AM public-range normalized counterfactual-advantage evidence is in
[docs/PHASE6AM_PUBLIC_RANGE_COUNTERFACTUAL_ADVANTAGES.md](docs/PHASE6AM_PUBLIC_RANGE_COUNTERFACTUAL_ADVANTAGES.md).
Submission gates are in
[docs/KAGGLE_COMPETITION_RUNBOOK.md](docs/KAGGLE_COMPETITION_RUNBOOK.md).

## Quick start

Python 3.11+ and PyTorch 2.x are required. From this project directory:

```powershell
python scripts/run_tests.py
python scripts/smoke_demo.py
python scripts/phase2_demo.py
python scripts/phase3_demo.py
python scripts/phase4_demo.py
python scripts/phase5_demo.py
python scripts/phase6_demo.py --repetitions 1
python scripts/phase6_foundation.py
python scripts/phase6_neural.py
python scripts/phase6_bounds.py
python scripts/phase6_resolving.py
python scripts/phase6_exploitation.py
python scripts/phase6_bridge.py
python scripts/phase6_trajectory.py
python scripts/phase6_trajectory_neural.py
python scripts/phase6_native_game.py
python scripts/phase6_native_blueprint.py
python scripts/phase6_mccfr_reference.py
python scripts/phase6_native_external_sampling.py
python scripts/phase6_outcome_sampling_reference.py
python scripts/phase6_native_plan_blueprint.py
python scripts/phase6_native_plan_iterations.py
python scripts/phase6_native_safety_admission.py
python scripts/phase6_native_depth_limited.py
python scripts/phase6_native_leaf_calibration.py
python scripts/phase6_native_belief_calibration.py
python scripts/phase6_native_leaf_model.py
python scripts/phase6_native_public_leaf.py
python scripts/phase6_root_heldout_leaf.py
python scripts/phase6_opponent_deviation_values.py
python scripts/phase6_native_counterfactual_reach.py
python scripts/phase6_aligned_counterfactual_values.py
python scripts/phase6_aligned_public_range.py
python scripts/phase6_official_prefix_reach.py
python scripts/phase6_joint_official_values.py
python scripts/phase6_official_range_calibration.py
python scripts/phase6_official_intent_calibration.py
python scripts/phase6_official_policy_generalization.py
python scripts/phase6_official_value_leaf.py --reuse-source
python scripts/phase6_expected_value_surfaces.py --reuse-source
python scripts/phase6_branch_conditioned_values.py --reuse-source
python scripts/phase6_semiparametric_branch_hybrid.py --reuse-validation
python scripts/phase6_expanded_root_mean.py --stage evaluate
python scripts/phase6_high_precision_confirmation.py --stage evaluate
python scripts/phase6_public_range_counterfactual_advantages.py --stage evaluate
```

Optional editable installation:

```powershell
python -m pip install -e .
rocketbot-smoke
rocketbot-phase2-demo
rocketbot-phase3-demo
rocketbot-phase4-demo
rocketbot-phase5-demo
rocketbot-phase6-demo
rocketbot-phase6-trajectory
rocketbot-phase6-trajectory-neural
rocketbot-phase6-native-game
rocketbot-phase6-native-blueprint
rocketbot-phase6-mccfr-reference
rocketbot-phase6-native-external-sampling
rocketbot-phase6-native-belief-calibration
rocketbot-phase6-native-leaf-model
rocketbot-phase6-opponent-deviation-values
rocketbot-phase6-native-counterfactual-reach
rocketbot-phase6-aligned-counterfactual-values
rocketbot-phase6-aligned-public-range
rocketbot-phase6-official-prefix-reach
rocketbot-phase6-joint-official-values
rocketbot-phase6-official-range-calibration
rocketbot-phase6-official-intent-calibration
rocketbot-phase6-official-policy-generalization
rocketbot-phase6-official-value-leaf
rocketbot-phase6-expected-value-surfaces
rocketbot-phase6-branch-conditioned-values
rocketbot-phase6-semiparametric-branch-hybrid
rocketbot-phase6-expanded-root-mean
rocketbot-phase6-high-precision-confirmation
rocketbot-phase6-public-range-counterfactual-advantages
```

The Phase 1-5 demos print deterministic JSON. The Phase 1 smoke verifies deck, state,
plan, strategic heads, aggregation, and tactical masking. The Phase 2 demo
verifies observation redaction, trajectory supervision, held-out calibration,
inverse action updates, behavior alternatives, neural/symbolic serialization,
integrated tensor shapes, and downstream decision conditioning. The Phase 3
demo verifies complete/distinct proposal, full-posterior behavior comparison,
autoregressive branch execution, risk-aware selection, branch safety,
rich-result serialization, and plan-level branching reduction. The Phase 4 demo
verifies learned gating, held-out calibration, selective search concentration,
compute-budget enforcement, net utility, and durable decision serialization. The
Phase 5 demo verifies self-play collection, independent admission, audited
counterfactuals, every multi-task gradient path, held-out distillation,
checkpoint restoration, and post-distillation expert-search retention.
The Phase 6 script uses the staged official competition engine to validate legal
decks, play the full ablation/opponent/style matrix, and persist JSON, Markdown,
native replays, and automated review rubrics under `artifacts/phase6_evidence`.
The matrix is reproducible, but the native engine exposes no match-seed control,
so individual shuffled outcomes are not byte-deterministic.
The Phase 6 foundation script deterministically converts the six selected
native replays into leakage-audited public episodes, verifies the exact analytic
microgame equilibrium, trains the tabular CFR reference, and exercises exact
counterfactual-bounded local re-solving.

The exact deck cannot currently be submitted: Prism Tower is absent from the
official engine. A local, baseline-only compatibility archive can be built only
with an explicit acknowledgement:

```powershell
python scripts/build_submission.py --allow-watchtower-compatibility
```

The archive is labeled as a heuristic/sampled-rollout integration baseline and
is not the final game-theoretic agent. Do not submit that changed deck without
owner approval. The official `cg` runtime is competition-use-only and must not
be redistributed.

## Repository layout

```text
src/rocketbot/
  data/naic_2026_juho_kallama_mewtwo_spidops.json
  deck.py       validated deck configuration
  state.py      immutable state and history schema
  features.py   deterministic strategic tensorization
  plan.py       persistent latent plan state and memory
  strategy.py   six value heads and meta-value aggregation
  policy.py     autoregressive tactical-policy interfaces
  observation.py      no-leakage public observations and action features
  beliefs.py          intent/resource posteriors and inverse planning
  belief_model.py     neural belief model, calibration, and metrics
  belief_training.py  trajectory labels, batching, and losses
  model.py            strategic agent plus learned search uncertainty
  plans.py            complete plan schema and Team Rocket proposer
  search.py           belief-aware plan search and rich results
  search_gate.py      learned criticality gate, calibration, and compute bank
  search_learning.py  validated teacher records and self-play collection
  distillation.py     multi-task student, losses, metrics, and checkpoints
  abstract_simulator.py  branch-safe strategic verification adapter
  game_theory.py     neural game-theoretic contracts and explicit safety stubs
  public_replay.py   truth-replay perspective conversion and leakage audit
  official_trajectory.py corrected logged information-set trajectory substrate
  trajectory_neural.py split-safe logged-data neural bootstrap and checkpoint
  official_search_game.py branchable native-search information-set boundary
  native_blueprint_features.py Phase 6I-compatible no-truth native features
  native_regret_blueprint.py restorable sampled-regret blueprint candidate
  external_sampling.py bounded MCCFR traversal and separate replay reservoirs
  mccfr_reference.py exact estimator audit and tabular convergence reference
  exact_game.py      enumerable microgame, best responses, CFR, exact resolver
  neural_blueprint.py deterministic microgame neural providers and checkpoints
  counterfactual_calibration.py calibrated value bounds and exact safety audit
  safe_resolving.py all-or-blueprint budget/deadline-safe exact resolver
  opponent_exploitation.py separate models and exact robust exploit certificate
  official_bridge.py one-step official observation/action bridge and shadow audit
  demo.py             deterministic Phase 1 integration demo
  phase2_demo.py      deterministic Phase 2 contract and decision demo
  phase3_demo.py      deterministic Phase 3 plan-search benchmark
  phase4_demo.py      deterministic Phase 4 selective-search benchmark
  phase5_demo.py      deterministic Phase 5 search-guided learning benchmark
  competition.py     official contract, baseline policy, safety, telemetry
  competition_search.py  bounded baseline native look-ahead
  official_evaluation.py native match runner and legal opponent panels
  evaluation.py      multidimensional metrics, slices, evidence report
  replay_review.py   replay triage and human expert rubric
  phase6_demo.py     official Phase 6 integration-baseline benchmark
  phase6_foundation.py deterministic Phase 6A-6B evidence
  phase6_neural.py  deterministic Phase 6C neural microgame evidence
  phase6_bounds.py  deterministic Phase 6D bound-calibration evidence
  phase6_resolving.py deterministic Phase 6E resolver evidence
  phase6_exploitation.py deterministic Phase 6F exploitation evidence
  phase6_bridge.py live Phase 6G official shadow-bridge evidence
  phase6_trajectory.py deterministic Phase 6H trajectory evidence
  phase6_trajectory_neural.py deterministic Phase 6I bootstrap evidence
  phase6_native_game.py live Phase 6J branchable-game evidence
  phase6_native_blueprint.py live Phase 6K regret-candidate evidence
  phase6_mccfr_reference.py deterministic Phase 6L estimator evidence
  phase6_native_external_sampling.py live Phase 6M endgame-wiring evidence
  phase6_outcome_sampling_reference.py deterministic Phase 6N estimator evidence
  phase6_native_plan_blueprint.py live Phase 6O turn-plan candidate evidence
  phase6_native_plan_iterations.py live Phase 6P iterative/variance evidence
  phase6_native_safety_admission.py deterministic Phase 6Q rejection evidence
  depth_limited_game.py generic abstract cutoff/leaf-value boundary
  native_plan_leaf.py confidence-zero native diagnostic leaf adapter
  phase6_native_depth_limited.py Phase 6R exact/live variance evidence
  phase6_native_leaf_calibration.py Phase 6S held-out leaf diagnostic
  native_public_range.py public-only native range adapter
  phase6_native_belief_calibration.py Phase 6T allocation/range diagnostic
  phase6_native_leaf_model.py Phase 6U low-noise leaf diagnostic
  native_public_leaf.py split-safe public-only ridge leaf
  phase6_native_public_leaf.py Phase 6V fresh-holdout leaf evidence
  phase6_root_heldout_leaf.py Phase 6W collection-heldout leaf audit
  phase6_opponent_deviation_values.py Phase 6X action-value substrate
  counterfactual_reach.py exact reach accounting and redacted trace primitives
  phase6_native_counterfactual_reach.py Phase 6Y native reach evidence
  phase6_aligned_counterfactual_values.py Phase 6Z aligned-root value evidence
  phase6_aligned_public_range.py Phase 6AA public-archetype range diagnostic
  phase6_official_prefix_reach.py Phase 6AB live redacted prefix-reach ledger
  phase6_joint_official_values.py Phase 6AC same-root truth/value evidence
  phase6_official_range_calibration.py Phase 6AD official range calibration
  phase6_official_intent_calibration.py Phase 6AE intent-proxy calibration
  phase6_official_policy_generalization.py Phase 6AF policy-transfer calibration
  phase6_official_value_leaf.py Phase 6AG official full-game point-value leaf
  phase6_expected_value_surfaces.py Phase 6AH repeated root/action value audit
  phase6_branch_conditioned_values.py Phase 6AI causal branch-value audit
  phase6_semiparametric_branch_hybrid.py Phase 6AJ fresh hybrid confirmation
  phase6_expanded_root_mean.py Phase 6AK expanded root-mean confirmation
  phase6_high_precision_confirmation.py Phase 6AL high-precision confirmation
  phase6_public_range_counterfactual_advantages.py Phase 6AM normalized advantage audit
tests/          integrity, shape, masking, and round-trip tests
artifacts/      official evidence, exact-game/neural evidence, baseline bundle
```

## Claim boundary

The Phase 2-5 fixtures validate research contracts, not tournament strength.
Phase 6 genuinely executes the official card-rule engine and provides broad
local integration evidence. Phase 6G also validates a one-decision public/private
observation boundary and exact action lifting while returning the baseline
selection unchanged. Phase 6J adds a branchable native-search transition
boundary under supplied hidden particles. Its game object remains abstract;
the neural equilibrium blueprint, safe resolver, and exploitation overlay
remain separate from official strategy control.
Phase 6H derives a causally aligned offline corpus from genuine native replays,
but its provenance is abstract and it contains no counterfactual branches.
Phase 6I trains a restorable representation/behavior model on that corpus, but
its behavior-imitation objective is explicitly not the final game-theoretic
strategy and it does not supply counterfactual values or official execution.
Phase 6J supplies counterfactual native action branches and terminal paths, but
not a calibrated belief prior, equilibrium checkpoint, safe bounds, or a
deployed planner.
Phase 6K trains a small full-game candidate from those terminal branches. It
validates semantic feature reuse, regret-target plumbing, legal support, and
checkpoint restoration; it does not provide reach-corrected Deep CFR,
full-game exploitability, equilibrium convergence, calibrated counterfactual
values, safe resolving, official strategy control, or improved play.
Phase 6L corrects that estimator gap and validates external-sampling MCCFR
exactly on the non-official microgame. Phase 6M connects the same traversal and
reservoir contracts to genuine native descendants only at five immediate-
terminal endgame roots. The resulting zero advantages, tiny scope, supplied
hidden completions, unseeded runtime, and absence of opponent decision nodes
mean neither milestone is a trained full-game blueprint, equilibrium result,
or strength claim.
Phase 6N validates a primary-source-aligned one-trajectory outcome-sampling
estimator, including nonzero baseline correction, on the exact microgame.
Phase 6O applies it to a five-action turn-plan abstraction over genuine native
descendants and trains separate player-regret and average-plan networks. The
plans, tactical executor, hidden prior, transformed neural objective, and model
remain abstractions. Heavy-tailed regrets, an uncalibrated prior, and no
full-game best-response or counterfactual-value audit preclude equilibrium,
safety, deployment, or performance claims.
Phase 6P verifies that the previous regret models can drive the next collection
round, but its substantially worse importance-weight tail fails the prerequisite
for counterfactual calibration. Safe resolving and exploitation remain off;
training-loop execution is not convergence evidence.
Phase 6Q formalizes that boundary in code: the candidate cannot expose
counterfactual bounds, a resolver, exploitation, or official strategy control
while any admission prerequisite is absent. Rejection is not a safety
certificate, but it prevents unsupported safety promotion.
Phase 6R controls trajectory horizon behind the same fail-closed boundary. The
exact cutoff audit passes and the live observed importance tail is much smaller,
but every live leaf is confidence-zero, full-range, and uncalibrated. Those
samples remain diagnostic and cannot enable bounds, resolving, exploitation,
official strategy control, deployment, or a performance claim.
Phase 6Y then validates the counterfactual importance-ratio convention exactly
on the abstract microgame and persists a redacted reach ledger beside a fresh
native Phase 6X-style collection. The observed unit weights are expected at the
first opponent decision and do not create a calibrated value: each public root
appears once, the hidden prior is supplied, and upstream official reach is not
certified. Counterfactual bounds, safe resolving, exploitation, and official
strategy control therefore remain off.
Phase 6Z replaces the one-particle-per-public-root shape with three hidden
allocations behind each of 18 later live roots and reserves entire root
collections for fitting, conformal calibration, and validation. This is genuine
native continuation evidence but still not counterfactual calibration: the
prior is supplied, acting-player hands are fixed at the exact observation, and
upstream official reach is unavailable. No result enters the resolver or
official policy.
Phase 6AA reuses those frozen roots to fit a public-only six-archetype range.
It detects held-out deck-identity signal but cannot claim an official public
posterior: holdouts are tiny, calibration error is high, feature orientation is
not production-bridged, and hidden hands/resources/intents are outside its
scope. The provider remains offline and cannot supply resolver bounds.
Phase 6AB independently recollects every declared deterministic player choice
from fresh official battle starts to later player-1 roots. The selections and
public chains are genuine, but internal chance-event likelihoods are not
exposed by the adapter. Its chance/full reach and counterfactual weights remain
`null`, and the independent roots cannot be joined to Phase 6Z values. This
closes no bound, resolver, exploitation, deployment, or strength gate.
Phase 6AC removes that root mismatch with a new joint collection: each prefix,
transient realized official truth sample, and five-plan value surface shares the
same live root. Native reinitialization matches the complete root observation
and a forced `END` transition matches the next public/private observation,
including the next draw. The target and sampler use the same opaque official
chance law, so that law cancels from the likelihood ratio without revealing its
absolute probability. This is still not value calibration: every information
set has one truth sample, rollout error reaches `0.666667`, and the five plans
and frozen continuations remain abstract. Bounds, resolving, exploitation,
official control, deployment, and strength therefore remain unvalidated.
Phase 6AD scales the official truth path to 180 fresh roots and explicitly
orients public features to player 0 even though the captured prompt belongs to
player 1. It passes preregistered held-out gates for six exact candidate lists
and 96 hidden hand/Prize/deck hash-bucket resources. The roots and transient
truth labels are genuine official execution; the observer adapter, exact-list
catalog, prototype classifier, uniform allocation prior, hash buckets, and
posterior interpretation remain research abstractions. The pass is limited to
this six-deck, baseline-policy, later-root distribution. It does not cover
opponent intent, absolute chance reach, counterfactual values, safe resolving,
exploitation, official control, deployment, or playing strength.
Phase 6AE independently crosses the same six candidate lists with six
controlled player-1 intent policies while player 0 remains fixed. The 324 fresh
official roots contain 8,474 selections and 875 public player-1 MAIN actions;
the untouched 108-root validation split passes preregistered declared-proxy,
joint-range, and resource gates. The intent labels, behavior rules, linear
model, candidate-intent independence assumption, and particles remain research
abstractions. Human latent intent, full belief, chance reach, values, bounds,
resolving, exploitation, official control, deployment, and strength remain
unvalidated.
Phase 6AF then holds policy implementation out explicitly. Canonical and
engine-first policies supply training roots, hand-first policies supply
calibration roots, and action-first policies remain untouched until validation.
All aggregate and per-intent transfer gates pass over 432 new official roots.
The authored policy family, labels, observer abstraction, model, particles, and
candidate-intent independence assumption remain non-official. Human latent
intent, complete belief, chance reach, values, bounds, resolving, exploitation,
control, deployment, and strength remain unvalidated.
Phase 6AG then completes 288 official games and trains a 208-input neural
outcome point estimate while keeping the `action_first` implementation
untouched. The point model strictly beats both preregistered episode-MSE
baselines, but the terminal-outcome conformal intervals under-cover and remain
too wide. It is an offline predictive leaf only: it is not a counterfactual
value provider or equilibrium blueprint, and every resolver, exploitation,
control, deployment, submission, and strength gate remains closed.
Phase 6AH then holds the same four policy implementations apart while replacing
one terminal label per state with 16 terminal returns for every root/action
target. Complete official matches, retained snapshots, exact native replay, and
descendant transitions are genuine; hidden allocations, five plans, tactical
execution, the continuation blueprint, and the regressor are abstractions.
Target precision passes, and aggregate point MSE improves, but the model fails
the preregistered action-contrast comparison and its mean intervals under-cover,
remain too wide, and miss the worst-slice threshold. This is evidence that
sampling noise is no longer the only bottleneck, not an expected-value or
counterfactual-value admission. Equilibrium, resolving, exploitation, official
control, deployment, submission, and strength remain unvalidated.
Phase 6AI then instruments the immediate native child behind every forced plan
using only public deltas. Its grouped model transfers root means but fails to
preserve the causal branch baseline's action ordering, so the neural
action-contrast and interval gates reject it. Phase 6AJ freezes a narrower
semiparametric hybrid before collecting 36 new `action_first` roots. The fresh
panel validates the action-sensitive point signal and a positive contrast
bootstrap, but interval width remains too large. This is a scoped predictive
admission only: it is neither an opponent counterfactual-value provider nor an
equilibrium blueprint, and every resolver, exploitation, official-control,
deployment, submission, and playing-strength gate remains closed.
Phase 6AK treats all prior Phase 6AI/6AJ targets as development data, freezes a
regularized residual root-mean model, freezes its conformal radius on 36 new
calibration roots, and evaluates once on another 36 roots. The fresh point MSE
beats Phase 6AJ and all frozen nonlearned baselines, while overall coverage and
width now pass. The preregistered worst-slice coverage still fails at
`0.666667`, so conditional expected means remain rejected. This is collection-
heldout evidence, not implementation-heldout evidence, and it opens no
counterfactual, equilibrium, resolver, exploitation, control, deployment,
submission, or strength gate.
Phase 6AL freezes that same mean without retraining and doubles the independent
replay budget to 32 returns per action on sequential 36-root calibration and
validation panels. Sampling precision passes and point MSE again beats every
frozen reference, but the preregistered action-contrast bootstrap, aggregate
coverage, and worst-slice coverage gates fail. It confirms that target noise
alone does not justify promotion. The official public-range adapter and
opponent-reach ledgers remain research inputs; no native counterfactual bound,
equilibrium, resolver, exploitation, official-control, deployment, submission,
or strength claim is enabled.
Phase 6AM uses those prior roots only as declared development evidence and
makes the counterfactual pivot directly. It combines the frozen Phase 6AD
public range with public-root, baseline-relative advantage intervals, then
evaluates once on 36 new roots. Range, coverage, slice, width, bootstrap,
reach, and holdout gates pass; maximum target SE `0.201688` misses the fixed
`0.20` ceiling. The bound matrix is rejected and cannot enter a resolver.
Genuine official matches, legal selections, snapshot restores, branches, and
descendants remain distinct from the abstract candidate range, five-plan game,
continuation blueprint, normalization, and intervals.
Phase 6A replay conversion, the Phase 6B
microgame/tabular resolver, the Phase 6C neural blueprint, and Phase 6D value
bounds plus Phase 6E all-or-blueprint resolving are validated only in their
declared exact-game scopes. Phase 6F certified exploitation is likewise a
synthetic exact-game result; a validated full-game equilibrium blueprint and
official resolver are still unimplemented. Their
tests do not establish official playing
strength. Four opponent decks are legality probes rather than expert
agents, the engine exposes no paired match seed, the exact deck is card-pool
blocked, human expert review is pending, and no new leaderboard result belongs
to this agent. The repository therefore reports `external_blocked`, an
untrained game-theoretic planner, and `superhuman_strength: not_established`.
