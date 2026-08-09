# Phase 6 official-integration and game-theoretic-pivot audit

Audit date: 2026-07-27  
Package version: 0.6.0  
Official benchmark: `official-cg-phase6-r1`  
Evidence fingerprint: `04c3aab4d467f7298a3f39dd769387235d4e2c964fd405d151c316a8ae34999f`  
Phase 6J evidence fingerprint: `f02f0ad7ca088acf42130cfd4631beea3c4db48e0b65079b1475ece54a1d56b4`  
Phase 6K evidence fingerprint: `ed9ce6c0b682b298e61006c19a4cfe81a08ca341160f15127f991a3d8b885236`  
Phase 6L evidence fingerprint: `1f1fd025939174b9ea9ad53b713791e7db63f5f5f2c6b8647632bc442324fa55`  
Phase 6M evidence fingerprint: `21d903ed43791ae24644924b4e32e8787f01338933f5af09a18a1bb2cf0bba5c`
Phase 6N evidence fingerprint: `c7f86852150488f15d7b8895ccfc7f87f859e6c1e719516e1eafdac9d0da9c0f`
Phase 6O evidence fingerprint: `3297ba3b78fae0541ca6125d43ced7c5f6e0f6552ff51135f72456050e2ae199`
Phase 6P evidence fingerprint: `0aa43252f3c50de925b60c02e99984331698848d0f443a5c48873b9580b06bc9`
Phase 6Q evidence fingerprint: `da885fe53aa1e21656528a69a628a5b27b590fc68d4dbfc9ae654f696fe004bd`
Phase 6R evidence fingerprint: `41051f5779163fb0e3630379508fd1cde741635147f5552b83132a8c8770d09c`
Phase 6S evidence fingerprint: `0acf5b4f2f246b4071ecfbf1f560acfc8a32534333807420f50cc7e72c805206`
Phase 6T evidence fingerprint: `b43545776432f09d33fe071731599c31818a27357d8bc65e801840cf8b795d37`
Phase 6U evidence fingerprint: `52fb90a7960dc73887fc92c0268e8cf3bac05a1322c3dcbaebe4a27e8791aa04`
Phase 6V evidence fingerprint: `d7d5af2fe343cd0ba73da8f601d81850ed80212a7f43a41b41eb99929710a900`
Phase 6W evidence fingerprint: `ab42c69fce00c0a5cc4b1da98b62160fd258c1b2cf5f34114ed22c9807be1b78`
Phase 6X evidence fingerprint: `48a14f5c70c2bfce446e01f58d38e1364867417ef1578f5abf40efcec637f67a`
Phase 6Y evidence fingerprint: `d57936de3a85e10b3704470e984a46c1461c9b38db3f0b5d31a9bfcf3d974f3c`
Phase 6Z evidence fingerprint: `762d6d1316c13486ac073664d9e3150139f65f2efc26f324f1f62810385892f0`
Phase 6AA evidence fingerprint: `9aebdacbea9530baab3753d07f13374d53389fed4607c9a4f136c8d97d8b4974`
Phase 6AB evidence fingerprint: `804eb749005aef537cb1cbfb96861369a76cc079e37c7896a2447cd0e6c97cdb`
Phase 6AC evidence fingerprint: `ddf1c9b79b2e0457405218aaada8cdf94f7daa6be6de370a3d0cedd8793d3702`
Phase 6AD evidence fingerprint: `021a3dda44cd198cf5df7d1fa52d48094f900255e1d88204c34d9c346f932dc9`
Phase 6AE evidence fingerprint: `06478e797fbc1c6b0621b49a6dae4754cf8facc7b264c2a5ff0121f7a4ee323f`
Phase 6AF evidence fingerprint: `45d5d9df545ebba82066dcdf1214493c7d9ddb0ea453612f2007480a6a719c35`
Phase 6AG evidence fingerprint: `2112d91fdb5fe6adfbaee72b86346846e87806818368f6054919fd3ecaf97673`
Phase 6AH evidence fingerprint: `414e85a4f0bf4112cac0e914194da587f7c300a5deb3a5714eaaa89c98004c1d`
Phase 6AI evidence fingerprint: `27b440477f099edf4b6a8e63be99785b707d23a5ae04aa30c4a2c7970d02961b`
Phase 6AJ evidence fingerprint: `9d78e4364d60ecfb032febbdae0ef90c0909ef1c5babc6971fdfc94bf8b4d359`
Phase 6AK evidence fingerprint: `edc94664e746f708ebbebf5805d9e4b3547ac3a64c58ba3e5c92db42b2bfe471`
Phase 6AL evidence fingerprint: `d896278588650ed424bf47870990a9ddcf7dd27594cbc1a1d759af5f84763663`
Phase 6AM evidence fingerprint: `7a0f1e28e31d017601ecd205e7af896ed40316a4dc2bc1855492b14cf68ae640`

## Outcome

The Phase 6 official integration baseline is implemented and runs against the
native competition engine. Its heuristic policy and sampled-rollout advisor are
not the final strategic approach. The game-theoretic pivot has executable
interfaces, leakage-audited official replay data, exact-game references, two
deterministic neural microgame blueprints, calibrated microgame value bounds,
an interruption-safe microgame resolver, and a separately certified synthetic
exploitation overlay. It now also has a one-decision official observation/action
bridge validated transparently in live shadow execution and a corrected
610-step logged information-set trajectory corpus with public/private separation
and causal supervision links. A deterministic logged-data neural bootstrap now
validates train-only normalization, representation learning, complete legal-
action scoring, auxiliary heads, and exact restoration. Its action objective is
heuristic behavior imitation, not the final strategy. A production-disabled
Phase 6J boundary now branches and traverses complete games through the licensed
native search API from explicit hidden-zone completions. The descendant
transitions are genuine engine execution; the completion prior and public
belief are supplied research abstractions. Phase 6K now initializes from Phase
6I and trains a small full-game sampled-regret candidate on terminal-valued
Phase 6J branches. Its held-out diagnostics are mixed, and it is not reach-
corrected CFR or equilibrium-validated. Phase 6L now implements and exactly
audits a reach-corrected external-sampling MCCFR estimator on the non-official
microgame. Phase 6M wires the traversal and separate reservoirs to genuine
native descendants at five immediate-terminal roots, where all recorded
advantages are zero. Phase 6N adds an exactly audited one-trajectory
outcome-sampling estimator, and Phase 6O uses it in a five-plan turn abstraction
to collect nonzero both-player native samples and train separate player-regret
and average-plan networks. Every descendant transition in Phase 6O executes
through native search, while the plan actions, fixed tactical executor, hidden
prior, samples, transformed heavy-tailed objective, and model remain research
abstractions. Phase 6P closes the iterative loop but fails its variance gate,
with raw-regret p95 `5.04e11`; it is not admitted to counterfactual calibration.
Phase 6Q records that rejection and enables no downstream capability.
Phase 6R validates an exact cutoff boundary and controls the live sampled
horizon, reducing the observed p95 to `17.90`, but its confidence-zero Phase 6K
leaf remains uncalibrated and therefore does not reopen the admission gate.
Phase 6S confirms why: its held-out empirical interval spans the full utility
range and the supplied public range is indistinguishable from uniform.
Phase 6T resamples allocation between hidden zones and evaluates the public-only
Phase 6I range head on native histories. On 72 held-out rows the neural range
is worse than uniform and far worse than the causal revealed-card baseline, so
belief calibration and every downstream capability remain false.
Phase 6U reduces the maximum leaf-target standard error below threshold with
ten continuations, but calibration selects a zero baseline and every held-out
interval remains full-range. Leaf model bias and counterfactual calibration
remain unresolved.
Phase 6V trains a public-only ridge leaf with training-only candidate-held-out
selection and evaluates it on a separate native allocation family. Its fresh
point RMSE beats all three references, but the calibrated interval is still
nearly full-range, so the empirical gate and every safety capability remain
false.
Phase 6W separates three complete native collections for training, calibration,
and validation. Its untouched validation collection passes coverage but the
public leaf loses to zero and retains an over-wide interval, confirming that
Phase 6V's six-row point improvement does not generalize.
Phase 6X replaces the pooled scalar target with five opponent plan-action
returns under each supplied hidden-allocation hypothesis. The action substrate
has nontrivial held-out signal at bounded rollout error, but it is conditional
on sampled reach rather than opponent-reach-weighted; counterfactual and safety
gates remain false.
Phase 6Y validates the target-player-excluding reach ratio on all 24 exact
microgame information-set/action rows at zero error, then binds redacted
root-relative reach ledgers to a fresh native collection. Its 600 terminal
continuations execute 70,741 genuine transitions. All native roots are already
the first opponent plan decision, all weights are `1.0`, and all 12 public
information sets are unique; reach semantics pass while native value and safety
gates remain false.
Phase 6Z fixes that collection shape with three distinct hidden allocations
behind each of 18 later public roots. Whole six-root collections are reserved
for training, conformal calibration, and untouched validation. The aligned
substrate gates pass, but the hypothesis-conditioned ridge is rejected:
validation coverage is `75.555556%` against `90%`, and RMSE `0.847985` loses to
the zero baseline's `0.771402`. The acting-player hand is fixed within each
exact observation, the prior is supplied, and upstream official reach remains
absent, so counterfactual and safety gates stay false.
Phase 6AA maps those frozen public roots into the existing `OpponentRange`
contract without candidate truth or private features at prediction time. Its
train-only prototypes and calibration-selected smoothing reach validation NLL
`0.800538` versus uniform `1.791759` and top-1 `5/6`. That narrow archetype-
signal gate passes, but ECE `0.476073`, six roots per holdout, absent hidden-
resource/intent scope, and an unvalidated observer orientation keep official
public-range and all downstream gates false.
Phase 6AB independently captures complete declared player-action prefixes for
18 fresh official roots. All 512 selections execute through `battle_select` and
their redacted public-state/history chains validate. Deterministic behavior and
sampling factors are exactly one, but the live adapter exposes no automatic
shuffle/draw/Prize/coin likelihoods. Official chance reach, full reach,
counterfactual weights, Phase 6Z value-root alignment, and every downstream
gate remain unavailable.
Phase 6AC recollects all three data components at the same 18 fresh official
roots: 568 redacted prefix selections, one transient realized truth sample per
root, and all five plan values over 270 terminal continuations. Native
reinitialization matches the complete live root and the public/private state
after a forced `END`/next-draw transition. The shared opaque official chance
law cancels from the on-policy likelihood ratio, but absolute chance reach is
still unknown. With one truth sample per information set and maximum rollout
standard error `0.666667`, calibrated counterfactual values and every downstream
gate remain false.
Phase 6AD scales the observer-oriented truth path to 180 fresh official roots,
5,178 genuine prefix selections, and 34,560 public-consistent allocations.
Train, calibration, and untouched validation each contain 60 roots. The
preregistered six-candidate and 96-resource hidden hand/Prize/deck gates pass.
This is a narrow range result: the exact-list catalog, player-0 observer adapter,
prototype model, allocation prior, hash buckets, and posterior interpretation
remain abstractions. Opponent intent, full official belief, absolute chance
reach, counterfactual values, safe resolving, exploitation, and official control
remain false.
Phase 6AE removes the same-policy seat confound with 324 fresh official roots,
8,474 genuine selections, 875 public player-1 MAIN actions, and 62,208
candidate-conditioned allocations. Player 0 is fixed while only player 1 varies
over six controlled proxy policies. The 108-root untouched validation split
passes declared-proxy intent, joint candidate-intent, and hidden-resource gates.
This validates controlled policy identification, not human latent intent. Full
belief, chance reach, values, bounds, resolving, exploitation, and official
control remain false.
Phase 6AF removes exact implementation identity as a shortcut with 432 fresh
official roots, 11,105 selections, 1,188 public player-1 MAIN actions, and
82,944 candidate-conditioned allocations. Two authored policies train, a third
calibrates, and a fourth remains untouched validation. Aggregate and all six
intent-family transfer gates pass. This is controlled implementation transfer,
not human latent intent; full belief, values, bounds, resolving, exploitation,
and control remain false.
Phase 6AG completes 288 official games and 30,668 genuine legal selections
under four split-isolated policy implementations. The untouched implementation
point MSE `0.938795` beats zero `1.000000` and the causal board heuristic
`1.019981`. The terminal-outcome intervals fail coverage and width gates, so
expected-value intervals, counterfactual values, safe resolving, exploitation,
and control remain false.
Phase 6AH completes 180 additional official games, retains and exactly replays
one late player-0 root per game, and executes 14,400 native terminal
continuations over 360 supplied hidden allocations and 900 five-plan targets.
Maximum target SE `0.258199` passes, while validation action-contrast MSE
`0.028905` loses to zero-contrast `0.026840`. Coverage `0.872222`, mean width
`1.826374`, and minimum slice coverage `0.733333` also fail. Repeated sampling
therefore resolves the declared target-precision prerequisite but admits no
conditional expected-value model, counterfactual value, safe resolver,
exploitation, or control.
Phase 6AI completes another 180-root panel with 1,800 immediate public native
branch observations and 14,400 terminal continuations. Its grouped neural
model passes aggregate point MSE but fails relative action contrast, the root
bootstrap, and all interval gates. The causal branch heuristic is retained as
a diagnostic only; the learned advantage is rejected.
Phase 6AJ freezes a semiparametric model and bounds before collecting 36 new
`action_first` roots. Its point MSE `0.582840` beats zero `0.764236`, root
heuristic `0.891487`, and standalone branch heuristic `0.853716`. Contrast MSE
`0.022868` beats zero `0.025278` with bootstrap lower bound `0.000753`, so the
narrow action-sensitive point signal is validated. Coverage `0.905556` and
worst-slice coverage `0.800000` pass, but mean width `1.839791` fails the
`1.75` ceiling. Conditional expected means and every counterfactual, safety,
exploitation, and control gate remain false.
Phase 6AK expands the root mean over all 216 declared Phase 6AI/6AJ
development roots, freezes that model before a 36-root calibration collection,
and freezes its conformal radius before a disjoint 36-root validation
collection. It records 720 immediate public branches and 5,760 terminal native
continuations across the two fresh panels. Validation point MSE `0.398773`
beats the frozen Phase 6AJ hybrid `0.503581`, zero `0.683333`, root heuristic
`0.661293`, and branch heuristic `0.663245`; contrast and bootstrap gates pass.
Overall coverage `0.916667` and width `1.676068` pass, but worst-slice coverage
`0.666667` fails the fixed `0.75` floor. The scoped point diagnostic is
validated; conditional expected means and every counterfactual, safety,
exploitation, and control gate remain false.
Phase 6AL freezes that model unchanged and doubles replay precision to 32
returns per action. Its 36 calibration roots are complete and serialized before
36 disjoint validation roots begin. Across the two panels it records 72
official matches, 1,440 immediate branches, 360 targets, and 11,520 terminal
native continuations. Maximum target standard error `0.179605` passes. Held-out
point MSE `0.430678` beats zero `0.664497`, root heuristic `0.661890`, branch
heuristic `0.630282`, and Phase 6AJ `0.615516`, but the contrast-bootstrap lower
bound is `-0.000512`, coverage `0.888889` misses `0.90`, and worst-slice
coverage remains `0.666667`. The preregistered protocol rejects point and
conditional-mean admission; every downstream capability remains false.
Phase 6AM stops extending the absolute-mean sequence. It freezes a public-root,
blueprint-relative branch center and `0.555019619275336` conformal advantage
radius from the entire Phase 6AI-6AL development ledger, then joins it to the
frozen Phase 6AD public range on 36 fresh official roots. The panel records
3,779 live selections, 720 immediate branches, 5,760 terminal continuations,
and 274,068 native transitions. Coverage `0.95`, worst-slice coverage
`0.833333`, width `1.110039`, the coverage bootstrap, public-range transfer,
reach semantics, and collection holdout all pass. Maximum target standard
error `0.201688` exceeds the fixed `0.20` ceiling, so normalized
counterfactual-advantage bounds are rejected and every downstream capability
remains false.
There is still no validated full-game
equilibrium blueprint, calibrated full-game counterfactual values, official
safe resolver, or official exploitation overlay. The end-to-end external
objective is not complete:
the immutable exact deck contains Prism Tower, which the current Kaggle engine
does not implement, no substituted deck was submitted without approval, human
expert review is pending, and superhuman strength is not established.

The repository reports this as `external_blocked`, not `ok`.

## Verification summary

- 219 tests pass, including corrected public replay alignment,
  logged trajectories, exact game, CFR, neural/value calibration, interruption
  safety, and Phase 6A-6AM tests.
- All 979 public package exports resolve with no duplicate names.
- A fresh PEP 517 wheel builds and installs successfully; the installed
  `rocketbot-phase6-public-range-counterfactual-advantages` entry point resolves
  to the checked module, and the installed package exposes all 979 exports. The
  712,887-byte
  wheel has SHA-256
  `aea80d04c72b1fc01fb5843360045e18973499d32ee77fd20daa43e956960196`.
- Re-evaluating Phase 6AM from the frozen saved panel changes zero of its four
  generated artifact hashes.
- The refreshed benchmark completed 144 native-engine matches with zero illegal
  selections, fallbacks, or soft timeouts.
- Two independent baseline archive builds produced the same 1,990,071-byte file
  and SHA-256
  `e491f86afe6a63974d430382070a3baa5921d939b231c27e1a5a88fb122e02c7`.
- After extraction, that archive completed a native-engine smoke match with 56
  target decisions, zero illegal selections, zero fallbacks, and no engine
  error. The result is packaging/integration evidence, not a strength claim.
- Phase 6A-6B schema-2 evidence is byte-identical across repeated runs. It
  contains 12 perspective-isolated episodes / 1,232 frames / 610 correctly
  aligned behavior labels, zero unknown event types or
  structural leakage violations, an analytic `-1/18` microgame value, CFR
  NashConv `0.00321571`, and an exact local resolve with zero measured opponent-
  bound violation.
- Phase 6C evidence and both neural checkpoints are byte-identical across two
  independent generations. The selected policy checkpoint restores with zero
  output drift and has exact microgame NashConv `0.00004637`; this is abstract-
  game evidence, not official-engine performance.
- Phase 6D evidence is byte-identical across two independent generations. Its
  12 calibration-heldout value targets have 100% coverage; six exact resolve
  roots have zero false-safe certificates and all high-risk gates fall back.
- Phase 6E evidence is byte-identical across two runs. All 12 roots complete a
  fixed exact grid with strict non-regression, five endgame roots change for a
  small positive conditional gain, and four interruption/gate probes return the
  blueprint exactly.
- Phase 6F evidence and both opponent-model artifacts are byte-identical across
  two runs. Six roots enable bounded exploitation; all 12 final strategies have
  exact robust certificates, including adversarial model slices.
- Phase 6G ran 12 live official-engine shadow matches and audited 566 target
  decisions. All 566 ordered selections round-trip exactly; 557/557 applicable
  own-private sensitivity checks pass; and there are zero bridge errors,
  unknown option fields, illegal actions, fallbacks, soft timeouts, or action
  changes. The native engine is unseeded, so this bundle is not claimed to be
  byte-identical across runs.
- Phase 6H derives 610 logged information-set steps from the corrected official
  replays. All behavior labels satisfy their source contracts; 616 paired
  common-public frames have zero state/history mismatches; all transitions and
  12 terminal outcomes are linked; and JSON restoration is exact. The corpus is
  explicitly offline, abstract, and non-branchable.
- Phase 6I trains a 10-epoch multi-task bootstrap on the match-disjoint 385/86/
  139 train/calibration/validation split. All task heads receive gradients,
  training loss falls from 1.93158436 to 1.13767178, and the serialized model
  restores with zero prediction drift and exact legal support across all 610
  steps. These are small-development-corpus wiring diagnostics, not strength or
  equilibrium evidence.
- Phase 6J starts six simultaneous hidden-completion roots from the same native
  decision, proves identical policy-facing root information across the selected
  truths, rejects tampered actions/chance, verifies distinct alternative and
  deterministic repeated children at root and midgame, and reaches six terminal
  games over 862 genuine native transitions. Raw hidden card IDs are absent from
  persisted manifests. The live unseeded bundle is not claimed byte-identical,
  and these traversal checks are not equilibrium or strength evidence.
- Phase 6K collects 24 information sets across both players and all six
  completion candidates, evaluating legal actions through 15,452 genuine native
  branch transitions. The Phase 6I-initialized regret candidate lowers train
  loss from 3.64769418 to 1.68179459, preserves exact legal support, and restores
  with zero drift. Its 19/2/3 information-set-disjoint split has mixed held-out
  deviation diagnostics; one rollout per action and absent reach correction or
  best responses preclude Deep CFR, exploitability, equilibrium, or strength
  claims. The live unseeded bundle is not claimed byte-identical.
- Phase 6L exhaustively audits 768 player/traverser expectation blocks on the
  exact game with maximum absolute estimator error `9.99e-16`. Its 20,000-
  iteration tabular run completes 40,000 traversals / 210,852 transitions and
  reaches NashConv `0.00461750` and exploitability `0.00230875`. This is
  deterministic abstract-game evidence only.
- Phase 6M admits five of 12 balanced/weakened native endgame probes across both
  players, executes 45 external-sampling transitions (9,875 native transitions
  including spine construction and admission checks), and serializes five
  advantage samples with no raw private-zone keys. All advantages are zero and
  no opponent node is encountered, so no strategy sample or useful neural
  training evidence is produced. Larger traversals exceeded budgets and were
  discarded without persisting partial samples.
- Phase 6N exhaustively enumerates all 60 outcome-sampling terminal paths under
  an arbitrary nonzero pre-action baseline and matches exact immediate
  counterfactual regrets to `4.16e-17`. Its deterministic 20,000-iteration
  abstract run completes 40,000 traversals / 133,624 transitions and reaches
  NashConv `0.03111612`. It is an estimator reference, not native evidence.
- Phase 6O completes 36 both-player trajectories across all six supplied
  hidden completions, executing 5,569 genuine native descendant transitions
  and retaining 238/214 regret samples plus 452 separate average-strategy
  samples. Three networks receive gradients, training loss falls from
  6.64556058 to 4.86231795, and restoration drift is zero. Raw regret p95 is
  `1.021e7` and maximum is `5.440e12`; the train-only `asinh` target changes
  magnitude and precludes an equilibrium/convergence claim.
- Phase 6P closes the collect–fit–update loop across three rounds and 5,562
  genuine native descendant transitions. Each round adds nonzero both-player
  samples, updates all three policies, and lowers its transformed training
  objective. The final raw-regret p95 is `5.04e11` and maximum `1.24e16`, so
  the candidate fails counterfactual-calibration and safe-resolver admission.
- Phase 6Q binds that rejection to the exact Phase 6P dataset/model/evidence
  fingerprints. Three provenance/redaction/restoration checks pass and 13
  data/variance/belief/held-out/value/equilibrium checks fail. It builds no
  bounds and runs no resolver, exploitation, or official strategy control.
- Phase 6R exactly audits the cutoff wrapper across 32 enumerated paths and 18
  pre-cutoff regret entries with maximum error `5.55e-17`. Its live six-step
  probe executes 1,474 genuine native tactical descendants across 36 abstract
  cutoff trajectories, retaining 54/54 regret and 108 strategy records. Raw-
  regret p95 / maximum is `17.90` / `134.92`, but every live leaf has
  confidence zero and the full `[-1, 1]` interval. No calibrated value, bound,
  resolver, exploitation, official control, or performance claim follows.
- Phase 6S covers all six candidates with 12 calibration and six disjoint
  validation order particles, then executes 54 terminal continuations over
  6,240 native transitions. Validation coverage is 100%, but mean interval
  width is the full `2.0`, informative fraction is zero, and maximum rollout
  standard error is `0.667`. Static-range NLL equals uniform NLL `1.791759`.
  Belief, leaf, counterfactual, and safety gates remain false.
- Phase 6T covers all six candidates with 24 calibration and 12 held-out
  allocation particles across two native roots. It records 216 public decision
  rows and 274 genuine native descendant transitions without raw private IDs or
  truth inputs. On 72 validation rows, neural / causal / uniform NLL is
  `1.810882` / `0.895880` / `1.791759`; neural ECE is `0.320267`. The neural
  allocation gate, public-belief calibration, and all safety gates remain
  false.
- Phase 6U covers all six candidates with 18 calibration and six held-out
  allocation particles, executing 240 terminal continuations over 29,690
  genuine native transitions. Maximum rollout standard error is `0.333333`,
  but zero beats the Phase 6I/6K learned heads on calibration. Validation MAE
  is `0.366667`; mean interval width is `2.0` and informative fraction is zero.
  Empirical, counterfactual, and safety gates remain false.
- Phase 6V trains on 12 declared Phase 6U rows, chooses `lambda=1000` without
  consulting validation data, and evaluates a new 24-particle/240-continuation
  native dataset containing 28,324 genuine transitions. On six untouched rows,
  public-ridge RMSE is `0.446013` versus the best reference's `0.489898`, but the
  conformal radius is `1.170885` and mean interval width is `1.997117`.
  Empirical, counterfactual, belief, resolver, and control gates remain false.
- Phase 6W uses 24 / 24 / 24 rows for collection-isolated training,
  calibration, and validation, with zero particle/state/history overlap. The
  fresh collection executes 240 continuations over 27,906 genuine transitions.
  Coverage is `91.667%`, but public-ridge RMSE is `0.595137` versus zero's
  `0.493288`, and mean interval width is `1.721623`. Every strategic gate
  remains false.
- Phase 6X covers all six opponent hypotheses in both allocation variants and
  branches five plans into 60 action targets, 600 terminal continuations, and
  73,538 genuine native transitions. Validation nontrivial-surface fraction is
  `100%`, mean action span `0.633333`, mean empirical deviation gain `0.286358`,
  and maximum SE `0.333333`. The conditional substrate gate passes while
  counterfactual, belief, resolver, exploitation, and control gates stay false.
- Phase 6Y exhaustively matches 24 exact reach/value rows with zero error, then
  records complete redacted ledgers for 12 native surfaces, 60 actions, 600
  terminal continuations, and 70,741 genuine native transitions. Native weights
  are all `1.0`, but maximum histories per public information set is one. Exact
  semantics and ledger gates pass; value, resolver, exploitation, and control
  gates remain false.
- Phase 6Z isolates six public roots in each of train, calibration, and
  validation. Three zero-substitution allocations per root produce 54 surfaces,
  270 plan-action targets, 810 terminal continuations, and 70,931 genuine native
  transitions. Root alignment/disjointness pass, while validation coverage
  `75.555556%` and RMSE `0.847985` versus zero `0.771402` reject the conditional
  ridge. Counterfactual, resolver, exploitation, and control gates remain false.
- Phase 6AA preserves those 6/6/6 root splits and uses public features only.
  Validation NLL `0.800538` and Brier `0.378643` beat uniform `1.791759` /
  `0.833333`, with top-1 `5/6`; archetype signal passes. ECE `0.476073`, both
  60-root sufficiency gates, hidden-resource scope, official range,
  counterfactual, resolver, exploitation, and control gates remain false.
- Phase 6AB captures 18 fresh official behavior prefixes ending at turn 4.
  Their 512 genuine selections comprise 358 player-0 and 154 player-1
  decisions, with 21 to 39 steps per trace and exact public-chain continuity.
  Behavior-action and declared deterministic player-reach gates pass. Official
  chance-probability coverage is `0.0`, so chance/full reach, counterfactual
  weights, Phase 6Z alignment, resolving, exploitation, and control remain
  unavailable.
- Phase 6AC captures 18 new same-root prefix/truth/value records. The 568
  prefix selections plus 18 forced replay-fidelity selections are genuine;
  90 five-plan targets produce 270 terminal continuations and 29,460 native
  descendants. Root/next-draw parity and common-chance cancellation pass.
  Absolute chance reach, repeated truth samples, calibrated values, resolving,
  exploitation, and control remain unavailable.
- Phase 6AD captures 180 new observer-oriented official truth roots and 5,178
  genuine prefix selections. The disjoint split is 60/60/60, and 32 allocations
  per candidate/root yield 34,560 particles. Validation candidate NLL is
  approximately `1.40e-8` versus uniform `1.791759`; resource presence Brier is
  `0.038424` versus uniform-candidate `0.100017`, with ECE `0.007916`. The
  in-scope hidden-card range passes, while intent, full belief, values, bounds,
  resolving, exploitation, and control remain false.
- Phase 6AE captures 324 new seat-decoupled official roots, 8,474 genuine
  selections, and 875 public player-1 MAIN actions. The split is 108/108/108,
  and 32 allocations per candidate/root yield 62,208 particles. Validation
  intent top-1 is `0.879630`, intent ECE is `0.042259`, joint ECE is `0.042195`,
  and resource ECE is `0.011843`. The controlled declared-policy proxy range
  passes; human latent intent, full belief, values, bounds, resolving,
  exploitation, and control remain false.
- Phase 6AF captures 432 new implementation-heldout official roots and 11,105
  genuine selections. The split is 144/144/144, with four policy
  implementations and 82,944 candidate-conditioned particles. Validation
  intent top-1 is `0.826389`, intent ECE is `0.090055`, joint ECE is `0.091196`,
  and every intent slice is at least `0.583333`. Controlled transfer passes;
  human latent intent, full belief, values, bounds, resolving, exploitation,
  and control remain false.
- Phase 6AG captures 288 complete official games, 576 perspective-isolated
  episodes, and 30,668 genuine selections. Train/calibration/validation contain
  144/72/72 matches, and every candidate/intent calibration stratum is balanced
  between model selection and conformal bounds. Untouched episode MSE
  `0.938795` passes both point baselines. Coverage `0.862791`, mean width
  `1.937685`, and endgame width `1.753315` reject interval admission; values,
  bounds, resolving, exploitation, and control remain false.
- Phase 6AH captures 180 complete official games and 20,388 live selections,
  then exactly replays 180 retained snapshots into 14,400 terminal native
  continuations and 605,351 native transitions. The 72/72/36 root split is
  implementation-isolated. Maximum target SE `0.258199` passes, but validation
  action-contrast MSE `0.028905` loses to `0.026840`, coverage is `0.872222`,
  width is `1.826374`, and the worst slice is `0.733333`; every downstream gate
  remains false.
- Phase 6AI captures 180 new roots, 19,270 live selections, 1,800 immediate
  branches, 14,400 terminal continuations, and 614,665 native transitions. The
  grouped model's validation point MSE `0.563726` beats its references, but
  contrast `0.038066` loses to zero `0.037569` and the standalone branch
  heuristic `0.035565`; bootstrap and interval gates reject promotion.
- Phase 6AJ freezes its model/bounds before opening 36 new roots, then records
  3,840 live selections, 360 immediate branches, 2,880 continuations, and
  117,341 native transitions. All point/contrast/bootstrap/precision/holdout
  gates pass. Mean interval width `1.839791` fails, so only the scoped point
  signal is admitted and all downstream capabilities remain false.
- Phase 6AK declares the 180 Phase 6AI and 36 Phase 6AJ roots as development,
  selects a residual ridge root mean across five complete folds, and serializes
  it before fresh collection. The 36-root calibration panel is serialized with
  radius `1.123981` before a distinct 36-root validation panel begins. Together
  they record 7,821 live selections, 720 immediate branches, 5,760 terminal
  continuations, and 254,479 native transitions. All point, contrast,
  bootstrap, precision, and collection-ledger gates pass. Aggregate coverage
  and width pass; worst-slice coverage `0.666667` rejects interval admission,
  so every downstream capability remains false.
- Phase 6AL declares all 288 Phase 6AI/6AJ/6AK roots as development and freezes
  the exact Phase 6AK mean before collection. Its 36 calibration roots use two
  replay batches and are serialized with radius `1.015211` before any of the 36
  validation roots begin. Together they record 7,541 live selections, 1,440
  immediate branches, 11,520 terminal continuations, and 538,446 native
  transitions. Sampling precision and point-reference comparisons pass, but
  the contrast bootstrap, aggregate coverage, and worst-slice gates reject both
  point and conditional-mean promotion. All downstream capabilities remain
  false.
- Phase 6AM declares all 360 Phase 6AI-6AL roots as development and freezes its
  counterfactual-advantage radius before the first of 36 new roots. Those roots
  contribute 720 branches and 5,760 continuations. Seven of eight aggregate
  admission gates pass; maximum target SE `0.201688` misses `0.20`, so the
  serialized 6-hypothesis x 5-action bound interface remains inspection-only
  and all downstream capabilities remain false.

## Requirement matrix

| Requirement | Evidence | Status |
|---|---|---|
| Reproducible integration baseline | Flat standard-library policy, official `agent(obs_dict)` entry point, deterministic baseline archive builder | Implemented |
| Legal action safety | 7,272 target decisions; 0 illegal actions; 0 runner or policy fallbacks | Validated |
| Timing/failsafe engineering | 0 soft timeouts; 0.684 ms mean match-level decision latency; 33.766 ms maximum | Validated locally |
| Tactical evaluation | Sequencing action counts, attack timing, retreat/evolution/attachment counts, 100% static-lethal conversion | Implemented |
| Strategic evaluation | Six values/weights, Prize margin, board, tempo, hand resources, optionality | Implemented |
| Belief evaluation | Archetype posterior against known panel labels; mean NLL 0.4609, Brier 0.1908 | Implemented at archetype level |
| Search-value evaluation | 191 invocations; 14 changed selections; advice bonus and ablation slices | Implemented |
| Robustness | Six archetypes, three adversarial styles, both seats, four ablations | Implemented with stated panel limits |
| Replay review | Six official replays, automated triage, stable expert rubric | Triage complete; human review pending |
| Competition packaging | Deterministic `.tar.gz`, manifest/hashes, all platform libraries, local archive smoke path | Implemented for explicit compatibility profile |
| Public-belief and opponent range | Serializable hypotheses plus six truth-replay conversions into 12 isolated perspectives; 923 event identities redacted | Phase 6A foundation validated |
| Abstract information-set boundary | Corrected opaque-state extensive-form protocol plus a 55-state/12-information-set exact non-official game | Phase 6B foundation validated |
| Exact equilibrium/CFR reference | Analytic NashConv effectively zero; deterministic 20,000-step CFR NashConv 0.00321571 | Phase 6B validated |
| Neural blueprint provider | Two exact-game candidates plus a Phase 6I-initialized full-game sampled-regret candidate with zero-drift restoration | Phase 6C exact-game gate and Phase 6K wiring validated; no full-game convergence/safety certificate |
| Counterfactual-value calibration | 12 calibration and 12 validation targets; 100% held-out coverage; componentwise uncertainty widening and fail-closed gates | Phase 6D exact-game gate validated |
| Safe subgame/endgame re-solving | Fixed-budget 102-mixture exact grid, strict per-hypothesis non-regression, iteration/node/deadline identity fallback | Phase 6E microgame validated; official full-game solver not implemented |
| Baseline-constrained exploitation | Separate synthetic models plus all-64-pure-continuation final certificates; max exact loss 0.00999815 under 0.01 tolerance | Phase 6F exact-game gate validated; official overlay disabled |
| Official information-set bridge | Allow-listed public/private split plus complete ordered action lifting across 566 live shadow decisions | One-step shadow boundary validated; no planner deployment |
| Logged full-game training substrate | 610 complete information-set/action records, observed continuations, returns, match-disjoint splits, and causal later reveals | Phase 6H offline corpus validated; not a branchable game or trained model |
| Logged neural representation bootstrap | Train-only normalizers, separate public/private encoders, complete legal-action imitation, range/outcome/next-public heads, exact checkpoint restore | Phase 6I offline bootstrap validated; heuristic teacher, not equilibrium blueprint |
| Branchable native-search game boundary | Six simultaneous completion roots, complete ordered actions, explicit completion/coin chance, repeatable alternative children, six terminal traversals, redacted manifests | Phase 6J boundary validated; native descendants genuine, hidden prior abstract, no equilibrium strategy or deployment |
| Full-game equilibrium-oriented training | 24 both-player/six-candidate information sets, 15,452 terminal native branch transitions, separate average-policy/advantage/value heads, exact-support restore | Phase 6K sampled-regret candidate wiring validated; not reach-corrected CFR, equilibrium convergence, deployable blueprint, or strength evidence |
| Reach-corrected MCCFR estimator | Exhaustive conditional-advantage equality over 768 exact-game blocks, bounded fail-closed traversal, separate strategy/advantage reservoirs, tabular convergence reference | Phase 6L validated on the exact non-official microgame; no full-game neural convergence claim |
| Native external-sampling integration | Five immediate-terminal roots, both players, exact semantic feature widths, redacted reservoir round trip, 45 genuine native descendant transitions | Phase 6M narrow wiring validated; all advantages zero, no opponent nodes, no training/equilibrium/strength evidence |
| Outcome-sampling estimator | All 60 exact sampled paths, arbitrary nonzero pre-action baseline, separate regret/strategy reservoirs, fail-closed traversal, tabular convergence | Phase 6N validated on the exact non-official microgame; no native convergence claim |
| Native turn-plan equilibrium candidate | Five persistent plans, fixed legal executor, both-player outcome memories, three split-safe/restorable networks, 5,569 native descendants | Phase 6O nontrivial wiring validated; heavy-tailed transformed targets, uncalibrated prior, and no exploitability/safety/deployment evidence |
| Iterative native plan learning | Three previous-model-driven rounds, cumulative separate memories, per-round policy drift, exact restore | Phase 6P loop validated but variance gate failed; no counterfactual calibration, safe resolving, deployment, or strength evidence |
| Native safety admission | Fingerprint binding, 16 necessary prerequisites, deterministic rejection, capability-bypass rejection | Phase 6Q correctly rejects the candidate; no bounds/resolver/exploitation/official control executed |
| Depth-limited native variance control | Exact pre-cutoff expectation audit; six-step live cutoff boundary; bounded leaf/baseline; redacted separate reservoirs | Phase 6R controls the observed importance tail, but the leaf remains uncalibrated and safety admission remains rejected |
| Held-out native leaf/range calibration | Disjoint order particles across all candidates; repeated terminal continuations; residual and uniform-range diagnostics | Phase 6S rejects the current leaf as vacuous and the static range as uninformative; no counterfactual or safety capability enabled |
| Hidden-allocation/native range calibration | Public-consistent allocation resampling; 216 public-only rows; neural/causal/uniform held-out comparison | Phase 6T rejects direct neural transfer; causal revealed-card range is the next baseline, not a calibrated official posterior |
| Low-noise allocation-heldout leaf calibration | Ten terminal continuations per leaf; calibration-only three-model selection; disjoint validation allocations | Phase 6U passes target precision but rejects both learned heads and retains a vacuous full-width interval; no counterfactual or safety capability |
| Purpose-trained public native leaf | Training-only leave-one-candidate-out ridge selection; separate 24-particle native holdout; six calibration and six validation rows | Phase 6V improves fresh point RMSE but rejects the nearly full-width interval; no counterfactual, belief, resolver, exploitation, or control capability |
| Collection-heldout public native leaf | Three complete identity-disjoint 24-particle collections; 24-row conformal calibration; untouched 24-row validation | Phase 6W rejects the public ridge against zero and on width; no counterfactual, belief, resolver, exploitation, or control capability |
| Opponent-hypothesis plan-deviation values | Five forced opponent plans per supplied allocation; ten paired continuations per action; all candidates in both splits | Phase 6X validates conditional action supervision, not opponent-reach-weighted counterfactual values; no bound, resolver, exploitation, or control capability |
| Counterfactual-reach semantics/native ledger | 24 exact information-set/action expectation checks; redacted root-relative reach for all 12 native surfaces and 60 action targets | Phase 6Y validates reach arithmetic and ledger coverage; unique public roots, supplied prior, and absent held-out calibration keep native values/bounds/resolving/control disabled |
| Public-root-aligned conditional action values | 18 disjoint later roots; three hidden allocations/root; full five-plan surfaces; train/calibration/validation root isolation | Phase 6Z validates the aligned substrate but rejects the ridge on coverage and zero-baseline RMSE; incomplete hand range, supplied prior, and absent upstream reach keep values/bounds/resolving/control disabled |
| Public-only deck-archetype range | Train-only candidate prototypes; calibration-only temperature/uniform shrinkage; untouched validation roots; normalized `OpponentRange` output | Phase 6AA validates narrow archetype signal but fails sample sufficiency/calibration/scope; no official hidden-state posterior, bounds, resolving, exploitation, or control capability |
| Official behavior-prefix reach | 18 fresh later roots; 512 genuine selections; contiguous redacted state/history chains; deterministic behavior factors | Phase 6AB validates player-action execution/reach only; zero exposed chance likelihoods and no Phase 6Z root join leave full/counterfactual reach, bounds, resolving, exploitation, and control unavailable |
| Joint official truth/value samples | 18 fresh same-root prefixes and realized truth samples; exact native root/next-draw replay parity; 90 five-plan targets and 270 terminal continuations | Phase 6AC validates the joined substrate and symbolic common-chance cancellation; one truth sample per information set, unknown absolute chance reach, and noisy abstract continuations keep values/bounds/resolving/exploitation/control unavailable |
| Observer-oriented official hidden-card range | 180 fresh roots; 60/60/60 splits; public-only player-0 features; 34,560 candidate-conditioned allocations; 96 hidden-resource buckets | Phase 6AD passes its six-list candidate/resource calibration gates only; intent, full official belief, absolute chance reach, values/bounds/resolving/exploitation/control remain unavailable |
| Seat-decoupled declared-policy intent range | 324 fresh roots; 108/108/108 splits; fixed player-0 policy; opponent-only coarse MAIN action inputs; 36 joint hypotheses; 62,208 allocations | Phase 6AE passes controlled-policy intent/joint/resource gates only; human latent intent, full belief, chance reach, values/bounds/resolving/exploitation/control remain unavailable |
| Policy-implementation-heldout intent families | 432 fresh roots; 144/144/144 splits; two train, one calibration, and one untouched validation implementation; leave-one-policy-out selection | Phase 6AF passes authored implementation transfer and per-intent gates only; human latent intent, full belief, values/bounds/resolving/exploitation/control remain unavailable |
| Implementation-heldout official full-game point leaf | 288 complete matches; 144/72/72 split; two training implementations, balanced selection/bounds halves of a third, untouched fourth; 208 public/private/range inputs | Phase 6AG point MSE passes zero and causal-heuristic baselines, but coverage/width reject interval admission; expected/counterfactual values, bounds, resolving, exploitation, and control remain unavailable |
| Repeated official root/action expected values | 180 complete matches; 72/72/36 roots; two supplied allocations/root; all five plans; eight paired continuations/allocation/action; 14,400 terminal paths | Phase 6AH passes target precision and improves aggregate MSE, but fails action-contrast, coverage, width, and worst-slice gates; conditional/counterfactual values, bounds, resolving, exploitation, and control remain unavailable |
| Causal branch-conditioned values | 180 fresh split-isolated roots; 1,800 immediate public branch observations; 900 action targets; 14,400 terminal paths | Phase 6AI improves point MSE but fails learned action contrast, bootstrap, and interval gates; no action-sensitive/conditional/counterfactual promotion |
| Semiparametric branch-value hybrid | Model/bounds frozen from Phase 6AI train/selection/bounds only; 36 fresh `action_first` roots; 360 branches; 180 action targets; 2,880 terminal paths | Phase 6AJ validates a scoped action-sensitive point signal, but interval width rejects conditional expected means; counterfactual values, bounds, resolving, exploitation, and control remain unavailable |
| Expanded semiparametric root mean | Five-fold ridge selection over all 216 declared development roots; model frozen before 36 calibration roots; radius frozen before 36 distinct validation roots; 720 fresh branches and 5,760 terminal paths | Phase 6AK validates a collection-heldout action-sensitive point signal and passes aggregate coverage/width, but worst-slice coverage rejects conditional expected means; it is not implementation-heldout, and counterfactual values, bounds, resolving, exploitation, and control remain unavailable |
| High-precision conditional-value confirmation | Frozen Phase 6AK mean; sequential 36/36 calibration/validation roots; two replay batches; 32 returns/action; 1,440 fresh branches and 11,520 terminal paths | Phase 6AL passes sampling precision and point-reference comparisons, but contrast-bootstrap, aggregate coverage, and worst-slice gates reject promotion; counterfactual values, bounds, resolving, exploitation, and control remain unavailable |
| Public-range normalized counterfactual advantages | All 360 prior value roots declared development; Phase 6AD public range; frozen `0.555019619275336` advantage radius; 36 fresh roots; two replay batches; 720 branches and 5,760 terminal paths | Phase 6AM passes coverage, slice, width, bootstrap, range, reach, and holdout gates, but maximum target SE `0.201688` fails the `0.20` ceiling; normalized bounds, absolute values, equilibrium, resolving, exploitation, and control remain unavailable |
| Exact deck in official engine | 58/60 cards resolve; Prism Tower CRI 80 absent | Externally blocked |
| Kaggle leaderboard movement | No unauthorized substituted-deck submission | Externally blocked |
| Superhuman strength | Local probes and unpaired ablations are insufficient; no expert or leaderboard validation | Not established |

## Official benchmark results

The refreshed 144-match panel scored 80.6% overall (95% Wilson interval
73.3%–86.2%).
This is evidence of functional strength against the specified local panel, not
a claim about the Kaggle population or expert human play.

| Ablation | Matches | Wins | Score rate | Mean Prize margin |
|---|---:|---:|---:|---:|
| Policy only | 36 | 27 | 75.0% | +0.81 |
| Strategic | 36 | 30 | 83.3% | +1.31 |
| Belief + strategic | 36 | 29 | 80.6% | +0.89 |
| Full with native search | 36 | 30 | 83.3% | +1.92 |

The full baseline observed one more win and a larger Prize margin than the
belief/strategic slice in this run, while native search changed 7.3% of searched
selections. The native API exposes no paired match seed, so this is not a causal
search-gain estimate and does not verify a strength improvement. Search is
retained as an integration baseline and diagnostic experiment, not credited
with unverified strength and not designated as the final planner.

| Opponent archetype | Matches | Score rate |
|---|---:|---:|
| Official sample Abomasnow | 24 | 66.7% |
| Alakazam legality probe | 24 | 91.7% |
| Crustle legality probe | 24 | 100.0% |
| Dragapult legality probe | 24 | 75.0% |
| Rocket compatibility mirror | 24 | 87.5% |
| Starmie legality probe | 24 | 62.5% |

Abomasnow and Starmie are the clear local failure clusters and dominate the
expert-review queue.

## Exact-deck incompatibility

The official NAIC page lists `2 Prism Tower CRI 80`. The official Kaggle CSV
contains 1,267 unique IDs and no Prism Tower; its current format is TEF–POR.
`resolve_official_deck()` therefore returns:

- 58 resolved IDs;
- unresolved: `Prism Tower`;
- exact playable: `false`.

The optional local compatibility resolution is:

- explicit substitution: `2 Prism Tower -> 2 Team Rocket's Watchtower`;
- 60 IDs;
- exact: `false`;
- resolution fingerprint:
  `af3bf9c3d2690c82f2326d2dfb900389ab769a58ee24255bab967e06ec7de315`.

The builder refuses to create that archive unless
`--allow-watchtower-compatibility` is supplied. Submission still requires the
owner's decision because it changes the requested deck.

## Claim boundary

| Claim | Status |
|---|---|
| Official engine executes the agent | Validated |
| Agent always returns legal selections in the benchmark | Validated |
| Exact NAIC deck works in the current Kaggle engine | Blocked |
| New Phase 6 agent moved the leaderboard | Blocked |
| Human expert endorsed replay quality | Pending |
| Phase 6A public replay foundation is validated | Yes; scoped leakage audit |
| Phase 6B exact game/tabular reference is validated | Yes; abstract microgame only |
| Neural game-theoretic blueprint is trained | Exact abstract microgame plus Phase 6K option-level and Phase 6O plan-level full-game candidates; no full-game equilibrium/convergence certificate or deployment |
| Neural counterfactual bounds are calibrated | Exact abstract microgame only; no replay/full-game calibration |
| Safe information-set re-solving is implemented | Exact microgame only; not official full game |
| Interrupted solving returns the blueprint | Validated for exact iteration/node/deadline/OOD probes |
| Opponent exploitation has a final-strategy certificate | Exact synthetic microgame only; official overlay disabled |
| Official engine is bridged to the abstract information-set game | One-step shadow bridge plus production-disabled Phase 6J branchable native-search boundary; hidden completion prior remains abstract and no strategy controls the agent |
| Replay action labels are causally aligned | 610/610 validate against their source prompt after schema-2 next-frame correction |
| Full-game logged trajectory corpus exists | Yes; abstract offline data only, with no counterfactual branches |
| Logged trajectory neural model exists | Yes; split-safe Phase 6I behavior/representation bootstrap only, not a blueprint provider |
| Branchable full-game transition boundary exists | Yes; six Phase 6J native-search completions reach terminal states under a supplied non-official hidden prior |
| Full-game regret-trained neural candidate exists | Yes; Phase 6K option-level wiring plus Phase 6O outcome-sampled plan-level networks, with no equilibrium/safety/deployment claim |
| Reach-corrected MCCFR estimator is validated | Yes; Phase 6L exhaustive exact-microgame expectation audit and tabular reference only |
| Reach-corrected native full-game training exists | Partial Phase 6O/6P turn-plan abstraction: nonzero iterative both-player samples and separate networks; Phase 6P variance gate failed, with no convergence or exploitability certificate |
| Same-root official prefix/truth/value collection exists | Yes; Phase 6AC joins 18 live roots and validates native root/next-draw parity, but has one truth sample per information set and no calibrated value or safety capability |
| Observer-oriented official hidden-card range exists | Yes, in scope; Phase 6AD passes held-out six-list candidate and hand/Prize/deck bucket gates, but intent/full belief/value/safety/control remain unavailable |
| Seat-decoupled declared-policy intent range exists | Yes, in scope; Phase 6AE passes six-controlled-policy classification and joint-range gates, but human latent intent/full belief/value/safety/control remain unavailable |
| Policy-implementation-heldout intent transfer exists | Yes, in scope; Phase 6AF passes across a fourth authored policy implementation, but human latent intent/full belief/value/safety/control remain unavailable |
| Implementation-heldout official full-game point leaf exists | Yes, offline only; Phase 6AG beats both point baselines on an untouched authored implementation, but its expected-value interval gate fails and it is not counterfactual, equilibrium, safe, or deployed |
| Repeated root/action expected-value substrate exists | Yes, offline only; Phase 6AH supplies 16 returns per root/action and passes target precision, but action-transfer and interval gates reject the model; it is not counterfactual, equilibrium, safe, or deployed |
| Agent is superhuman | Not established |

Native `cg` matches, legal options, returned selections, telemetry, replay
capture, and the Phase 6G shadow wrapper's observed decision inputs are genuine
official-engine execution. The Phase 6G public/private split and exact action
lifting are validated only at that one-decision boundary. Phase 6H records
historical transitions that genuinely occurred, but its derived information-set
corpus has abstract provenance and cannot branch. Full-game transitions are now
available through the production-disabled Phase 6J native-search boundary:
after a supplied hidden-zone completion chooses a root, each descendant is
genuinely executed by the licensed engine. The completion chance prior, static
public-belief representation, equilibrium blueprint strategies,
counterfactual bounds, safe re-solving, and opponent exploitation remain non-
official abstractions until separately trained and validated. Phase 6I consumes
the abstract corpus and imitates logged
heuristic behavior; its exact restoration does not make it official or game-
theoretically safe. Phase 6K consumes genuine terminal native branches but its
derived regret dataset and neural strategy remain abstract. The candidate is
trained, yet its one-rollout objective, missing reach correction, mixed held-out
diagnostics, and absent full-game best responses leave equilibrium convergence,
safe counterfactual values, deployment, and strength unvalidated. Phase 6L
corrects and exactly validates the external-sampling estimator only in the
abstract microgame. Phase 6M's admitted descendant transitions are genuine
native execution, while its supplied hidden completions, fixed particles,
behavior policy, features, information sets, and MCCFR samples remain research
abstractions. Its zero advantages are not a training or performance result.
Phase 6N validates the outcome-sampling estimator only in the exact abstract
game. Phase 6O's descendants genuinely execute native search, but its plan
actions, fixed executor, hidden prior, features, sampled regrets, transformed
training objective, and neural provider are abstract. The provider remains
outside official strategy control and refuses counterfactual-value access.
Phase 6P iterates that training loop but materially worsens the importance-
weighted tail. Its failed admission gate prevents Phase 6D/E's exact-game
calibration and safety claims from being transferred to this model.
Phase 6Q makes that rejection executable and fingerprint-bound. Passing its
engineering thresholds in a future run would permit only the next calibration
audit; it would not itself be a safety certificate.
Phase 6R controls the estimator horizon but deliberately uses an uncalibrated
leaf. Its smaller observed tail is not value correctness, bound coverage,
equilibrium convergence, or safety evidence.
Phase 6S independently confirms that the current empirical interval is
full-range and that within-zone order particles are not enough to establish
belief or counterfactual calibration.
Phase 6T broadens those particles to hidden-zone allocation. Its descendant
observations are genuine native execution, but the allocations, candidate
labels, causal updater, neural range, and comparative metrics are abstractions.
The neural range fails its held-out gate, and the stronger causal baseline is
not promoted to an official posterior or safety input.
Phase 6U's ten-rollout means are lower-noise fixed-profile empirical targets,
not opponent counterfactual values. Its precision threshold passes, but both
learned heads lose to zero and the conformal interval remains vacuous. This
cannot enable bounds, resolving, exploitation, or official strategy control.
Phase 6V's fresh dataset contains genuine official-native descendants and
terminal continuations. Its hidden completions, cutoff sampler, fixed policy
profile, public feature bridge, ridge scalar, and conformal interval are
abstractions. The scalar's fresh point improvement does not convert its
fixed-profile return into an opponent counterfactual value or safety bound.
Phase 6W adds another genuine official-native dataset but changes no boundary:
its collection split, public ridge, fixed rollout profile, interval, and gates
are abstractions. A root-family generalization failure cannot enable official
strategy control or be interpreted as an engine result about the model itself.
Phase 6X's 73,538 descendant transitions are genuine native execution. The
hidden completion, cutoff history, forced plan, continuation blueprint, paired
sampling protocol, and conditional return surface are abstractions. Because
opponent/chance reach is absent, these values cannot satisfy the full-game
counterfactual-value provider or safety-bound contract.
Phase 6Y's 70,741 descendant transitions and terminal outcomes are likewise
genuine native execution. Its exact microgame audit, hidden prior, five-plan
game, root-relative reach ratio, and admission decision are abstractions. The
empty native prefix explains the unit weights but cannot substitute for
repeated private hypotheses at aligned public roots or collection-disjoint
value/bound calibration.
Phase 6Z's 18 live anchors, 70,931 descendant transitions, and 810 terminal
outcomes are genuine official-engine execution. The hidden allocation prior,
five-plan actions, fixed continuation blueprint, ridge model, conformal
intervals, and admission gates are research abstractions. Exact observations
fix the acting player's hand, and no upstream official reach is recorded. The
failed conditional-model diagnostic cannot enable counterfactual bounds,
resolving, exploitation, or strategy control.
Phase 6AA executes no new native transition. Its source roots retain genuine
official-battle provenance, while public-example conversion, prototype fitting,
temperature/shrinkage selection, and posterior interpretation are offline
abstractions. The provider covers six deck labels only and cannot be treated as
a hidden-state posterior or official belief input.
Phase 6AB's 18 battle starts, 512 selections, and internal rule advancement are
genuine official-engine execution. The deterministic `PriorityBaseline`, its
probability-one policy factors, redaction schema, and collection strata are
project declarations. The engine does not expose chance likelihoods, and this
fresh collection does not share a root token with Phase 6Z. Execution
provenance therefore cannot be promoted to counterfactual probability or value
provenance.
Phase 6AC's 18 battle starts, 586 live selections, transient realized hidden
samples, exact root/next-step parity checks, and 29,460 descendant transitions
are genuine official/native execution. The visualizer-to-completion adapter,
shared opaque chance-law assumption, five-plan actions, frozen tactical and
continuation policies, three-rollout value surface, information-set grouping,
and admission decision are research abstractions. Symbolic chance cancellation
does not disclose absolute chance reach, and a single truth sample cannot
support public-belief or counterfactual-value calibration.
Phase 6AD's 180 battle starts, 5,178 prefix selections, and transient realized
player-1 truth labels are genuine official execution. The observer reorientation,
six exact candidate lists, 32-bucket card representation, public-consistent
allocation sampler, prototype classifier, calibration metrics, and range
interpretation are research abstractions. The in-scope pass does not make those
components official or enable intent, values, bounds, resolving, exploitation,
or strategy control.
Phase 6AE's 324 battle starts, 8,474 selections, public opponent MAIN actions,
and transient realized resource labels are genuine official execution. The six
intent policies and labels, action-token representation, linear classifier,
calibration, candidate-intent independence assumption, allocation particles,
and range interpretation are research abstractions. The proxy pass does not
establish human latent intent or enable values, bounds, resolving, exploitation,
or strategy control.
Phase 6AF's 432 battle starts, 11,105 selections, and public opponent action
sequences are genuine official execution. The four policy implementations,
intent-family labels, leave-one-policy-out model selection, particles, and
joint range remain research abstractions. Passing transfer among authored
policies does not establish human latent intent or enable values, bounds,
resolving, exploitation, or strategy control.
Phase 6AG's 288 battle executions, prompts, selections, transitions, terminal
results, and truth replay frames are genuine official execution. Perspective
conversion, causal coarse-history counts, authored collection policies,
candidate ranges, neural outcome model, heuristic blend, and conformal
intervals are research abstractions. The point pass does not turn a realized
on-policy outcome predictor into an expected or counterfactual-value provider.
Phase 6AH's 180 complete matches, legal selections, retained engine snapshots,
snapshot replay, and descendant transitions are genuine official execution.
The two-allocation law, five plans, tactical executor, continuation blueprint,
neural model, and empirical intervals are research abstractions. Passing target
precision and lower aggregate MSE do not establish transferable action values,
counterfactual values, equilibrium, or safe control.
Phase 6AI's complete matches, selections, retained snapshots, immediate
branches, and terminal descendants are genuine official execution. Its hidden
allocation law, plan boundary, tactical executor, continuation policy, grouped
network, bootstrap, and intervals are abstractions. Phase 6AJ uses the same
provenance split on 36 fresh roots; its neural root mean, public branch
heuristic, semiparametric composition, and bounds are abstractions. Passing the
point gate does not make that hybrid a counterfactual-value or strategy
provider.
Phase 6AL follows the same provenance boundary across 72 fresh complete
matches, 1,440 immediate branches, and 11,520 terminal descendants. The native
matches, legal selections, retained snapshots, restores, branches, and
descendants are genuine official-engine execution. The supplied allocation
law, five-plan boundary, tactical executor, continuation policy, fixed mean,
bootstrap, and conformal intervals remain abstractions. Higher sampling
precision does not turn those abstractions into official counterfactual values
or a strategy controller.
Phase 6AM follows that boundary across 36 further complete matches. Its 3,779
legal selections, retained snapshots, 720 immediate branches, 5,760 terminal
continuations, and 274,068 descendant transitions are genuine official-engine
execution. The six-list candidate/resource range, hidden allocation sampler,
five-plan information-set game, fixed continuation blueprint, centered branch
heuristic, common-chance normalization, bootstrap, and conformal intervals are
research abstractions. The range's perfect in-panel top-1 result does not make
it a full official posterior, and passing seven gates does not override the
failed precision gate.
Phase 6AK's 72 fresh complete matches, selections, snapshots, immediate
branches, and terminal descendants are genuine official execution. Its
candidate/intention policies, supplied allocation law, five-plan game,
tactical executor, continuation policy, fixed branch heuristic, ridge mean,
bootstrap, and intervals are abstractions. The collection-heldout point pass is
not implementation-heldout validation and does not make the model a
conditional expected-mean, counterfactual-value, blueprint, or strategy
provider.
Contract tests do not turn abstract models or stubs into official strategy
evidence.

The previously observed account submission
`HelbrosRocketSpidopsProV3.tar.gz` scored 571.8. It predates this Phase 6 agent
and is retained only as external context, never attributed to the new code.

## Durable evidence

- `docs/PHASE6AM_PUBLIC_RANGE_COUNTERFACTUAL_ADVANTAGES.md` - frozen
  development ledger/radius, public-range and common-chance semantics, fresh
  held-out precision rejection, fingerprints, and fail-closed boundary.
- `artifacts/phase6_public_range_counterfactual_advantages/` - frozen
  reference, 36-root validation panel, evaluation, and JSON/Markdown evidence;
  deterministic re-evaluation is byte-stable while native collection is not
  claimed seeded or reproducible.
- `docs/PHASE6AL_HIGH_PRECISION_CONFIRMATION.md` - frozen-before-collection
  reference, sequential calibration/validation protocol, high-precision held-
  out rejection, fingerprints, and downstream fail-closed boundary.
- `artifacts/phase6_high_precision_confirmation/` - frozen model reference,
  fresh 36-root calibration panel and bounds, fresh 36-root validation panel,
  evaluation, and JSON/Markdown evidence; native collection is unseeded and
  not claimed byte-deterministic.
- `docs/PHASE6AK_EXPANDED_ROOT_MEAN.md` - frozen development/model protocol,
  sequential fresh calibration and validation, point pass, worst-slice interval
  rejection, hashes, and downstream fail-closed boundary.
- `artifacts/phase6_expanded_root_mean/` - frozen ridge model, fresh 36-root
  calibration panel and bounds, fresh 36-root validation panel, evaluation,
  and JSON/Markdown evidence; native collection is unseeded and not claimed
  byte-deterministic.
- `docs/PHASE6AJ_SEMIPARAMETRIC_BRANCH_HYBRID.md` - frozen-before-validation
  protocol, fresh point-signal pass, interval-width rejection, provenance,
  fingerprints, and downstream fail-closed boundary.
- `artifacts/phase6_semiparametric_branch_hybrid/` - frozen hybrid/bounds,
  fresh 36-root validation dataset, evaluation, and JSON/Markdown evidence;
  native recollection is unseeded and not claimed byte-deterministic.
- `docs/PHASE6AI_CAUSAL_BRANCH_VALUES.md` - causal immediate-branch protocol,
  grouped-model rejection, diagnostic baseline analysis, hashes, and boundary.
- `artifacts/phase6_branch_conditioned_values/` - Phase 6AI fresh 180-root
  dataset, grouped model, calibration, and fail-closed evidence.
- `docs/PHASE6AH_EXPECTED_VALUE_SURFACES.md` - preregistered repeated-sampling
  protocol, genuine/abstract boundary, rejection metrics, hashes, and next
  action-sensitive value dependency.
- `artifacts/phase6_expected_value_surfaces/` - Phase 6AH frozen 180-root
  dataset, conditional model, calibration, and fail-closed evidence; the native
  runtime is unseeded and recollection is not claimed byte-deterministic.
- `docs/PHASE6AG_OFFICIAL_VALUE_LEAF.md` - complete-game split protocol,
  public/private/range boundary, point pass, interval rejection, artifact
  hashes, and next expected-value dependency.
- `artifacts/phase6_official_value_leaf/` - Phase 6AG frozen 288-match derived
  trajectory corpus, compact manifest/model/calibration/evidence, and retained
  rejected-v1 records; the runtime is unseeded and recollection is not claimed
  byte-deterministic.
- `docs/PHASE6AF_POLICY_IMPLEMENTATION_GENERALIZATION.md` - split-isolated
  policy protocol, robust selection, public-input boundary, metrics, hashes,
  and strict latent/downstream claim boundary.
- `artifacts/phase6_official_policy_generalization/` - Phase 6AF official trace
  corpus, robust intent model, joint calibration rows, and evidence; the runtime
  is unseeded and the bundle is not claimed byte-deterministic.

- `docs/PHASE6AE_OFFICIAL_INTENT_PROXY.md` - seat control, proxy-policy
  semantics, public-action leakage boundary, held-out metrics, hashes, and
  strict latent/downstream claim boundary.
- `artifacts/phase6_official_intent_calibration/` - Phase 6AE official trace
  corpus, intent model, joint calibration rows, and evidence; the runtime is
  unseeded and the bundle is not claimed byte-deterministic.

- `docs/PHASE6AD_OFFICIAL_HIDDEN_CARD_RANGE.md` - observer orientation, live
  collection protocol, hidden-resource abstraction, held-out metrics, hashes,
  and strict downstream claim boundary.
- `artifacts/phase6_official_range_calibration/` - Phase 6AD redacted truth
  dataset, public range model, calibration rows, and evidence; the native runtime
  is unseeded and the bundle is not claimed byte-deterministic.

- `docs/PHASE6AC_JOINT_OFFICIAL_VALUE_SAMPLES.md` - same-root collection,
  transient-truth redaction, native root/next-draw fidelity, common-chance
  cancellation, hashes, and fail-closed calibration boundary.
- `artifacts/phase6_joint_official_values/` - Phase 6AC redacted live dataset
  and evidence; the native runtime is unseeded and the bundle is not claimed
  byte-deterministic.

- `docs/PHASE6AB_OFFICIAL_PREFIX_REACH.md` - live prefix protocol, redaction,
  player-reach semantics, missing chance boundary, hashes, and next admissible
  joint-collection requirement.
- `artifacts/phase6_official_prefix_reach/` - Phase 6AB redacted live dataset
  and fail-closed evidence; the native runtime is unseeded and the bundle is not
  claimed byte-deterministic.

- `docs/PHASE6AA_ALIGNED_PUBLIC_RANGE.md` - public-only split discipline,
  prototype range, signal metrics, sufficiency/calibration failures, hashes,
  and next range requirement.
- `artifacts/phase6_aligned_public_range/` - deterministic Phase 6AA dataset,
  model, calibration, and evidence derived from the frozen Phase 6Z roots.

- `docs/PHASE6Z_PUBLIC_ROOT_ALIGNED_VALUES.md` - later-root protocol, aligned
  hidden allocations, split-safe conditional model, failure metrics, hashes,
  claim boundary, and next admissible step.
- `artifacts/phase6_aligned_counterfactual_values/` - Phase 6Z redacted
  dataset, model, calibration, and evidence; native portions are unseeded and
  not claimed byte-deterministic.

- `docs/PHASE6Y_COUNTERFACTUAL_REACH_LEDGER.md` - estimator convention, exact
  expectation audit, native ledger result, hashes, blockers, and next step.
- `artifacts/phase6_native_counterfactual_reach/` - Phase 6Y exact audit, fresh
  source values, redacted ledger, admission, and evidence; native portions are
  unseeded and not claimed byte-deterministic.

- `docs/PHASE6X_OPPONENT_DEVIATION_VALUES.md` - hypothesis/action surface,
  paired-rollout precision, reach-weighting boundary, hashes, and next step.
- `artifacts/phase6_opponent_deviation_values/` - Phase 6X redacted dataset,
  calibration, and evidence; unseeded and not claimed byte-deterministic.

- `docs/PHASE6W_COLLECTION_HELDOUT_LEAF.md` - three-collection split,
  untouched generalization rejection, hashes, and claim boundary.
- `artifacts/phase6_root_heldout_leaf/` - Phase 6W model, fresh redacted
  validation dataset, calibration, and evidence; unseeded and not claimed
  byte-deterministic.

- `docs/PHASE6V_PUBLIC_NATIVE_LEAF.md` - split-safe training, fresh native
  holdout, point-estimate improvement, interval rejection, and claim boundary.
- `artifacts/phase6_native_public_leaf/` - Phase 6V model, fresh redacted
  dataset, calibration, and evidence; unseeded and not claimed byte-
  deterministic.

- `docs/PHASE6U_LOW_NOISE_NATIVE_LEAF.md` - ten-rollout target precision,
  calibration-only model selection, vacuous-interval rejection, and claim
  boundary.
- `artifacts/phase6_native_leaf_model/` - live redacted Phase 6U dataset,
  calibration, and evidence; unseeded and not claimed byte-deterministic.

- `docs/PHASE6T_NATIVE_ALLOCATION_RANGE.md` - public-consistent allocation
  sampling, public-only neural range boundary, comparative rejection, and
  strict claim limits.
- `artifacts/phase6_native_belief_calibration/` - live redacted Phase 6T
  dataset, calibration, and evidence; unseeded and not claimed byte-
  deterministic.

- `docs/PHASE6S_NATIVE_LEAF_CALIBRATION.md` - held-out order-particle design,
  terminal-continuation results, vacuous-interval failure, uniform-range
  comparison, and strict claim boundary.
- `artifacts/phase6_native_leaf_calibration/` - live redacted Phase 6S dataset,
  calibration, and evidence; unseeded and not claimed byte-deterministic.

- `docs/PHASE6R_DEPTH_LIMITED_NATIVE_PLAN.md` - exact cutoff audit, live
  bounded-horizon probe, provenance split, fingerprints, and next calibration
  gate.
- `artifacts/phase6_native_depth_limited/` - live redacted Phase 6R dataset and
  evidence; unseeded native runtime and not claimed byte-deterministic.

- `docs/PHASE6Q_NATIVE_SAFETY_ADMISSION.md` - necessary prerequisites,
  deterministic rejection, hashes, and explanation of why no vacuous bounds
  are built.
- `artifacts/phase6_native_safety_admission/` - deterministic Phase 6Q
  admission/evidence bound to the live Phase 6P fingerprints.

- `docs/PHASE6P_ITERATIVE_NATIVE_PLAN_MCCFR.md` - iterative loop, policy drift,
  failed variance gate, live hashes, and next admissible research boundary.
- `artifacts/phase6_native_plan_iterations/` - live Phase 6P dataset, final
  model, training report, and evidence; unseeded and not claimed byte-
  deterministic.

- `docs/PHASE6O_NATIVE_TURN_PLAN_BLUEPRINT.md` - plan abstraction, genuine-
  descendant boundary, both-player memories, neural diagnostics, hashes, and
  strict claim limits.
- `artifacts/phase6_native_plan_blueprint/` - live Phase 6O dataset, model,
  training report, and evidence; unseeded and not claimed byte-deterministic.
- `docs/PHASE6N_OUTCOME_SAMPLING_MCCFR.md` - estimator derivation, nonzero-
  baseline expectation audit, tabular reference, hashes, and scope.
- `artifacts/phase6_outcome_sampling_reference/` - deterministic Phase 6N
  expectation, solver, and evidence artifacts.

- `docs/PHASE6M_NATIVE_EXTERNAL_SAMPLING.md` - native admission boundary,
  fail-closed budgets, redacted reservoirs, live hashes, and zero-regret claim
  limits.
- `artifacts/phase6_native_external_sampling/` - live Phase 6M dataset and
  evidence; unseeded and not claimed byte-deterministic.
- `docs/PHASE6L_REACH_CORRECTED_MCCFR.md` - estimator derivation, exhaustive
  expectation audit, tabular convergence reference, hashes, and scope.
- `artifacts/phase6_mccfr_reference/` - deterministic Phase 6L expectation,
  solver, and evidence artifacts.

- `docs/PHASE6K_NATIVE_REGRET_BLUEPRINT.md` — semantic feature bridge,
  terminal-branch regret objective, live diagnostics, hashes, and strict claim
  limits.
- `artifacts/phase6_native_blueprint/` — live Phase 6K sampled-regret dataset,
  neural checkpoint, and evidence; unseeded and not claimed byte-deterministic.

- `docs/PHASE6J_BRANCHABLE_NATIVE_SEARCH_GAME.md` — native-search lifecycle,
  redacted completion prior, information-set invariants, live traversal evidence,
  and provenance limits.
- `artifacts/phase6_native_search_game/` — live Phase 6J evidence and redacted
  completion manifests; unseeded and not claimed byte-deterministic.

- `docs/PHASE6H_OFFICIAL_TRAJECTORY_SUBSTRATE.md` — corrected action alignment,
  public/private trajectory schema, deterministic range baseline, and limits.
- `artifacts/phase6_official_trajectory/` — deterministic 610-step offline
  corpus and Phase 6H evidence.
- `docs/PHASE6I_LOGGED_NEURAL_BOOTSTRAP.md` — split-safe offline architecture,
  deterministic diagnostics, artifact hashes, and claim boundary.
- `artifacts/phase6_trajectory_neural/` — deterministic Phase 6I model and
  JSON/Markdown evidence.
- `docs/PHASE6G_OFFICIAL_SHADOW_BRIDGE.md` — live one-step boundary,
  redaction, action-lifting, and claim limits.
- `artifacts/phase6_official_bridge/` — 12-match live shadow evidence and
  566 decision audits; unseeded and not claimed byte-deterministic.
- `artifacts/phase6_evidence/phase6_evaluation.json` — all 144 match records and metrics.
- `artifacts/phase6_evidence/phase6_evaluation.md` — readable report.
- `artifacts/phase6_evidence/replays/` — six official truth-level replays.
- `artifacts/phase6_evidence/replay_reviews/` — automated triage plus expert rubric.
- `PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md` — target architecture, safety boundary, and Phase 6A-6AK sequencing.
- `docs/PHASE6A_B_GAME_THEORY_FOUNDATION.md` — replay-redaction and exact-game design plus evidence boundaries.
- `artifacts/phase6_game_theory_foundation/` — deterministic public replay dataset and Phase 6A-6B JSON/Markdown evidence.
- `docs/PHASE6C_NEURAL_BLUEPRINT.md` — neural architecture, exact metrics, deterministic restoration, and claim boundary.
- `artifacts/phase6_neural_blueprint/` — deterministic Phase 6C evidence plus both neural candidates and the selected checkpoint.
- `docs/PHASE6D_COUNTERFACTUAL_BOUNDS.md` — residual calibration, uncertainty gates, and exact false-safety audit.
- `artifacts/phase6_counterfactual_bounds/` — deterministic Phase 6D calibration and evidence.
- `docs/PHASE6E_SAFE_RESOLVING.md` — exact strict-nonregression resolver and interruption behavior.
- `artifacts/phase6_safe_resolving/` — deterministic Phase 6E resolver evidence.
- `docs/PHASE6F_CERTIFIED_EXPLOITATION.md` — separate opponent models, robust certificate, and adversarial slices.
- `artifacts/phase6_certified_exploitation/` — deterministic Phase 6F models and evidence.
- `artifacts/baselines/rocketbot_phase6_integration_baseline_watchtower_compatibility.tar.gz` — local baseline-only compatibility bundle; do not submit without accepting the deck change.
