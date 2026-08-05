# Kaggle competition runbook

This runbook prevents an accidental claim or submission with a changed deck.
The current archive is an official-integration baseline only: its heuristic
policy and sampled-rollout advisor are not the final strategic approach. **Do
not submit the current archive.**

## 1. Verify the exact deck gate

```powershell
python -c "from rocketbot import resolve_official_deck; print(resolve_official_deck().to_dict())"
```

Expected current result: 58 IDs and unresolved `Prism Tower`. A strict build
must fail.

```powershell
python scripts/build_submission.py
```

Do not work around that failure by inventing a card ID.

## 2. Run tests and official evidence

```powershell
python scripts/run_tests.py
python scripts/phase6_demo.py --repetitions 1
```

Review:

- `artifacts/phase6_evidence/phase6_evaluation.md`
- every file under `artifacts/phase6_evidence/replay_reviews/`
- the matching native replay before approving a candidate.

## 3. Optional local compatibility baseline

Only after explicitly accepting the two-card change:

```powershell
python scripts/build_submission.py --allow-watchtower-compatibility
```

This creates the local-only baseline:

`artifacts/baselines/rocketbot_phase6_integration_baseline_watchtower_compatibility.tar.gz`

The archive contains `main.py`, `deck.csv`, the flat standard-library policy,
bounded search adapter, manifest, and official multi-platform `cg` runtime. Its
manifest labels the profile `watchtower_compatibility_integration_baseline`,
states that the strategy is not final, and records both deck fingerprints.

## 4. Local archive smoke

Extract into a temporary directory, import `main.agent`, confirm the initial
call returns 60 IDs, then play at least one complete official-engine match.
Reject the bundle for any import error, native-engine rejection, illegal
selection, fallback, timeout, or missing manifest hash.

## 5. Future external submission gate

The baseline archive named above is not a submission candidate. A future
candidate needs a different artifact name and manifest after the neural
game-theoretic planner is trained and validated.

Before consuming a daily Kaggle submission:

- user/owner has explicitly approved the Watchtower compatibility deck or the
  official engine has added Prism Tower;
- all tests pass;
- archive smoke passes;
- the full-agent panel does not regress the frozen evidence;
- a trained neural blueprint is bound to the reviewed deck and abstract-game
  fingerprints;
- the official information-set bridge passes replay equivalence and action-
  lifting coverage tests;
- any enabled re-solving has calibrated counterfactual bounds and safety
  certificates, and any exploitation overlay certifies the final distribution;
- the six queued replays have human expert disposition;
- Kaggle quota and current leaderboard are rechecked;
- submission message names the exact version and compatibility profile.

The Phase 6G live shadow audit satisfies an initial one-decision redaction and
observed action-lifting check only. It does not satisfy full replay/transition
equivalence, trained-blueprint, safe-resolver, expert-review, exact-deck, or
submission-approval gates.

Phase 6H corrects replay action alignment and supplies logged training
trajectories. Because those trajectories contain no unobserved counterfactual
branches, they do not satisfy the branchable-game, trained-blueprint,
counterfactual-bound, or matched-ablation gates either.

Phase 6I validates a restorable neural bootstrap over those trajectories. Its
action head imitates the existing heuristic behavior and its transition head
predicts only observed continuations. It therefore does not satisfy the
equilibrium-blueprint, branchable-game, counterfactual-bound, safe-resolver,
matched-ablation, or submission-approval gates.

Phase 6J satisfies the branchable-transition boundary only: the licensed
native search API executes genuine descendant transitions from explicit hidden-
zone completions, while the completion prior and public belief remain supplied
research abstractions. It does not satisfy calibrated-belief, equilibrium-
blueprint, full-game counterfactual-bound, safe-resolver, matched-ablation,
exact-deck, expert-review, or submission-approval gates.

Phase 6K trains a production-disabled neural candidate on terminal-valued
Phase 6J branches. Its one-rollout sampled-regret objective and tiny disjoint
held-out slices validate training/restoration wiring only; held-out deviation
diagnostics are mixed. It does not satisfy reach-corrected equilibrium training,
full-game exploitability, calibrated-belief, counterfactual-bound, safe-resolver,
matched-ablation, exact-deck, expert-review, or submission-approval gates.

Phase 6L exactly validates a reach-corrected external-sampling MCCFR estimator
and tabular convergence on the small non-official microgame. It does not satisfy
the full-game neural-blueprint, calibrated-belief, counterfactual-bound, safe-
resolver, matched-ablation, exact-deck, expert-review, or submission-approval
gates.

Phase 6M exercises the estimator and redacted reservoir contracts on five
immediate-terminal native roots. Its 45 descendant transitions are genuine
licensed-engine execution, but all five retained advantages are zero, no
opponent decision node is traversed, and the hidden completions remain supplied
abstractions. It does not satisfy full-game training, exploitability,
calibrated-belief, counterfactual-bound, safe-resolver, matched-ablation, exact-
deck, expert-review, or submission-approval gates.

Phase 6N exactly validates a one-terminal-history outcome-sampling MCCFR
estimator, including a nonzero pre-action baseline correction, on the small
non-official microgame. It does not satisfy native equilibrium, calibrated-
belief, counterfactual-bound, safe-resolver, matched-ablation, exact-deck,
expert-review, or submission-approval gates.

Phase 6O collects nonzero both-player targets through a five-plan turn
abstraction over genuine native descendants and trains separate regret and
average-plan models. Its plan actions, fixed tactical executor, hidden prior,
transformed heavy-tailed objective, and model are abstract and production-
disabled. It does not satisfy equilibrium convergence, calibrated-belief,
counterfactual-bound, safe-resolver, matched-ablation, exact-deck, expert-
review, or submission-approval gates.

Phase 6P runs three iterative collect–fit–update rounds, but its final raw-
regret p95 is `5.04e11` and maximum `1.24e16`. It therefore fails admission to
counterfactual calibration and safe resolving. The model remains outside the
archive and official action path, and it does not satisfy equilibrium, safety,
matched-ablation, exact-deck, expert-review, or submission-approval gates.

Phase 6Q binds that failure to the exact model/dataset/evidence fingerprints.
It passes only provenance, redaction, and restoration and rejects 13 remaining
prerequisites. It builds no bounds and executes no resolver, exploitation, or
official action control. Phase 6Q is a guardrail, not a submission candidate or
safety certificate.

Phase 6R demonstrates bounded-horizon variance-control plumbing, but all live
continuations end at confidence-zero, uncalibrated abstract leaves. Its smaller
observed regret tail does not reopen any Phase 6Q gate and is not submission,
strength, equilibrium, or safety evidence. Phase 6R remains outside the archive
and official action path.

Phase 6S adds held-out native order-particle continuations, but its leaf
intervals are full-range and its public-range NLL equals the uniform baseline.
The empirical leaf, public-belief, counterfactual-value, and safety gates remain
false. Phase 6S is also excluded from packaging and submission.

Phase 6T adds public-consistent hidden-allocation particles and a public-only
native range evaluation. The replay-trained neural head is worse than uniform
and substantially worse than the causal revealed-card baseline on held-out
allocation rows. Public-belief calibration remains false, and neither range is
packaged into the agent. Phase 6T is excluded from packaging and submission.

Phase 6U lowers empirical leaf-target standard error with ten continuations,
but calibration selects a zero baseline and all validation intervals remain
full-range. It establishes neither a usable leaf nor counterfactual values and
is excluded from packaging and submission.

Phase 6V's purpose-trained public scalar improves point RMSE on a separate
native holdout, but its 90% conformal interval remains almost the full utility
range. It is a fixed-profile research predictor, not a counterfactual-value
bound, and is excluded from packaging, strategy control, and submission.

Phase 6W uses an entirely untouched third native collection and rejects the
same public-leaf family against the zero baseline as well as on interval width.
It confirms that the Phase 6V point gain does not generalize. Phase 6W is also
excluded from packaging, strategy control, and submission.

Phase 6X collects opponent-plan conditional action returns and passes only a
supervision-substrate gate. It has no opponent/chance reach weighting and hence
no calibrated counterfactual values or safety bounds. It is excluded from
packaging, strategy control, and submission.

Phase 6Y validates the counterfactual reach formula on the exact microgame and
attaches a redacted root-relative probability ledger to a fresh native
collection. All native roots are already the first opponent decision, all
weights are one, and every public information set appears only once. This is
reach-accounting evidence, not a calibrated native value or safety bound, and
is excluded from packaging, strategy control, and submission.

Phase 6Z captures later live roots and evaluates three distinct hidden
allocations behind each shared public information set. Complete root families
are isolated across model fitting, conformal calibration, and untouched
validation. The native continuations are genuine, but the supplied prior,
fixed acting-player hand, abstract plan actions, absent upstream reach, and
diagnostic ridge model prevent counterfactual or safety admission. The model is
also empirically rejected: validation coverage is `75.555556%`, and RMSE
`0.847985` loses to zero `0.771402`. Phase 6Z is excluded from packaging,
strategy control, and submission.

Phase 6AA fits a public-only six-archetype range diagnostic from the frozen
Phase 6Z roots. It beats uniform on untouched validation, but only six roots
exist per holdout, ECE is `0.476073`, hidden hands/resources/intents are outside
scope, and no production observer-orientation bridge exists. It is not an
official posterior and is excluded from packaging, counterfactual values,
resolving, strategy control, and submission.

Phase 6AB captures 18 fresh official behavior prefixes and 512 genuine
selections, but the live adapter exposes no likelihoods for automatic shuffle,
Prize, draw, or coin events. Only declared deterministic player-policy reach is
available; chance/full reach and counterfactual weights are `null`, and the new
roots are not aligned with Phase 6Z value surfaces. It is excluded from
packaging, resolving, exploitation, strategy control, and submission.

Phase 6AC recollects the prefix, transient realized official truth, and all five
plan values at the same 18 roots. Exact native root and forced-next-draw replay
parity pass, and the identical target/sampling chance law cancels symbolically.
Absolute chance reach remains unavailable, each information set has one truth
sample, and the values come from a frozen abstract plan/continuation profile
with only three rollouts per action. It supplies no calibrated belief, value
bound, safety certificate, or strength evidence and is excluded from packaging,
resolving, exploitation, strategy control, and submission.

Phase 6AD collects 180 new official roots and passes preregistered held-out
candidate/resource calibration only for six exact declared deck lists, the
baseline-policy panel, and one later-root distribution. The roots and transient
truth labels are genuine official execution; the player-0 observer adapter,
prototype classifier, public-consistent allocation prior, 96 hash buckets, and
range interpretation remain offline abstractions. Opponent intent, full belief,
counterfactual values, safe resolving, exploitation, and official control are
false. Phase 6AD is excluded from packaging, strategy control, and submission.

Phase 6AE collects 324 additional roots with player 0 fixed and only player 1
varied across six controlled declared-intent proxy policies. Its 108-root
untouched validation split passes intent-proxy top-1 (`0.879630`), intent ECE
(`0.042259`), joint ECE (`0.042195`), and resource gates. The official roots,
legal selections, public action sequences, and transient realized resource
labels are genuine execution. The proxy policies/labels, linear classifier,
candidate-intent product, particles, and range interpretation are offline
abstractions. Human latent intent, full belief, chance reach, values, safety,
exploitation, and official control remain false. Phase 6AE is excluded from
packaging, strategy control, and submission.

Phase 6AF collects 432 further roots across four split-isolated authored
player-1 policy implementations. Canonical and engine-first policies train,
hand-first policies calibrate, and action-first policies remain untouched until
validation. Validation intent top-1 is `0.826389`, intent ECE is `0.090055`,
joint ECE is `0.091196`, resource ECE is `0.006552`, and all intent-family
slices exceed `0.50`. This validates controlled policy-family transfer only.
Human latent intent, full belief, values, safety, exploitation, and official
control remain false. Phase 6AF is excluded from packaging, strategy control,
and submission.

Phase 6AG completes 288 official games across the same six candidates, six
controlled intents, four split-isolated implementations, and two replicas.
Canonical and engine-first matches train, disjoint halves of hand-first
matches select the model and calibrate intervals, and action-first matches stay
untouched until validation. The neural/blended point leaf reaches episode MSE
`0.938795` versus zero `1.000000` and the causal heuristic `1.019981`, so its
narrow point gate passes. The 90% terminal-outcome intervals fail: overall
coverage is `0.862791`, overall mean width is `1.937685`, and endgame mean width
is `1.753315`. This is not an expected-value or counterfactual-value bound.
Safe resolving, exploitation, official control, packaging, submission, and
strength claims remain false. Phase 6AG is excluded from the archive and every
Kaggle action path.

Submission commands are intentionally omitted while these gates are open. Once
the owner authorizes a separately built candidate, record its reference,
status, score, timestamp, archive SHA-256, and leaderboard delta in a new
immutable evidence record. Do not rewrite the local 144-match result to make it
appear paired with Kaggle outcomes.

## 6. Competition-only license

The official `cg` engine is not open source. Use it only for this competition,
do not redistribute it, and delete it when the competition ends. The official
rules and bundled license govern.
