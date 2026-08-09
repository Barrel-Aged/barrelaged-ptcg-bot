# Phase 6K native regret blueprint candidate

Phase 6K is the first full-game neural strategy candidate trained from
counterfactual branches rather than logged heuristic action labels. It closes a
training-wiring gap, not the equilibrium, safety, deployment, or strength gate.

The candidate is initialized from the Phase 6I representation checkpoint. At
selected Phase 6J information sets it branches every legal action, follows each
branch to a terminal state through the licensed native search runtime, computes
acting-player action values, accumulates sampled regrets, and fits separate
average-policy, advantage, and state-value heads.

## Provenance boundary

Two layers remain deliberately distinct:

1. Every recorded descendant transition and terminal utility in the Phase 6K
   branch corpus genuinely executes in the licensed `cg` native search engine.
2. The six-particle hidden-completion prior, feature bridge, derived regret
   dataset, neural checkpoint, and strategy distribution are non-official
   research abstractions.

The neural model never receives the selected completion ID. The completion ID
is retained only in a field named `source_candidate_id_evaluation_only` for
coverage auditing. Policy inputs contain common public features, hash-bucketed
agent-visible own-private cards, and complete legal-action features. The root
policy view is identical across all six selected completion truths.

## What was implemented

### Semantic feature bridge

`native_blueprint_features.py` supplies a fixed, leakage-audited schema:

- 111 public features, including board/resource counts and public-card hash
  buckets;
- 75 acting-player-private features, including own-hand and visible-search hash
  buckets; and
- 11 semantic features for every complete ordered legal action.

Those names and widths exactly match the Phase 6I train-only normalizers. Raw
card IDs never enter the persisted dataset. `InformationSet.private_features`
provides an optional generic seam for policy-safe own-private tensors; existing
exact games and replay contracts retain their previous behavior through the
empty default.

Schema compatibility is not a claim of identical data distribution. The live
bridge uses currently visible opponent cards where the replay corpus could keep
a revealed-card high-water mark, and it sets replay-interval private-event count
to an explicit zero because that interval is unavailable at a live decision.
These semantic shifts are another reason the checkpoint remains non-deployable
until OOD behavior is measured and the live history encoder is strengthened.

### Neural candidate

`NativeRegretBlueprintNetwork` copies all compatible Phase 6I public, private,
state, action, policy, value, and range parameters, then adds a distinct
action-conditioned advantage head. Its optimized losses are:

- cross-entropy to the accumulated regret-matched strategy target;
- normalized cumulative-regret regression;
- sampled state-value regression; and
- a small policy/advantage consistency term.

`NativeRegretBlueprintProvider` emits an exact-support legal distribution and a
restorable `BlueprintCheckpoint`. The checkpoint truthfully records that it was
trained with an equilibrium-oriented regret objective. Its capability flags
remain false for equilibrium convergence, deployment, calibrated
counterfactual values, safe resolving, opponent exploitation, official strategy
control, and exact-deck execution.

The provider exposes its scalar rollout estimate only in `predict()`. Its formal
counterfactual-value method fails closed because no per-hypothesis values have
been calibrated; the scalar can never silently enter safe resolving.

### Sampling and splits

The live evidence run uses two collect/train rounds. Each round selects one
bounded-action information set for each player in each of six hidden-completion
scenarios. All legal actions at that information set receive one terminal
rollout. The second round uses the first-round candidate as its behavior and
continuation policy.

Information-set identities—not individual rows—are assigned to train,
calibration, or validation. Repeated appearances can never cross a split. New
second-round information sets enter training while the original held-out sets
remain held out.

This is sampled neural regret matching, not Deep CFR. The current run has only
one rollout per action, no counterfactual-reach correction, no reservoir-scale
advantage sampling, no full best response, and no full-game NashConv or
exploitability calculation.

## Live evidence

The checked-in run is unseeded at the native battle root and therefore is not
claimed byte-deterministic across independent generations.

- 2 regret collection/training rounds;
- 24 samples / 24 distinct information sets;
- all 6 completion candidates represented;
- 12 player-0 and 12 player-1 samples;
- 19 train / 2 calibration / 3 validation samples;
- 15,452 genuine native branch transitions;
- 24 Phase 6I-initialized parameter tensors;
- gradients through 22 trainable parameter tensors, covering both semantic
  encoders, state/action encoders, average-policy head, advantage head, and
  value head;
- training loss `3.64769418 -> 1.68179459`;
- exact legal support on every train/calibration/validation sample;
- 12 sampled regret targets change the behavior-policy argmax; and
- exact serialized-model restoration with maximum prediction drift `0.0`.

| Split | Samples | Policy CE | Advantage RMSE | Value RMSE | Candidate sampled deviation | Behavior sampled deviation |
|---|---:|---:|---:|---:|---:|---:|
| Train | 19 | 1.418405 | 0.648182 | 0.459019 | 0.686232 | 0.678030 |
| Calibration | 2 | 1.450983 | 0.696104 | 0.636515 | 1.305089 | 0.982507 |
| Validation | 3 | 1.429511 | 0.313584 | 0.697322 | 0.556841 | 0.564173 |

These tiny-sample diagnostics are intentionally reported even where the
candidate is worse than its behavior policy. They do not establish a gain. The
zero deviation of the constructed regret target is tautological on its sampled
one-rollout action values and is not presented as exploitability evidence.

## Durable artifacts

- evidence fingerprint:
  `ed9ce6c0b682b298e61006c19a4cfe81a08ca341160f15127f991a3d8b885236`;
- dataset fingerprint:
  `276c2dd193355206b3cff2c8108f8ce7b9df9d284880c96728239487b319ea7f`;
- model fingerprint:
  `e9cf893ccf28eeee656af9791de389196d7a869d9bdae090e0e15b7dc5f83232`;
- feature-schema fingerprint:
  `60b0422a9a221ca5c200ed6de4f8e55d5d83f78437858acd8aaf8282559ec6ea`;
- runtime fingerprint:
  `9a62073d859273c86926a63c30a9a28dc16c698ad759ade3d1a7563b05af1634`;
- `native_regret_dataset.json` SHA-256:
  `caab4493de746e8c518014910375d202145db476db242dbd9ed126c2843801f9`;
- `native_regret_blueprint.json` SHA-256:
  `4899e7e57f629e6ed84759b03aa0c6214ccb622714fbcd91b8793db2fc27171b`;
- `phase6k_evidence.json` SHA-256:
  `7cb0d8c5ac7d34c6a41fc6c1126998b8ad1da7dfff3b3d4dad37c0ed64b3722d`;
- `phase6k_evidence.md` SHA-256:
  `29dda55bea17003f210e870b6406c44eb0bd07e83e92bd9224352dd2dea859cc`.

Reproduce with:

```powershell
python scripts/phase6_native_blueprint.py
```

## Next gate

Phase 6L subsequently validates a reach-corrected external-sampling estimator
and separate replay reservoirs exactly on the non-official microgame. Phase 6M
wires those contracts to five immediate-terminal native roots, but produces
only zero advantages and no opponent-node strategy samples. See
`PHASE6L_REACH_CORRECTED_MCCFR.md` and
`PHASE6M_NATIVE_EXTERNAL_SAMPLING.md`.

Phase 6N subsequently validates a primary-source-aligned outcome-sampling
alternative, and Phase 6O uses a five-plan turn abstraction to collect
nontrivial both-player regret and strategy memories and train three separate
networks. See `PHASE6N_OUTCOME_SAMPLING_MCCFR.md` and
`PHASE6O_NATIVE_TURN_PLAN_BLUEPRINT.md`.

The next blueprint gate remains a substantially larger, stable neural
MCCFR-style experiment over nontrivial native information sets, with
iteration-separated held-out evaluation and a defensible sampled-regret trend.
Phase 6O's train-only `asinh` target changes regret magnitude and therefore is
not a convergence result.
The public hidden-completion prior must also become calibrated and history-
conditioned. Only after a baseline strategy has credible full-game validation
should Phase 6 calibrate per-hypothesis counterfactual-value bounds and connect
the all-or-blueprint safe resolver. The current candidate remains disabled in
the official agent and must not be submitted to Kaggle.
