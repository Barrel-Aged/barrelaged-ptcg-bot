# Phase 6I logged-trajectory neural bootstrap

Phase 6I proves that the corrected Phase 6H trajectory schema can support a
restorable, split-safe neural training loop. It does **not** train the final
strategic policy. Its action head is supervised imitation of the preserved
heuristic behavior in the native replays, and its transition head sees only the
single continuation that actually occurred.

The durable strategic target remains a neural information-set blueprint trained
with an equilibrium-oriented objective, safe public subgame/endgame re-solving,
and a separately constrained opponent-model overlay. Phase 6J subsequently
supplies a production-disabled, branchable native-search boundary under an
explicitly supplied hidden-state prior; it does not itself train that target.

## Provenance boundary

Two statements are simultaneously true and must not be collapsed:

1. The source matches and recorded transitions genuinely ran in the official
   competition engine.
2. The neural model consumes a derived offline dataset whose provenance is
   `abstract_information_set_game`, whose own `official_engine_executed` flag is
   false, and which has no counterfactual branches.

The model is therefore an offline representation/behavior bootstrap, not an
official engine, a `BlueprintStrategyProvider`, a counterfactual-value provider,
or a safe resolver.

## Architecture

`trajectory_neural.py` implements four named, train-only-normalized inputs and
heads:

- a 111-wide common-public vector encoded without private observations;
- a separate 75-wide acting-player-private vector containing hashed card
  buckets rather than raw card IDs;
- a variable-cardinality legal-action scorer over 11-wide action features;
- a public-only six-candidate opponent-range head;
- an acting-player terminal-outcome head; and
- a next-public-feature delta head initialized to the identity transition.

The public and private embeddings are combined for behavior, outcome, and
transition prediction. The opponent-range head receives only the public
embedding. Legal actions are represented as a flat ragged batch, so no action
outside the exact information-set support can receive probability.

Losses cover:

- cross-entropy against the logged native behavior selection;
- MSE against the terminal outcome from the acting player's perspective;
- cross-entropy against the supervision-only opponent-deck candidate; and
- normalized next-public-state MSE against the observed continuation.

Only the 385-step train partition fits weights and feature normalizers. The
86-step calibration partition selects policy and range temperatures from a
fixed grid. The 139-step validation partition is evaluation-only in the run,
but its diagnostics were inspected while building this scaffold; it is not a
pristine final test set.

## Deterministic evidence

The checked-in run uses seed `20260726`, 10 full-batch epochs, hidden width 32,
and the match-disjoint Phase 6H split:

| Split | Matches | Steps | Policy NLL (uniform) | Outcome MSE (constant) | Range NLL (uniform) | Next-public MSE (identity) |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4 | 385 | 1.077407 (1.343878) | 0.010723 (0.979592) | 0.376016 (1.791759) | 0.049698 (0.052069) |
| Calibration | 1 | 86 | 1.364963 (1.614884) | 0.014340 (0.914096) | 0.758862 (1.791759) | 0.119395 (0.121140) |
| Validation | 1 | 139 | 1.281160 (1.401432) | 0.005730 (1.047130) | 1.225283 (1.791759) | 0.031766 (0.032861) |

These are small-corpus wiring diagnostics, not population estimates or playing-
strength claims. In particular, favorable imitation metrics do not turn the
logged heuristic teacher into the desired game-theoretic strategy.

Validated integrity properties:

- training loss decreases from `1.9315843561` to `1.1376717802`;
- all four task heads and their shared encoders receive non-zero gradients;
- train/calibration/validation match overlap is zero;
- all three normalizers record `fit_split: train` and bind to train-row
  fingerprints;
- policy and range temperatures are selected on calibration only (`1.50` and
  `1.25`);
- JSON checkpoint restoration has zero action, value, range, and transition
  prediction drift across all 610 steps; and
- restored policy support equals the complete legal action set for all 610
  information sets.

Artifact fingerprints and hashes:

- Phase 6H dataset fingerprint:
  `de42c3abb2ab32e334a7547189391be08b98c2148666572204421919a4d4203a`;
- Phase 6I model fingerprint:
  `bb9876945b902a485d809daf969c7f15f31230b9760c86f5d7e58bb9be25914d`;
- Phase 6I evidence fingerprint:
  `a6b11390a14fe91fcc10315198072db6f90641fcaf15def8ce49475bbae2c58e`;
- model JSON SHA-256:
  `584b3ba5ef6407c0614640be310d106b5f263aa58cbeb2bdbf70aff1695b14e0`;
- evidence JSON SHA-256:
  `d8bfc2912b99815335452dd9089170d01007f226107f5ac733085400141ca78d`;
- evidence Markdown SHA-256:
  `101486c711575099e456a72b15c79bff282e418d36ff3293440ae20034e30ad2`.

Two independent generations produced the same three file hashes.

## Interfaces and artifacts

- `LoggedTrajectoryBootstrapNetwork` — public/private representation, ragged
  legal-action scorer, and auxiliary heads.
- `NamedFeatureNormalizer` — named train-only normalization with a fitted-row
  fingerprint.
- `TrajectoryBootstrapArtifact` — weights, schemas, split bindings,
  temperatures, flags, and deterministic fingerprint.
- `LoggedTrajectoryBootstrapProvider` — legal-support prediction interface
  intentionally distinct from `BlueprintStrategyProvider`.
- `restore_logged_trajectory_bootstrap` — strict architecture/tensor restore.
- `run_phase6_trajectory_neural_bootstrap` — evidence gates and exact roundtrip.
- `scripts/phase6_trajectory_neural.py` — durable artifact generator.
- `artifacts/phase6_trajectory_neural/` — model and JSON/Markdown evidence.

## Next gate

Phase 6I is an initialization substrate only. The next strategic gate is to
use the Phase 6J licensed native-search boundary to train and evaluate a full-
game blueprint with an equilibrium-oriented objective. Phase 6K now validates
that connection with a small terminal-branch sampled-regret candidate, but not
reach-corrected CFR, convergence, or improved play. The next gate is to scale
and validate that equilibrium-oriented training. Only after that should the
project calibrate full-game
counterfactual-value bounds, enable all-or-blueprint safe re-solving, attach the
separate exploitation overlay, and run matched-compute official-engine
ablations. The heuristic and sampled-rollout policies remain integration
baselines, not the final method.
