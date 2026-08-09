# Phase 6AH preregistration: repeated expected-value surfaces

Status: **fresh panel complete; conditional expected-value model rejected; all downstream gates remain closed**

## Checked result

The preregistered run completed all 180 official matches, 360 supplied hidden-
allocation surfaces, 900 aggregated root/action targets, 14,400 terminal native
continuations, 20,388 live selections, and 605,351 native search transitions.
The train/calibration/validation root counts are exactly 72/72/36.

Repeated sampling solved the immediate precision problem: maximum target
standard error is `0.258199`, below the frozen `0.30` threshold. The selected
32-hidden-unit model uses weight decay `1e-4` and a `0.50` neural/heuristic
blend. On the untouched `action_first` implementation, point MSE is `0.704402`,
better than zero `0.812674` and heuristic `0.895724`. That aggregate improvement
does not pass the point gate, because root-centered action-contrast MSE
`0.028905` is slightly worse than the zero-contrast reference `0.026840`.

The conformal interval gate also rejects the model. Overall empirical-mean-CI
coverage is `0.872222` versus the required `0.90`; mean width is `1.826374`
versus the maximum `1.75`; and the minimum action/candidate/intent slice
coverage is `0.733333` versus the required `0.75`. The worst slice is the
`resource_conservation` declared intent. The calibrated radius is `1.333081`.

Therefore Phase 6AH establishes a useful negative result: sixteen returns per
root/action are enough for the declared target-precision gate, but this feature
and objective stack does not learn transferable action-sensitive values. It
admits no conditional expected-mean provider, counterfactual values,
equilibrium claim, safe resolver, exploitation overlay, official strategy
control, submission, or playing-strength claim.

| Artifact | Fingerprint |
|---|---|
| Dataset | `db4716917936b9491208842126cf81083f8913238221587d6702069ffa3afb9b` |
| Model | `87306708c804612cd60f9373083988d6db3f4bb55ffbd60d79c14c28ee880d0b` |
| Calibration | `8a9df559bfb6e001e6c728c330eada229c88385f462feb89830db255e7bf843d` |
| Evidence | `414e85a4f0bf4112cac0e914194da587f7c300a5deb3a5714eaaa89c98004c1d` |

All 189 repository tests pass, including six Phase 6AH artifact tests.
The root shards remain under `work/phase6ah_root_shards` for deterministic
reconstruction; the checked compact artifacts are under
`artifacts/phase6_expected_value_surfaces`.

## Purpose

Phase 6AG found implementation-heldout point signal in complete official games,
but its one terminal result per trajectory state did not produce admissible
intervals. Phase 6AH tests a narrower question: can repeated native terminal
continuations provide useful conditional expected-value estimates for the five
abstract turn plans at fresh, later official-game roots?

This is not an equilibrium, counterfactual-value, safe-resolving, exploitability,
deployment, or playing-strength test. A positive result only admits a
conditional mean model under the supplied hidden-allocation law and frozen
continuation blueprint.

## Fixed collection panel

The panel contains 180 complete official-engine matches:

| Opponent policy implementation | Role | Replicas per candidate/intent |
|---|---|---:|
| `canonical` | train | 1 |
| `engine_first` | train | 1 |
| `hand_first` | calibration | 2 |
| `action_first` | untouched validation | 1 |

Every implementation is crossed with all six frozen candidate decks and all
six declared intent families. `hand_first` replica 0 selects hyperparameters;
replica 1 supplies conformal bounds. `action_first` is never used for fitting,
selection, or interval construction.

Player 0 uses the deterministic tempo priority policy. Player 1 uses the
declared-intent priority policy for the assigned implementation. Direction is
deterministically alternated by stratum and retry. No outcome-dependent policy
choice is allowed.

## Root selection

Each match runs to an official terminal result. During the match the collector
retains the first player-0 `MAIN` prompt from every distinct player-0 turn after
at least two distinct opponent `MAIN` turns have been observed. After the match
is finished, it chooses the third-to-last eligible root, or the earliest root
when fewer than three are available. This lookback was fixed from feasibility
probes before the panel was generated; no prize or outcome filter is applied.

The retained observation contains the engine-issued native search snapshot.
After `battle_finish`, Phase 6AH reopens that snapshot through the official
native branch adapter and requires exact public-state and acting-player private
observation parity. The model sees no future match events or terminal result
from the completed match; completion is used only to establish that the source
match genuinely terminated and to make root selection deterministic.

## Repeated value targets

At every retained root:

1. Sample two independent public-consistent hidden allocations from the
   realized opponent candidate deck. This allocation law is supplied by the
   research adapter and is not an official belief distribution.
2. For each allocation, force each of the five abstract turn-plan actions.
3. From each forced child, execute eight terminal native continuations under
   the frozen Phase 6P average plan blueprint and frozen tactical executor.
4. Use paired/common rollout seeds across all actions and both allocations at
   a root.

The fixed totals are 180 roots, 360 allocation surfaces, 900 aggregated
root/action targets, and 14,400 terminal rollouts. Each root/action target is
the mean of 16 terminal utilities. Its standard error is computed directly
from those 16 returns; the maximum permitted target standard error is 0.30.

## Model and holdout protocol

The neural regressor receives only:

- public root features;
- acting-player private root features;
- causal coarse action counts from the observed prefix;
- a one-hot supplied candidate hypothesis; and
- a one-hot abstract plan action.

The declared intent label and opponent policy-implementation label are retained
for audit slices only and are forbidden model inputs. A bounded MLP is fit on
the two train implementations. Hidden width (`32`, `64`), weight decay (`0`,
`1e-4`), and neural/heuristic blend (`0.50`, `0.75`, `1.0`) are selected only on
the first calibration replica.

The point gate requires untouched-validation MSE to beat both the zero baseline
and the fixed public-state heuristic. It also requires root-centered action
contrast MSE to beat the zero-contrast baseline.

The second calibration replica constructs a finite-sample conformal radius from
`abs(prediction - empirical_mean) + 1.645 * target_SE`. On untouched validation,
the resulting clipped interval must cover the entire empirical 90% mean
confidence interval, not merely its point estimate. Admission thresholds are:

- overall coverage at least 0.90;
- mean interval width at most 1.75;
- every action, candidate, and intent slice coverage at least 0.75; and
- maximum target standard error at most 0.30.

## Provenance boundary and fail-closed interpretation

Genuine official-engine execution consists of complete matches, legal live
selections, engine-issued retained snapshots, snapshot replay, and every native
descendant transition. Research abstractions consist of the two-allocation
prior, five plan actions, tactical executor, frozen continuation blueprint,
neural model, and empirical intervals.

Regardless of the measured result, Phase 6AH keeps the following false:

- official hidden-allocation prior;
- equilibrium-converged continuation strategy;
- opponent counterfactual values;
- safe-resolver compatibility;
- opponent-exploitation enablement;
- official strategy control; and
- Kaggle submission or performance claims.

Root shards are written independently so interrupted collection can resume
without changing the frozen grid or seeds.
