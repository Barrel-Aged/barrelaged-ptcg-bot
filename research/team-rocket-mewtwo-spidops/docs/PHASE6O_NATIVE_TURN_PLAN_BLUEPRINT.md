# Phase 6O: native turn-plan blueprint candidate

Phase 6O applies the audited outcome-sampling estimator to a bounded,
nontrivial native-backed game and trains three separate neural models: a
player-0 regret head, a player-1 regret head, and an average-plan strategy head.
This closes the empty/zero-regret Phase 6M wiring gap. It does not close the
equilibrium, calibration, safety, deployment, or strength gates.

## Why the action abstraction changed

The native engine exposes many procedural selection prompts. Direct
outcome-sampling through those option-index decisions completes a terminal path
in roughly 143–194 transitions, but the importance-weighted regret estimates
reached magnitudes near `1e65`. That is not a usable neural target.

The new information-set game instead chooses one of the project's persistent
plans at each main-turn boundary:

- board setup;
- energy engine;
- prize push;
- recovery; or
- control.

A fixed plan-conditioned tactical executor maps the chosen plan to legal native
selections until chance, the next player's main turn, or terminal state. This
reduces the game-theoretic horizon while retaining a strategically meaningful
choice.

## Official execution versus abstraction

Genuine licensed-engine execution:

- every descendant tactical selection is one of the native state's legal
  selections;
- every descendant transition is executed through the native `search_step`
  boundary;
- terminal results and utilities come from the native engine; and
- all 5,569 native descendant transitions in the checked artifact completed
  through that boundary.

Research abstractions:

- the five plan actions are not official option indices;
- the plan-conditioned tactical executor is fixed research behavior;
- the six equal-weight hidden-zone completions are a supplied, uncalibrated
  prior;
- the Phase 6K scalar state value is used only as a pre-action variance
  baseline, not a counterfactual value;
- public/private/action features and all neural predictions are derived
  research objects; and
- the resulting strategy is production-disabled and does not control the
  official agent.

`GameBoundaryProvenance` therefore marks the plan game as an abstract
information-set game with `official_engine_executed=False`, while each sample
separately records that its source descendants genuinely executed the engine.

## Checked evidence

The live, unseeded engine run covered six hidden-completion candidates, three
iterations, and both traversers:

| Check | Result |
|---|---:|
| Complete trajectories | 36 |
| Player-0 / player-1 trajectories | 18 / 18 |
| Macro / native / tactical transitions | 927 / 5,569 / 5,547 |
| Player-0 / player-1 regret samples | 238 / 214 |
| Average-strategy samples | 452 |
| Nonzero regret action targets | 2,260 |
| Train / calibration / validation samples | 335 / 75 / 42 |
| Training loss | 6.64556058 → 4.86231795 |
| Checkpoint restore drift | 0.0 |
| Serialized raw private-card fields | 0 |

All three networks receive gradients. Restored providers preserve exactly the
five-action support, and attempts to promote equilibrium, calibrated-value,
safe-resolver, or deployable capability flags fail closed.

## Variance and neural target limitation

Raw absolute sampled regrets have median `378.164`, p95 `10,209,941.165`, and
maximum `5,440,035,687,151.638`. The neural loss therefore uses a scale computed
from training-split regrets only and fits `asinh(regret / scale)`. This preserves
the sign and ordering of each target, but it changes regret magnitude. No CFR
convergence theorem or exploitability conclusion is claimed for the trained
network.

Fresh native starts are unseeded, so this live bundle is not claimed to be
byte-deterministic. The checked fingerprints identify this run only.

## Durable artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| `native_plan_outcome_dataset.json` | 20,271,383 | `c401621479c45b6b6cd946db80edeae821624dcdd24d634549d49e514a70d855` |
| `native_plan_blueprint.json` | 893,789 | `b7d52645f7bcd81c8ed8ce383ae2082f46207de1155a4007204bb6139256f61b` |
| `native_plan_training_report.json` | 4,444 | `fcfaa024e061edf5743b2b5182e0dd9ccb4a9bb275771fe6ae855b4b5557e83d` |
| `phase6o_evidence.json` | 2,899 | `1284b1d59bb9c9ffad9956c450dd1e07dfe7a5ae0db15bdee65c75c5baa1b7bf` |
| `phase6o_evidence.md` | 2,446 | `1cfca42f76fa8dc6db712816dfd3c4f09cd87f216d3cabe25f0dabaa878826dc` |

Dataset fingerprint:
`f5c0ba2fbf2be083fa97ef29248c8a51d3311cfe8824dfdb560228a83da2f25f`.
Model fingerprint:
`3e621c1adeceff3cee10bec22aaaa6c5bc4eacbc9224bf874488fa6ccc50132f`.
Evidence fingerprint:
`3297ba3b78fae0541ca6125d43ced7c5f6e0f6552ff51135f72456050e2ae199`.

## Next safety gate

Before any plan policy can be called a safe blueprint, the project still needs
held-out native histories and particles large enough to calibrate
per-hypothesis opponent counterfactual-value error. A resolver must then show
componentwise non-regression within those bounds and return the baseline
exactly on OOD, low-confidence, incomplete-support, timeout, or budget failure.
Only after that may a separate opponent-model overlay be evaluated against the
same safe baseline. No Kaggle submission or performance claim follows from
Phase 6O.
