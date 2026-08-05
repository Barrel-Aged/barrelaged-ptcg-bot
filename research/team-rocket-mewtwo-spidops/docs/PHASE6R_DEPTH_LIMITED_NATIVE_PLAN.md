# Phase 6R: depth-limited native turn-plan variance control

Phase 6R adds a depth-limited information-set-game wrapper and an explicitly
uncalibrated native leaf-value adapter. Its purpose is to control the extreme
importance-weight horizon exposed by Phase 6P without pretending that a neural
bootstrap is an official terminal result or a counterfactual-value bound.

## What was implemented

- `DepthLimitedInformationSetGame` wraps any information-set game at a fixed
  source-depth delta. Genuine source terminals still delegate to the source
  game; cutoff terminals delegate to a `LeafValueProvider`.
- `LeafValueEstimate` requires a finite point value, containing interval,
  confidence, calibration flag, and source description. A cutoff estimate is
  structurally forbidden from claiming official terminal execution.
- `ExactProfileLeafValueProvider` supplies exact continuation values only for
  the enumerable non-official `RocketPrizeDuel` audit.
- `UncalibratedNativePlanLeafValueProvider` clamps the Phase 6K scalar to the
  native utility range. It always returns the full `[-1, 1]` interval,
  confidence zero, and `calibrated=False`.
- `NativeDepthLimitedPlanDataset` stores separate player-regret and average-
  strategy reservoirs, binds all source/wrapper/model/runtime fingerprints,
  rejects raw private-zone fields, and requires the cutoff-valued samples to
  remain abstract.

## Exact estimator audit

The exact audit enumerates the depth-limited wrapper's own chance-state IDs.
This matters because wrapper IDs intentionally differ from source-game IDs;
forcing source IDs would leave chance sampling uncontrolled.

At source-depth delta 3, 32 enumerated paths cover 18 pre-cutoff regret entries.
Exact continuation leaves reproduce the full-game immediate counterfactual
regret expectations with maximum absolute error `5.55e-17`. The eight exact
leaf calls have point intervals and confidence one. This validates cutoff
estimator semantics on the small non-official game only.

## Live native probe

The checked-in probe uses:

- the Phase 6P final player-regret strategy provider;
- the Phase 6K tactical provider;
- all six supplied hidden-completion candidates;
- both traversers for three iterations, for 36 trajectories;
- a six-macro-transition horizon;
- leaf values and pre-action baselines clamped to `[-1, 1]`.

All 36 trajectories end at an explicit cutoff rather than an official terminal.
They execute 1,474 genuine licensed-engine tactical descendant transitions and
216 abstract macro transitions, retaining 54 player-0 regret records, 54
player-1 regret records, and 108 separate strategy records. No raw private card
identity field is persisted.

Raw absolute regret median / p95 / maximum is `0.643773` / `17.895890` /
`134.916423`. The corresponding ratios to the earlier Phase 6P artifact are
`3.55e-11` and `1.09e-14` for p95 and maximum. This is a deterministic wiring
comparison across different horizons, not a controlled performance estimate,
convergence result, or claim that the leaf target is correct.

## Provenance boundary

| Item | Classification |
|---|---|
| Native tactical descendant selections | Genuine licensed official-engine execution |
| Five turn-plan actions and fixed executor | Research abstraction |
| Six-particle hidden completion prior | Supplied, uncalibrated abstraction |
| Depth-limited target boundary | Abstract; `official_engine_executed=False` |
| Phase 6K cutoff scalar | Uncalibrated approximation, confidence zero |
| `[-1, 1]` leaf interval | Full utility range, not a useful counterfactual bound |
| Persisted regret/strategy samples | Abstract learning data |

Because the live leaves are uncalibrated, Phase 6R does not train a replacement
blueprint and does not alter the Phase 6Q rejection. Counterfactual bounds, safe
resolving, exploitation, official strategy control, packaging, and Kaggle
submission remain disabled.

## Reproduce

```powershell
python scripts/phase6_native_depth_limited.py
python -m unittest tests.test_depth_limited_game tests.test_native_depth_limited -v
```

Durable outputs are under `artifacts/phase6_native_depth_limited/`. The live
native runtime is not claimed byte-deterministic; fingerprints bind the exact
observed run.

## Next admissible step

Phase 6AG now supplies a held-out full-game terminal-outcome point leaf without
selected hidden-truth features. Its point MSE beats both preregistered baselines,
but its progress-stratified intervals fail coverage and width, so it does not
replace this confidence-zero cutoff with an admissible expected-value bound.

The next step is repeated independent native continuation from fresh
implementation-heldout root/action/hypothesis families. Estimate conditional
expected values before calibrating intervals, verify coverage and abstraction
error, and only then reconsider the Phase 6Q counterfactual-value gates. Until
those gates pass, these bounded-horizon samples are diagnostic and must not feed
a claimed safe resolver.
