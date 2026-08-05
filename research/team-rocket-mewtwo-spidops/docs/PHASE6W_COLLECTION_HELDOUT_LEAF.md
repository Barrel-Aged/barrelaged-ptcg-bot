# Phase 6W: collection-heldout public leaf audit

Phase 6W tests whether Phase 6V's point-estimate improvement survives a stronger
split. Training, conformal calibration, and validation each use a complete,
identity-disjoint 24-particle native collection. The entire validation
collection is generated only after the model and calibration protocol are
frozen.

## Three-collection protocol

| Role | Source | Rows | Use |
|---|---|---:|---|
| Training A | Phase 6U native collection | 24 | Ridge fitting and leave-one-candidate-out regularization selection |
| Calibration B | Phase 6V fresh native collection | 24 | Conformal radius only |
| Validation C | New Phase 6W native collection | 24 | Untouched final scoring only |

Every collection contains variants 0-3 for all six candidate decks. Pairwise
overlap is zero across hidden-particle fingerprints, cutoff-state identifiers,
and public-history fingerprints. The training normalizer sees collection A
only. Leave-one-candidate-out cross-validation selects `lambda=0.1`; collection
B cannot influence that choice or the fitted weights.

Using 24 calibration rows raises the 90% conformal rank to 23, so the radius no
longer has to equal the single largest residual. Collection C supplies 240 new
terminal continuations and 27,906 genuine official-native transitions.

## Untouched result

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Collection-trained public ridge | `0.482329` | `0.595137` |
| Phase 6I logged value | `1.254123` | `1.322351` |
| Phase 6K tactical value | `0.662991` | `0.765785` |
| Zero baseline | `0.400000` | `0.493288` |

| Calibration diagnostic | Result |
|---|---:|
| Calibration rows / conformal rank | `24 / 23` |
| Conformal radius | `1.047599` |
| Calibration coverage | `95.833%` |
| Untouched validation coverage | `91.667%` |
| Mean clipped interval width | `1.721623` of `2.0` |
| Informative interval fraction | `87.5%` |
| Maximum rollout standard error | `0.333333` |
| Beats best reference RMSE | `false` |
| Empirical fixed-profile leaf gate | `false` |

Coverage and rollout-noise thresholds pass. The leaf fails both predictive
improvement and interval-width criteria: its RMSE is worse than the zero
baseline and its mean width exceeds the predeclared `1.25` limit. This stronger
root-family holdout shows that Phase 6V's six-row point improvement did not
generalize.

## Official execution versus abstraction

Genuine official-engine evidence consists of the initial observations,
descendant transitions, and terminal game outcomes produced by the licensed
runtime. In particular, all 27,906 validation transitions and 240 validation
continuations are native executions.

The hidden completions, collection roles, cutoff sampler, fixed rollout profile,
public feature representation, ridge weights, empirical return target,
conformal interval, and every gate are research abstractions. The model artifact
therefore records `official_engine_executed=false` even though its datasets
contain native descendants.

Phase 6W estimates one fixed profile's empirical return. It does not produce
opponent counterfactual values, calibrate the public belief, establish
equilibrium convergence, certify a safe resolver, enable an exploitation
overlay, control the official agent, or demonstrate Pokémon strength.

## Consequence

The public ridge remains useful only as an ablation. The next model should not
be promoted by adding more variants of the same single-vector regression.
Future work needs many more independent public histories, causal range inputs,
and hypothesis-conditioned opponent value targets. Counterfactual calibration
must then occur on untouched collection families before the existing safe
resolver interface can be considered.

Phase 6X supplies the first per-hypothesis, per-opponent-plan conditional
targets and validates their action sensitivity and sampling precision. It still
lacks the reach correction required for counterfactual values. See
`docs/PHASE6X_OPPONENT_DEVIATION_VALUES.md`.

No package, agent policy, benchmark result, or Kaggle submission was changed.

## Durable artifacts

Evidence fingerprint:
`ab42c69fce00c0a5cc4b1da98b62160fd258c1b2cf5f34114ed22c9807be1b78`  
Model fingerprint:
`a542c12f89fdf55ebff08a45ab45071a83ac87b66faa324d4b010572fdd2815d`  
Calibration fingerprint:
`6800f0d9d0f6ba95606325d1082253e694f6d8aff37cd6df29680796d5331d6e`  
Validation dataset fingerprint:
`f8cba7d2ce3d4b3affcade967623c76acfde8528e678fcd03732d9cbc4b1eece`

| File | Bytes | SHA-256 |
|---|---:|---|
| `fresh_validation_leaf_dataset.json` | 420,371 | `ec797e3bbbe3a906468e3fad34139bf97dcafd0cc6a5213615e6dc00d02dfffc` |
| `root_heldout_public_leaf.json` | 13,211 | `b3261da838c73e7929a42c12da7ed3dcf4de970776ea30e5e942fb22f635a3e4` |
| `root_heldout_leaf_calibration.json` | 6,349 | `4ae66043e8a3e21a6b0215e12aacfe18a5724dca5dc7b40599afaa431710fe7d` |
| `phase6w_evidence.json` | 2,431 | `90289d5346882e01dfb338c84ecaa648c87c57e237be194fba7cb4d86fe1c520` |
| `phase6w_evidence.md` | 1,045 | `1aae1a14223dbcc4e6b0fc2998b0e83b32bf23155adb8930e6a445a5dd624bcd` |

Reproduce the expensive validation collection with:

```powershell
python scripts/phase6_root_heldout_leaf.py
```

Run artifact-only regression checks with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_root_heldout_leaf.py" -v
```

The competition runtime is unseeded, so independent executions are not claimed
byte-deterministic. The fingerprints above bind this observed run.
