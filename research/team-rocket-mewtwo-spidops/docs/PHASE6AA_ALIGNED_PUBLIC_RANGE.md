# Phase 6AA: aligned public-archetype range diagnostic

Phase 6AA uses the frozen Phase 6Z root collections to test whether public
state alone carries reusable signal about the six candidate deck archetypes.
It produces the project's `OpponentRange` interface, but only as an offline
deck-archetype diagnostic. It is not a hidden-hand/resource/intent posterior
and is not admitted to counterfactual values or safe resolving.

## Split and leakage discipline

The 18 Phase 6Z public roots remain intact:

| Split | Roots | Candidate strata | Use |
|---|---:|---:|---|
| Train | 6 | 6 | Fit normalizer and one prototype per candidate |
| Calibration | 6 | 6 | Select temperature and uniform shrinkage |
| Validation | 6 | 6 | Untouched final diagnostic |

Each example contains only the 111 named native public features and opaque
public/root fingerprints. The true candidate identifier is a supervision or
evaluation label only. It is never included in the feature vector or passed to
`predict()`. No private features or raw card identities are serialized.

## Provider

`PrototypePublicRangeModel` normalizes features using train roots only. Because
there is one train root per class, each normalized row is the corresponding
candidate prototype. Prediction computes negative squared distance to all six
prototypes, applies a softmax temperature, and mixes toward uniform:

```text
q(c | public) = (1 - alpha) softmax(-distance(c) / temperature)
                + alpha / 6
```

The calibration-only search covers 11 temperatures and nine uniform-mixture
weights. It selects temperature `30.0` and uniform mix `0.05` by minimum
calibration NLL. The provider emits six normalized `RangeHypothesis` records
with empty hidden-resource fields and explicit `unknown` intent/style values.

## Result

| Metric | Result |
|---|---:|
| Calibration NLL | `1.550591` |
| Validation NLL | `0.800538` |
| Uniform NLL | `1.791759` |
| Validation Brier | `0.378643` |
| Uniform Brier | `0.833333` |
| Validation top-1 accuracy | `0.833333` (5/6) |
| Validation expected calibration error | `0.476073` |
| Held-out archetype-signal gate | **pass** |
| Minimum calibration roots | 60 required / 6 observed — **fail** |
| Minimum validation roots | 60 required / 6 observed — **fail** |
| Official public range calibrated | **false** |
| Hidden-hand/resource posterior | **false** |
| Counterfactual / safe-resolver compatibility | **false / false** |

The improvement over uniform is evidence that later public board state contains
candidate-archetype signal under these three style-defined collections. It is
not a calibration claim. Six roots per holdout cannot support the required
coverage, ECE is high, and one class error materially moves every reported
rate.

## Official execution versus abstraction

Genuine official-engine provenance:

- the 18 source observations were reached through legal live battles in Phase
  6Z; and
- their public-state fingerprints bind this artifact to those native roots.

Research abstraction:

- conversion from a Phase 6Z root to a target-player archetype example;
- prototype fitting, temperature/mixing selection, and all evaluation metrics;
- the six-archetype support and target-player orientation; and
- every interpretation of the emitted `OpponentRange`.

Phase 6AA executes no new engine transition. The source is native evidence; the
fit and posterior interpretation are offline research artifacts.

## Why the range is not deployable

Four blockers are serialized and machine-enforced:

1. six calibration and six validation roots are below the predeclared 60/60
   minimum;
2. the provider covers deck archetypes but not hidden hands, resources, Prize
   allocations, or strategic intents;
3. Phase 6Z's player-1-relative public feature orientation has no production-
   validated observer adapter; and
4. the supplied hidden-completion prior and upstream official reach are still
   unresolved.

No output enters a counterfactual-value provider, safe resolver, exploitation
overlay, packaged policy, or official action. No Kaggle submission or Pokémon-
strength claim is made.

## Durable artifacts

Evidence fingerprint: `9aebdacbea9530baab3753d07f13374d53389fed4607c9a4f136c8d97d8b4974`  
Dataset fingerprint: `336c28e1d933de72c790dc6a5230fe7644f5ae8bf0c272ae6f0a7894b217baca`  
Model fingerprint: `e7dcf68fd4b4cfa666a8de2c3811e471cef6190f6ccb5422d469a28b53cf4ee7`  
Calibration fingerprint: `6ea1e621786a198a4afe0619481d34ea174513b46bbbbb519a2c8f1cdcde2d5d`

| File | Bytes | SHA-256 |
|---|---:|---|
| `aligned_public_range_dataset.json` | 161,995 | `21eeb8fbc99c1c6e154ef0ca57c3c01e616ba681f56fda9fd3e8b31dd4b65967` |
| `prototype_public_range_model.json` | 44,134 | `fcf9294e9863335f9f6ba627249d819297b83dcbab3767935afff3c7a0015f91` |
| `aligned_public_range_calibration.json` | 20,197 | `2c0abee2feea8b67de1085296da69b6f2a7457e1cd874fae52109dc2c05684c6` |
| `phase6aa_evidence.json` | 2,032 | `21f08098c41cdaaf82e0f1424964150914d7c8941e1c5e6f917f9af1708fd83d` |
| `phase6aa_evidence.md` | 1,067 | `01a8fb13e95d5bd73b239e3dbaf1b6eb8d16e4c3545db54819e8e412aaa94f25` |

Reproduce deterministically from the frozen Phase 6Z artifact with:

```powershell
python scripts/phase6_aligned_public_range.py
```

Run artifact-only tests with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_aligned_public_range.py" -v
```

The next range experiment needs many independently collected public roots,
match-root-disjoint archetype/style/opponent splits, causal public histories,
and an observer-orientation bridge. Hidden-hand/resource hypotheses then need
separate later-reveal calibration before the range can be considered for
counterfactual value learning. Phase 6AB records fresh genuine behavior
prefixes, but its missing chance likelihoods and lack of a joint Phase 6Z value
root mean it does not change this admission boundary. Phase 6AC joins new
prefixes, realized truth, and plan values at the same roots, but one truth sample
per information set cannot calibrate a public range. It therefore leaves this
range admission boundary unchanged.
