# Phase 6V: purpose-trained public native leaf

Phase 6V tests whether the Phase 6U failure was specific to reusing value heads
trained for other jobs. It fits a small ridge model directly to low-noise native
leaf returns, uses public features only, freezes every training choice before a
fresh official-native holdout, and keeps the result outside strategy control.

## Leakage-safe protocol

The checked Phase 6U dataset is the only training source:

- variants 0 and 1 provide 12 training rows across all six candidate decks;
- leave-one-candidate-out cross-validation chooses among ridge penalties
  `0.1`, `1`, `10`, `100`, and `1000`;
- each fold fits its normalizer on that fold's training rows only;
- the winning penalty is `1000`, after which the final normalizer and weights
  are fit on the 12 declared training rows;
- Phase 6U variants 2 and 3 are not used for model selection, calibration, or
  reported Phase 6V validation.

A separate native run then creates 24 new hidden-allocation particles and 240
terminal continuations. Its variants 2 and 3 supply six calibration and six
validation rows respectively. The original and fresh particle families have no
overlap. Fresh variants 0 and 1 are retained for audit but are not scored.

Candidate identity, selected hidden truth, private feature vectors, and future
reveals are not model inputs. Candidate labels exist only to define the
leave-one-candidate-out training folds.

## Fresh validation result

The purpose-trained public model improves point prediction on the six untouched
validation allocations, but its uncertainty interval is still effectively
vacuous:

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Public ridge leaf | `0.338766` | `0.446013` |
| Phase 6I logged value | `1.176184` | `1.263804` |
| Phase 6K tactical value | `0.633707` | `0.684181` |
| Zero baseline | `0.400000` | `0.489898` |

| Calibration diagnostic | Result |
|---|---:|
| Fresh terminal continuations | `240` |
| Fresh genuine native transitions | `28,324` |
| Maximum rollout standard error | `0.333333` |
| 90% conformal radius | `1.170885` |
| Validation coverage | `100%` |
| Mean clipped interval width | `1.997117` of `2.0` |
| Informative interval fraction | `50%` |
| Beats every reference RMSE | `true` |
| Empirical fixed-profile leaf gate | `false` |

The empirical gate requires fresh-root coverage, a mean interval width no more
than `1.25`, rollout standard error no more than `0.35`, disjoint allocations,
and improvement over every reference predictor. Width is the failing criterion.
The point-estimate improvement is therefore a useful research signal, not a
deployable leaf-value result.

## Execution and abstraction boundary

What is genuine official-engine execution:

- the fresh initial observation is produced by the licensed competition
  runtime;
- every descendant transition used to reach a cutoff or terminal state is
  executed by that runtime;
- all 240 terminal continuations and 28,324 recorded native transitions are
  runtime executions.

What remains an abstraction:

- hidden-zone completions and their candidate supervision labels;
- cutoff sampling and the fixed rollout policy profile;
- the public feature bridge, ridge model, empirical target, and conformal
  interval;
- every comparison, gate, and interpretation in this document.

The ridge artifact correctly records `official_engine_executed=false`: the data
contains native descendants, while the fitted scalar itself is not the official
engine. It estimates one fixed policy profile's empirical return. It does not
estimate opponent counterfactual values, calibrate a public belief posterior,
certify a resolver, enable an exploitation overlay, or control official play.

## Fail-closed consequence and next step

Phase 6V does not replace the packaged heuristic baseline and is not included in
the Kaggle archive. No submission was attempted. The next value experiment
needs substantially more independent roots and allocation families, an explicit
hypothesis-conditioned opponent counterfactual target, and enough untouched
calibration examples to obtain informative bounds. The causal revealed-card
range from Phase 6T remains the stronger belief baseline until a learned public
range beats it on genuinely independent native histories.

Phase 6W subsequently assigns complete native collections to training,
calibration, and untouched validation. The public ridge loses to zero on that
third collection and still has an over-wide interval, so the Phase 6V point
gain is not treated as generalizing evidence. See
`docs/PHASE6W_COLLECTION_HELDOUT_LEAF.md`.

## Durable artifacts

Evidence fingerprint:
`d7d5af2fe343cd0ba73da8f601d81850ed80212a7f43a41b41eb99929710a900`  
Training dataset fingerprint:
`4ed11fa80126e65a6637f9e5d2d333a21dc9087824d99e8fca7b02956ef12d27`  
Fresh holdout dataset fingerprint:
`82d4ca97dd63641248cb6b95715d89f884e9fa74740159d6746a7d06c9a0edf1`  
Model fingerprint:
`39cd27dc8a9e0209972d6475510b0cfa20bb110c55ea4ae2c6e87a01dcc2cd4a`  
Calibration fingerprint:
`37b3c674a5ffc8211ce4c9f401b768c4d50a8c10948b7aa2e0dc5e583a2eed21`

| File | Bytes | SHA-256 |
|---|---:|---|
| `fresh_native_leaf_dataset.json` | 420,315 | `227c44e254da1577b9c0344d75e30e38fea7289d1b21ba2e4da71c854028e394` |
| `public_ridge_leaf.json` | 12,218 | `85eee364b043d40934f790bda0054047aedd490f5bd4b8815f12ced4e046f717` |
| `public_leaf_calibration.json` | 3,272 | `5423d4246dcd1c3965eef6e4fbc436ece65dd41e0db68977adce6793e95b3f24` |
| `phase6v_evidence.json` | 2,208 | `23f8761e12fe0e0a9b0ef1f32f8e1897fecf7d2c03ed32eaf5b9c1ae4cd6f52c` |
| `phase6v_evidence.md` | 1,090 | `939874f9ea004aae2da6eb35dd53c68db86038ab19cfbe21e436bc1561b06d4f` |

Reproduce the expensive native run with:

```powershell
python scripts/phase6_native_public_leaf.py
```

Run the artifact-only regression checks with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_native_public_leaf.py" -v
```

The competition runtime is unseeded, so independent executions are not claimed
byte-deterministic. The fingerprints above bind this observed run.
