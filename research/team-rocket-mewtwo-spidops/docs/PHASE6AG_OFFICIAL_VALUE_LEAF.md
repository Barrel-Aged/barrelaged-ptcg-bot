# Phase 6AG: implementation-heldout official full-game value leaf

Phase 6AG replaces the invalid Phase 6R cutoff heuristic with an offline neural
point-value leaf trained on complete official-engine games. The neural point
estimate passes its preregistered untouched-implementation baseline gate. Its
terminal-outcome conformal intervals do not pass coverage or width gates, so
the value leaf is **not** admitted for counterfactual bounds, resolving,
exploitation, official strategy control, packaging, or submission.

Status:
`implementation_heldout_official_point_leaf_validated_intervals_rejected_downstream_fail_closed`

Evidence fingerprint:
`2112d91fdb5fe6adfbaee72b86346846e87806818368f6054919fd3ecaf97673`

## Dependency addressed

Phase 6R showed that bounded-horizon native search reached uncalibrated
confidence-zero leaves. Phases 6S-6W then rejected several progressively more
careful native leaf families. Phase 6AG tests a different substrate:

1. finish complete games in the licensed official runtime;
2. convert each truth replay into two isolated observer perspectives;
3. train a neural value point estimate on public state/history, the acting
   player's own private observation, and the public candidate range;
4. hold a complete opponent-policy implementation out until validation; and
5. keep all counterfactual, equilibrium, safety, control, and performance
   capabilities disabled unless their separate gates pass.

The learned target is the realized on-policy terminal utility in `{-1, 0, 1}`.
It is not a counterfactual value and is not conditioned on an alternative
action surface.

## Preregistered collection and split protocol

The panel crosses six exact opponent candidates, six controlled intent proxy
families, four split-isolated opponent-policy implementations, and two replicas
per stratum:

| Role | Player-1 implementation | Matches | Permitted use |
|---|---|---:|---|
| Train | `canonical`, `engine_first` | 144 | Normalizer and model fitting |
| Calibration selection | half of `hand_first` | 36 | Architecture/blend selection |
| Calibration bounds | other half of `hand_first` | 36 | Conformal radii only |
| Untouched validation | `action_first` | 72 | Final point and interval gates only |

Player 0 uses the same tempo policy in every match. Player 1 alone varies by
candidate, declared intent proxy, implementation, and tie-break direction.
Each candidate/intent calibration stratum contributes exactly one replica to
selection and one to bound calibration using
`(replica + candidate_index + intent_index) mod 2`. This prevents replica
identity from being confounded with calibration role.

The resulting frozen corpus contains:

- 288 complete official matches;
- 576 perspective-isolated episodes;
- 30,668 leakage-safe decision steps;
- 30,668 genuine live legal selections; and
- 144 / 72 / 72 train, calibration, and validation matches.

The official runtime is not seedable. These files fingerprint this completed
collection; recollection is not claimed byte-deterministic.

## Inputs and neural model

The model has 208 named inputs:

| Input family | Width | Boundary |
|---|---:|---|
| Public state and causal history | 127 | Public state scalars plus cumulative coarse MAIN-action counts |
| Acting-player private observation | 75 | Only that observer's visible hand/search information |
| Public opponent range | 6 | Candidate probabilities, never the realized candidate label |

Policy implementation ID, declared intent label, realized candidate label,
truth resource counts, terminal outcome, raw selection indices, and opponent
private card identities are not model features.

The selected deterministic configuration is a two-hidden-layer tanh MLP with
width 32, weight decay `1e-4`, 40 epochs, and a `0.5` neural / `0.5` causal
board-heuristic blend. The grid was width `{32, 64}`, weight decay
`{0, 1e-4}`, and blend `{0.5, 0.75, 1.0}`. Selection reads only the dedicated
half of `hand_first` calibration matches. Training weights each
match-perspective equally rather than letting long games dominate.

## Replay-conversion extension

The original replay converter fingerprint remains
`a6b863b88400b5eb1e66cc3ade167a9036be455208303704bc1fcdadb54ca308`.
Phase 6AG opts into a distinct converter configuration that:

- allow-lists the official public `Coin.head` event exposed by the larger
  panel; and
- emits `CoarseMainAction(playerIndex, optionType)` on the next frame, where
  the official visualizer records the prior prompt's chosen option.

The synthesized event is causal and public. It stores neither card identity,
target identity, private truth, nor a future action. The enhanced trajectory
updater receives its own fingerprint; legacy Phase 6H datasets and converter
fingerprints remain unchanged.

## Point result

The untouched `action_first` implementation was never read during fitting,
selection, or conformal calibration.

| Untouched validation metric | Result |
|---|---:|
| Neural/blended episode MSE | `0.9387949616` |
| Zero baseline episode MSE | `1.0000000000` |
| Causal board-heuristic episode MSE | `1.0199809467` |
| Point gate | **pass** |

The point gate requires strict improvement over both preregistered baselines.
It passes on this controlled implementation holdout. This supports the narrow
claim that the frozen point model contains transferable predictive signal for
this authored official-game panel. It does not establish equilibrium quality,
human-opponent generalization, Kaggle strength, or safe action selection.

## Interval rejection

Ninety-percent split-conformal radii are fitted separately for opening,
midgame, and endgame progress buckets using only the reserved `hand_first`
bound-calibration matches.

| Interval metric | Required | Observed | Result |
|---|---:|---:|---|
| Overall coverage | implied by all buckets `>= 0.90` | `0.8627906977` | fail |
| Overall mean width | `<= 1.90` | `1.9376846364` | fail |
| Opening coverage | `>= 0.90` | `0.8505263158` | fail |
| Midgame coverage | `>= 0.90` | `0.9907834101` | pass |
| Endgame coverage | `>= 0.90` | `0.8587699317` | fail |
| Endgame mean width | `<= 1.50` | `1.7533149472` | fail |

One realized terminal outcome is a noisy label for an underlying expected
value. The observed near-full-range intervals are consistent with substantial
outcome noise and model error; the experiment does not identify their relative
shares. Phase 6AG therefore does not reinterpret these intervals as expected-
value bounds and does not pass them to a resolver.

## Genuine execution versus abstractions

| Component | Classification |
|---|---|
| Battle initialization, prompts, legality, selected options, transitions, terminal results, and truth replay frames | Genuine official `cg` execution |
| Perspective conversion and coarse public-history extraction | Audited research adapter |
| Six candidate ranges and six declared intent policies | Controlled authored abstractions |
| Player-0 tempo policy and four player-1 implementations | Hand-authored collection policies |
| 208 features, MLP, heuristic blend, and conformal buckets | Offline research abstractions |
| Counterfactual-value bounds and equilibrium blueprint | Not established |
| Safe resolver, exploitation overlay, official control, and Kaggle submission | Machine-disabled / not attempted |

Truth replays are converted in memory. Raw truth replays are not persisted.
The frozen trajectory artifact contains perspective-isolated derived rows and
must not be described as direct official observations without this boundary.

## Serialization correction

The larger artifact exposed a floating-point restoration defect in
`OpponentRange.from_dict`: deserializing already-normalized weights normalized
them a second time, introducing tiny weight drift and breaking the enclosing
public-belief fingerprint. Deserialization now preserves serialized normalized
weights exactly while the `OpponentRange` constructor still requires the sum
to be one within tolerance. Tests bind the restored range and public-belief
fingerprints exactly.

## Rejected predecessor retained

The first Phase 6AG snapshot omitted causal opponent behavior history. Its
balanced point model reached episode MSE approximately `0.981961`, beating the
zero baseline but not producing admissible intervals. Its compact manifest,
model, calibration, evidence JSON, and evidence Markdown are retained as
`rejected_v1_*` files. They are historical negative evidence, not an alternate
candidate. The final frozen corpus was recollected after the public-history
schema was defined.

## Checked artifacts

| Artifact | Bytes | File SHA-256 | Semantic fingerprint |
|---|---:|---|---|
| `implementation_heldout_trajectory_dataset.json` | 907,103,751 | `427163a69a223d99198f30360ffbfa258d2360d1948c7363d18b6a8c110bc9c1` | `be1fdfba69e1048101ac5d39f72e98c628f7b474b0c45862e3c70b01063b207b` |
| `official_value_leaf_manifest.json` | 237,880 | `01da39945b3efb3a0d1303633bad6a044b1a23920592054c5231c02901a8d3d4` | `9c92da92ec8696f1daa450547d18e3a225ab06544f69b92075b8ab1a47fd9236` |
| `official_value_leaf_model.json` | 270,665 | `fd4ef5fff140249ef9d0424825fa425bad464a7143149a21c46b28b8cae56f39` | `08523989536022fa6a1447143a587de1749817c62da2923a98f924eae272b763` |
| `official_value_leaf_calibration.json` | 9,864 | `e11f916a1b049dbf76765a153b9f09f07863d9aa9529c183b2d5453f8106f328` | `c8bb41b6d5606a925a0e5ddc53cd1b8f322507c5c1d4dce21779cce6122783c8` |
| `phase6ag_evidence.json` | 2,231 | `543b83cb228679ec51450582285ea7c45065f96f148902209990567322bfda5c` | `2112d91fdb5fe6adfbaee72b86346846e87806818368f6054919fd3ecaf97673` |

The 907 MB dataset is deliberately excluded from routine unit-test parsing.
Compact tests restore and fingerprint-bind the manifest, model, calibration,
and evidence artifacts.

## Next admissible dependency: Phase 6AH

The next experiment should collect repeated independent native continuations
from fresh implementation-heldout information-set roots. Repetitions must
estimate the conditional expected outcome for each root/action/hypothesis,
rather than treating one terminal result as the expectation. Complete root
families must remain isolated across training, calibration, and untouched
validation.

Only if root-relative expected-value intervals meet preregistered coverage and
width gates may Phase 6AH propose a counterfactual-value provider. Even then,
opponent/chance reach alignment, equilibrium training, resolver safety, matched
ablations, official strategy control, and Kaggle submission remain separate
gates.

## Reproduce without recollection

```powershell
$env:PYTHONPATH = "src"
python scripts/phase6_official_value_leaf.py --reuse-source
python -m unittest tests.test_official_value_leaf -v
```

Omitting `--reuse-source` runs a new unseeded official collection. No Phase 6AG
command submits to Kaggle.
