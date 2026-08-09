# Phase 6AD: observer-oriented official hidden-card range

Phase 6AD scales the Phase 6AC truth path into a split-isolated range experiment.
It collects 180 fresh official roots, views each root from player 0 (the Rocket
Mewtwo seat), and uses the realized player-1 hidden state only as transient
supervision. The checked result passes the preregistered **in-scope** candidate
and hidden-card calibration gates. It does not calibrate opponent intent, a full
official public belief, counterfactual values, a safe resolver, exploitation, or
official strategy control.

Status: `in_scope_hidden_card_range_calibrated_full_belief_fail_closed`  
Evidence fingerprint: `021a3dda44cd198cf5df7d1fa52d48094f900255e1d88204c34d9c346f932dc9`

## Scope and collection design

The collection boundary is player 1's second MAIN turn. The live procedural
prompt therefore belongs to player 1, while the range is deliberately oriented
to player 0. `native_observer_public_feature_items` makes that distinction
explicit and reads neither player's private hand identities.

The six opponent candidates are `abomasnow`, `alakazam`, `crustle`, `dragapult`,
`rocket_mewtwo`, and `starmie`. Each split contains ten independent roots per
candidate:

| Split | Roots | Candidate strata | Roots per candidate |
|---|---:|---:|---:|
| Train | 60 | 6 | 10 |
| Calibration | 60 | 6 | 10 |
| Validation | 60 | 6 | 10 |
| **Total** | **180** | **6** | **30** |

The roots contain 5,178 genuine `battle_select` executions. Their complete
redacted behavior prefixes retain the Phase 6AB/6AC public-state and public-
history chains. The native runtime does not expose a match seed, so a new live
run is not claimed byte-deterministic.

## Genuine execution versus research abstractions

| Component | Provenance |
|---|---|
| Battle starts, official selections, rule advancement, and realized hidden states | Genuine official-engine execution |
| Prefix public-state/history chain and live/visualizer parity checks | Derived directly from genuine official observations |
| Candidate-deck catalog and player-0 observer adapter | Research declaration/adapter |
| 32-bucket hand, Prize, and deck labels | Privacy-preserving abstraction of transient official truth |
| Candidate prototypes and calibration grid | Offline statistical abstraction |
| Public-consistent hidden-zone allocation particles | Supplied candidate-deck prior and uniform-allocation abstraction |
| `OpponentRange` interpretation and admission gates | Research interface and evaluation decision |

The dataset sets `official_battle_roots_executed=true` and
`model_transformation_official_engine_executed=false`. Deserializers and tamper
tests reject promotion of the model, range interpretation, or downstream
capabilities to official-engine provenance.

## Leakage and hidden-resource representation

Only the player-0-oriented public feature vector enters the model. The true
candidate identifier and realized resource counts are supervision fields, not
features. Raw private card IDs are not serialized.

Each hidden state is represented by 96 counts:

- 32 SHA-256-stable card buckets for the opponent hand;
- 32 buckets for opponent Prize cards; and
- 32 buckets for the remaining opponent deck.

For every root and every candidate, 32 public-consistent allocations estimate
expected bucket counts and bucket-presence probabilities. This produces 34,560
allocation particles. Publicly impossible candidate decks retain explicit
evidence-adjustment counts. The realized candidate requires zero adjustment at
every checked root.

The allocation prior is not a learned posterior over card order. It is a
candidate-conditioned marginal approximation whose calibration is measured only
for this six-deck, baseline-policy, later-root distribution.

## Public-only range model

`OfficialPublicRangeModel` fits one normalized public-feature prototype per
candidate using train roots only. The calibration split selects temperature,
uniform shrinkage, and a penalty for candidate decks that conflict with public
evidence. The selected configuration is:

| Hyperparameter | Selected value |
|---|---:|
| Temperature | `3.0` |
| Uniform mixture | `0.0` |
| Public-evidence adjustment penalty | `16.0` |

Prediction consumes only the public feature vector and the public-consistent
candidate summaries. It emits the existing normalized `OpponentRange` contract;
each hypothesis carries 96 rounded expected resource counts and retains
`intent_id="unknown"`.

## Held-out results

The minimum calibration and validation sizes were fixed at 60 roots each. The
candidate ECE ceiling was `0.15`, and the resource-presence ECE ceiling was
`0.10`.

| Metric | Calibration | Validation | Reference / limit |
|---|---:|---:|---:|
| Candidate NLL | `0.000000026` | `0.000000014` | Uniform `1.791759` |
| Candidate Brier | `0.000000000000024` | `0.000000000000015` | Uniform `0.833333` |
| Candidate top-1 accuracy | `1.000000` | `1.000000` | — |
| Candidate ECE | `0.000000026` | `0.000000014` | At most `0.15` |
| Hidden-resource presence Brier | `0.043324` | `0.038424` | Uniform-candidate `0.105863` / `0.100017` |
| Oracle-candidate resource Brier | `0.043324` | `0.038424` | Diagnostic lower reference |
| Hidden-resource ECE | `0.010059` | `0.007916` | At most `0.10` |
| Hidden-resource count MAE | `0.179774` | `0.173372` | Diagnostic |

Candidate separation is nearly perfect because the six exact declared deck
lists have strongly distinguishing public card evidence at this collection
boundary. That result must not be extrapolated to unknown lists, variants,
different policies, earlier turns, intent, or tournament strength.

## Admission result

| Gate | Result |
|---|---:|
| Root-disjoint train/calibration/validation | **pass** |
| At least 60 calibration roots | **pass** |
| At least 60 validation roots | **pass** |
| Explicit player-0 observer orientation | **pass** |
| Held-out six-candidate calibration | **pass** |
| Held-out hidden-resource bucket calibration | **pass** |
| In-scope official hidden-card range | **pass** |
| Opponent intent range | **false** |
| Full official public belief | **false** |
| Counterfactual-value compatibility | **false** |
| Safe resolver compatibility | **false** |
| Opponent exploitation | **false** |
| Official strategy control | **false** |

Passing the in-scope gate makes this provider a useful input candidate for the
next value experiment. It does not satisfy the broader `PublicBeliefState`
contract by itself and does not authorize any action-selection path.

## Next admissible work

The next stage must retain this observer-oriented range while collecting or
estimating opponent intent and opponent-reach-weighted values behind shared
public abstractions. It must then calibrate counterfactual-value intervals on
collection-disjoint roots. Only empirically covered bounds may enter the
existing safe resolver, and the resolver must still return the baseline
blueprint exactly on missing coverage, OOD input, interruption, or failure.

Phase 6AE now completes only the first controlled diagnostic in that sequence:
it identifies six seat-decoupled declared policy proxies in scope. Those labels
are not human latent intent, and opponent-reach-weighted aligned values remain
unavailable. See `docs/PHASE6AE_OFFICIAL_INTENT_PROXY.md`.
Phase 6AF additionally holds authored policy implementation out, but preserves
the same latent-intent and value limitations.

Absolute official chance reach is still undisclosed. Phase 6AD does not repair
that Phase 6AB/6AC limitation, train an equilibrium blueprint, or validate the
five-plan continuation abstraction.

## Durable artifacts and reproduction

| File | Bytes | SHA-256 |
|---|---:|---|
| `official_public_range_model.json` | 158,316 | `f3d3ca1d8a089bf2e4df01ad1b92d5b6134bcf28704320a35f39add2992bb2fb` |
| `official_range_calibration.json` | 1,956,529 | `77c516398c0ed30b5e4dcded0ac5b10a3d3a842fe1051f6c617152ba6ba0ba74` |
| `official_range_truth_dataset.json` | 28,176,568 | `73f04a48122e312749842b0c25ba239a534f684edcffe88e5bdb299b8834e1c0` |
| `phase6ad_evidence.json` | 2,469 | `2a5faafd43bc474848f5e4fc64758fac6727851a93ea1f6ba3b4d7d36637ab0b` |
| `phase6ad_evidence.md` | 1,595 | `3305f5328989619c96c7868c1bd1c6e8d96fee732700a41b4e2f856c1afe068b` |

Recollect from the staged official runtime with:

```powershell
python scripts/phase6_official_range_calibration.py
```

Artifact-only validation:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_official_range_calibration -v
```

No Kaggle submission was made, and no performance or superhuman claim follows
from this diagnostic.
