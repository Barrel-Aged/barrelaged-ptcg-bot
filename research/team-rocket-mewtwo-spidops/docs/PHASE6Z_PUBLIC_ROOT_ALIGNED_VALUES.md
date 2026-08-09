# Phase 6Z: public-root-aligned conditional action values

Phase 6Z repairs the public-root coverage defect exposed by Phase 6Y. It
captures later player-1 decisions in genuine official battles, freezes each
public root, samples several hidden allocations behind that same observation,
forces every abstract turn plan, and follows each branch to a native terminal
outcome under the frozen Phase 6P average-plan blueprint.

This is a collection and modeling milestone, not a safe-resolving milestone.
The fitted outputs are hypothesis-conditioned conditional rollout values—not
official posterior-weighted counterfactual values.

## Experimental design

The collection is declared before fitting:

| Split | Live baseline style | Public roots | Hidden allocations per root |
|---|---|---:|---:|
| Train | tempo | 6 | 3 |
| Calibration | attack rush | 6 | 3 |
| Validation | control | 6 | 3 |

Each split has one root for each of the six native candidate profiles. A root
is captured at player 1's second distinct MAIN decision. Complete roots—not
individual particles or action rows—are assigned to splits. The run rejects
duplicate root identities, any public mismatch among a root's allocations,
any hidden-allocation collision, and any completion that needs a public-card
substitution.

For every allocation the experiment:

1. exposes the five Phase 6O plan actions at the shared information set;
2. records the frozen Phase 6P blueprint probability for each action;
3. forces each plan through the fixed tactical lifting policy;
4. uses common rollout seeds across actions;
5. executes three complete continuations per action through native search; and
6. stores only public features, opaque digests, action probabilities, returns,
   and transition accounting.

No raw deck, Prize, hand, search token, observation JSON, or native search ID
is serialized.

## Aligned information-set invariant

The three allocation variants behind a root must have one shared:

- sanitized public-state fingerprint;
- public-history fingerprint;
- public-belief fingerprint;
- abstract information-set identifier; and
- public feature vector.

Their hidden-allocation digests must be distinct. This supplies repeated
private-world outcomes for one public decision instead of Phase 6Y's one-root-
per-particle shape.

The range is still incomplete. An exact agent observation reveals the acting
player's own hand, so that hand cannot be changed while retaining the same
information set. Phase 6Z varies unseen deck/Prize allocation and the opposing
hidden zones; it does not claim full acting-player private-hand support.

## Hypothesis-conditioned model

The diagnostic model is a ridge predictor over:

```text
native public features
+ candidate-hypothesis one-hot
+ forced-plan-action one-hot
```

Only train roots fit the normalizer and weights. Regularization is selected by
leave-one-public-root-out cross-validation inside the train collection. The
calibration collection supplies separate split-conformal residual radii for
each plan action, and the validation collection is untouched until final
metrics are computed.

This identifier is an explicit hypothesis input. It is not smuggled into the
public feature vector and is not described as known truth at deployment.

## Result

The checked-in run contains:

| Quantity | Result |
|---|---:|
| Disjoint later public roots | 18 |
| Hidden-allocation surfaces | 54 |
| Forced plan-action targets | 270 |
| Genuine terminal continuations | 810 |
| Genuine native transitions | 70,931 |
| Validation interval coverage | `75.555556%` |
| Validation RMSE | `0.847985` |
| Zero-baseline validation RMSE | `0.771402` |
| Maximum rollout standard error | `0.666667` |
| Public-root alignment gate | pass |
| Collection-disjoint gate | pass |
| Conditional-model diagnostic gate | **fail** |
| Native counterfactual-value gate | **fail** |
| Safe-resolver / official-control gates | **fail / fail** |

The model gate is only an empirical diagnostic over the supplied fixed-profile
experiment. It fails both coverage (`75.56% < 90%`) and point prediction
(`0.847985` RMSE versus zero's `0.771402`). This rejects the small ridge family
without invalidating the aligned collection substrate. The selected
regularization was `1000.0`; all five action-specific calibration ranks use the
most conservative 18th of 18 residuals, yet held-out coverage still fails.

## Official execution versus abstraction

Genuine official-engine execution:

- each anchor is reached by legal selections in a live official battle;
- all descendant state transitions and terminal outcomes execute through the
  licensed native search API; and
- tactical lifting uses the existing legality boundary.

Research abstractions:

- the hidden-completion particles and their uniform prior;
- the five turn-level plan actions and fixed tactical executor;
- the frozen continuation blueprint and paired sampling protocol;
- the candidate-hypothesis input, ridge predictor, conformal intervals, and
  admission thresholds.

The native transitions are genuine. The particle prior, information-set game,
model, and safety interpretation are not official engine semantics.

## Why downstream gates remain closed

Phase 6Z does not provide `CounterfactualValueBounds` because:

1. the hidden-completion prior is supplied, not learned and calibrated as an
   official public posterior;
2. the acting player's observed hand is fixed within each public root;
3. the root collection omits upstream official chance and opponent reach; and
4. the fitted conditional predictor is not an equilibrium-trained
   counterfactual-value provider.

Consequently it cannot enter the safe resolver, certify an exploitation
overlay, or control official actions. It is excluded from the competition
archive and no Kaggle submission or Pokémon-strength claim is made.

## Reproduction

```powershell
python scripts/phase6_aligned_counterfactual_values.py
```

Artifact-only validation:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_aligned_counterfactual_values.py" -v
```

The competition runtime does not expose match seeding, so independent native
runs are not claimed byte-deterministic. Durable fingerprints and file hashes
for the checked-in run are recorded below after generation.

## Durable artifacts

Evidence fingerprint: `762d6d1316c13486ac073664d9e3150139f65f2efc26f324f1f62810385892f0`  
Dataset fingerprint: `b1dfb9f067ba751b86599d882fc884bddf3ac4e19b0541c770acce980706e6f1`  
Model fingerprint: `a6edc06568638e9ed9648351f70d338f8e8680405b649cc82eaaafe959179b86`  
Calibration fingerprint: `808d3c7e887e6c9adff11bc0ae5b448a9b70cc2e58c2e364bd6416734cdbfc9c`

| File | Bytes | SHA-256 |
|---|---:|---|
| `public_root_aligned_value_dataset.json` | 835,672 | `1ce2a565f231193e2fe3bf64621ed98b533d8f99b84567c8be1b1ff1286d02ab` |
| `hypothesis_conditioned_value_model.json` | 19,361 | `46fe352b80d97df1557dd0e802cb12c97de95b121143aa1a6ed40f6f5767681f` |
| `aligned_value_calibration.json` | 3,328 | `306362d250d77d2d787ad06fe0664b3da4d60dd692391e4c1e3746c4b705c423` |
| `phase6z_evidence.json` | 1,918 | `c3786685739c7ab441fe611c5f4d74788cbe36cc311cb14e98e521a0c48c6ffb` |
| `phase6z_evidence.md` | 1,120 | `bf5c6dfb82806d0e755b3b87bcdc6759151dd2fbd315d1d01e0c85e51937bf82` |

Phase 6AA subsequently finds public deck-archetype signal on these frozen
roots, but it does not calibrate hidden hands/resources/intents or a production
observer adapter. Phase 6AB subsequently captures genuine behavior prefixes,
but the live API exposes no chance-event likelihoods and the new prefixes are
not jointly bound to these value roots. Phase 6AC then validates a fresh joint
prefix/truth/value collector with exact root and next-draw replay parity. Its
common target/sampling chance law cancels symbolically, but every information
set has only one realized truth sample and absolute chance reach remains
unavailable. The next admissible step is therefore repeated same-information-
set truth sampling, a sufficiently large calibrated public hidden-state range,
broader acting-player private support, lower-noise continuation targets, and a
counterfactual value provider whose bounds cover completely disjoint public-
root families. Only that object can be considered for the
existing safe re-solving interface.
