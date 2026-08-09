# Phase 6AC: joint official truth/value samples

Phase 6AC closes the root-alignment gap left by Phase 6AB without promoting the
result to a calibrated counterfactual-value model. For each collection stratum,
one live official battle now supplies all three parts of a single record:

1. the complete redacted player-action prefix to player 1's second MAIN turn;
2. a transient realized hidden-state sample from that exact official root; and
3. terminal rollout values for all five abstract turn plans from a native state
   reinitialized with that sample.

The same-root join is genuine and replay-checked. The five plans, frozen
tactical executor, average-plan continuation policy, and empirical rollout
values remain research abstractions. The collector is production-disabled and
does not control the competition agent.

## Collection grid

The checked-in live run contains one root for every split/candidate pair:

| Split | Baseline style | Candidate roots |
|---|---|---:|
| Train | `tempo` | 6 |
| Calibration | `attack_rush` | 6 |
| Validation | `control` | 6 |

The candidate strata are Abomasnow, Alakazam, Crustle, Dragapult, Rocket
Mewtwo, and Starmie. These labels are collection conditions, not predictions
from an official opponent-range model.

Each root branches the five plan actions `board_setup`, `energy_engine`,
`prize_push`, `recovery`, and `control`. Every action receives three terminal
continuations under the frozen Phase 6P average-plan blueprint. The tactical
executor is restored from the Phase 6K checkpoint. Both checkpoint
fingerprints, the Phase 6Z split/candidate metadata, the runtime fingerprint,
and the compatibility-deck fingerprint are bound into the dataset.

## Exact root and next-step fidelity

The collector reads the native visualizer only inside the live-root callback.
It converts the realized hidden zones into a transient `HiddenZoneCompletion`
and immediately initializes an `OfficialNativeSearchGame` at that exact root.
Raw truth is never passed to the policy and no raw private card identity is
serialized.

Four checks distinguish a plausible reconstruction from an exact one:

- visualizer public state matches the live root;
- the visualizer's acting-player hand/deck/Prize zones match the live root;
- the reconstructed native root matches the complete official public and
  private observation; and
- a unique legal `END` selection is executed in both live and reconstructed
  games, after which the complete next public and private observations match.

The forced `END` probe deliberately avoids a card effect that reshuffles a
deck. It advances through the next mandatory draw, so matching the subsequent
private observation validates future hidden deck order as well as root counts.
All 18 roots pass every check.

The visualizer's prompt metadata is frame-shifted: its displayed `selected`
field belongs to the following visual frame. Phase 6AC therefore does not use
that field for root identity. It uses the live observation for the information
set and compares only visualizer public state plus the acting player's private
zones at the visualizer boundary. Full public/private equality is checked after
native reinitialization against the live official observation.

## Common opaque chance law

The official adapter executes shuffle, Prize placement, draw, and coin events
but does not report their probabilities. Phase 6AC therefore keeps absolute
chance reach and full sampling reach unavailable.

There is nevertheless one auditable cancellation. For target player `i`, an
outcome-sampled counterfactual contribution has likelihood ratio

```text
    pi_c(h) * pi_-i(h)
    -------------------
    q_c(h) * q_0(h) * q_1(h)
```

When target and sampling use the same official chance law pointwise,
`pi_c(h) == q_c(h)` and the opaque chance factors cancel. The retained ratio is
`pi_-i(h) / (q_0(h) * q_1(h))`. The checked-in deterministic behavior/target
policies match, so every retained ratio is `1.0`.

`CommonChanceLawReachRatio` records the shared-law fingerprint, both player
factors, the derived ratio, and an explicit `absolute_chance_reach: null`.
Deserialization rejects any attempt to promote that symbolic cancellation into
an absolute chance-reach claim. This ratio can support a sampled contribution;
it is not itself a counterfactual value, reach mass, confidence interval, or
safety bound.

## Result

| Metric | Result |
|---|---:|
| Same-root live official roots | 18 |
| Candidate strata | 6 |
| Redacted prefix selections | 568 |
| Forced next-step fidelity selections | 18 |
| Total genuine live selections | 586 |
| Five-plan action targets | 90 |
| Terminal continuations | 270 |
| Genuine native descendant transitions | 29,460 |
| Macro transitions | 5,418 |
| Maximum rollout standard error | `0.666667` |
| Nontrivial action-surface fraction | `0.833333` |
| Maximum truth samples per information set | 1 |
| Same-root prefix/truth/value gate | **pass** |
| Official truth-sample gate | **pass** |
| Root/next-step replay-fidelity gate | **pass** |
| Common-chance likelihood ratio available | **true** |
| Absolute chance reach available | **false** |
| Calibrated counterfactual values | **false** |
| Safe resolver / exploitation / official control | **false / false / false** |

Dataset fingerprint:
`2b6f619be9ba52f7394113a2b728a14bde6dc5aabd12f2cd840290175faa19d5`  
Evidence fingerprint:
`ddf1c9b79b2e0457405218aaada8cdf94f7daa6be6de370a3d0cedd8793d3702`

## Official execution versus abstraction

Genuine official/native execution:

- all 18 battle starts and 586 live selections;
- realized official hidden-state samples at the captured roots;
- native root and next-step public/private replay parity; and
- all 29,460 licensed native descendant transitions and terminal outcomes.

Research declarations and abstractions:

- the train/calibration/validation strata and candidate labels;
- redaction, fingerprints, and the reconstructed information-set schema;
- the assumption that target and sampling refer to the same opaque official
  chance law;
- the five turn-plan action abstraction;
- the frozen tactical and continuation policies; and
- empirical three-rollout action values and all admission gates.

The native search object is still an abstract game boundary initialized from a
realized official sample. Genuine descendant execution does not make the plan
abstraction, policy, probability model, or derived values official.

## Why the result stays fail-closed

Every public information set has only one realized truth sample. Phase 6AC can
therefore demonstrate a joined data path, but it cannot estimate or calibrate a
hidden-state range at an information set. The three rollouts per plan also have
maximum standard error `0.666667`, and the fixed continuation profile is not an
equilibrium continuation. Absolute chance reach remains undisclosed even
though a shared chance law cancels in the on-policy likelihood ratio.

The following remain disabled and machine-enforced:

1. official hidden-state belief calibration;
2. calibrated native counterfactual values and bounds;
3. safe subgame or endgame re-solving;
4. opponent-model exploitation;
5. official competition strategy control; and
6. any performance, leaderboard, or superhuman claim.

Phase 6AD completes the first range-specific part of that path with 180
independent official truth roots, an explicit player-0 observer orientation,
and collection-disjoint candidate/resource calibration. Its six-list hidden-card
range passes only in the declared baseline-policy, later-root distribution.
Complete opponent/behavior reach, opponent intent, collection-disjoint
counterfactual-value calibration, and materially lower-noise continuation
targets are still required. Only after empirical bound coverage passes may the
existing safe-resolver interface be considered. See
`docs/PHASE6AD_OFFICIAL_HIDDEN_CARD_RANGE.md`.

## Durable artifacts and reproduction

| File | Bytes | SHA-256 |
|---|---:|---|
| `joint_official_value_dataset.json` | 966,066 | `bda23d21c3f71db52670d64dc8e88037240b7d438d678f3966feb9827b304558` |
| `phase6ac_evidence.json` | 1,820 | `e2c0f44dd008339db9119263942e6196c43e186035fea73446af94f45d3bd576` |
| `phase6ac_evidence.md` | 1,456 | `1adcd8a162a912f5f49704fd5992efe22642ff0edb61b5b964cbcc9ba08e01e1` |

Recollect from the live official runtime with:

```powershell
python scripts/phase6_joint_official_values.py
```

Artifact-only validation:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_joint_official_values tests.test_counterfactual_reach -v
```

The competition runtime exposes no match seed, so independent live runs are not
claimed byte-deterministic. No Kaggle submission was made.
