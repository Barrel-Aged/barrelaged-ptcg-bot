# Phase 6AM preregistration: public-range normalized counterfactual advantages

Status: **completed; normalized counterfactual-advantage bounds rejected by the frozen precision gate**

## Question

Phase 6AL shows that more replay precision does not make the frozen learned
absolute action-value mean sufficiently calibrated. Phase 6AM therefore does
not fit another post-hoc mean and does not treat sampled rollouts as the final
strategy. It asks a narrower game-theoretic question: can the already admitted
in-scope public opponent range be joined to baseline-relative, public-root
normalized counterfactual **advantage bounds** with held-out coverage?

The strategic baseline remains the frozen native neural turn-plan blueprint.
Repeated native continuations are used only to calibrate and audit a safety
certificate around its five-plan action advantages. They do not select the
production action, establish equilibrium convergence, or control the official
agent.

## Frozen sources and development ledger

The following artifacts are fixed before collection:

- Phase 6AD candidate/resource range model:
  `f4574ce0b5facec0ab2e6e2c08063d575c7f92112a8c109d04e5a81665ec04ca`;
- Phase 6AD evidence:
  `021a3dda44cd198cf5df7d1fa52d48094f900255e1d88204c34d9c346f932dc9`;
- frozen native plan blueprint:
  `b416597d44ad31329acf46bb59a517ffa09a722e4e4e7cf14c79176a38c343d9`;
- frozen tactical blueprint:
  `e9cf893ccf28eeee656af9791de389196d7a869d9bdae090e0e15b7dc5f83232`;
- Phase 6AL evidence:
  `d896278588650ed424bf47870990a9ddcf7dd27594cbc1a1d759af5f84763663`.

All 360 Phase 6AI-6AL value roots are declared development evidence. In
particular, all 72 Phase 6AL roots and their outcomes motivated and calibrate
this protocol; none may be reused as validation evidence.

## Public range and belief boundary

At each fresh player-0 MAIN root, Phase 6AM constructs the player-0-oriented
public feature vector and 32 public-consistent hidden-resource particles for
each of the six declared candidate decks. The frozen Phase 6AD provider emits a
normalized six-hypothesis `OpponentRange`, which is embedded in a serialized
`PublicBeliefState`. Candidate truth is retained only as an audit label and is
never selected as provider input.

This is the admitted Phase 6AD exact-list/resource range on a new, later-root
distribution. It is not a full official hidden-state posterior. The six deck
lists, uniform hidden-zone allocation law, 32-bucket resource representation,
and range interpretation remain research abstractions. Range transfer must pass
again on the new panel.

Opponent intent is deliberately excluded from the safety bounds. Phase 6AF's
declared-policy intent proxy remains available only for a later, separate
exploitation overlay; human latent intent remains uncalibrated.

## Counterfactual semantics

Every collection prefix uses deterministic declared target and opponent
policies, so both recorded player-policy reach factors are one. Target and
sampling use the same opaque official chance law at the same retained public
root. The chance likelihood therefore cancels pointwise in the sampled ratio,
while absolute official chance reach remains unavailable.

For each candidate hypothesis and plan action, the target is player 0's
conditional advantage over the frozen blueprint mixture at that public root:

`A(h, a) = V_0(h, a) - sum_b blueprint(b) * V_0(h, b)`.

The target-player's own upstream reach is excluded; opponent reach is one under
the declared deterministic prefix; and the common positive opaque chance factor
is normalized away. These are public-root conditional counterfactual
advantages, not absolute tournament-reach values.

## Frozen advantage model and calibration

The point center is the existing causal public branch heuristic, centered by
the frozen blueprint distribution at the same root. It consumes only the
pre-action information set, public child deltas, legal plan identity, and
blueprint probabilities. It consumes no future return, true candidate label,
intent label, policy-implementation label, split label, hidden completion, or
raw opponent card identity.

All 72 Phase 6AL development roots provide 360 independent 32-return action
targets. For each target the frozen nonconformity score is:

`abs(centered_branch_prediction - empirical_advantage) + 1.96 * target_SE`.

The finite-sample 97.5% split-conformal rank is
`ceil((360 + 1) * 0.975) = 352`. The resulting frozen global advantage radius
is `0.555019619275336`. No candidate, intent, action, or validation label
selects a radius. Intervals are clipped only to the possible advantage range
`[-2, 2]`; their unclipped width is `1.110039238550672`.

## Fresh held-out panel

The validation namespace is `phase6am-validation`. Abstraction-side replay
batch seeds are `6060` and `1006060`; range-particle sampling uses seed `6061`;
the root bootstrap uses seed `6062` with 10,000 root resamples.

The panel contains one new `action_first` complete match for each cross of six
candidate decks and six declared intent-policy families: 36 official matches.
Each retained root uses two independently seeded replay batches. Each batch has
two independently sampled hidden allocations, all five plan actions, and eight
paired terminal continuations per allocation/action. The panel therefore has:

- 36 public beliefs and candidate ranges;
- 72 replay batches;
- 720 immediate native branches;
- 180 action-advantage targets;
- 32 terminal returns per action; and
- 5,760 native terminal continuations.

All 36 root IDs must be unique and disjoint from the 360-root development
ledger. A source reference containing the exact development ledger, source
fingerprints, calibration rank, and radius must be serialized before the first
Phase 6AM root is collected.

## Frozen admission gates

The normalized counterfactual-advantage provider is admitted only if all gates
pass on the untouched 36-root panel:

1. maximum target standard error is at most `0.20`;
2. overall advantage-interval coverage is at least `0.95`;
3. every action, candidate, and declared-intent slice covers at least `0.80`;
4. mean clipped interval width is at most `1.25`;
5. the root-bootstrap 90% lower bound for coverage-minus-`0.90` is strictly
   positive;
6. candidate top-1 accuracy is at least `0.95`;
7. candidate expected calibration error is at most `0.15`;
8. every candidate slice has top-1 accuracy at least `5/6`;
9. every root has exact common-chance cancellation ratio one, with absolute
   chance reach still null; and
10. the development and validation ledgers are disjoint, raw private-card
    persistence is zero, and every official/native execution claim validates.

Coverage means that the complete empirical 95% mean interval
`mean +/- 1.96 * SE` lies inside the frozen conformal advantage interval. No
threshold, model component, range model, radius, or bootstrap rule may change
after validation begins.

## Fail-closed boundary

Even a pass admits only a supplied-range, five-plan, fixed-continuation,
public-root normalized counterfactual-advantage provider for shadow research.
Phase 6AM always keeps false:

- absolute official chance reach or absolute counterfactual values;
- a full official hidden-state or human-intent belief;
- equilibrium convergence of the native blueprint;
- safe subgame/endgame resolver enablement;
- opponent-exploitation enablement;
- official strategy control or deployment;
- Kaggle submission; and
- tournament-strength or superhuman claims.

A later independently preregistered phase may use admitted advantage bounds to
test a baseline-identity-safe one-step resolver. Missing range support, OOD
input, bound failure, interruption, or any uncertainty violation must return
the frozen blueprint exactly.

## Frozen held-out result

The full panel completed without retries: 36 distinct official matches, 3,779
genuine live selections, 720 immediate native branches, 5,760 terminal native
continuations, and 274,068 native descendant transitions. All 36 serialized
roots round-trip exactly, have distinct IDs, remain disjoint from the 360-root
development ledger, persist zero raw private-card keys, and use common opaque
chance-law ratio one with absolute chance reach unavailable.

| Gate | Frozen result | Threshold | Pass |
|---|---:|---:|:---:|
| Maximum target SE | `0.20168842355216715` | `<= 0.20` | **no** |
| Overall interval coverage | `0.950000` | `>= 0.95` | yes |
| Worst action/candidate/intent slice | `0.833333` | `>= 0.80` | yes |
| Mean clipped width | `1.110039238550672` | `<= 1.25` | yes |
| Bootstrap lower bound for coverage minus `0.90` | `0.022222` | `> 0` | yes |
| Candidate top-1 | `1.000000` | `>= 0.95` | yes |
| Candidate ECE | `0.000002975361` | `<= 0.15` | yes |
| Worst candidate top-1 | `1.000000` | `>= 5/6` | yes |
| Reach semantics | ratio `1`; absolute reach null | exact | yes |
| Collection holdout/provenance | complete and disjoint | exact | yes |

Nine of 180 advantage rows are uncovered. The worst slice is `abomasnow` at
`25/30 = 0.833333`, which still passes. The sole failed gate comes from the
`abomasnow` / `prize_race` / `plan:control` row: its target SE is
`0.20168842355216715`, exceeding the frozen ceiling by approximately
`0.001688`. The threshold is not relaxed and no extra replay is added after
seeing the result.

Consequently the normalized counterfactual-advantage provider is **not
admitted**, even in shadow mode. The implementation can serialize a complete
six-hypothesis by five-action matrix for research inspection, but unsupported
hypotheses receive only vacuous `[-2, 2]` bounds and the failed panel may not
feed a resolver. Absolute counterfactual values, equilibrium convergence, safe
resolving, exploitation, and official strategy control remain false.

## Artifact identity

- frozen reference fingerprint:
  `dd2b85d1abcba2fc53ddc11ec3b1b8fc664c9f27723e6b40e8e8c5529cbf6e6d`;
- validation panel fingerprint:
  `f769537cdb6388e3b2efaa0e0133179381b9285f6e3e45d0d3b308bceab11e85`;
- evaluation fingerprint:
  `36bf46a4610b9c10fc004f33957aaa0ad01f8e0a8e79124067ba2c39cd12cb46`;
- evidence fingerprint:
  `7a0f1e28e31d017601ecd205e7af896ed40316a4dc2bc1855492b14cf68ae640`.

Re-running evaluation from the sealed panel changes none of the four generated
panel/evaluation/evidence JSON-or-Markdown file hashes. The native collection
itself is not claimed reproducible because the licensed official runtime does
not expose all chance seeds.

Verification after integration:

- all 219 repository tests pass, including eight Phase 6AM artifact, shape,
  leakage, gate-consistency, and tamper-rejection tests;
- all 979 public exports are unique and resolve;
- source, scripts, and tests compile;
- the fresh 712,887-byte wheel has SHA-256
  `aea80d04c72b1fc01fb5843360045e18973499d32ee77fd20daa43e956960196`;
  and
- an isolated installation imports all exports and resolves
  `rocketbot-phase6-public-range-counterfactual-advantages --help`.
