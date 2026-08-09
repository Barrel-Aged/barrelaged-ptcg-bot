# Phase 6AE: seat-decoupled official intent-proxy range

Phase 6AE adds a narrowly calibrated public-behavior component to the Phase 6AD
observer-oriented hidden-card range. It validates classification of six
**declared deterministic proxy policies** under a controlled official-engine
panel. It does not establish that a human opponent has one of these latent
intent labels, and it does not enable counterfactual values, resolving,
exploitation, strategy control, packaging, submission, or performance claims.

## Question and result

Phase 6AD used the same behavior style in both seats, so its candidate/resource
result could not support an opponent-intent claim. Phase 6AE asks a narrower
question: with player 0 fixed to one tempo policy, can player 0's public root
and player 1's preceding public MAIN action types identify which of six
controlled player-1 policies generated the trace?

Yes, within this panel. The untouched 108-root validation split reaches intent
top-1 accuracy `0.879630`, intent ECE `0.042259`, joint candidate-intent top-1
accuracy `0.879630`, joint ECE `0.042195`, and hidden-resource ECE `0.011843`.
All preregistered in-scope gates pass. The stronger claim that real opponent
latent intent or a full official public belief is calibrated remains false.

## Collection protocol

- Observer: player 0.
- Target policy: fixed player-0 tempo baseline for every intent stratum.
- Opponent: player 1; only its declared proxy policy varies.
- Candidate decks: the six exact candidate lists admitted by Phase 6AD.
- Intent proxies: setup, prize race, lock/disruption, board denial, resource
  conservation, and combo assembly.
- Root boundary: player 1's second distinct MAIN turn.
- Splits: train, calibration, and untouched validation.
- Strata: `3 splits x 6 candidates x 6 intents x 3 roots = 324` roots.
- Allocation support: 32 public-consistent particles per candidate and root,
  or `324 x 6 x 32 = 62,208` candidate-conditioned particles.

The controlled policies combine strategic action-type priorities with explicit
per-turn MAIN action budgets. Conservation passes immediately; denial attempts
one high-impact action; setup and prize-race lines take at most two; disruption
takes three; combo assembly can invest up to six. Prize-race policies prioritize
attacking and combo policies may cash out by attacking, while setup and
lock/disruption voluntarily end before a low-priority attack. These rules make
the proxy behavior operationally distinct without claiming human semantics.

## Genuine execution versus abstraction

| Component | Status |
|---|---|
| Battle starts, legal prompts, 8,474 selections, public transitions, and 875 recorded player-1 MAIN actions | Genuine official `cg` execution |
| Realized hidden hand/Prize/deck sample used transiently as supervision | Genuine official state exposed only through the offline visualizer boundary |
| Player-0 public-state sanitation and action-type tokenization | Research adapter |
| Six deterministic intent policies and their labels | Controlled abstractions, not official or human latent truth |
| Linear softmax intent model and calibration | Offline research model |
| Phase 6AD candidate particles and hidden-resource buckets | Public-consistent allocation abstraction |
| Candidate-intent product producing 36 hypotheses | Explicit independence approximation |
| Chance reach, counterfactual values, safety, exploitation, and control | Unavailable and machine-disabled |

The official runtime does not expose shuffle, Prize, draw, or coin
probabilities and is unseeded. Checked-in telemetry and fingerprints describe
this evidence run; a fresh collection is not claimed byte-deterministic.

## Leakage boundary

The intent model consumes only:

1. a small allow-list of player-0-oriented public root scalars; and
2. counts, positions, transitions, and ten sequence slots derived from player
   1's selected coarse public MAIN action types.

It does not consume raw option indices, either private hand, private card IDs,
the candidate label, the intent label, realized hidden-resource counts, or
player 0's action semantics. Serialized decision steps keep fingerprints of the
selection contract and selected indices but do not persist those indices. The
transient realized allocation is retained only as offline resource supervision.

## Split discipline and calibration

The normalizer and six-class linear model fit on train roots only. Calibration
selects L2 penalty, temperature, and uniform shrinkage lexicographically by:

1. maximum calibration top-1 accuracy;
2. minimum calibration ECE;
3. minimum calibration NLL; and
4. minimum calibration Brier score.

The selected checked-in configuration is L2 `1e-5`, temperature `1.0`, and
uniform mixture `0.05`. Validation labels never participate in fitting or
selection. The proxy and calibration protocol was frozen before this fresh
checked collection; an earlier exploratory collection is not presented as
evidence. The frozen Phase 6AD candidate/resource model is evaluated on the
fresh Phase 6AE roots, then multiplied by the intent distribution to emit 36
normalized `RangeHypothesis` entries with preserved resource marginals.

## Checked evidence

- Roots: 324 (`108/108/108`).
- Live official selections: 8,474.
- Public opponent MAIN actions: 875.
- Candidate-conditioned allocation particles: 62,208.
- Validation candidate NLL / ECE: `0.0000665` / `0.0000664`.
- Validation intent NLL / top-1 / ECE: `0.383061` / `0.879630` / `0.042259`.
- Validation joint NLL / top-1 / ECE: `0.383127` / `0.879630` / `0.042195`.
- Validation resource presence Brier / ECE: `0.042899` / `0.011843`.
- Dataset fingerprint:
  `46b1b5326681c4fe79aa4b2c6283d231944ce7fd8cdc4b33c48fce7d739e9eec`.
- Intent-model fingerprint:
  `949965ebd15092935a45c9763cab9e80860b745c280127d74d03c4b55142d328`.
- Joint-calibration fingerprint:
  `7869d827f072ff2b754ae7dfb062997cb017b08f501250d5ce67694c18bcf777`.
- Evidence fingerprint:
  `06478e797fbc1c6b0621b49a6dae4754cf8facc7b264c2a5ff0121f7a4ee323f`.

## What this unlocks and what remains closed

Phase 6AE admits an **in-scope joint candidate / declared-policy intent /
hidden-resource range** for this six-deck, six-proxy, later-root distribution.
It is suitable for offline diagnostics and as a typed input to future research.

It does not admit:

- latent human-opponent intent calibration;
- broader deck, policy, matchup, or turn-distribution generalization;
- a joint model that learns candidate-intent dependence;
- absolute official chance reach or counterfactual reach weights;
- calibrated hypothesis-conditioned counterfactual values or bounds;
- safe subgame/endgame resolving;
- baseline-constrained opponent exploitation;
- official strategy control, Kaggle submission, or playing-strength claims.

The next honest step is an independently collected policy-diverse public-belief
panel with repeated public information sets, followed by aligned conditional
value calibration. Resolver or exploitation work must remain disabled until
those upstream gates pass.

Phase 6AF now completes the policy-implementation-heldout part of that request
across four authored policies. It still does not supply human latent intent,
repeated hidden completions behind shared public information sets, or aligned
counterfactual values. See
`docs/PHASE6AF_POLICY_IMPLEMENTATION_GENERALIZATION.md`.

## Reproduce

From the project root:

```powershell
python scripts/phase6_official_intent_calibration.py
python -m unittest tests.test_official_intent_calibration -v
```

The command writes
`artifacts/phase6_official_intent_calibration/official_intent_truth_dataset.json`,
`official_intent_range_model.json`, `official_joint_belief_calibration.json`,
and the Phase 6AE JSON/Markdown evidence summaries. It makes no network request
and does not submit to Kaggle.
