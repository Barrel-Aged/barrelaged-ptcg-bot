# Causal empirical-pair recovery protocol

Status: frozen before any admission labels were summarized or used for model
selection.

## Restart boundary

The contextual pairwise neural head was rejected on its opened 90-game
selection panel. It achieved 49.2% non-tied pair accuracy at its best epoch and
no intervention threshold met the selection gate. Its untouched admission
panel was not read by the trainer. The failure showed strong distribution shift
between the older one-rollout data and the new common-coin, two-rollout data.

The rejected neural run and all older one-rollout data are excluded from this
model. The three former selection shards are now opened development evidence:

- `pairwise_selection_rank1a_v1.jsonl`
- `pairwise_selection_rank1b_v1.jsonl`
- `pairwise_selection_rank2_v1.jsonl`

They contain 90 complete second-seat matches, 179 eligible early-main roots,
and two matched official-engine terminal rollouts per candidate. The three
independent source processes are the cross-validation folds.

The admission evidence remains the separately launched and frozen files:

- `pairwise_test_rank1a_frozen_v1.jsonl`, SHA-256
  `9f8c376c04d2d1fd8c65fa5ea16b323b9d9309dbcb8edc50a0ea722d8087a3be`;
- `pairwise_test_rank1b_frozen_v1.jsonl`, SHA-256
  `6e226da1e715a9c64c42c3d1c7072def2412f299a704b204cec538415e91ff03`;
- `pairwise_test_rank2_frozen_v1.jsonl`, SHA-256
  `6dc79bda3569160f0c0c883e812d9b5976c314008bc11b480b01767890baa733`.

Freezing parsed those files only to verify schema, complete rollout counts,
unique roots, and hashes. No test action deltas, rule performance, or promotion
metric has been displayed or used. The admission program may open them only if
the frozen leave-one-shard-out development gate passes.

## Model

This is a deliberately low-capacity opponent-exploitation overlay, not an
equilibrium solver and not a replacement for the neural blueprint. An action
has the public semantic signature:

`(option type, hand card id, target in-play area, attack id)`.

A rule maps an exact `(blueprint action, alternative action)` signature pair to
observed official terminal deltas. Candidate actions remain restricted to the
frozen blueprint's semantically distinct top four. At a matching root the rule
with highest mean delta wins, followed by lowest harm rate, greatest support,
and deterministic option index. Otherwise the blueprint action is unchanged.

Rules are fit with this single frozen configuration:

- at least 5 total observations;
- evidence in at least 2 independent source shards;
- at least 3 strictly positive observations;
- mean terminal delta at least `+0.10`;
- observed harm rate at most `0.20` over all observations;
- precision at least `0.75` among decisive non-tied observations; and
- nonnegative mean delta in every source shard where the rule appears.

The configuration was selected from the declared finite development-only grid
in `scripts/explore_empirical_causal_rules.py`. It is now frozen; there is no
admission-data sweep.

## Development gate

Use leave-one-source-shard-out predictions: fit rules on two source processes
and evaluate the third, repeated for all three folds. With 10,000 game-clustered
bootstrap samples and seed `615801`, the concatenated predictions must have:

- 8 or more interventions and no more than 35% of roots;
- at least 90% non-harm rate, counting exact terminal ties as non-harm;
- at least 80% precision among decisive outcomes;
- positive mean paired improvement across all roots; and
- a strictly positive game-clustered 95% bootstrap lower bound.

Failure rejects the model without opening admission deltas.

## One-time admission gate

If development passes, fit the same frozen rule criteria once on all three
development shards, then open the three named admission files exactly once.
Admission requires:

- at least 8 interventions and no more than 35% of roots;
- at least 85% non-harm rate;
- at least 70% precision among decisive outcomes;
- positive mean paired improvement across all roots; and
- a strictly positive game-clustered 95% bootstrap lower bound.

Passing admits only a runtime candidate for fresh complete-match evaluation.
It does not establish arena strength, a public-information-set value,
counterfactual safety, or equilibrium play. Runtime remains restricted to the
first eligible main-phase root per turn through turn 8, only when Rocket is
second into publicly identified Marnie. Official legality, full-match gates,
and the arena remain authoritative.
