# Target-80 randomized advantage V1 learning lock

Locked: 2026-08-05, after corpus admission and before cache construction or
outcome-model fitting  
Status: **training authorized; no official-engine candidate authority**

## Admitted data

- exact corpus: 1,200 games in 18 game-disjoint shards;
- corpus SHA-256:
  `3dde57e3420070bf43f31ffd9fa38ac6a9015c6fa6cc3878a63dff776eb69602`;
- corpus audit SHA-256:
  `72cb9266742d80c2443c987e1d0c1527e3ec8cb47e9586d69439d586b1bd8680`;
- corpus auditor SHA-256:
  `db3cd060f4fa4878ddbf365692a7fa5ca013190377424a4f9a60d2733dd55fb6`;
- 57,741 eligible known-propensity decisions;
- 8,620 realized exploratory decisions;
- 74,694/74,694 Rocket observations passed the hidden-hand boundary;
- all six opponent/seat cells contain realized interventions;
- zero duplicate replay names or digests.

No outcome was reported by the corpus audit. Dataset construction and
intervention gates were therefore resolved independently of win/loss balance.

## Frozen learner

Script SHA-256:
`14859418a975bf6305c421b34a06191a102de8bafd5f6cb95a6f2858cce27182`

The cache builder must reproduce the frozen v10 public state and option tokens,
the logged reference logits/probabilities within `2e-5`, the exact behavior
support, action, and chosen propensity. It processes all neural decisions in
sequence to preserve the deployed belief state, but caches only the eligible
one-choice mixture decisions. Packaged guard and empty decisions correctly do
not advance the neural belief extractor because they did not do so online.

Each independently initialized head predicts:

- a bounded public-state value;
- a state-option interaction over frozen relational tokens;
- option advantages centered to zero under the exact logged behavior
  distribution;
- a reference-logit residual initialized at zero.

Training uses terminal outcomes `+1 / 0 / -1`, with draws treated as non-wins
for calibration, equal base mass per game, clipped inverse-propensity weights
(`1 / p`, maximum 20), deterministic 80% bootstrap masks, twin heads, and
advantage shrinkage. Runtime proposals will be restricted to the exact logged
support (reference argmax plus alternatives that had at least 0.03 reference
probability). A change is possible only when both heads choose the same
non-baseline action and both margins clear the selected threshold.

## Frozen configurations

All use width 96, 10 epochs, batch size 256, learning rate `3e-4`, weight decay
`1e-4`, and the exact train cache only.

| Name | Advantage regularization | Seed |
|---|---:|---:|
| `low_reg` | 0.01 | 854001 |
| `balanced` | 0.03 | 854002 |
| `high_reg` | 0.08 | 854003 |

No additional configuration may be trained after validation is opened in V1.

## Validation selection

Evaluate all three configurations on the exact 240-game validation cache. A
configuration is eligible only if:

1. the overall and every opponent/seat inverse-propensity effective-sample
   ratio are at least 0.25;
2. on realized exploratory actions, Q AUC strictly exceeds the calibrated
   reference-logit-only AUC;
3. on realized exploratory actions, Q Brier score is no worse than the
   reference-logit-only Brier score;
4. first-eligible-per-game Q AUC is no worse than the reference-logit AUC;
5. at least one frozen margin in `0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20`
   yields a twin-agreed non-baseline change rate between 0.5% and 5%; and
6. all metrics are finite and every proposed change is support-valid.

Among eligible configurations, select the largest exploratory-action AUC gain;
break ties by lower exploratory-action Q Brier, then stronger advantage
regularization. For the selected configuration, choose the largest frozen
margin whose validation change rate is still at least 0.5% (and at most 5%).
Write the selection and exact checkpoint hash before opening test.

If no configuration qualifies, V1 closes without test-driven tuning.

## One-shot test gates

The selected configuration and margin receive one evaluation on the exact
240-game test cache. It advances only if:

- every effective-sample ratio remains at least 0.25;
- exploratory-action Q AUC strictly exceeds reference-logit AUC;
- exploratory-action Q Brier is no worse than reference-logit Brier;
- first-eligible-per-game Q AUC is no worse than reference-logit AUC;
- the frozen threshold changes between 0.5% and 12% of supported test choices;
- twin heads agree on every changed action;
- checkpoint reload reproduces all test Q values, best actions, and proposed
  changes byte-for-byte.

Offline passage authorizes only implementation of a fail-closed runtime overlay
and a new preregistered official-engine development panel. It does not
authorize packaging, arena submission, or an 80% claim.

