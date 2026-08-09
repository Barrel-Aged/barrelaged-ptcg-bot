# Target-80 residual advantage V2 result

Completed: 2026-08-05  
Decision: **closed at development validation; no confirmation, runtime overlay,
package, or submission**

## What worked

- The five game-level state folds covered all 34,402 train choices without
  in-fold prediction leakage.
- The cross-fitted state baseline achieved AUC 0.826 and Brier 0.170.
- Square-root propensity stabilization repaired V1's effective-sample failure:
  the minimum V2 development opponent/seat ESS ratio was 0.295 and the overall
  ratio was 0.482.
- All three residual heads trained and serialized under their frozen seeds.
- Every frozen configuration had at least one twin-agreement threshold with a
  0.5%-to-5% development change rate.

## Decisive validation failure

The action heads were required to improve over the frozen state-only model,
not merely over raw policy logits. None did.

| Config | Min cell ESS | Exploratory Q − state AUC | State − Q Brier | Zero − learned residual MAE | First-choice Q − state AUC | Eligible |
|---|---:|---:|---:|---:|---:|---|
| low regularization | 0.295 | -0.0099 | -0.0047 | -0.0280 | +0.0010 | no |
| balanced | 0.295 | -0.0063 | -0.0033 | -0.0217 | -0.0013 | no |
| high regularization | 0.295 | -0.0044 | -0.0028 | -0.0193 | -0.0059 | no |

Negative AUC differences mean adding the learned action residual made
exploratory-action outcome ranking worse than the public-state baseline.
Negative Brier and residual-MAE gains also favor the state-only/zero-advantage
baselines. High regularization was least harmful, but it still missed four
independent gates. It is not a candidate.

The development cache was used exactly as prospectively declared. Because no
configuration qualified, no fresh confirmation corpus was collected and no
checkpoint was integrated into an agent.

## Interpretation

The randomized corpus has strong support and the state representation predicts
outcomes well, but 1,716 development interventions still do not identify a
generalizable per-decision action effect from terminal outcomes. Sequential
credit assignment dominates: each game contains many choices, later states
depend on earlier randomized choices, and a single terminal label is too weak
to assign causal value to one option token.

Another terminal-outcome head, different regularization coefficient, or looser
offline gate is not justified. The next learning attempt needs denser targets:
paired official-engine continuations from selected public roots, short-horizon
strategic milestone rewards, or a genuinely sequential off-policy estimator
with trajectory-level importance handling. Until such evidence exists, the
current neural control remains safer than either V1 or V2 overlay.

The Wobbuffet-free deck substrate, official-engine corpus, telemetry,
serialization, and all submitted V5 bytes remain unchanged.

