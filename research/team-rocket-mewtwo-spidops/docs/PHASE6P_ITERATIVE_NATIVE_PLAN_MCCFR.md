# Phase 6P: iterative native turn-plan MCCFR candidate

Phase 6P closes the collect–fit–update wiring gap left by Phase 6O. Each round
samples from the preceding player-specific regret model, adds the resulting
outcome-regret and average-strategy records to separate cumulative memories,
then refits all three networks. The experiment succeeds as an implementation
audit and fails the variance gate required before counterfactual calibration.

## Iteration contract

The starting policy is the checked Phase 6O model. For each of three rounds:

1. every one of the six supplied hidden completions is traversed once for each
   player;
2. the acting player's current strategy comes from that player's regret head;
3. outcome sampling mixes the traverser strategy with 0.25 uniform exploration;
4. the Phase 6K tactical state value remains a variance baseline only;
5. regret and average-strategy records enter separate bounded reservoirs;
6. fresh player-0, player-1, and average-plan networks are fitted to the
   cumulative memories; and
7. the new regret heads drive the next round.

Every round adds nonzero targets, lowers its training objective, changes both
player policies and the average policy on newly seen states, and preserves
exact five-plan support.

| Round | Trajectories | P0 added | P1 added | Strategy added | Loss | Mean TV drift P0 / P1 / average |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 69 | 114 | 183 | 8.451102 → 7.750219 | 0.3192 / 0.1892 / 0.2647 |
| 2 | 12 | 67 | 79 | 146 | 8.610624 → 7.968815 | 0.2046 / 0.0838 / 0.0802 |
| 3 | 12 | 81 | 66 | 147 | 9.352509 → 8.864545 | 0.2346 / 0.0781 / 0.0852 |

The live run executes 5,562 native, 959 macro, and 5,556 tactical transitions.
It retains 217 player-0 regrets, 259 player-1 regrets, and 476 separate
average-strategy records. All checkpoint tensors restore exactly, and the
serialized dataset contains zero raw private-card fields.

## Variance gate: failed

The iterative strategy is not admitted to counterfactual-value calibration:

| Raw absolute regret statistic | Result |
|---|---:|
| Median | 8,540.421 |
| p95 | 504,409,388,223.769 |
| Maximum | 12,427,707,138,910,762 |

The p95 is roughly 49,000 times the Phase 6O p95, and the maximum is over 2,000
times larger. This is the predictable long-horizon importance-weight tail, not
a numerical serialization defect. The `asinh` target makes optimization finite
but changes regret magnitude and cannot manufacture a convergence guarantee.

Consequently:

- the final network still refuses `counterfactual_values()`;
- calibrated-counterfactual-value and safe-resolver capability flags remain
  false;
- the provider stays outside official strategy control;
- no safe resolving or opponent exploitation is run from these values; and
- no playing-strength, equilibrium, or Kaggle claim is made.

The next research gate must reduce horizon/variance before collecting an
independent calibration set. Plausible candidates are a bounded public-subgame
boundary with a separately calibrated leaf-value model, a more effective
action-dependent baseline, or a primary-source-supported variance-reduced
sampling method. Any cutoff leaf value must be calibrated before it can support
a safety certificate.

## Official execution versus abstraction

Genuine execution is limited to legal descendant transitions and terminal
results produced by the licensed native search engine. The five plan actions,
fixed tactical executor, supplied hidden-completion prior, current/average
strategies, estimator records, training loss, and neural model are research
abstractions. The abstract game is never labeled official.

## Durable artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| `iterative_native_plan_dataset.json` | 21,345,183 | `cc43167fb07d2836b3671ebfdb8504fbac9ad31da6290df72b5bf36b8d7d70a6` |
| `iterative_native_plan_blueprint.json` | 893,799 | `e01ae0684b00aaec576dd27f3146a2a5243764a8a34d7da2f93c2817c1665e54` |
| `iterative_native_plan_training_report.json` | 4,449 | `46bc93b5085133fb6b9ce5b1c44291c805949f63880ba29a122ab629812baf4d` |
| `phase6p_evidence.json` | 4,364 | `b1fc6e5ef853420a2dcfe127cecd5e92c3784fd84109feb4e638f3abb023debd` |
| `phase6p_evidence.md` | 1,873 | `ab29b98c2b0ab98da4db50cb07bb1d5ad45cb4b21df3b07bf4855b0ff161a22f` |

Dataset fingerprint:
`7853e314f5f05d2555c645e74ebe549ca0337bce2e1bc7a4e10faf33818b098e`.
Model fingerprint:
`b416597d44ad31329acf46bb59a517ffa09a722e4e4e7cf14c79176a38c343d9`.
Evidence fingerprint:
`0aa43252f3c50de925b60c02e99984331698848d0f443a5c48873b9580b06bc9`.

The native battle root is unseeded, so a fresh run is not claimed byte-
deterministic. Reproduce the workflow with:

```powershell
python scripts/phase6_native_plan_iterations.py
```
