# V1 Persistent Plan Validation Report

Protocol: `rocket-persistent-plan-causal/v1`  
Locked candidate: `locked_admitted_axes_blend_v1`  
Games: 4,000 fresh complete local official-engine matches  
Allocation: 1,000 candidate and 1,000 control games against each of two top Marnie opponents  
Correctness: zero illegal actions, zero overlay exception fallbacks, one deck/model/library/runtime hash throughout

## Result

The persistent semantic overlay failed validation.

| Metric | Candidate | Control | Difference |
|---|---:|---:|---:|
| Equal-opponent win rate | 43.30% | 43.25% | +0.05 pp |
| V110 | 43.10% (431/1,000) | 43.00% (430/1,000) | +0.10 pp |
| Recovery V4 | 43.50% (435/1,000) | 43.50% (435/1,000) | 0.00 pp |

The locked 50,000-replicate stratified bootstrap interval was `[-3.00, +3.15]` percentage points. The overlay intervened in 64.2% of candidate games, averaging 1.20 changed decisions per game, so the null result is not explained by an inactive gate.

## Gate disposition

- Point improvement at least +3 pp: **failed** (+0.05 pp).
- 95% lower bound greater than 0: **failed** (-3.00 pp).
- Neither opponent worse than -2 pp: passed.
- At least 10% intervention games: passed (64.2%).
- Zero correctness failures: passed.

Overall V1 status: **failed**.

Per the preregistered stop rule, no persistent-plan test is collected or opened, no candidate from this family is packaged, and no arena submission is made. The immutable analysis is `artifacts/team_rocket_recovery/persistent_plan_v1_validation/analysis.json`, SHA-256 `b4573f86249d911b1d5f445660bb8cd0b0c520fa778f2e90ca81836892d576d5`.

## Research conclusion

The D1 and D2 positive estimates were selection noise. A bounded hand-authored semantic layer can change many actions without changing win probability. The missing capability is not another fixed bias vector; it is a policy that learns long-horizon credit from complete games while retaining a persistent latent state. The next family therefore trains a recurrent neural residual against complete official-engine outcomes at orders of magnitude more scale than the earlier PPO pilots.

