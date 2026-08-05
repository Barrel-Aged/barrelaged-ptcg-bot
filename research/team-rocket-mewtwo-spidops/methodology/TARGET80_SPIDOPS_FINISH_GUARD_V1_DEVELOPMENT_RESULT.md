# Target-80 Spidops-finish guard V1 development result

Completed: 2026-08-05  
Decision: **failed; exact action guard closed with no broad-panel authority**

## Integrity

- 200/200 direct candidate-versus-control games completed in the genuine local
  official engine.
- The candidate occupied each engine side exactly 100 times and started first
  exactly 100 times.
- All 200 games ended normally; zero agent errors, tracebacks, step-limit
  results, or draws occurred.
- The frozen Wobbuffet-free deck and neural checkpoint were byte-identical
  between candidate and control.
- The guard made 51 real interventions across turns 3-18.
- Match output SHA-256:
  `74a841271a9ef185e9d35e66d209cde522e0925df58290ae3fe513439fbc53a4`.
- Decision trace SHA-256:
  `9d44de6a0e5f0ed616a85785c02fca68f60d2e06e40528aa04ae072544a57757`.

The analyzer required two documented mechanical pre-output corrections: its
local tree digest initially included an extra file-size field, and it initially
treated a pregame trace row with `firstPlayer == -1` as the start-order row.
Neither correction changed an outcome, candidate, gate, or scoring rule. See
`TARGET80_SPIDOPS_FINISH_GUARD_V1_ANALYZER_AMENDMENT.md`.

## Locked gates

| Gate | Result | Required | Pass |
|---|---:|---:|:---:|
| Exact games / sides | 200; 100 each side | exact | Yes |
| Candidate wins | 106/200 | at least 110 | **No** |
| Overall win rate | 53.0% | at least 55% | **No** |
| Starts first | 61/100 | at least 50% | Yes |
| Starts second | 45/100 | at least 50% | **No** |
| Guard interventions | 51 | at least 5 | Yes |
| Execution errors | 0 | zero | Yes |

The immutable analysis records `passes_development: false` and
`authority: none; close Spidops finish guard V1` at
`work/target80_spidops_finish_guard_v1_development/analysis.json`.

## Interpretation

The narrow guard was legal and directionally positive overall, but its +3
percentage-point direct margin was too small and start-order performance was
not robust enough for advancement. The result is consistent with the paired
milestone evidence finding a useful local action without demonstrating a large
match-level effect.

The candidate is not eligible for broad development, untouched confirmation,
packaging, arena submission, or a strength claim. The live V5 identity remains
unchanged.
