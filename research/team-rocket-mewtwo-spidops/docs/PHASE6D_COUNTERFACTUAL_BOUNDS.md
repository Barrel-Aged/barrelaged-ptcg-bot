# Phase 6D counterfactual-value bounds

Date: 2026-07-26  
Status: exact abstract microgame gate validated  
Evidence fingerprint: `829dc9714e55779b93f6189bbed0a1a75ae6ea9e341423e912ac827c37dafc50`

## Outcome

Phase 6D adds a conservative, fail-closed interval layer around the selected
Phase 6C neural blueprint's opponent counterfactual-value estimates. It is
calibrated and exhaustively audited on `RocketPrizeDuel`, not on the official
Pokémon engine.

The deterministic information-set split contains 12 calibration targets and 12
calibration-heldout targets. “Held out” here means held out from interval
calibration; the small neural blueprint was trained on every microgame
information set. The calibrated symmetric residual radius is
`0.0001929184468`, and coverage on the disjoint validation split is 100%.

## Bound construction

For each opponent private-state hypothesis, the provider starts from the neural
counterfactual-value prediction and applies a symmetric half-width:

```text
calibrated residual radius
+ declared abstraction-error upper bound
+ OOD score × OOD margin scale
+ incomplete-search fraction × interruption margin scale
```

Each component is recorded. In the deterministic widening probe, the base
half-width `0.0001929184468` expands to `0.0451929184468` after adding a 0.01
abstraction allowance, OOD score 0.20, and incomplete-search fraction 0.25.

The provider refuses to return bounds when:

- the blueprint/game/deck fingerprints disagree;
- held-out exact coverage fails;
- requested confidence exceeds the calibrated 90% level;
- the OOD score exceeds 0.50; or
- the incomplete-search fraction exceeds 0.50.

Those cases return an explicit blueprint-fallback decision. The protocol-only
`bounds()` method raises `BoundGateFailure` so a caller cannot accidentally
treat missing certification as valid bounds.

## Exact false-safety audit

Six held-out information-set roots were re-solved over 101 candidate mixtures.
Every approximate interval and every resolver certificate was checked against
enumerated continuation values for every opponent private-card hypothesis.

| Metric | Result |
|---|---:|
| Held-out hypotheses | 12 |
| Baseline interval coverage | 1.000 |
| Resolver certificates audited | 6 |
| False-safe certificates | 0 |
| Maximum exact interval violation | 0 |
| Maximum certified-bound violation | 0 |
| Maximum true baseline regression | 0.0000231559 |
| Regression beyond declared uncertainty | 0 |

The small positive baseline regression is inside the declared estimation-
uncertainty envelope. It must not be described as strict zero-regression safety.
Phase 6E is responsible for interruption-safe resolver behavior and stronger
non-regression policies.

## Reproducibility

Two independent generations produced byte-identical files:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `counterfactual_calibration.json` | 8,771 | `b3cf3813a1e934abe16b3a319a97f383e132077bcf3fc674eab97c13fdb0104a` |
| `phase6d_evidence.json` | 2,205 | `4a79ee3a4d387c04fc94b229b52372d93749ff062736c71e70ffd706aae1ca95` |
| `phase6d_evidence.md` | 1,819 | `40e8af00527c6fa33165b9eec592a4b68109b6730c8455172427137f04ad06a2` |

At the Phase 6D checkpoint, the project suite passed 73 tests and all 315
then-current public exports resolved without duplicates.

Reproduce from the project root with:

```powershell
python scripts/phase6_bounds.py
```

## Boundary and next work

This milestone does not calibrate values on replay-derived official states,
estimate Pokémon abstraction error, implement a full-game information-set
transition bridge, or certify an official/full-game resolver. Phase 6G now
validates a one-step official shadow boundary only. Phase 6E must add a genuinely
budget-aware resolver whose deadline, node, and iteration interruption paths
return the blueprint exactly, then audit non-regression under adversarial budgets.
