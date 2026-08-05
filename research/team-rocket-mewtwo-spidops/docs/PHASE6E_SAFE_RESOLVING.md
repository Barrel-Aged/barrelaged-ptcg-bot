# Phase 6E interruption-safe resolving

Date: 2026-07-26  
Status: exact abstract microgame gate validated  
Evidence fingerprint: `369c083fc221933be4702dc9e86c43359b50233a42799369f8dc1708794955c3`

## Outcome

Phase 6E replaces the earlier unbudgeted validation resolver with an “all or
blueprint” exact resolver for `RocketPrizeDuel`. A completed search evaluates the
full 102-mixture grid and every opponent private-state hypothesis. If the
iteration, node, deadline, or value-bound gate prevents complete certification,
the resolver discards all partial work and returns the neural blueprint
distribution exactly.

This remains an abstract-game implementation. It does not resolve official
Pokémon endgames or run behind the official observation/action boundary.

## Safety rule

A candidate is eligible only when exact enumeration shows, for every opponent
private-card hypothesis, that it:

- does not exceed the calibrated opponent counterfactual-value upper bound; and
- does not improve the opponent relative to the exact blueprint continuation
  value beyond the declared `1e-12` tolerance.

The second constraint is intentionally stricter than Phase 6D's uncertainty
envelope. Partial candidate grids never produce a strategy.

## Fixed-budget exact evidence

Every root uses the same budget: 102 iterations, 206 conceptual value nodes,
1,000 ms, and `1e-12` safety tolerance.

| Scope | Roots | Complete grids | Strategy changes | Exact conditional actor-value gain | Max opponent baseline regression |
|---|---:|---:|---:|---:|---:|
| Endgame, depth 2–3 | 9 | 9 | 5 | 0.00003695798 | 2.22e-16 |
| Continual root, depth 1 | 3 | 3 | 0 | 1.11e-16 | 2.22e-16 |

Aggregate conditional gain is `0.00003695797756`. It is a small exact microgame
quantity, not a win-rate or playing-strength result. The continual-root pass
selects the blueprint at all three roots, which is a valid non-regression result
rather than evidence of continual-resolving improvement.

All returned distributions have exact legal support, all completed certificates
are safe, and all 12 roots satisfy strict per-hypothesis opponent non-regression
within tolerance.

## Interruption and gate behavior

Deterministic probes validate four independent fail-closed paths:

| Trigger | Result | Partial strategy exposed |
|---|---|---:|
| 101/102 iteration preflight | exact blueprint | no |
| 205/206 node preflight | exact blueprint | no |
| injected mid-search deadline | exact blueprint | no |
| OOD score above Phase 6D gate | exact blueprint without invoking resolver | no |

The deadline test uses an injected monotonic clock so the interruption path is
reproducible rather than dependent on workstation speed.

## Reproducibility

Two independent generations produced byte-identical artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `phase6e_evidence.json` | 3,117 | `b24e02c20668702a36ce243ac57e002f1c87c297b0964c5eb28809909d6cd533` |
| `phase6e_evidence.md` | 1,737 | `af0399881247b4f072ccf08fa2b00c76c5df6223f2412e59d79bace4b3dd3369` |

At the Phase 6E checkpoint, the project suite passed 77 tests and all 325
then-current public exports resolved without duplicates.

Reproduce from the project root with:

```powershell
python scripts/phase6_resolving.py
```

## Boundary and next work

Phase 6F may add opponent-specific exploitation only as a separately evaluated
overlay. A support/KL engineering guard is insufficient: the final mixed
strategy must be re-certified against exact worst-case values, and adversarial
misclassification must return the safe strategy. Phase 6G subsequently
validates a one-step official shadow bridge; full-game transition semantics and
official resolving remain later work.
