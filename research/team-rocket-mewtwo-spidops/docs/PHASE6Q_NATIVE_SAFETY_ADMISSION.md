# Phase 6Q: native safety admission rejected

Phase 6Q prevents the Phase 6P training candidate from being mistaken for a
safe blueprint. It binds the exact dataset, model, and Phase 6P evidence
fingerprints, evaluates explicit necessary prerequisites, and enables nothing
unless every check passes.

The checked candidate is rejected before any counterfactual-value bounds are
built. No resolver, exploitation overlay, official strategy control, benchmark
claim, or Kaggle submission is executed.

## Result

Three of 16 necessary checks pass:

- the source descendants genuinely executed the licensed native search engine;
- no raw private-card identities were serialized; and
- the model checkpoint restores with zero drift.

Thirteen checks fail:

| Failed gate | Observed | Required by this gate |
|---|---:|---:|
| Sampling iterations | 3 | at least 20 |
| Player-0 regret samples | 217 | at least 5,000 |
| Player-1 regret samples | 259 | at least 5,000 |
| Average-strategy samples | 476 | at least 5,000 |
| Raw-regret p95 | `5.044e11` | at most `1e5` |
| Raw-regret maximum | `1.243e16` | at most `1e8` |
| Public-belief calibration | absent | required |
| Held-out-particle validation | absent | required |
| Full-game regret/exploitability proxy | absent | required |
| Independent counterfactual calibration | absent | required |
| Model counterfactual-value capability | false | required |
| Safe-resolver compatibility flag | false | required |
| Equilibrium-convergence validation | false | required |

The numerical thresholds are conservative engineering admission thresholds,
not a theorem and not a sufficient safety certificate. Passing them in a future
run would only permit the next independent calibration audit; it would not by
itself enable deployment. Public-belief calibration, held-out abstraction
fidelity, a defensible regret proxy, and independent counterfactual-value
coverage remain mandatory regardless of sample count.

`NativePlanSafetyAdmission.require_admitted()` raises with every failed check.
Serialized capability tampering is rejected, and an admission/model/dataset
fingerprint mismatch fails closed. The current provider continues to raise on
`counterfactual_values()`.

## Why no wide fallback interval was built

A trivially wide utility interval could cover the observed outcomes while
placing no useful restriction on the opponent's value. Labeling that interval
safe would hide rather than solve the problem. Phase 6Q therefore builds no
counterfactual bounds at all and does not call the safe resolver. The preserved
official heuristic integration baseline remains the only executable agent path;
it is a legal fail-soft baseline, not a game-theoretically certified strategy.

## Official execution versus abstraction

The native execution claim applies only to the source descendant transitions
recorded in Phase 6P. The admission gate, thresholds, hidden-completion prior,
plan game, model, and rejection decision are research abstractions. A rejected
admission is safety engineering evidence, not official-engine playing or
strength evidence.

## Durable artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| `native_plan_safety_admission.json` | 5,219 | `eabfc51cd7f9d1e34f82fbcdee7af91795869e9b7e96582ee76e83ed640765ff` |
| `phase6q_evidence.json` | 6,327 | `af746866b2237f3c2d9e98835bf53bd01c689c6240568d523c006a8bd4873706` |
| `phase6q_evidence.md` | 2,311 | `9318fe75d88c56a7c845dc07fc73721fd21ba458692e525b47abc246a7b6a3dc` |

Admission fingerprint:
`8aaf918f46d1f51c7c479d88def6b06e97a35ef42e63ce2a775fd42b658cf122`.
Evidence fingerprint:
`da885fe53aa1e21656528a69a628a5b27b590fc68d4dbfc9ae654f696fe004bd`.

Reproduce the deterministic admission decision from the checked Phase 6P
artifacts with:

```powershell
python scripts/phase6_native_safety_admission.py
```
