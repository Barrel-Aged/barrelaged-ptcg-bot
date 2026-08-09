# Phase 6X: opponent-hypothesis plan-deviation values

Phase 6X replaces the rejected pooled leaf scalar with the first native value
target shaped for an opponent-response model. For each supplied opponent hidden
allocation, it reaches a player-1 turn-plan decision, forces all five abstract
plans, and estimates the opponent's conditional return under the frozen
average-plan continuation.

This is a supervision substrate, not a counterfactual-value certificate.

## Collection protocol

The native run covers all six candidate deck hypotheses with two independently
sampled public-consistent hidden allocations each:

- variant 0 is the calibration allocation;
- variant 1 is the validation allocation;
- each of the five persistent plans is forced once at the opponent decision;
- each forced plan receives ten full terminal continuations;
- the same ten rollout seeds are used across the five actions at a surface,
  providing common random numbers for lower-noise action comparison;
- later decisions for both players follow the frozen Phase 6P average-plan
  network through the existing tactical executor.

The resulting artifact contains 12 hypothesis-conditioned surfaces, 60 forced
action targets, 600 terminal continuations, and 73,538 genuine native
transitions. Each action target stores its ten returns, empirical opponent
value, standard error, and transition/depth accounting. Public features and
hashed identities are retained; raw private card identities are rejected.

## Signal and precision result

| Diagnostic | Calibration | Validation |
|---|---:|---:|
| Candidate hypotheses | 6 | 6 |
| Forced action targets | 30 | 30 |
| Nontrivial action-surface fraction | `83.333%` | `100%` |
| Mean action-value span | `0.500000` | `0.633333` |
| Maximum action-value span | `1.200000` | `0.800000` |
| Mean empirical deviation gain | `0.275031` | `0.286358` |
| Maximum empirical deviation gain | `0.645421` | `0.411626` |
| Maximum rollout standard error | `0.333333` | `0.333333` |
| Blueprint-top plan matches empirical best | `0%` | `0%` |

The conditional-deviation substrate gate passes: both splits cover all six
hypotheses, the ten-rollout precision threshold passes, and action sensitivity
is nontrivial. The zero agreement between the blueprint's highest-probability
plan and the empirical best plan is a diagnostic on this small supplied-
hypothesis collection, not an exploitability or strength estimate.

## Why these are not counterfactual values

An extensive-form opponent counterfactual value must weight histories by the
other player's and chance's reach while excluding the opponent's own reach.
Phase 6X instead conditions on one sampled cutoff history and one supplied
hidden completion, forces a plan, and follows a fixed continuation profile.
Therefore:

- `conditional_on_sampled_reach=true`;
- `opponent_reach_weighted=false`;
- `exact_counterfactual_values=false`;
- the counterfactual-value and safe-resolver gates remain false.

Phase 6Y subsequently adds that redacted reach ledger and audits its expectation
on the exact microgame. The native audit reveals that all Phase 6X-style roots
already begin at the first opponent plan decision and that each public
information set appears only once. The next admissible step is therefore to
construct repeated, public-root-aligned private-hypothesis families at later
decisions and calibrate a hypothesis-conditioned action-value model on complete
collection-disjoint native data. Only bounds that cover every opponent
hypothesis may enter the existing safe resolver interface.

## Official execution versus abstraction

Genuine official-engine evidence:

- initial observations and every descendant transition come from the licensed
  competition runtime;
- all 600 terminal continuations and 73,538 recorded native transitions are
  genuine engine executions;
- every tactical selection is checked and executed through the existing native
  legality boundary.

Research abstractions:

- hidden completions and candidate supervision labels;
- the five turn-plan actions and tactical mapping;
- cutoff-history sampling, forced deviations, common-random-number protocol,
  frozen continuation strategy, empirical values, and substrate gate.

No resolver, exploitation overlay, official strategy controller, package,
benchmark policy, or Kaggle submission changed. No Pokémon-strength claim is
made.

## Durable artifacts

Evidence fingerprint:
`48a14f5c70c2bfce446e01f58d38e1364867417ef1578f5abf40efcec637f67a`  
Dataset fingerprint:
`66be824968cfb591947c04463d08f80717e2b36cfb4ee87ea6e8c2f05cff8a3c`  
Calibration fingerprint:
`e8532cc2645b2f08b11964b4451f328e3d0a6cf7f16eca89d59cde1c96fa329a`  
Particle-family fingerprint:
`9659606eddad91939ef309bc4d15a4356d487dfb7214899466a34bcbae1dc27f`

| File | Bytes | SHA-256 |
|---|---:|---|
| `opponent_deviation_dataset.json` | 172,906 | `05e3feaff0811dffca007dd431a5fee8fa2eb624db7e50da2069a727197f7a84` |
| `opponent_deviation_calibration.json` | 2,328 | `59de589ed15958d6efb32705c30aa7ec63259f6e7d47b71e1dcf3c64aacdc8a8` |
| `phase6x_evidence.json` | 1,901 | `cf0a415c940940623d74f4dd90934c04afec6b410b853a035488f5a736a49f9c` |
| `phase6x_evidence.md` | 1,051 | `f864c3a323c02e7a3600cff1a56c78befcd87a655c90842e174e4c7dddb4f86c` |

Reproduce the expensive native collection with:

```powershell
python scripts/phase6_opponent_deviation_values.py
```

Run artifact-only checks with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_opponent_deviation_values.py" -v
```

The competition runtime is unseeded, so independent executions are not claimed
byte-deterministic. The fingerprints above bind this observed run.
