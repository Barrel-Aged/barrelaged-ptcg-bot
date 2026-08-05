# Phase 4 completion audit

Audit date: 2026-07-26

Authoritative scope: the Phase 4 section of `PROJECT_PLAN.md`.

| Requirement | Implementation evidence | Verification evidence | Result |
|---|---|---|---|
| Learn whether extra computation is valuable | `SearchGateNetwork` learns a binary profitable-search logit and an expected-gain head from `SearchGateSupervision` | Loss test proves gradients reach both learned heads; demo trains 480 examples and evaluates a disjoint test set | Complete |
| Supervise from policy-versus-search outcomes | `SearchGateSupervision.from_outcome` records chosen search value minus routine value and labels gain above cost | Dataset invariants and canonical JSON round-trip are tested | Complete |
| Detect low policy confidence | Extractor emits normalized policy uncertainty, entropy, and legal-action pressure | Unit test checks high uncertainty from a low-confidence pivotal decision | Complete |
| Detect disagreement among strategic value heads | Extractor computes six-head population deviation and range | Pivotal fixture has 0.447 disagreement versus 0.008 on the routine fixture | Complete |
| Detect high opponent-belief uncertainty | Plan, resource, and retained-hypothesis entropies remain separate normalized inputs | Feature-shape and normalization test covers the full 24-signal vector | Complete |
| Detect public critical thresholds | Explicit knockout, prize, evolution, disruption, board-collapse, combined KO, and combined prize signals use only agent-visible state | Feature extractor is exercised on an integrated Team Rocket combo/disruption observation | Complete |
| Detect close plans with different optionality | Cheap proposal priors produce ambiguity/entropy; plan axis alignment produces optionality spread | Integrated pivotal fixture exposes 0.874 ambiguity and 0.600 optionality spread | Complete |
| Calibrate invocation probabilities | Validation-only temperature search minimizes binary cross-entropy and always includes the uncalibrated baseline | Calibration test proves selected listed temperature does not worsen held-out BCE; calibrator round-trips | Complete |
| Choose a utility-aware search threshold | Threshold search maximizes realized gain minus configured search cost on validation data | Test metric is positive and retains at least 85% of oracle net utility | Complete |
| Concentrate search on pivotal states | `SelectiveSearchPolicy` calls the Phase 3 planner only after learned approval | Test split invokes on 98.2% of profitable cases and 0% of unprofitable cases | Complete |
| Leave routine sequencing to the policy | Declined decisions return the complete `RoutineDecision` and never attach a search result | End-to-end demo returns `routine_policy` and leaves compute unchanged | Complete |
| Respect a finite search budget | Immutable `SearchComputeBank` validates, spends, refills, serializes, and records before/after state | Primitive test covers spend/refill/overdraft; empty-bank demo blocks a positive gate request | Complete |
| Preserve rich expert-search evidence | Approved decisions retain the full Phase 3 `PlanSearchResult`; routine and expert modes have strict invariants | Routine, expert, and budget-limited decisions all canonical-JSON round-trip | Complete |
| Expose interpretable trigger evidence | Gate records explicit reason codes plus probability, gain, threshold, cost, and all features | Pivotal demo reports policy, value-disagreement, disruption, and optionality reasons | Complete |

## Verification commands

```powershell
python -B scripts/run_tests.py
python -B scripts/smoke_demo.py
python -B scripts/phase2_demo.py
python -B scripts/phase3_demo.py
python -B scripts/phase4_demo.py
```

Observed final result: 39/39 tests pass; every Phase 1–4 demo returns
`"status": "ok"`; package import/export, Python syntax, metadata, local Markdown
links, the 60-card deck invariant, and generated-cache cleanup all pass.

Expected deterministic Phase 4 acceptance values:

- 480 train, 110 validation, and 110 test examples;
- 99.1% held-out gate accuracy;
- 98.2% profitable-case invocation and 0% unprofitable-case invocation;
- 0.104 mean realized net utility versus 0.106 oracle net utility;
- routine bypass, one pivotal search invocation, and budget-limited fallback;
- all durable dataset, calibrator, and decision artifacts round-trip.

## Scope boundary confirmed

The checked-in benchmark uses deterministic synthetic supervision and the
Phase 3 strategic abstraction adapter. It proves signal coverage, trainability,
calibration, selective invocation, compute accounting, and serialization. It is
not evidence of Pokemon rules correctness, real-match search value, latency, or
tournament strength. Search-generated teacher collection and distillation were
outside this historical Phase 4 audit and are now implemented in Phase 5;
card-rule integration and competition evidence remain Phase 6.
