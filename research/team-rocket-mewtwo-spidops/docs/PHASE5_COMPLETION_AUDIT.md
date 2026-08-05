# Phase 5 completion audit

Audit date: 2026-07-26

Authoritative scope: the Phase 5 section of `PROJECT_PLAN.md`.

| Requirement | Implementation evidence | Verification evidence | Result |
|---|---|---|---|
| Run self-play in which validated search improves selected decisions | `SearchGuidedSelfPlayEnvironment` protocol and `SearchGuidedSelfPlayRunner` evaluate roots, invoke Phase 4, thread compute, request paired validation, collect, and advance episodes | Demo runs two disjoint two-turn episodes; all four low-confidence roots invoke the full Phase 3 planner and receive complete-horizon paired evidence | Complete |
| Validate search rather than trusting its own estimate | `OutcomeValidation` is independent of `PlanSearchResult`; admission requires gain, confidence, samples, terminal/horizon attestation, environment identity, bounded uncertainty, and positive net utility | Four strong reports pass; a weak report is rejected for gain, utility, confidence, sample-count, and terminal checks | Complete |
| Save states where search meaningfully changes action or strategic plan | `SearchTeacherRecord` embeds observation, routine, search, changed flags, modal legal teacher action, targets, evidence, and provenance; collector requires action or plan change | All four admitted records change both action and plan; duplicate roots/IDs are rejected; full dataset canonical-JSON round-trips | Complete |
| Keep search-generated data reproducible | `ArtifactProvenance`, model/deck/environment/search fingerprints, seed, episode, schemas, and dataset fingerprint | Every record matches its deck and validation environment; dataset and derived gate dataset round-trip | Complete |
| Train the tactical policy | Teacher candidate sets and selected indices drive the real `NeuralAutoregressiveTacticalPolicy` through its recurrent decoder | Tactical-policy parameters receive finite nonzero gradients; held-out token accuracy remains 100% while NLL falls | Complete |
| Train persistent plan memory | Symbolic chosen plans receive deterministic fixed-width latent targets; MSE trains `PlanMemory` | Plan-memory parameters receive gradients; held-out plan-target MSE falls from 0.291 to below 0.001 | Complete |
| Train opponent beliefs | Scenario posterior is aggregated into soft six-intent and ten-resource targets | Belief model receives gradients; held-out intent cross-entropy falls from 1.864 to 0.225 and resource MAE from 0.475 to below 0.001 | Complete |
| Train strategic and aggregate values | Chosen plan posterior expectations supervise all six heads and meta-value | Both value subsystems receive gradients; held-out six-head MAE falls from 0.709 to 0.005 and meta MAE from 0.509 to 0.004 | Complete |
| Train uncertainty | Agent owns a four-output nonnegative search-uncertainty head supervised from the full result | Uncertainty parameters receive gradients; held-out uncertainty MAE falls from 0.527 to 0.006 | Complete |
| Add carefully audited counterfactual labels for a shortlist | `CounterfactualAuditor` checks same-root support, normalization, rollout completion, finite labels, bounded uncertainty, and score gap before weighting | Two alternatives per record pass every check; eight accepted labels train value, score, uncertainty, and ranking outputs | Complete |
| Absorb high-level search discoveries into routine policy | `PlanValuePolicyHead` scores plan descriptors without rollouts; chosen/alternative values and scores plus ranking loss train it | Held-out plan top-1 and counterfactual ranking accuracy improve from 50% to 100%; plan-value MAE falls from 0.461 to 0.010 | Complete |
| Retain search for genuinely hard states | Records distinguish provisional distill/retain dispositions; `SearchRetentionTarget` requires post-training paired revalidation and feeds refreshed gate supervision | Absorbed root has 0.02 remaining gain below 0.08 cost and bypasses search; hard root has 0.18 gain and invokes search exactly once | Complete |
| Keep compute and utility semantics separate | Teacher roots and records retain `compute_cost` and `search_value_cost` independently | Unit and end-to-end checks use compute cost for bank spend and value cost for admission/retention labels | Complete |
| Preserve train/held-out separation and restorable checkpoints | Dataset splits by episode; `DistillationCheckpointManifest` fingerprints all artifacts and disallows overlapping episode IDs | Held-out metrics use a separate episode; restored state dict has the exact student fingerprint; manifest canonical-JSON round-trips | Complete |
| Deliver a policy that no longer permanently depends on search | Student trains all named systems; refreshed calibrated gate chooses routine versus expert based on remaining utility | Held-out loss falls 95.0%; easy state returns routine, hard state returns full expert search, total post-training search invocations equal one | Complete |

## Verification commands

```powershell
python -B scripts/run_tests.py
python -B scripts/smoke_demo.py
python -B scripts/phase2_demo.py
python -B scripts/phase3_demo.py
python -B scripts/phase4_demo.py
python -B scripts/phase5_demo.py
```

Observed final result: 49/49 tests pass; all five Phase 1–5 demos return
`"status": "ok"`; Python syntax, v0.5.0 metadata/imports, 149 unique public
exports, local Markdown links, the exact 60-card deck, roadmap integrity, and
generated-cache cleanup all pass.

Deterministic Phase 5 acceptance values:

- two episode-disjoint train/held-out records each;
- four admitted validated searches and one deliberately rejected weak report;
- four action changes, four plan changes, and eight accepted counterfactuals;
- finite nonzero gradients to all seven named module groups;
- held-out multi-task loss reduction above 80% (observed 95.0%);
- held-out tactical, plan top-1, and alternative-ranking accuracy of 100%;
- exact student state-dict and checkpoint-manifest restoration;
- post-distillation gate positive invocation 100%, negative invocation 0%; and
- routine mode on the absorbed root, expert-search mode on the hard root.

## Scope boundary confirmed

The self-play runner, teacher admission, counterfactual audit, losses,
checkpointing, and retention logic are simulator-neutral. The checked-in
evidence uses deterministic strategic paired replay and the Phase 3 abstraction
adapter. It proves contracts, invariants, trainability, and selective knowledge
transfer—not Pokémon rules correctness, real match strength, opponent-panel
generalization, latency, or Kaggle performance. Those remain Phase 6 evidence.
