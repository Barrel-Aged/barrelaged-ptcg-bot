# Phase 2 completion audit

Audit date: 2026-07-26

Authoritative scope: the Phase 2 section of `PROJECT_PLAN.md`.

| Requirement | Implementation evidence | Verification evidence | Result |
|---|---|---|---|
| Keep agent-visible information separate from simulator truth | `observation.py`: `AgentObservation`, `ObservedAction`, `redact_state_for_observer` | `test_observation.py` redacts a named secret hand card, both decks/prizes, and a private opponent draw while preserving counts | Complete |
| Infer a compact distribution over strategic plans | `beliefs.py`: six `OpponentIntent` values and normalized `OpponentPlanPosterior`; `belief_model.py`: plan head | Shape/normalization tests and Phase 2 demo cover all six classes | Complete |
| Calibrate plan probabilities | `BeliefCalibrator.fit` uses held-out temperature selection; `evaluate_beliefs` reports NLL, Brier, ECE, and accuracy | Calibration unit test proves the selected temperature cannot worsen its fit objective; demo uses separate validation/test fixtures | Complete |
| Condition downstream decisions on plan/resource beliefs | `Phase2BeliefAwareAgent` conditions state, persistent plan, value heads, meta-value, and tactical policy; tactical policy accepts a 16-value belief context | Integrated shape/ablation test; held-out decision probe improves accuracy from 0.167 to 1.0 | Complete |
| Train from trajectories, revealed outcomes, and action patterns | `BeliefSupervision`, `TrajectoryBeliefDataset`, `BeliefBatchBuilder`, masked action-history encoder, and joint belief loss | Dataset round-trip, partial-reveal mask, and gradient tests cover both heads | Complete |
| Generate plausible opponent lines | `PlausibleLineGenerator` protocol plus deterministic `IntentTemplateLineGenerator` reference | Template generator exercised for every intent in belief and demo tests | Complete at Phase 2 abstraction level; card-legal line generation is the documented simulator adapter seam |
| Infer minimum enabling private resources | `StrategicLineHypothesis.minimum_resources`, `ResourceVector`, `MinimumEnablingHypothesisGenerator` | Beam tests validate normalized hypotheses and resource-vector invariants | Complete |
| Reweight hypotheses from observed decisions | `InversePlanningBeliefUpdater` combines feasibility, intent utility, style rationality, action-sequence emission, and random-action floor | Observed setup line raises the basic-Pokemon marginal from 0.323 to 0.756 and makes setup the leading plan | Complete |
| Update from later hidden-information reveals | `HiddenResourcePosterior.condition_on_reveal` | Unit test verifies a positive reveal further raises the relevant marginal | Complete |
| Preserve multiple hypotheses including imperfect play and deception | `OpponentStyle` rational, imperfect, and deceptive variants plus an unmodeled-play floor | Beam and end-to-end tests require all three styles and 19 retained hypotheses | Complete |
| Expose durable plan and resource beliefs | `OpponentBeliefState`, `HiddenResourcePosterior`, calibrator, observation, and dataset canonical JSON | Round-trip tests cover every durable Phase 2 artifact and neural/symbolic fusion | Complete |
| Beliefs improve decisions, not only belief metrics | Autoregressive policy directly consumes calibrated plan/resource context | Separate held-out synthetic decision probe requires at least +0.50 accuracy and currently obtains +0.833 | Complete as a wiring/trainability gate; real match strength remains a Phase 6 evaluation requirement |

## Verification commands

```powershell
python -B scripts/run_tests.py
python -B scripts/smoke_demo.py
python -B scripts/phase2_demo.py
```

Observed result: 26/26 tests pass; both demos return `"status": "ok"`.

## Scope boundary confirmed

Phase 2 does not claim card-rule simulation, card-legal opponent search, our own
plan proposer, tree search, search gating, self-play, or Kaggle packaging. Those
remain Phases 3–6. The synthetic decision probe proves that calibrated beliefs
are learned, persist, update, serialize, and materially affect decisions; it is
not presented as real-game performance evidence.
