# Final roadmap re-read

Review date: 2026-07-27

This review was performed after the exact packaged artifact was promoted to the
internal arena and completed games. It re-reads `PROJECT_PLAN.md`, the six-phase
acceptance matrix, and `PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md` against what
actually runs in `main.agent`, rather than accepting source-only interfaces as a
finished bot.

## Outcome first

The six-phase implementation and submission plan is complete. The submitted bot
is faithful to the intended architecture: a trained, frozen, dependency-free
public-belief neural blueprint owns strategic evaluation and action scoring;
beliefs, persistent plans, six value heads, the meta-network, an autoregressive
decoder, a selective fail-closed resolver, and official legality all execute on
the live path. The old heuristic and sampled-rollout controller are not the
final strategy provider.

The north-star performance goal is not complete. The 30-game internal-arena
snapshot is 4–26 at rating 355.2, and the replay audit shows severe strategic
output collapse. “Plan complete” here means the promised research/build/train/
evaluate/submit cycle ran to its honest conclusion—not that the resulting bot
is superhuman.

## Phase-by-phase re-read

### Phase 1 — strategic substrate

Re-read verdict: **complete**.

The submitted controller consumes structured official observations covering
cards, zones, attachments, damage, prizes, legal options, own-private data, and
public history. Its relational encoder hashes card and attack identities into
learned token features, runs token self-attention, and pools them into the
action query. `PlannerMemory` carries a five-plan latent across turns and resets
when the arena asks for the deck. Six separately trained strategic values feed a
state-conditioned softmax meta-network. Ordered multi-option selections are
decoded autoregressively with prefix features and native legality validation.

Evidence:

- `src/rocketbot/integrated_planner.py`
- `artifacts/integrated_controller/integrated_training_report.json`
- `artifacts/integrated_controller/runtime_parity_report.json`
- `tests/test_integrated_planner.py`
- `tests/test_submission_archive.py`

The held-out tactical ablation is material: full top-1 accuracy is 0.8273,
versus 0.7178 without relational tokens, 0.6929 without beliefs, and 0.6929
without plan inputs. Runtime parity against the PyTorch calculation passes at
`3.55e-15` maximum absolute error.

### Phase 2 — learn strategic intent

Re-read verdict: **complete as a deployed model; calibration is weak**.

The live model returns a distribution over six opponent intents, a six-archetype
range, and ten coarse resource probabilities. It retains multiple
minimum-enabling hidden-resource alternatives, including rational, imperfect,
and deceptive play styles, and reweights intent from observed behavior. Those
outputs condition both the plan bridge and tactical action query.

Evidence:

- `artifacts/integrated_controller/integrated_planner.json`
- `artifacts/integrated_evaluation/integrated_evaluation_report.json`
- `artifacts/integrated_evaluation/official_match_records.json`

The model should not be oversold. Untouched intent accuracy is 33.3%, and the
official robustness panel reports intent NLL 1.9000 and archetype NLL 1.9876.
The downstream held-out action ablation shows that belief inputs affect choices,
but it does not prove accurate human-intent inference.

### Phase 3 — search over strategic plans

Re-read verdict: **complete mechanically and on the live path; learned strategy
quality is poor**.

Five complete Team Rocket plans are scored under the public belief, persistent
plan state, and strategic values. The evaluator emits plan priors and values,
beliefs, hidden-resource hypotheses, uncertainty, six strategic values and
weights, an aggregate value, selected objective, gate result, and safety result.
The autoregressive action policy lifts the selected mixture into official legal
sequences.

Evidence:

- `src/rocketbot/integrated_planner.py`
- `PHASE6AN_INTEGRATED_CONTROLLER_PROTOCOL.md`
- `artifacts/internal_arena/HelbrosMewtwoSpidopsGTWTV1_replay_analysis.json`

The arena audit reveals the central failure: `board_setup` was selected on
1,235/1,382 decisions, while `prize_push` and `recovery` never appeared. The
bot implements plan-level reasoning, but the learned representation has
collapsed toward one plan under the arena distribution.

### Phase 4 — selective expert search

Re-read verdict: **complete after two failed iterations and a required restart**.

The first integrated gate failed to put more available search improvement in
gated than routine states. Phase 6AO also failed its concentration gate and was
kept only as rejected evidence. Phase 6AP retrained a monotone gate on
uncertainty, archetype entropy, strategic disagreement, and turn. On fresh
validation roots it gated 10/36 states with mean available improvement 0.1627,
versus 0.0753 for routine states.

Evidence:

- `PHASE6AO_SELECTIVE_SEARCH_GATE_PROTOCOL.md`
- `PHASE6AP_MONOTONE_SELECTIVE_GATE_PROTOCOL.md`
- `artifacts/integrated_controller/phase6ao_selective_search_gate_report.json`
- `artifacts/integrated_controller/phase6ap_monotone_selective_gate_report.json`

In 30 arena games the resolver was attempted 620 times but changed the safe
mixture only 15 times. This is fail-closed behavior, though it also shows that
the expensive component contributes little under the observed field.

### Phase 5 — search-guided learning loop

Re-read verdict: **complete for this frozen training cycle**.

The final planner was trained from 14,400 official terminal continuations over
180 roots with disjoint train, selection, bounds, and untouched partitions. Its
autoregressive policy used 27,776 official decisions, including 23,918 fresh
compatibility-deck teacher decisions and 3,858 archived ProV3 decisions. All
named layers were frozen into the dependency-free artifact. Five-action value
surfaces supplied counterfactual alternatives; split-conformal calibration
admitted an empirical absolute-advantage radius of 0.3824 with 95.56% untouched
marginal coverage.

The low-confidence tactical safety mixture was separately calibrated. At its
0.68255 threshold it obtained 94.17% exact-sequence accuracy on held-out data
while retaining neural control on 68.13% of nontrivial selections. It is a
tactical sequence provider only; the neural blueprint remains responsible for
strategic plans, beliefs, values, and safe resolving.

Evidence:

- `artifacts/integrated_controller/integrated_training_report.json`
- `artifacts/integrated_controller/tactical_safety_mixture_report.json`
- `artifacts/integrated_controller/gate_tactical_retrain_merge_report.json`

The search-guided cycle is technically closed, but its arena generalization is
not good enough. A future cycle should use the pulled arena distribution as a
held-out adversarial target and add anti-collapse objectives before further
distillation.

### Phase 6 — evaluation and game-theoretic pivot

Re-read verdict: **complete as an evaluation/submission loop; superhuman target
rejected by evidence**.

The fresh local evaluation contains 240 official-engine games, zero runner
illegal fallbacks, zero controller exceptions, and zero engine errors. The
60-game archetype/style robustness panel was 47–13; head-to-head results were
18–22 against both the legacy heuristic and tactical teacher. Component
ablations were mixed: full beat `no_beliefs` 12–8 and `no_safe_resolve` 16–4,
but lost 8–12 to both `no_plan` and `no_meta` in their small samples. These are
diagnostics, not proof of component value.

The game-theoretic safety pivot is live within a deliberately limited claim
boundary:

- the blueprint is learned and belief-conditioned but no equilibrium is
  claimed;
- counterfactual intervals are empirical conditional guards, not equilibrium
  safety bounds;
- the resolver changes play only when those scoped bounds admit it;
- the separate exploitation candidate went 23–37 with a lower 95% utility
  bound of −0.4815 against the −0.02 requirement, so it was rejected and its
  deployed weight is zero; and
- official engine transitions and native legality are genuine, while public
  belief, plans, value surfaces, and bounds are documented research
  abstractions.

Evidence:

- `PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md`
- `artifacts/integrated_controller/opponent_exploitation_overlay_report.json`
- `artifacts/integrated_evaluation/deployed_artifact_binding_report.json`
- `artifacts/submissions/HelbrosMewtwoSpidopsGTWTV1_smoke_report.json`
- `INTERNAL_ARENA_SUBMISSION.md`

## Submission fidelity decision

**Pass.** The exact archive submitted to the internal arena traces from
`main.agent` to `IntegratedBeliefPlannerAgent`, loads the frozen model named
above, constructs the public-belief/planning evaluation on every selection, and
uses the official engine selection contract. The packaged heuristic cannot
replace the plan, belief, value, meta, gate, or resolver; it is consulted only
as an admitted low-confidence tactical sequence provider. Sampled-rollout
search is absent from the final strategy path. The exploitation overlay is
safely disabled after failing admission.

This passes the user's “faithful bot submission” condition, so another automatic
restart is not triggered. It fails the aspirational strength condition, which
is recorded as the principal result rather than hidden.

## Next-cycle priorities

These are follow-on research priorities, not unfinished requirements of this
execution:

1. prevent plan/meta collapse with entropy floors, balanced plan targets, and
   state-conditional diversity checks;
2. train on arena-like opponent/deck distributions rather than the narrow local
   proxy panel;
3. calibrate beliefs against richer revealed-card outcomes and policy-family
   holdouts;
4. focus counterfactual labels on main-action disagreements and zero-attack
   losses; and
5. require a preregistered arena improvement gate before promoting a successor.
