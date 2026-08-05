# Six-phase completion matrix

Final execution snapshot: 2026-07-27

This is the acceptance contract used for the completed six-phase research run.
“Pass” means that the promised component is trained or calibrated where the
roadmap requires it, connected to the submitted official action path, and backed
by checked-in evidence. It does **not** mean that the north-star goal of
superhuman strength was reached. The first internal-arena sample decisively says
it was not.

The research deck remains the exact 60-card NAIC transcription. The current
official engine lacks Prism Tower, so the submitted arena build is explicitly a
non-exact compatibility list replacing its two Prism Tower copies with two Team
Rocket's Watchtower copies.

## Phase 1 — strategic substrate

| Requirement | Deployed evidence | Result |
|---|---|---|
| Structured cards, zones, attachments, damage, prizes, and history | `planner_feature_items`, relational Pokémon tokens, public/private/history feature groups in `src/rocketbot/integrated_planner.py`; exercised on every packaged decision | Pass |
| Hybrid relational encoder | Learned token query/key/value attention and state-conditioned pooling; held-out top-1 falls from 0.8273 to 0.7178 under `no_relational` | Pass |
| Persistent latent plan | `PlannerMemory.plan_latent` updates across decisions, resets on the deck query, serializes in telemetry, and affects the live plan distribution | Pass |
| Six strategic value heads | setup, tempo, resource efficiency, prize control, board control, and optionality heads are frozen in `integrated_planner.json` and logged live | Pass |
| State-conditioned meta-value | Learned convex weights and aggregate value execute in the dependency-free runtime; runtime/PyTorch parity maximum error is `3.55e-15` | Pass |
| Autoregressive tactical policy | Ordered prefix decoding over current legal options, with native legality validation; 225 tests and 12 extracted-archive games pass | Pass |
| Explain strategic objective | Every live telemetry row contains the selected plan, dominant strategic axis, all values, meta-weights, beliefs, and search trace | Pass |

## Phase 2 — strategic intent and hidden information

| Requirement | Deployed evidence | Result |
|---|---|---|
| Opponent plan posterior | Six-intent posterior plus six-archetype range trained from the official corpus and used by the action query and plan bridge | Pass, weak calibration |
| Inverse hidden-resource model | Ten category probabilities and a public-consistent multi-hypothesis beam with rational, imperfect, and deceptive alternatives | Pass |
| Preserve uncertainty and alternatives | Entropy, posterior, and up to eight hidden-resource hypotheses persist in live telemetry without private opponent inputs | Pass |
| Beliefs affect decisions | Held-out action top-1 is 0.8273 full versus 0.6929 for `no_beliefs`; the 240-game official ablation panel is retained | Pass |

The untouched planner split achieved 69.4% archetype accuracy but only 33.3%
intent accuracy. The final 60-game robustness panel recorded archetype NLL
1.9876 and intent NLL 1.9000. These are evidence of a working but poorly
calibrated belief model, not a claim of human-intent recovery.

## Phase 3 — strategic plan search

| Requirement | Deployed evidence | Result |
|---|---|---|
| Propose a few complete plans | Five persistent Team Rocket plans with learned priors and values execute on every decision | Pass |
| Belief-conditioned comparison | Belief-to-plan bridge, plan values, retained hidden hypotheses, and behavior-likelihood intent updates feed the live evaluator | Pass |
| Rich result | Priors, final plan mixture, six values, meta-weights, uncertainty, beliefs, hypotheses, gate reasons, and safety result are serialized | Pass |
| No heuristic/sampled-rollout final strategy | Submitted `main.py` instantiates `IntegratedBeliefPlannerAgent`; manifest and extracted trace mark the frozen neural blueprint as strategy provider | Pass |

Arena replay review found a serious quality failure: 1,235 of 1,382 decisions
selected `board_setup`; only three of five plans appeared. The machinery is on
the action path, but its learned plan distribution is too collapsed for strong
play.

## Phase 4 — selective expert search

| Requirement | Deployed evidence | Result |
|---|---|---|
| Trigger on uncertainty, belief entropy, value disagreement, critical thresholds, and close plans | The deployed monotone gate uses uncertainty, archetype entropy, strategic disagreement, and turn; telemetry retains the complete trigger explanation | Pass |
| Concentrate computation where measured benefit is higher | `phase6ap_monotone_selective_gate_report.json`: gated mean improvement 0.1627 versus routine 0.0753 on fresh validation roots | Pass after restart |
| Fail closed | Unadmitted bounds, interrupted evaluation, or insufficient advantage return the neural blueprint exactly | Pass |

The earlier Phase 6AN and 6AO gates failed concentration and were rejected. The
plan was restarted at this step; only the Phase 6AP gate was admitted.

## Phase 5 — search-guided learning

| Requirement | Deployed evidence | Result |
|---|---|---|
| Official continuation/search supervision | 14,400 official terminal continuations across 180 roots, split 72/36/36/36 for training, selection, bounds, and untouched evaluation | Pass |
| Train all named heads and tactical policy | Frozen planner, beliefs, plan bridge, plan/value/meta/uncertainty/gate heads, relational encoder, and autoregressive action layers; all restore in the dependency-free runtime | Pass |
| Counterfactual alternatives | All five plan values per root plus 95% split-conformal absolute-advantage residual radius 0.3824; untouched marginal coverage 95.56% and worst archetype coverage 83.33% | Pass within empirical scope |
| Distill tactical sequencing | 27,776 official action examples; held-out top-1 0.8273 and NLL 0.5553 | Pass |
| Preserve a tactical safety layer | Low-confidence threshold 0.68255 raises held-out exact-sequence accuracy to 94.17% while leaving 68.13% of nontrivial actions under neural control | Pass |

The tactical provider is a bounded sequence-level safety component. It does not
choose the strategic plan, beliefs, values, gate, or safe-resolve mixture, and
the manifest explicitly forbids treating it as the strategy provider.

## Phase 6 — evaluation and game-theoretic pivot

| Requirement | Deployed evidence | Result |
|---|---|---|
| Tactical, strategic, belief, search, and robustness scorecards | `artifacts/integrated_evaluation/integrated_evaluation_report.json`, 240 official games and 12 captured replays | Pass |
| Public-belief information-set boundary | Structured public state/history and own-private inputs, complete native legal support, opponent range, and hidden hypotheses execute in the packaged controller | Pass as an abstraction over official execution |
| Neural blueprint is the live baseline strategy | Frozen standard-library model fingerprint `de6db2bc…afe4d`; no equilibrium claim; no heuristic or sampled rollout marked final | Pass |
| Counterfactual-value bounds | Empirical split-conformal bounds are admitted only in their measured scope and explicitly labelled “not equilibrium safety” | Pass within stated scope |
| Safe re-solving | Selective resolver uses admitted bounds and otherwise returns the blueprint exactly; full versus `no_safe_resolve` was 16–4 in the local ablation sample | Pass |
| Baseline-constrained exploitation | Candidate overlay went 23–37, lower 95% utility bound −0.4815 versus required −0.02; it was correctly rejected and deployed at weight zero | Pass by safe rejection |
| Package and official smoke | Archive SHA-256 `01c02185…dc315`; 17 manifest file hashes; 12 games, zero illegal fallbacks, engine errors, or controller exceptions | Pass |
| Internal-arena submission and replay review | `HelbrosMewtwoSpidopsGTWTV1` live; preliminary 30-game snapshot 4–26, rating 355.2; 1,382/1,382 replay decisions reproduced | Pass, strength poor |

## Completion verdict

All six implementation, integration, evidence, packaging, and submission phases
were executed. The submitted artifact is faithful to the requested
belief-aware, plan-conditioned neural architecture and the game-theoretic safety
pivot. The extracted-archive audit therefore passes and does not trigger another
automatic restart.

The strength target does **not** pass: the initial arena record is 4–26, its
95% Wilson win-rate interval is approximately 5.3%–29.7%, and replay telemetry
shows severe plan/objective collapse. This run completes the plan; it does not
establish a superhuman bot. The next research cycle should target plan/meta
diversity, stronger matchup-conditioned value targets, and arena-distribution
training rather than relabelling this result as strong.
