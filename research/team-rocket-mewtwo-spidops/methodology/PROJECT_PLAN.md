# Project plan: superhuman Team Rocket's Mewtwo / Spidops bot

Last updated: 2026-07-27

This is the durable roadmap for the Pokemon TCG Kaggle simulation project. It
captures the full set of eighteen research ideas recorded in the coordinator
conversation, the selected target architecture, the six delivery phases, the
starting deck, and the intended implementation order. It is intentionally more
ambitious than the Phase 1 code. The current strategic pivot makes a trained
neural game-theoretic blueprint plus safe information-set re-solving the final
planning target; heuristic and sampled-rollout search remain integration
baselines. The durable pivot contract is
`PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md`.

## Final execution result

The six-phase run described below was completed on 2026-07-27 and produced the
internal-arena submission `HelbrosMewtwoSpidopsGTWTV1`. The submitted archive
uses a frozen public-belief neural blueprint as its strategy provider, with a
persistent plan, six strategic values, state-conditioned meta-weights,
autoregressive legal-action decoding, a selectively gated fail-closed resolver,
and a separately tested exploitation overlay that was rejected and disabled.
The packaged heuristic is limited to an admitted low-confidence tactical
sequence safety role; sampled-rollout search is not the final controller.

The official engine lacks Prism Tower, so the arena artifact is the explicitly
non-exact two-Watchtower compatibility build. Its archive SHA-256 is
`01c02185df68ae161d5883bbca2cb32431023e721e8064673f83200b548dc315`.

The outcome is scientifically negative on strength: the preliminary 30-game
arena snapshot is 4–26 at rating 355.2, and replay analysis finds collapsed
plan/meta behavior. This completes the roadmap's build/train/evaluate/submit
cycle but does not reach the superhuman north star. The authoritative closeout
documents are `FINAL_ROADMAP_REVIEW.md`, `FINAL_INDEPENDENT_AUDIT.md`, and
`INTERNAL_ARENA_SUBMISSION.md`; their final evidence supersedes older
“current repository status” paragraphs retained below as research history.

## Mission and north star

The goal is top-of-leaderboard, superhuman card play rather than a narrowly
incremental bot. The target is a **belief-aware neural game-theoretic planner**
that:

1. maintains a theory of what the opponent is trying to accomplish;
2. maintains calibrated hypotheses about hidden cards and resources that would
   support those plans;
3. operates on a provenance-tracked abstract information-set game;
4. obtains a baseline-safe mixed strategy from a trained neural blueprint;
5. safely re-solves pivotal subgames/endgames with opponent counterfactual-value
   bounds;
6. applies opponent-specific exploitation only through a separately constrained
   overlay; and
7. lifts the chosen strategy through a strong autoregressive card-sequencing
   policy and the official legality contract.

Validated search may still act as a teacher, but teacher targets must now pass
information-set, abstraction-fidelity, counterfactual-value, and safety audits.
Over time, routine competence moves into the blueprint while expensive safe
re-solving remains available for pivotal states.

```mermaid
flowchart LR
    H["Official public history, board, own hand, legal options"] --> B["Public-belief state and opponent range"]
    B --> I["Validated abstract information-set game"]
    I --> N["Trained neural blueprint strategy"]
    N --> C["Opponent counterfactual-value bounds"]
    I --> S["Safe subgame or endgame re-solving"]
    C --> S
    N --> S
    S --> E["Baseline-constrained exploitation overlay"]
    B --> E
    E --> T["Autoregressive action lifting"]
    T --> L["Official legality and selection contract"]
    L --> X["Official engine action"]
    X --> R["Telemetry, replay, and training evidence"]
```

The intended reasoning order is:

1. What is the opponent likely trying to accomplish?
2. Which hidden cards or resources would make that behavior rational?
3. What baseline-safe strategy does the blueprint assign to this information set?
4. Can the state be safely re-solved within counterfactual-value bounds?
5. Is any opponent-specific deviation both calibrated and safety-constrained?
6. What exact legal card sequence executes the selected distribution?

## The eighteen research ideas

These items preserve the original numbering from the coordinator conversation.
They are options in a shared research program, not eighteen commitments that
must be implemented simultaneously.

1. **Inverse hidden-hand belief modeling.** Generate the opponent's strongest
   plausible turn, infer the minimum private information needed to enable it,
   and update opponent beliefs from that explanation instead of relying only on
   forward-sampled hands. Retain multiple weighted hypotheses, allow imperfect
   play through a soft rationality model, and begin with card-category beams and
   short horizons.

2. **Plan-level belief modeling.** Infer the opponent's strategic intent before
   reconstructing individual cards, derive likely resources from the inferred
   plan, and reason over a smaller latent space such as setup, prize race,
   lock/disruption, board denial, resource conservation, or combo assembly.

3. **Counterfactual inverse planning.** Compare the observed opponent move with
   a theoretical best move under each hidden-state hypothesis, then weight
   hidden states by how well the observed behavior is explained. Include
   suboptimal, stylistic, and deceptive-play alternatives rather than assuming
   perfect rationality.

4. **Hierarchical reinforcement learning.** Separate strategic planning from
   tactical execution: a high-level controller selects an objective or plan,
   and a low-level policy performs the exact card sequencing.

5. **Multi-objective reinforcement learning.** Avoid optimizing a single
   intermediate objective per turn. Jointly model attacking setup, resource
   development, tempo, prize pressure, and board positioning while retaining
   terminal match outcome as the ultimate objective.

6. **Independent strategic value heads plus a meta-network.** Learn separate
   estimates for setup quality, tempo, resource efficiency, prize control, and
   board control, then learn state-conditioned weights that combine them. This
   project adds optionality as a sixth explicit head so it can be inspected and
   bounded rather than hidden inside the aggregate.

7. **Optionality reward shaping.** Reward useful future decision flexibility:
   preserving multiple attackers, conserving search resources, and maintaining
   multiple playable lines. Keep it bounded and validate that flexibility is
   not rewarded when immediate conversion is required.

8. **Counterfactual value prediction.** Estimate how the game would have
   changed under another legal action to provide denser supervision than final
   win/loss alone. Treat labels as uncertain and expensive; validate them before
   using them as training targets.

9. **Persistent latent game-plan representation.** Carry an evolving internal
   plan embedding across turns and condition decisions on it instead of treating
   each turn as independent. Keep serializable snapshots so plans can be audited,
   replayed, and eventually supervised.

10. **Hybrid neural reinforcement learning with tree search.** Use a neural
    policy and value system for priors and evaluation, then refine hard decisions
    with search, adapting the AlphaZero philosophy to imperfect information.

11. **Selective tree search.** Let the learned policy handle routine positions
    and invoke additional computation when confidence is low, value estimates
    disagree, belief entropy is high, or the state is strategically critical.

12. **Search over plans rather than individual actions.** Have the learned model
    propose a handful of complete strategic lines and search only those lines,
    using a tactical rollout policy to reduce the raw action branching factor.

13. **Search-generated training targets.** When a validated search process finds
    a materially stronger choice than the policy, store it as supervised data so
    the policy gradually absorbs search knowledge.

14. **A rich reinforcement-learning/search interface.** Exchange policy priors
    and scalar values plus uncertainty, strategic head values, belief
    distributions, inferred plans, high-level objectives, and explanatory traces.

15. **Transformer-based state encoding.** Replace hand-designed feature
    interactions with attention over cards, zones, attachments, damage, prizes,
    and history so the model can learn relational effects. Introduce this only
    when data and compute justify it; the Phase 1 interface does not require a
    particular encoder family.

16. **Policy-value architecture with selective expert search.** Use the neural
    system for most decisions and search as an expert on selected states, gaining
    strategic strength without paying full search cost on every choice.

17. **Belief-aware tree search.** Search over a maintained hidden-state and plan
    distribution instead of determinizing a single hidden state. This is better
    matched to Pokemon's partial observability but is the most technically
    demanding search component.

18. **Search-guided self-play.** Improve selected self-play moves with search,
    train the policy on stronger outputs, and gradually reduce routine reliance
    on search as the policy improves.

## Target deck specification

The fixed starting point is the user-supplied transcription of Juho Kallama's
7th-place NAIC 2026 Team Rocket box list. The machine-readable source of truth is
`src/rocketbot/data/naic_2026_juho_kallama_mewtwo_spidops.json`.

### Pokemon — 15

- 2 Team Rocket's Mewtwo ex
- 4 Team Rocket's Tarountula
- 4 Team Rocket's Spidops
- 2 Team Rocket's Articuno
- 2 Team Rocket's Mimikyu
- 1 Lillie's Clefairy ex

### Trainers — 34

- 4 Lillie's Determination
- 4 Team Rocket's Ariana
- 3 Team Rocket's Giovanni
- 2 Team Rocket's Proton
- 1 Team Rocket's Petrel
- 4 Team Rocket's Transceiver
- 4 Ultra Ball
- 3 Night Stretcher
- 2 Bug Catching Set
- 2 Lucky Helmet
- 2 Prism Tower
- 1 Team Rocket's Factory
- 1 Energy Switch
- 1 Maximum Belt

### Energy — 11

- 4 Team Rocket's Energy
- 6 Basic Grass Energy
- 1 Basic Psychic Energy

**Integrity invariant:** 15 + 34 + 11 = 60 cards. Later experiments may branch
from this baseline, but every branch must record its own immutable deck
fingerprint so results cannot silently mix configurations.

## Six-phase delivery roadmap

### Phase 1 — Strategic substrate

Build representations and contracts shared by all later experiments:

- versioned deck configuration;
- structured state, zones, attachments, damage, prizes, actions, and turn history;
- deterministic state features with an encoder seam that can later accept a
  relational or transformer implementation;
- persistent and serializable latent game-plan state;
- separate setup, tempo, resource-efficiency, prize-control, board-control, and
  optionality value heads;
- a state-conditioned meta-value aggregator; and
- an autoregressive tactical-policy interface for multi-step legal selections.

Deliverable: a runnable policy/value scaffold that exposes which strategic
dimensions it is optimizing, with integrity, shape, masking, and serialization
tests. No training, tree search, simulator rules, or opponent-hand beliefs yet.

**Current repository status:** implemented and deterministically testable.

### Phase 2 — Learn strategic intent and beliefs

Build two complementary models.

**Opponent plan model**

- Infer a calibrated distribution over compact strategic intents.
- Condition policy and future search on that distribution.
- Train from trajectories, revealed outcomes, and action patterns.

**Inverse hidden-information model**

- Generate strong plausible opponent lines.
- Infer minimum enabling card categories and private resources.
- Reweight hypotheses by how well they explain observed decisions.
- Preserve imperfect-play, style, and deception alternatives.

Deliverable: calibrated plan and hand-category beliefs that improve decisions,
not merely belief metrics.

**Current repository status:** implemented and deterministically testable. The
neural and inverse-planning paths are exercised on held-out synthetic trajectory
fixtures, including calibration and tactical decision improvement. Real
simulator trajectories and matchup evaluation remain Phase 6 evidence, not a
claim made by the fixture.

### Phase 3 — Search over strategic plans

- Add a high-level plan proposer that returns a few complete, distinct lines.
- Search those plans under the current belief distribution instead of expanding
  every raw action.
- Use the tactical policy to execute or roll out each plan.
- Compare predicted and observed opponent behavior under each intent/hidden-state
  hypothesis.
- Return the rich interface: priors, strategic values, aggregate value,
  uncertainty, beliefs, inferred plans, and objective.

Deliverable: a belief-aware planner that selects among meaningful game plans.

**Current repository status:** implemented and deterministically testable. The
Team Rocket proposer returns complete distinct plans; every retained Phase 2
hypothesis is behavior-reweighted and searched; the tactical policy executes
each branch autoregressively; and the planner returns a canonical rich result
with priors, six-head values, aggregate values, uncertainty, beliefs, inferred
intent, objectives, rollouts, and search-cost statistics. The checked-in
strategic abstraction validates planner mechanics but does not replace a future
card-rule simulator adapter.

### Phase 4 — Selective expert search

Learn when extra computation is valuable. Candidate triggers include:

- low policy confidence;
- disagreement among strategic value heads;
- high opponent-belief entropy;
- nearby knockout, prize, evolution, disruption, or board-collapse thresholds;
- close expected values with sharply different optionality.

Deliverable: a calibrated search gate that concentrates search budget on pivotal
turns and leaves routine sequencing to the policy.

**Current repository status:** implemented and deterministically testable. A
24-signal extractor covers every listed trigger without running rollouts. A
learned two-head network predicts profitable invocation and expected search
gain; held-out temperature scaling calibrates its probabilities and a
utility-aware threshold maximizes realized gain net of search cost. An immutable
compute bank constrains approved searches, while `SelectiveSearchPolicy` bypasses
the Phase 3 planner on routine states and preserves a reasoned, serializable
decision trace. The synthetic benchmark demonstrates selective-compute mechanics
and trainability; real card-rule trajectories remain the source of deployment
labels and competition evidence for Phase 6.

### Phase 5 — Search-guided learning

- Run self-play in which validated search improves selected decisions.
- Save states where search changes the preferred action or strategic plan by a
  meaningful margin.
- Train policy, plan memory, beliefs, values, and uncertainty from those targets.
- Add carefully audited counterfactual labels for a shortlist of alternatives.
- Distill routine improvements while retaining search for genuinely hard states.

Deliverable: a policy that absorbs reliable strategic discoveries instead of
permanently depending on search.

**Current repository status:** implemented and deterministically testable. An
environment protocol and episode runner execute Phase 4 selective search inside
self-play, preserve compute-bank transitions, and send every searched root
through independent paired-outcome admission. Versioned teacher records bind the
routine alternative, full search result, modal legal tactical trace, plan,
belief, value, uncertainty, and audited counterfactual targets to deck,
environment, model, search, seed, and episode fingerprints. Multi-task
distillation trains the tactical policy, persistent plan memory, opponent belief
model, six value heads, meta-value, search uncertainty, and a cheap plan policy.
After training, paired revalidation relabels absorbed roots as routine while
retaining expert search where remaining gain still exceeds its value-equivalent
cost. The reference evidence uses deterministic strategic paired replay; real
card-rule self-play and tournament strength remain Phase 6 claims.

### Phase 6 — Neural game-theoretic planning and competition loop

Evaluate strength by dimension and end outcome:

- **Tactical accuracy:** sequencing, search order, attachments, evolution timing,
  retreat decisions, and lethal conversion.
- **Strategic quality:** setup preservation, prize mapping, board liabilities,
  tempo, resources, and optionality.
- **Belief quality:** calibration after actions and eventual private-card reveals.
- **Search value:** improvement over policy alone, concentrated in critical states.
- **Robustness:** distinct archetypes, adversarial styles, and counter-strategies.
- **External outcome:** Kaggle leaderboard movement and expert replay review.

Deliverable: a reproducible competition agent plus evidence showing which
components create strength and where failures remain.

**Current repository status:** the Phase 6 software and local evidence loop are
implemented against the official competition engine as an **integration
baseline**. A fail-soft standard-library agent maintains a persistent plan, six
strategic diagnostics, an archetype posterior, and optional deadline-bounded
native look-ahead while validating every returned selection. The heuristic
policy and sampled rollout are not the final strategy. A 144-match matrix spans
four ablations, six legal archetype profiles, three adversarial styles, and both
seats; it records tactical, strategic, belief, search, robustness, latency, and
replay evidence.

The game-theoretic pivot now supplies executable, serializable contracts for a
public-belief opponent range, provenance-enforced abstract information-set game,
neural blueprint provider, counterfactual-value bounds, safe resolver, and a
separate baseline-constrained exploitation overlay. The only concrete
full-game neural candidate is Phase 6K's sampled-regret wiring; Phase 6L now
validates the corrected external-sampling estimator exactly, and Phase 6M
wires it to a tiny immediate-terminal native slice. Phase 6N adds an exactly
audited, linear-trajectory outcome-sampling alternative, and Phase 6O applies
that estimator to five persistent turn-plan actions over genuine native
descendants, producing both-player regret memories, a separate strategy
memory, and three restorable neural models. Its transformed heavy-tailed
objective is still not equilibrium evidence. Phase 6P closes the iterative
collect–fit–update loop but worsens the raw-regret tail enough to fail the
counterfactual-calibration admission gate. No validated full-game
equilibrium blueprint exists. Phase 6Y validates the missing target-player-
excluding reach ratio exactly and persists a redacted native reach ledger.
Phase 6Z repairs its one-history-per-information-set shape with repeated hidden
allocations behind disjoint later roots and a split-safe conditional-value
diagnostic. The supplied prior, fixed acting-player hand, and absent upstream
reach still keep counterfactual calibration closed. Phase 6AA adds a public-
only six-archetype `OpponentRange` diagnostic over those frozen roots. It finds
held-out archetype signal but lacks the root count, calibration, observer
orientation, and hidden-resource scope needed for an official posterior. Phase
6AB then recollects fresh official prefixes and records all deterministic player
choices to the same later-root shape. Its 512 selections are genuine engine
executions, but automatic chance-event likelihoods are absent from the live API;
chance/full reach and counterfactual weights remain unavailable and no Phase 6Z
value-root join exists. Phase 6AC now performs that join in a fresh collection:
18 prefixes, transient realized official truth samples, and all five plan
values share exact live roots, with native root and next-draw replay parity.
The shared opaque official chance law cancels from the on-policy likelihood
ratio, but absolute chance reach is still unavailable, every information set
has one truth sample, and the frozen five-plan rollout profile remains abstract.
This validates a collection substrate, not calibrated counterfactual values or
safety. Phase 6AD scales the official truth path to 180 new roots and introduces
an explicit player-0 observer orientation. A train-only six-candidate provider
and public-consistent allocation marginals pass preregistered candidate and
hidden hand/Prize/deck bucket gates on 60 calibration and 60 untouched
validation roots. That pass is limited to the six exact candidate lists,
baseline policies, and later-root distribution. Phase 6AE then removes the
same-policy seat confound: player 0 remains fixed while six controlled player-1
intent proxies are crossed with all six lists over 324 fresh roots. Held-out
declared-proxy, joint-range, and resource calibration pass, but the labels and
policies remain controlled abstractions. Phase 6AF then fits across two policy
implementations, calibrates on a third, and keeps a fourth untouched until
validation. Its 432-root aggregate and per-intent transfer gates pass. Human
latent intent, full official belief,
absolute chance reach, counterfactual values, safety, exploitation, and control
remain unavailable. Phase 6AG next collects 288 complete official games under
the same four-way implementation isolation and fits a public-plus-acting-private
neural terminal-outcome leaf. Its untouched-implementation point MSE
`0.938795` beats zero `1.000000` and the causal board heuristic `1.019981`.
The 90% progress-bucket intervals fail coverage and width gates, so expected-
value intervals, counterfactual values, resolving, exploitation, and control
remain unavailable. Phase 6A converts
six native truth replays into 12 isolated player perspectives with allow-list
redaction, chained fingerprints, later-reveal supervision, and zero detected
structural leakage. Schema 2 corrects the official visualizer's next-frame
`selected` semantics, yielding 610/610 contract-valid source-prompt labels and
excluding behavior targets from feature fingerprints. Phase 6B adds a 55-state
exact imperfect-information game,
analytic equilibrium and best-response metrics, deterministic tabular CFR, and
a genuine per-hypothesis bounded resolver for that microgame. Phase 6C adds two
deterministic neural candidates, exact best-response evaluation, fingerprinted
JSON checkpoints, and exact restoration; the selected policy reaches
`0.00004637` NashConv on that microgame. A one-step official
observation/action bridge now runs transparently in shadow; its 12-match native
panel round-trips all 566 observed target decisions without changing an action.
Reach-corrected full-game equilibrium training, replay-derived counterfactual
calibration, and official full-game safe re-solving remain future work. Phase
6J later supplies a branchable native-search boundary under a supplied hidden
prior, Phase 6K exercises it with a small sampled-regret candidate, Phase 6L
repairs and exactly validates the sampling estimator, and Phase 6M validates
only narrow native endgame plumbing. Phase 6D calibrates the
microgame neural value residuals on 12 examples and achieves 100% coverage on a
12-target disjoint calibration split. It also validates componentwise widening,
fail-closed uncertainty gates, and zero false-safe certificates over six exactly
enumerated resolve roots.
Phase 6E adds a budget-aware all-or-blueprint resolver. Nine endgame and three
earlier roots complete the same 102-mixture exact grid with strict per-hypothesis
opponent non-regression; five endgame strategies change for a small positive
aggregate exact conditional gain. Iteration, node, deadline, and OOD failures
return the blueprint exactly and expose no partial-search strategy.
Phase 6F trains two separate synthetic opponent models and evaluates final
strategies against all 64 pure opponent continuations. Six roots enable bounded
exploitation; model-matched held-out gain is positive, adversarial
misclassification and distribution shift stay within the declared 0.01 per-root
tolerance, and weak/wrong-seat models return the safe strategy.
Phase 6G validates the one-decision agent-visible information boundary,
allow-list redaction, complete ordered action enumeration, and exact action
lifting in live official-engine shadow execution. This does not supply a
full-game abstract transition model or deploy the neural/safe/exploit stack.
Phase 6H converts the corrected replays into 610 typed logged information-set
steps with complete ordered actions, observed continuation links, terminal
returns, causal later-reveal supervision, deterministic train/calibration/
validation splits, and a public-only six-deck range baseline. It is explicitly
offline and non-branchable, so full-game game-theoretic training still requires
a counterfactual transition abstraction or generative official-search bridge;
Phase 6J later supplies the latter boundary.
Phase 6I adds deterministic multi-task pretraining over those logged steps:
train-only named normalization, separate public/private encoders, complete
legal-action behavior scoring, public-only range prediction, outcome and
observed next-public auxiliary heads, and exact checkpoint restoration. This is
explicitly heuristic behavior imitation and representation initialization, not
the full-game equilibrium blueprint or a deployable strategic policy.
Phase 6J adds the missing branchable full-game boundary using the licensed
native search API. An explicit chance root selects among six supplied hidden-
zone particles, retained native states support alternative counterfactual
actions, manual coins are chance nodes, and all six exercised paths reach
native terminal states with both players represented. Native descendants are
genuine engine transitions; the particle prior remains an uncalibrated research
abstraction, so the game object is not labeled official and no strategy is
deployed.
Phase 6K initializes a full-game neural candidate from the Phase 6I checkpoint
and trains separate average-policy, advantage, and value heads on terminal-
valued Phase 6J branches. Its two-round live corpus has 24 information sets and
15,452 genuine native branch transitions with exact restoration and legal
support. It is sampled regret-matching wiring, not reach-corrected Deep CFR;
held-out diagnostics are mixed, the particle prior remains uncalibrated, and
the model is neither equilibrium-validated, safe, deployable, nor credited with
a strength gain.
Phase 6L implements external-sampling MCCFR with separate advantage and
average-strategy reservoirs and fail-closed traversal budgets. Exhaustive
enumeration on the exact non-official microgame matches conditional advantages
to `9.99e-16`; a 20,000-iteration tabular reference reaches NashConv
`0.00461750`. This validates the estimator, not full-game convergence.
Phase 6M reuses the genuine native descendant boundary for five admitted
immediate-terminal roots across both players. It records five zero-advantage
samples over 45 external-sampling native transitions, with exact semantic
feature widths and redacted reservoir restoration. Broader traversals exceeded
their budgets and were discarded; there are no opponent-node strategy samples,
useful nonzero regrets, neural training result, equilibrium certificate, or
strength claim.
Phase 6N implements outcome-sampling MCCFR with a corrected nonzero pre-action
baseline, separate regret/average-strategy reservoirs, and fail-closed budgets.
Its exhaustive 60-trajectory microgame audit has maximum expectation error
`4.16e-17`; a deterministic 20,000-iteration reference reaches NashConv
`0.03111612`. This is a higher-variance estimator reference only.
Phase 6O changes the native game-theoretic action from procedural option indices
to five persistent turn plans, with a fixed plan-conditioned tactical executor.
Thirty-six trajectories across both players and all six hidden completions
produce 238/214 regret samples and 452 strategy samples over 5,569 genuine
native descendant transitions. Three separate networks train and restore
exactly. Raw regrets remain extremely heavy-tailed and the `asinh` neural target
changes their magnitude, so the model is not equilibrium-validated, safe,
deployable, or credited with strength.
Phase 6P then closes the iterative collect–fit–update loop: each of three rounds
uses the previous player-regret models to generate the next trajectories and
fits all three networks on cumulative separate memories. The implementation
works, but raw-regret p95 grows to `5.04e11` and the maximum to `1.24e16`.
That failed variance gate blocks counterfactual calibration and safe resolving;
it is a diagnostic result, not convergence.
Phase 6Q makes the block executable: a fingerprint-bound admission audit passes
only provenance/redaction/restoration and rejects 13 other prerequisites. It
constructs no counterfactual bounds and runs no resolver or exploitation.
Phase 6R introduces a general cutoff boundary and first verifies it exactly on
the non-official microgame: 18 pre-cutoff regret expectations match to
`5.55e-17`. A six-step live probe then bounds the observed raw-regret p95 /
maximum to `17.90` / `134.92` across 36 trajectories and 1,474 genuine native
descendant transitions. The leaf is still an uncalibrated Phase 6K scalar with
confidence zero and the full utility interval, so the resulting samples are a
variance diagnostic rather than admissible counterfactual targets.
Phase 6S evaluates that leaf on disjoint within-zone hidden-order particles.
Eighteen cutoff examples branch into 54 genuine terminal continuations across
all six candidates. The residual interval attains nominal held-out coverage
only by spanning the full `[-1, 1]` utility range; informative fraction is zero,
and the static public range is identical to uniform NLL. This rules out
promoting either the leaf or range as calibrated and identifies hidden-zone
allocation diversity, history-conditioned range learning, and lower-noise
continuation targets as the next prerequisites.
Phase 6T then resamples allocation between hidden zones and evaluates the
Phase 6I public-only range head on 216 native decision rows. On 72 held-out
allocation rows the neural NLL (`1.8109`) is worse than uniform (`1.7918`) and
far worse than the causal revealed-card updater (`0.8959`). This rejects direct
neural transfer and promotes the causal updater only to the next experimental
baseline. Belief calibration, leaf calibration, and safety remain false.
Phase 6U then raises every allocation-heldout leaf to ten terminal
continuations. Maximum rollout standard error falls to `0.333`, below the
precision gate, but calibration selects a zero baseline over the Phase 6I and
Phase 6K value heads. The conformal interval remains full-width. This isolates
model bias/representation as the remaining leaf blocker rather than sampling
precision; no counterfactual-value or safety claim follows.
Phase 6V then trains a public-only ridge leaf specifically on Phase 6U variants
0-1, selecting regularization with leave-one-candidate-out training folds. A
new 24-particle native run supplies untouched calibration and validation
allocations. The new leaf beats every reference point predictor on fresh
validation (`0.446013` RMSE versus zero's `0.489898`), but its conformal interval
is still nearly full-width (`1.997117`). The empirical gate remains false; the
result narrows the next problem to data breadth, representation, and
hypothesis-conditioned counterfactual targets rather than merely swapping in a
generic value head.
Phase 6W strengthens the split to three complete, identity-disjoint native
collections: 24 rows for training, 24 for conformal calibration, and 24 from a
newly generated collection for untouched validation. Coverage passes at
`91.667%`, but public-ridge RMSE degrades to `0.595137` versus the zero
baseline's `0.493288`, and mean interval width remains `1.721623`. This rejects
the Phase 6V point improvement as non-generalizing and rules out further
promotion of the same single-vector ridge family.
Phase 6X changes the target shape rather than fitting another scalar. At a
player-1 decision for each of 12 supplied hidden allocations, it forces all five
opponent plans and runs ten paired continuations per action. The resulting 60
action targets cover all candidates in calibration and validation; validation
mean action span is `0.633333`, mean empirical deviation gain is `0.286358`,
and maximum standard error is `0.333333`. The action-conditioned substrate gate
passes, but the values remain conditional on sampled reach and are explicitly
not opponent-reach-weighted counterfactual values.
Phase 6Y makes that reach convention executable. All 24 exact-microgame
information-set/action rows reproduce direct counterfactual masses and values
at zero error. A fresh Phase 6X-style native run binds redacted probability
ledgers to 12 surfaces, 600 terminal continuations, and 70,741 genuine native
transitions. Every macro root is already the first opponent plan decision, so
the strategic prefix is empty and all importance weights are `1.0`. Because
all 12 public information sets are unique and the hidden prior is supplied,
this validates reach bookkeeping only; native counterfactual values and safe
resolving remain uncalibrated and disabled.
Phase 6Z repairs the public-root shape with 18 later live roots assigned wholly
to train, calibration, and validation. Each root contains three distinct,
zero-substitution hidden allocations that share the same public belief,
history, information set, and feature vector; all five plans receive three
paired terminal continuations. A hypothesis-and-action-conditioned ridge model
is trained only on train roots and receives per-action conformal calibration on
the second collection. On the untouched third collection, coverage is only
`75.555556%` against a `90%` target and RMSE `0.847985` loses to zero's
`0.771402`; the ridge diagnostic is rejected. This is still conditional fixed-
profile supervision: the supplied prior, fixed acting-player hand, absent
upstream reach, and abstract five-plan boundary keep counterfactual values and
safe resolving disabled.
Phase 6AA derives a public-only six-archetype diagnostic from those frozen
roots. Its narrow held-out signal does not satisfy sample-count, calibration,
hidden-resource, intent, or observer-orientation requirements. Phase 6AB then
collects 18 new live behavior prefixes and 512 genuine official selections.
The deterministic player-policy factors are exactly one, but the adapter does
not expose shuffle, draw, Prize, or coin probabilities. Therefore chance reach,
full sampling reach, counterfactual importance weights, and alignment with the
separately collected Phase 6Z values remain explicitly unavailable. Phase 6AC
uses a new same-root collector to bind each complete prefix to one transient
realized official hidden-state sample and 15 terminal plan continuations. All
18 reconstructed native roots and forced next-draw observations match their
live official counterparts. Common target/sampling chance cancels symbolically,
but absolute chance reach, repeated truth samples, calibrated values, bounds,
resolving, exploitation, and control remain unavailable.
Phase 6AD separately collects 180 fresh official roots and transient player-1
truth labels behind a player-0 public view. Its 5,178 prefix selections are
genuine execution. The six-list prototype model and 34,560 allocation particles
are research abstractions. Held-out candidate/resource calibration passes only
within that declared distribution; opponent intent, complete belief, values,
bounds, resolving, exploitation, and strategy control remain false.
Phase 6AE separately fixes player 0 to one policy and varies only player 1 over
six declared strategic-intent proxies. Its 324-root official collection contains
8,474 selections and 875 public player-1 MAIN actions. The train-only intent
model and frozen Phase 6AD candidate/resource model form a calibrated 36-entry
range in this controlled panel. The proxy labels are not human latent truth;
complete belief, chance reach, values, bounds, resolving, exploitation, and
strategy control remain false.
Phase 6AF expands the controlled family to four split-isolated implementations.
Robust selection cross-fits the two training policies, uses a third for
calibration, and never reads the fourth validation policy. The 432-root panel
passes aggregate intent/joint/resource and per-intent slice gates. This removes
exact-policy memorization as evidence but still cannot establish human latent
intent or enable values, bounds, resolving, exploitation, or strategy control.
Phase 6AG uses 288 complete official matches to validate a narrow neural point-
prediction gate on a fourth untouched implementation. It does not admit its
terminal-outcome intervals: overall coverage is `0.862791`, overall mean width
is `1.937685`, and endgame width is `1.753315`. The next admissible experiment
must estimate conditional expected values from repeated independent native
continuations at fresh implementation-heldout roots.
Phase 6AH executes that experiment with 180 complete matches, two supplied
allocations per retained root, all five plans, and eight paired continuations
per allocation/action. Maximum target SE `0.258199` passes, but action-contrast
MSE loses narrowly to zero-contrast and conformal coverage, width, and worst-
slice thresholds fail. The next admissible value experiment must improve
action-sensitive representation/objectives and demonstrate held-out contrast
signal before increasing rollout volume or approaching counterfactual bounds.
Phase 6AI records the immediate public child behind every forced plan on 180
new roots. Its grouped neural model improves aggregate point error but fails
the preregistered action-contrast, bootstrap, and interval gates. A standalone
causal public branch heuristic shows promising action ordering only as a
diagnostic; it is not retroactively promoted from that panel.
Phase 6AJ freezes a semiparametric hybrid—neural/root-heuristic state mean plus
the centered causal branch heuristic—using only Phase 6AI training/selection/
bounds partitions. On 36 new `action_first` roots, hybrid point MSE `0.582840`
beats all three frozen baselines, contrast MSE `0.022868` beats zero `0.025278`,
and the 90% root-bootstrap lower bound is positive (`0.000753`). This validates
the narrow action-sensitive point signal. Conditional expected-mean admission
still fails because mean interval width `1.839791` exceeds `1.75`; no
counterfactual, equilibrium, safety, exploitation, or control capability is
enabled.
Phase 6AK next declares all 216 Phase 6AI/6AJ roots to be development data and
selects a strongly regularized residual ridge root mean by five source/policy
folds. The model is frozen before 36 new calibration roots, and the conformal
radius is frozen before a distinct 36-root validation panel. Fresh point MSE
`0.398773` beats Phase 6AJ `0.503581` and every frozen nonlearned reference;
contrast, bootstrap, precision, and collection-holdout gates pass. Coverage
`0.916667` and mean width `1.676068` pass, but worst-slice coverage `0.666667`
misses the `0.75` floor. The scoped point model is validated while conditional
expected means and every counterfactual, equilibrium, safety, exploitation,
and control capability remain rejected.
Phase 6AL freezes the exact Phase 6AK mean and tests whether target noise caused
that rejection. It preregisters two independent replay batches per fresh root,
32 terminal returns per action, a 92% label-free conformal calibration target,
and unchanged validation gates. The 36-root calibration panel is completed and
frozen before the disjoint 36-root validation panel begins. Maximum target
standard error falls to `0.179605` and point MSE `0.430678` beats every frozen
reference, but the contrast-bootstrap lower bound is `-0.000512`, overall
coverage is `0.888889`, and worst-slice coverage is `0.666667`. The frozen
protocol therefore rejects action-sensitive point and conditional expected-mean
admission. No counterfactual, equilibrium, safety, exploitation, or control
capability is enabled.
Phase 6AM then stops treating another learned absolute rollout mean as the next
strategic step. Before collecting any new root, it declares all 360 Phase
6AI-6AL roots as development evidence and freezes a baseline-relative causal
branch center plus a `0.555019619275336` conformal advantage radius. It joins
that certificate to the admitted in-scope Phase 6AD public range and evaluates
36 fresh roots with two replay batches. Coverage `0.95`, worst-slice coverage
`0.833333`, width `1.110039`, bootstrap, range, reach, and holdout gates pass.
Maximum target SE `0.201688` misses the preregistered `0.20` ceiling, so the
normalized counterfactual-advantage bound matrix is rejected and may not feed
safe resolving. This negative result preserves the architectural pivot while
leaving absolute values, equilibrium, resolver, exploitation, and control
capabilities disabled.
See `PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md` and
`docs/PHASE6A_B_GAME_THEORY_FOUNDATION.md` plus
`docs/PHASE6C_NEURAL_BLUEPRINT.md` and
`docs/PHASE6D_COUNTERFACTUAL_BOUNDS.md` plus
`docs/PHASE6E_SAFE_RESOLVING.md` and
`docs/PHASE6F_CERTIFIED_EXPLOITATION.md` plus
`docs/PHASE6G_OFFICIAL_SHADOW_BRIDGE.md` plus
`docs/PHASE6H_OFFICIAL_TRAJECTORY_SUBSTRATE.md` plus
`docs/PHASE6I_LOGGED_NEURAL_BOOTSTRAP.md` plus
`docs/PHASE6J_BRANCHABLE_NATIVE_SEARCH_GAME.md` plus
`docs/PHASE6K_NATIVE_REGRET_BLUEPRINT.md` plus
`docs/PHASE6L_REACH_CORRECTED_MCCFR.md` plus
`docs/PHASE6M_NATIVE_EXTERNAL_SAMPLING.md` plus
`docs/PHASE6N_OUTCOME_SAMPLING_MCCFR.md` plus
`docs/PHASE6O_NATIVE_TURN_PLAN_BLUEPRINT.md`.
The iterative variance result is in
`docs/PHASE6P_ITERATIVE_NATIVE_PLAN_MCCFR.md`.
The fail-closed admission boundary is in
`docs/PHASE6Q_NATIVE_SAFETY_ADMISSION.md`.
The bounded-horizon diagnostic is in
`docs/PHASE6R_DEPTH_LIMITED_NATIVE_PLAN.md`.
The held-out leaf/range diagnostic is in
`docs/PHASE6S_NATIVE_LEAF_CALIBRATION.md`.
The hidden-allocation/public-range diagnostic is in
`docs/PHASE6T_NATIVE_ALLOCATION_RANGE.md`.
The low-noise allocation-heldout leaf diagnostic is in
`docs/PHASE6U_LOW_NOISE_NATIVE_LEAF.md`.
The purpose-trained public native-leaf diagnostic is in
`docs/PHASE6V_PUBLIC_NATIVE_LEAF.md`.
The collection-heldout public native-leaf audit is in
`docs/PHASE6W_COLLECTION_HELDOUT_LEAF.md`.
The opponent-hypothesis plan-deviation substrate is in
`docs/PHASE6X_OPPONENT_DEVIATION_VALUES.md`.
The exact/native counterfactual-reach ledger is in
`docs/PHASE6Y_COUNTERFACTUAL_REACH_LEDGER.md`.
The public-root-aligned conditional-value experiment is in
`docs/PHASE6Z_PUBLIC_ROOT_ALIGNED_VALUES.md`.
The aligned public-archetype range diagnostic is in
`docs/PHASE6AA_ALIGNED_PUBLIC_RANGE.md`.
The redacted official-prefix reach ledger is in
`docs/PHASE6AB_OFFICIAL_PREFIX_REACH.md`.
The joint official truth/value substrate is in
`docs/PHASE6AC_JOINT_OFFICIAL_VALUE_SAMPLES.md`.
The observer-oriented official hidden-card range is in
`docs/PHASE6AD_OFFICIAL_HIDDEN_CARD_RANGE.md`.
The seat-decoupled official intent-proxy range is in
`docs/PHASE6AE_OFFICIAL_INTENT_PROXY.md`.
The policy-implementation-heldout transfer audit is in
`docs/PHASE6AF_POLICY_IMPLEMENTATION_GENERALIZATION.md`.
The implementation-heldout official point-value leaf and rejected interval
gate are in `docs/PHASE6AG_OFFICIAL_VALUE_LEAF.md`.
The repeated root/action expected-value experiment and rejected action/interval
gates are in `docs/PHASE6AH_EXPECTED_VALUE_SURFACES.md`.
The causal branch-conditioned experiment and rejected learned action model are
in `docs/PHASE6AI_CAUSAL_BRANCH_VALUES.md`.
The fresh semiparametric point-signal confirmation and rejected intervals are
in `docs/PHASE6AJ_SEMIPARAMETRIC_BRANCH_HYBRID.md`.
The expanded-root-mean collection-heldout confirmation and worst-slice interval
rejection are in `docs/PHASE6AK_EXPANDED_ROOT_MEAN.md`.
The preregistered high-precision confirmation and its held-out rejection are in
`docs/PHASE6AL_HIGH_PRECISION_CONFIRMATION.md`.
The public-range normalized counterfactual-advantage preregistration and
held-out rejection are in
`docs/PHASE6AM_PUBLIC_RANGE_COUNTERFACTUAL_ADVANTAGES.md`.

The external end state is explicitly blocked rather than claimed complete. The
exact NAIC deck uses two Prism Tower CRI 80, while the current TEF–POR Kaggle
engine has no Prism Tower. The only runnable local profile substitutes two Team
Rocket's Watchtower cards and is fingerprinted/labeled non-exact. No such deck
was submitted without owner approval. Human expert review is pending, and the
local panel does not establish superhuman strength. See
`docs/PHASE6_COMPLETION_AUDIT.md`.

## Intended sequencing

The phases describe architectural maturity. Within them, use this experiment
order so every new capability has a useful standalone role:

1. Multi-head strategic values plus persistent plan memory. *(Phase 1)*
2. High-level plan proposer paired with the tactical executor. *(early Phase 3,
   initially without opponent beliefs)*
3. Plan-level opponent belief model. *(Phase 2)*
4. Selective plan search under public-state uncertainty as a proposal/teacher
   and integration baseline. *(Phases 3–4)*
5. Action-conditioned, multi-hypothesis hidden-hand inference. *(Phase 2)*
6. Full belief-aware plan-search baseline. *(Phases 3–4)*
7. Search-guided self-play and audited counterfactual training. *(Phase 5)*
8. Preserve the official-engine legality, telemetry, replay, packaging, and
   benchmark boundary. *(Phase 6 integration baseline)*
9. Build leakage-proof public-belief replay datasets and a small exactly
   enumerable information-set validation game. *(Phase 6A–6B; validated)*
10. Train and evaluate a neural blueprint against exact exploitability/NashConv
    references before full-game scaling. *(Phase 6C exact-game gate validated;
    full-game scaling pending)*
11. Calibrate opponent counterfactual-value bounds and deploy safe endgame
    re-solving before continual resolving. *(Phase 6D exact-game calibration
    and Phase 6E all-or-blueprint exact resolver validated; official/full-game
    scaling pending)*
12. Add opponent-specific exploitation only as a separately certified overlay
    around the baseline-safe strategy. *(Phase 6F exact-game gate validated;
    official opponent model remains untrained)*
13. Validate the official information-set bridge and run matched-compute native
    engine ablations; require owner approval before submission. *(Phase 6G
    one-step shadow boundary validated; trained pipeline, matched ablations,
    expert review, and submission remain pending)*
14. Correct native replay action alignment and build causally isolated logged
    information-set trajectories for full-game representation/belief/value
    training. *(Phase 6H validated; offline and non-branchable)*
15. Validate split-safe representation and behavior pretraining on the logged
    corpus without promoting the heuristic teacher to the final policy.
    *(Phase 6I validated; offline bootstrap only)*
16. Implement a branchable full-game transition boundary with explicit hidden-
    completion and coin chance nodes. *(Phase 6J validated; native descendant
    transitions with a supplied, uncalibrated particle prior)*
17. Initialize and exercise a full-game neural candidate with terminal-valued
    counterfactual branches rather than heuristic action labels. *(Phase 6K
    sampled-regret wiring validated; no convergence or strength claim)*
18. Validate a reach-corrected external-sampling estimator exactly, then wire
    it to genuine native descendants without admitting partial traversals.
    *(Phase 6L exact reference validated; Phase 6M immediate-terminal native
    wiring validated only as a partial integration gate)*
19. Scale sampling to nontrivial native information sets and train separate
    neural advantage and average-strategy models on calibrated public-belief
    particles. *(Phase 6N validates outcome sampling and Phase 6O supplies a
    nontrivial turn-plan candidate across both players; calibrated beliefs,
    equilibrium convergence, and a defensible full-game regret proxy remain
    pending)*
20. Calibrate full-game opponent counterfactual-value bounds, implement safe
    re-solving, and only then run matched blueprint/safe/exploit ablations.
    *(Phase 6R controls the sampled horizon. Phase 6S shows the current leaf
    interval is vacuous. Phase 6T adds hidden-allocation particles and rejects
    direct Phase 6I neural-range transfer; the causal revealed-card updater is
    the stronger next baseline. Phase 6U fixes the rollout-precision threshold
    but rejects both existing learned value heads. Phase 6V's purpose-trained
    public scalar improves fresh point RMSE but still produces a nearly
    full-width interval. Phase 6W then rejects that improvement on a wholly
    held-out collection. A broader hypothesis-conditioned counterfactual model
    remains pending. Phase 6X supplies the first nontrivial per-hypothesis,
    per-action conditional targets, and Phase 6Y validates the reach formula
    while exposing its one-root-per-particle defect. Phase 6Z then constructs
    repeated hypotheses behind 18 disjoint later public roots and performs
    collection-heldout hypothesis-conditioned modeling. Its range remains
    incomplete because acting-player hands are fixed; the prior is supplied and
    upstream official reach is absent. Phase 6AA detects deck-archetype signal
    on the same split roots but has only six calibration/validation examples,
    high ECE, no hidden-hand/resource/intent support, and no production observer
    bridge. Phase 6AB captures the missing player-action prefixes through 512
    genuine selections, but the live adapter exposes zero chance likelihoods
    and the new roots are not jointly collected with Phase 6Z values. Full
    reach is unavailable. Phase 6AC then recollects the prefix and realized
    official truth jointly with all five plan values, validates exact root and
    next-draw replay fidelity, and records the common opaque chance-law
    cancellation. Each information set still has only one truth sample, the
    continuation profile remains fixed and abstract, and absolute chance reach
    is undisclosed. Counterfactual calibration and every safety/control gate
    therefore remain unavailable. Phase 6AD then collects 180 independent
    official truth roots, validates the player-0 observer boundary, and passes
    preregistered six-list candidate plus hidden hand/Prize/deck bucket
    calibration on 60 calibration and 60 validation roots. Its exact-list prior,
    public-consistent allocation sampler, and prototype provider remain narrow
    abstractions. Phase 6AE then validates a seat-decoupled six-policy intent
    proxy and 36-hypothesis candidate-intent product on 108 untouched roots.
    Phase 6AF adds implementation-heldout transfer across four authored policies
    and 144 untouched roots. Phase 6AG finishes 288 full official games and
    validates a neural point leaf on the untouched fourth implementation, but
    rejects its single-terminal-outcome intervals. Phase 6AH then completes
    repeated root/action continuations at 180 fresh implementation-heldout
    roots. Target precision passes, but transferable action contrast and all
    interval gates fail. Phase 6AI adds causal public branch deltas but rejects
    the learned advantage. Phase 6AJ then confirms the frozen semiparametric
    action ordering on 36 new roots while still rejecting interval width.
    Phase 6AK expands and regularizes the root mean over the complete prior
    development corpus, then uses sequential fresh calibration and validation
    panels. Point transfer and aggregate interval width improve, but the frozen
    worst-slice coverage gate still rejects conditional expected means.
    Controlled labels are not human latent truth, and a sufficiently sharp
    conditional interval model remains required, so full belief,
    opponent-reach-weighted values,
    counterfactual bounds, equilibrium convergence, safe resolving,
    exploitation, and official control remain pending.
    Phase 6AL then freezes that mean and doubles the independent replay budget.
    Target precision passes, but the bootstrap and interval-coverage failures
    persist on a new sequential holdout, so the model is not promoted. The next
    admissible experiment must use the public-range and opponent-reach boundary
    directly and preregister per-hypothesis counterfactual calibration without
    treating Phase 6AL validation as fresh evidence. Phase 6AM performs that
    pivot with baseline-relative normalized advantages and a fresh 36-root
    holdout. Its coverage, range, and uncertainty-width evidence is encouraging,
    but maximum target SE misses the frozen ceiling by `0.001688`; the bound
    provider is therefore rejected. A future retry must be independently
    preregistered, use genuinely new roots, and improve sampling precision
    without tuning the observed Phase 6AM panel.)*

This order retains a functional, legal baseline while making a trained
information-set blueprint and safe re-solving—not sampled rollout—the strategic
destination. It also leaves room to test a transformer or other relational
encoder behind a stable representation contract instead of rewriting downstream
systems.

## Research tracks and decision rules

| Track | Central question | Primary evidence |
|---|---|---|
| Strategic representation | Can the agent represent what it is trying to accomplish across turns? | Head calibration, plan stability, action improvement |
| Belief modeling | Do observed actions reveal plans and hidden resources? | Log loss, calibration, downstream decision gain |
| Plan search | Does plan-level belief-aware search beat action-level or policy-only play efficiently? | Matched-root decisions, win rate, compute |
| Learning | Can search discoveries be distilled without losing generality? | Held-out replay and opponent-panel gain |
| Game understanding | Which dimensions predict winning play in each matchup? | Ablations and expert replay review |
| Neural game theory | Does the blueprint reduce regret/exploitability, and does re-solving preserve opponent counterfactual-value bounds? | Exact validation game, NashConv/exploitability, bound coverage, safety certificates |
| Safe exploitation | Can calibrated opponent adaptation gain value without exceeding the declared worst-case tolerance? | Adversarial opponent suite, final-strategy certificate, held-out gain |
| Official bridge | Does every abstract information set/action faithfully map to agent-visible native state and legal selections? | Replay equivalence, action coverage, official-engine legality |

Prior workstation or teammate results are context about particular simulators,
budgets, seeds, and opponents—not declarations that an idea is a dead end. Bold
ideas remain in scope, but they earn mainline complexity through reproducible
downstream decision quality. Belief calibration, internal intent accuracy, or
search depth alone are not sufficient if match decisions do not improve.

## Reproducibility requirements for future phases

- Version state, action, deck, feature, plan, and training-example schemas.
- Save deck and environment fingerprints with every trajectory/checkpoint.
- Separate simulator truth from agent-visible observations and beliefs.
- Record random seeds, opponent panels, compute budgets, and search gates.
- Keep policy-only, search-only, and belief/search ablations reproducible.
- Never treat inferred hidden cards as facts; retain posterior uncertainty.
- Store strategic-head values and meta-weights for replay diagnosis.
- Validate any search-generated or counterfactual label before distillation.
- Label every result as official-engine execution or abstract-game execution;
  never pool them into one performance claim.
- Bind blueprint and value checkpoints to the exact abstract-game, deck, and
  public-belief updater fingerprints.
- Require counterfactual-value coverage and an explicit safety certificate
  before reporting a strategy as safely re-solved.
