# Causal Q recovery protocol

Status: frozen before games 71-100 of any collection shard were generated or
scored. This protocol is a recovery experiment, not a relaxation of the neural
game-theoretic target and not an arena performance claim.

## Failure being addressed

The strongest live Team Rocket bot is materially weaker going second into the
dominant Marnie's Grimmsnarl field. Outcome-weighted imitation, terminal PPO,
small sampled hidden-hand searches, top-policy transfer, and low-dimensional
deck edits did not reproduce a held-out improvement. Those methods confound a
decision with the favorable state in which it happened or use too few hidden
particles to give stable action ordering.

This experiment instead branches legal actions at a live official-engine root.
The engine visualizer's transient full state is used only to reconstruct that
one realized hidden allocation through the licensed search API. Policies see
only normal redacted observations. Raw hidden cards are not serialized. A
label is conditional on one independently reached hidden state; it is not an
information-set expectation, a counterfactual-reach value, or an equilibrium
certificate.

## Frozen collection

Three independent 100-game, second-seat collections are used:

1. `live_truth_causal_rank1a_v1.jsonl` against `AshMarnieGeneralV110`;
2. `live_truth_causal_rank1b_v1.jsonl` against the same policy in a separate
   engine process; and
3. `live_truth_causal_rank2_v1.jsonl` against
   `HelbrosMarnieGeneralRecoveryV4`.

Each game retains the first eligible main-phase root on each of two distinct
turns. The frozen blueprint and up to three semantically distinct alternatives
are rolled to a terminal official result. Candidate order is randomized. Any
incomplete root, illegal selection, visualizer/public parity failure, or raw
truth serialization is rejected.

Complete source games are split by their immutable per-file game ID:

- games 1-70: training;
- games 71-85: threshold selection; and
- games 86-100: untouched admission.

No model, feature, checkpoint, or threshold is selected using games 86-100.

## Frozen model and optimization

The deployed `scale_value_calibrated_v1` network remains frozen as the safe
blueprint. A separate Q head receives its 384-wide contextual global token,
the contextual token for a shortlisted legal option, and the blueprint logit.
The Q head is a layer-normalized 769-to-192-to-96-to-1 MLP. It is trained with
mean squared error on each action's terminal utility minus the frozen
blueprint action's terminal utility at that same root, plus a 0.75-weight
pairwise ordering loss. The final layer is initialized to zero so the initial
overlay predicts no action advantage,
AdamW at `3e-4`, weight decay `1e-4`, root batches of 24, seed `615401`, at most
80 epochs, and validation-MSE early stopping with patience 12.

The selected checkpoint minimizes only selection-split MSE. The margin gate is
then selected only on that split. An eligible threshold must intervene on at
least five and at most 40% of roots, have positive paired mean improvement, and
have at least 60% intervention precision. If none qualifies, the gate selects
no actions and the candidate is rejected.

## Untouched offline admission

On games 86-100, all of these must hold:

- at least five interventions;
- intervention rate no greater than 40%;
- intervention precision at least 60%;
- positive mean utility change versus the frozen blueprint; and
- the game-clustered 95% bootstrap lower bound for paired utility change is
  strictly above zero.

Passing admits only an experimental second-seat Marnie reranker. It does not
establish game-theoretic safety or arena strength.

## Runtime constraint and official strength gates

If the offline gate passes, the Q overlay may inspect only actions already in
the frozen blueprint's four-action semantic shortlist. It may act only at a
single-choice main-phase decision, while Team Rocket is going second against a
publicly identified Marnie deck, through turn 8, and only above the frozen
margin. Every other decision is returned by the unchanged blueprint. Legality
is enforced after routing, and every intervention is logged.

Before arena submission, the extracted archive must pass deterministic import,
60-card deck, no-illegal-action, serialization, and runtime-parity checks. It
must then pass fresh official-engine match gates against both top Marnie bots
from the second seat and a balanced non-regression gate against the current
live Rocket archive. A local pass is not called an arena gain. Arena submission
occurs only for a candidate that beats the live archive in those frozen gates;
the live rating after at least 300 games remains the final strength authority.

The heuristic blueprint remains a legal fail-soft fallback. This causal Q
overlay is an opponent-model exploitation experiment constrained by that
baseline; it is not the project's final strategic architecture. The intended
final approach remains a public-belief neural blueprint with calibrated
information-set values and safe resolving.
