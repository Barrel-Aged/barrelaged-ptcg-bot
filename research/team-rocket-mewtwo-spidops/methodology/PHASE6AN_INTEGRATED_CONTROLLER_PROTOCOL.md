# Phase 6AN: Integrated Controller Pre-Registered Protocol

Status: locked before the Phase 6AN integrated model is fit, calibrated, or
evaluated. This protocol does not revise the earlier Phase 6AM thresholds
after seeing its rejected result.

## Purpose

Phase 6AN turns the completed research components into one official-engine
controller. The deployed action path must be a frozen neural public-belief
blueprint with a persistent plan state, six strategic values, a
state-conditioned meta-value, a calibrated selective-search gate, a
fail-closed safe resolver, and an autoregressive legal action policy. The
existing heuristic is allowed only as a tactical prior feature and emergency
exception fallback. Sampled rollout search is not the final controller.

The exact Juho Kallama list is the research target. Because Prism Tower is not
present in the current official TEF-POR engine card pool, official execution
uses the already documented two-card Prism Tower to Team Rocket's Watchtower
compatibility substitution. No result from that compatibility list will be
described as an exact-deck result.

## Frozen data boundaries

The macro-plan model uses
`artifacts/phase6_expected_value_surfaces/expected_value_surface_dataset.json`.
Its existing partitions remain fixed:

- `train`: fit model parameters.
- `selection`: select only optimization checkpoint and temperatures.
- `bounds`: construct the conformal counterfactual-advantage radius.
- `untouched`: one final admission test; never used for fitting or selection.

Each plan target is the mean of the 16 licensed native terminal continuations
already collected for that public root and action (two public-consistent hidden
allocations times eight returns). The collection is genuine official-engine
execution. The five plans, sampled hidden allocations, continuation policy,
belief labels, and neural estimator are research abstractions.

The tactical action model uses two source groups:

1. Archived decisions made by `HelbrosRocketSpidopsProV3`, split by complete
   replay file before examples are expanded.
2. Fresh official-engine games in which that frozen policy controls the
   Watchtower-compatibility target list against the five existing official
   panel decks and three deterministic opponent styles. Complete game files
   are split by match before fitting.

No action from a validation or test match may be used for fitting or
normalization. The archived policy is a behavioral teacher, not the deployed
strategy provider and not evidence that the new deck has inherited its arena
rating.

## Model and targets

The macro network consumes the 111 public, 75 acting-private, and 16 turn
history features plus the previous five-way plan latent. It jointly predicts:

- six opponent archetype probabilities;
- six opponent intent probabilities;
- ten hidden-resource-category marginals;
- a five-plan average-strategy blueprint;
- five official-continuation plan values;
- setup, tempo, resource efficiency, prize control, board control, and
  optionality values;
- six state-conditioned meta weights;
- value uncertainty; and
- probability that search-guided improvement is material.

The plan-policy target is a full-support mirror-descent update from the frozen
source blueprint using the repeated official continuation values. The tactical
network uses relational Pokemon tokens, the macro outputs, the current legal
options, the selected prefix, and a frozen heuristic tactical-prior score. It
predicts the next option or stop token autoregressively. Native legality is
always enforced after decoding.

## Counterfactual bounds and resolver admission

For every root/action, define the counterfactual advantage as the action value
minus the source-blueprint-weighted value at that root. The calibration score
is the absolute error of the neural predicted advantage. The radius is the
finite-sample 95% split-conformal quantile on the frozen `bounds` partition.

The certificate is admitted only if all of the following hold on `untouched`:

- marginal action-level coverage is at least 0.90;
- each opponent-archetype slice has coverage at least 0.80;
- every reported count is nonzero and all fingerprints reproduce; and
- the archive's independent runtime-parity test matches the training forward
  pass within `1e-6` for logits, probabilities, values, and action scores.

The resolver is fail closed. If the certificate is absent, malformed, or not
admitted, it returns the blueprint exactly. If admitted, it may add at most
0.50 probability mass to a plan only when that plan's predicted
counterfactual-advantage lower bound is no worse than -0.01 and its point
advantage is at least 0.05. Failure to find such a plan leaves the blueprint
unchanged. This is a conditional empirical non-regression guard, not a proof
of equilibrium safety.

## Opponent exploitation overlay

The separate exploitation query is trained only as a residual action scorer.
It is admitted only if its 95% lower confidence bound for paired held-out game
utility versus the same controller with exploitation disabled is at least
-0.02, and its configured weight is at most 0.15. Otherwise its deployed
weight is exactly zero. This preserves the baseline public-belief strategy as
the safe strategy.

## Tactical and strength acceptance gates

Before an internal-arena submission, all of these must pass:

- deck resolution is exactly 60 cards and the two compatibility substitutions
  are disclosed;
- the full unit suite and archive extraction smoke test pass;
- no illegal native selection and no uncaught controller exception in the
  evaluation suite;
- autoregressive held-out top-1 accuracy exceeds a uniform legal-action
  baseline and held-out negative log likelihood is finite;
- the learned plan distribution differs from the source blueprint on at least
  one untouched root, proving that search-guided learning reached the frozen
  policy;
- paired official games cover both seats, all five panel archetypes, and all
  three deterministic styles;
- the integrated controller's paired win rate is reported against both the
  legacy Phase 6 heuristic and the tactical teacher; and
- belief calibration, strategic telemetry, search-gate concentration, replay
  review, latency, and robustness slices are recorded whether favorable or
  unfavorable.

Internal-arena submission is an external strength estimate, not an acceptance
gate that may be rewritten after seeing the rating. The submitted archive and
the locally audited archive must have the same SHA-256 digest. There will be no
Kaggle submission and no unverified performance claim.

## Completion review

After submission, the original six-phase plan is re-read line by line against
durable evidence. A separate audit then extracts the submitted archive and
traces `main.agent` through the frozen model, beliefs, plan blueprint, resolver,
and autoregressive policy. If either review finds that the live submission is
not faithful to the plan, Phase 6AN is rejected and the implementation cycle
restarts rather than relabeling the heuristic baseline.
