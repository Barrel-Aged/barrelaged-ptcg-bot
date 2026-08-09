# Phase 6N: outcome-sampling MCCFR reference

Phase 6N adds a one-terminal-history-per-traversal MCCFR estimator so native
research is no longer forced to expand every traverser action at every future
information set. It is a scalable estimator contract, not evidence that the
full Pokémon game has converged.

## Method and exact audit

At a traverser node, the sampling policy mixes the current strategy with a
uniform exploration policy. At opponent nodes it samples from the current
strategy, and at chance nodes it samples the declared chance distribution. The
sample records the player reach, opponent-and-chance reach, full sampling
reach, current and sampling strategies, selected action probability, estimated
action values, policy value, and the resulting importance-corrected regrets.
Average-strategy samples are stored separately from regret samples.

The recursion supports a pre-action baseline. For sampled action `a` with
sampling probability `q(a)`, baseline `b(a)`, and sampled child estimate `v`,
the action estimate is

```text
b(a) + (v - b(a)) / q(a)
```

while unselected actions retain their baseline values. An exhaustive
enumeration over all 60 sampled terminal histories of `RocketPrizeDuel` uses a
deliberately nonzero action-dependent baseline. Its 24 audited expectation
rows match the exact immediate counterfactual regrets with maximum absolute
error `4.163336342344337e-17`.

Hard transition and depth limits fail closed: an over-budget trajectory raises
before any partial samples are returned. Deterministic hash-priority reservoirs
provide bounded, serializable regret and average-strategy memories.

## Reference training result

The deterministic 20,000-iteration tabular reference executes 40,000 traverser
trajectories and 133,624 transitions. Exact enumeration reports:

| Metric | Result |
|---|---:|
| Uniform NashConv | 0.916666667 |
| Trained NashConv | 0.031116118 |
| Trained exploitability | 0.015558059 |
| Trained player-0 value | -0.063604116 |
| Regret samples | 45,794 |
| Average-strategy samples | 45,794 |

Outcome sampling is substantially noisier here than the external-sampling
reference. The result validates the estimator and serialization semantics only
on the exact non-official microgame.

## Primary methodology sources

- [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://proceedings.neurips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf)
- [Corrected MCCFR appendix](https://webdocs.cs.ualberta.ca/~bowling/papers/09nips-mccfr-appendix.pdf)
- [OpenSpiel outcome-sampling MCCFR implementation](https://raw.githubusercontent.com/google-deepmind/open_spiel/master/open_spiel/algorithms/outcome_sampling_mccfr.cc)

## Durable artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| `expectation_audit.json` | 8,607 | `25cbba386fdfcc1b1928ed7dc25a4f410da67874e9198a3db38de3f5a2dedd0d` |
| `training_result.json` | 2,679 | `e178c117e5f95ede6d2634af9dd694270249511beaf66b274d07e119ec0fbe86` |
| `phase6n_evidence.json` | 2,048 | `7c7b45c38a4aef9c12c0039d26b3c034da8130289c94ded2d33d6b4422dad118` |
| `phase6n_evidence.md` | 2,127 | `016993dcfb0c6349f394227bc9be381b0da0b5f5072395a1358c34f95b75ed04` |

The evidence fingerprint is
`c7f86852150488f15d7b8895ccfc7f87f859e6c1e719516e1eafdac9d0da9c0f`.

## Claim boundary

This reference does not execute the licensed native engine. It does not claim
Deep CFR, native equilibrium convergence, calibrated full-game
counterfactual values, safe re-solving, official strategy control, Pokémon
strength, or a Kaggle result.

