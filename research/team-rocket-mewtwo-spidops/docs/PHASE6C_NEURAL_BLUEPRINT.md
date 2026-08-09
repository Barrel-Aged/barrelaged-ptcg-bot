# Phase 6C neural blueprint validation

Date: 2026-07-26  
Status: validated on the exact abstract microgame only  
Evidence fingerprint: `ba58c83e4b83f0f072dd53cb8fdd3f08aa714a80fbc7058f4040950bf39a9be8`

## Outcome

Phase 6C now has a deterministic neural-blueprint checkpoint path that consumes
an information set and public opponent range, emits a legal mixed strategy, and
estimates per-opponent-private-state counterfactual values. Two candidates were
trained and evaluated against exhaustive pure best responses in
`RocketPrizeDuel`, the 55-state, three-card, explicitly non-official validation
game introduced in Phase 6B.

The policy candidate is selected because its exact NashConv is lower. This is a
microgame correctness gate, not an official-engine strength result. No full-game
neural blueprint or official safe resolver has been trained or implemented.
Phase 6G subsequently validates only a one-step official shadow bridge; it does
not change this model's abstract scope.

## Representation and model

`DuelInformationEncoder` produces 14 deterministic features:

- acting-player identity;
- the acting player's own private card;
- public betting history;
- the public-belief range over the opponent's three possible private cards; and
- public pot/push indicators.

`DuelBlueprintNetwork` uses a shared multilayer perceptron with two outputs:

- a masked policy head over the four global abstract actions; and
- a conditional-value head with one estimate for each opponent private card.

Illegal actions are masked at inference. The serialized artifact records the
feature schema, model dimensions, deck and game fingerprints, training target,
seed, metrics, and every tensor. `NeuralDuelBlueprintProvider` restores that
artifact behind the generic `BlueprintStrategyProvider` contract.

## Candidates and exact evaluation

Both candidates use deterministic CPU training and the same information-set
features. Exact exploitability is computed by enumerating every deterministic
best response over the complete game.

| Candidate | Supervision target | Epochs | Policy L1 | Conditional-value RMSE | NashConv | Exploitability |
|---|---|---:|---:|---:|---:|---:|
| Policy | Analytic equilibrium strategy | 2,000 | 0.00029493 | 0.00006604 | 0.00004637 | 0.00002319 |
| Regret matching | Positive regret-matching encoding of the 20,000-iteration tabular CFR average | 4,000 | 0.00000000 | 0.00000000 | 0.00321571 | 0.00160785 |

The regret candidate is supervised from the tabular CFR result. It validates a
distinct regret-matching inference parameterization, but it is not Deep CFR and
must not be described as such. Its residual exploitability comes from the CFR
teacher rather than neural reconstruction error.

The selected policy artifact has model fingerprint
`2dab9a2db7675610839feeb76bac4fd7202927a582638db352998a2405e4e291`.

## Restoration and reproducibility

Both JSON checkpoints were deserialized into fresh networks and reevaluated.
Maximum strategy-probability delta, counterfactual-value delta, and NashConv
delta were all exactly zero for both candidates.

Two independent end-to-end generations produced byte-identical artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `phase6c_evidence.json` | 3,851 | `de94f616ee70f11b2c19bb4209036f4c1744c0412c85f7ad12dc2a6b8b81b97d` |
| `phase6c_evidence.md` | 1,883 | `b1c9247c565a9139ab55559730e435d4eeaeb770a7e55a7f073f00978d4d274b` |
| `policy_candidate.json` | 53,784 | `ad4bb27c6faf573345c0fd0d424b71197a1f61b8f36640c9f17612297922f801` |
| `regret_candidate.json` | 52,941 | `a5f7ba180d95bd362da2e3ec7a4a662aa8120b249cb87f8de59e0040113beb41` |
| `selected_neural_blueprint.json` | 53,784 | `ad4bb27c6faf573345c0fd0d424b71197a1f61b8f36640c9f17612297922f801` |

The selected artifact intentionally duplicates the selected policy candidate so
consumers have a stable filename without losing the two-candidate comparison.

## Verification

At the Phase 6C checkpoint, the standard-library suite passed 69 tests. Phase
6C tests cover:

- feature width and legal-action masks;
- exact NashConv and conditional-value tolerances;
- provider checkpoint status and fingerprints;
- JSON round-trip restoration with exact output equality;
- tamper rejection; and
- honest Phase 6 claim/status labels.

Reproduce the milestone from the project root with:

```powershell
python scripts/phase6_neural.py
```

Artifacts are written to `artifacts/phase6_neural_blueprint/`.

## Claim boundary and next gate

This milestone establishes that the neural provider and serialization design can
reproduce game-theoretic targets on a completely enumerable testbed. It does not
show that the representation is faithful to Pokémon TCG, that the model is
calibrated on official states, that safe re-solving remains safe under function
approximation, or that playing strength improved.

Phase 6D must next construct and calibrate conservative opponent
counterfactual-value intervals, including explicit widening for approximation,
out-of-distribution ranges, and interrupted solving. A resolver may receive a
safe certificate only after exact-game false-safety tests pass. The official
full-game transition bridge and full-game training remain later gates. Phase 6H
supplies a corrected logged trajectory corpus, and Phase 6J subsequently adds a
production-disabled branchable native-search boundary under a supplied hidden-
state prior. Phase 6K later adds a small full-game sampled-regret checkpoint,
but none of these milestones provides a validated full-game equilibrium
strategy, reach-corrected CFR result, or safety certificate.
