# Phase 6A-6B: public replay and exact-game foundation

Last updated: 2026-07-26  
Status: locally validated foundation; Phase 6 remains incomplete

## Purpose

The game-theoretic pivot needs two trustworthy foundations before neural
training:

1. agent-perspective training data that cannot read official simulator truth;
2. a small imperfect-information game whose value, best responses,
   exploitability, counterfactual values, and safety bounds can be checked
   exhaustively.

Phase 6A-6B now provides both. This work does not train the full-game neural
blueprint and does not connect the abstract solver to official observations.

## Phase 6A: official replay conversion

Native visualizer replays contain both players' complete hands, decks, prizes,
card serials, and hidden draw identities. `public_replay.py` treats those files
as truth-level evidence sources, never as observations. A whitelist converter
creates a separate episode for each player perspective.

Each public frame retains:

- public board, Stadium, discard, damage, attachments, conditions, turn flags,
  and zone counts;
- the observer's own hand and temporarily visible deck-search cards;
- only the acting observer's legal options and chosen indices;
- public action events; and
- chained history and observation fingerprints.

It removes:

- opponent hand, deck, prize, and search identities;
- both players' unknown deck/prize identities;
- opponent hidden draw and hidden-zone move identities;
- every physical card serial; and
- opponent-only legal options or private chosen indices.

Cards that become public later produce `RevealTarget` labels in a separate
`supervision_only_reveal_targets` envelope. Each target records the reveal
frame and may supervise only prefixes ending before that frame. Targets never
enter `PublicReplayFrame`, its observation fingerprint, or its history chain.
Player perspectives share no training frame, and the dataset explicitly sets
`cross_perspective_join_allowed: false`.

The official visualizer records a prompt's `selected` indices on the following
frame. Schema 2 shifts that label back to its source prompt, validates it against
the source count/distinctness/range contract, removes terminal phantom prompts,
and excludes the behavior target from observation/history fingerprints.

### Leakage audit

The audit checks every serialized frame for raw hidden-zone keys, serial fields,
opponent hand identities, non-actor selections, and redacted events that retain
card IDs. Synthetic canary tests place unique names and serials in every hidden
zone and verify none enters frame serialization. Unknown native event types
fail the durable Phase 6A evidence build until their visibility is reviewed.

The current six replay-review matches yield:

| Measure | Value |
|---|---:|
| Isolated perspective episodes | 12 |
| Public frames | 1,232 |
| Acting-observer decision frames | 610 |
| Redacted hidden event identities | 923 |
| Supervision-only later reveals | 320 |
| Unknown event types | 0 |
| Structural leakage violations | 0 |

The source frames are genuine official-engine replay output. The converted
frames are reconstructed training records, not direct proof that the official
bridge is faithful at every possible live decision. Phase 6G separately audits
566 observed decisions, still without a full-game transition abstraction.

## Corrected extensive-form boundary

The initial `AbstractInformationSetGame` protocol incorrectly allowed a solver
to transition directly from one information set to another. Information sets
are generally not Markov states in imperfect-information games. The corrected
boundary separates:

- opaque solver states, which may contain enumerated or sampled private truth;
- public-belief states, which contain no true hidden state;
- player information sets, which include a fingerprint of the player's own
  private observation; and
- transitions over opaque states with chance outcomes and legal actions.

Policies consume only the latter two public/player-visible representations.
Solver-only state IDs are explicitly prohibited as policy features.

## Phase 6B: RocketPrizeDuel

`exact_game.py` implements a three-card Kuhn-style zero-sum endgame. The private
strength labels are Mimikyu, Spidops, and Mewtwo. Players choose whether to
hold, push, concede, or contest. These labels are thematic only; the game has
no official Pokemon card-rule fidelity.

The complete tree contains:

- 6 ordered private deals;
- 55 states;
- 30 terminal states;
- 12 information sets; and
- 24 information-set actions.

An analytic equilibrium is included from the standard parameterized Kuhn
equilibrium family. Exact enumeration checks each profile against every pure
information-set-consistent best response.

| Profile | Player-0 value | NashConv | Exploitability |
|---|---:|---:|---:|
| Uniform baseline | +0.12500000 | 0.91666667 | 0.45833333 |
| Analytic equilibrium | -0.05555556 | 1.32e-16 | 6.59e-17 |
| Full-tree CFR, 20,000 iterations | -0.05555511 | 0.00321571 | 0.00160785 |

The analytic game value is `-1/18`. CFR's absolute value error is
`4.48e-7`, and its NashConv is below the declared `0.005` gate. The solver is a
deterministic full-tree tabular reference, not the planned neural blueprint.

## Exact bounded local re-solving

`ExactGridSafeResolver` enumerates 101 local mixed strategies at a two-action
information set. For each candidate it evaluates every opponent private-card
hypothesis and rejects any strategy that gives the opponent more conditional
value than the supplied exact upper bound.

The checked evidence begins from a deliberately weak uniform strategy at
`p0:spidops:root`. It changes `hold/push` from `0.5/0.5` to `0.0/1.0`, improving
the conditional actor value from `+0.125` to `+0.500`, while the maximum
measured opponent-bound violation remains `0.0` across both private-card
hypotheses.

This is a genuine exact certificate for that microgame and those bounds. It is
not a certificate for official Pokemon, a learned value approximation, an
abstracted full-game subgame, or the exploitation overlay.

## Durable evidence and reproduction

Run:

```powershell
python scripts/run_tests.py
python scripts/phase6_foundation.py
```

Artifacts:

- `artifacts/phase6_game_theory_foundation/public_replay_dataset.json`
- `artifacts/phase6_game_theory_foundation/phase6ab_evidence.json`
- `artifacts/phase6_game_theory_foundation/phase6ab_evidence.md`

The corrected schema-2 generation is deterministic:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Public replay dataset | 7,235,664 | `3f989efc3c214445a6b831919a7eb6257dca8ef955555ab4d10fe5e91c71093a` |
| JSON evidence | 4,029 | `2c523bf1ad7aaf0b984111597812a90e167af02bd870e73b9aa2e0f7bb1f52aa` |
| Markdown evidence | 2,459 | `597e8b2ffb17c9678db6e7c53b8c1a3ae67ce031ed625aeb1be67bc739275873` |

The corrected dataset fingerprint is
`3a9e8d1b740c0b79f986e10369823a668de57e9b1453c79384dbd1baacce4df7`;
the regenerated foundation evidence fingerprint is
`60e3c49b3d6881dc2f868b8ac41e0b2e82db587b308a72741ab90a7bc521bcd8`.

## Remaining Phase 6 gates

- Use the Phase 6H logged information-set trajectories to train full-game
  representation, belief, policy/value, and transition models without behavior-
  target or later-reveal leakage.
- Extend the Phase 6G one-step official observation/action-lifting bridge into
  a verified full-game transition abstraction.
- Train/evaluate on full-game data, obtain human replay review, resolve the
  exact Prism Tower card-pool blocker, and seek separate submission approval.

Accordingly, the durable evidence reports `phase6_complete: false`.
