# Phase 6J branchable native-search information-set game

Phase 6J closes the earlier “no branchable full-game boundary” gap without
pretending that the project has an official belief distribution or an
equilibrium strategy. The licensed competition search API can reconstruct a
native state from an exact agent observation plus caller-supplied hidden-zone
contents. Each retained native search ID is an immutable branch root: applying
different legal selections creates independent child states, and applying the
same selection again reproduces the same observable child.

`OfficialNativeSearchGame` wraps that capability behind the existing
`AbstractInformationSetGame` protocol.

## Exact provenance distinction

The boundary has two layers:

- **Genuine native execution:** every descendant transition calls the official
  `search_step` implementation. Complete games use the engine's card rules,
  legality, state changes, terminal results, and search-state cloning.
- **Research abstraction:** the root chance distribution over hidden decks,
  hands, prizes, deck order, and face-down Active cards is supplied by this
  project. It is not sampled or calibrated by the official engine.

For that reason the game's provenance remains
`abstract_information_set_game`, its own `official_engine_executed` flag is
false, and a separate `source_official_engine_executed` flag records that its
descendant transitions are native. This prevents native rule execution from
being misreported as an official belief model or strategy.

## Information-set safety

The opaque solver state retains the native search ID and selected hidden
particle. Policy-facing objects receive only:

- an allow-listed common public state and chained public-history fingerprint;
- a range over opponent candidate identities and weights, never the selected
  truth particle;
- a fingerprint of the acting player's own private hand/search observation;
  and
- every contract-valid ordered native selection.

At the common root, all six truth particles produce the same
`PublicBeliefState` fingerprint and the same `InformationSet` ID. Native search
IDs and particle identities do not appear in those payloads. As play reveals
cards or produces different private draws, information sets diverge through
legitimate public or acting-player-private observations.

The persistent hidden-completion manifests contain candidate/deck
fingerprints, probabilities, zone counts, adjustment counts, and source labels;
they contain zero raw private card IDs.

## Chance and legal actions

The game begins with an explicit chance node over six equal-weight local deck
particles. Within a particle, supplied deck order makes draws deterministic.
The runtime's `manual_coin=True` mode turns each coin flip into a two-outcome
chance node with probability 0.5 for heads and tails. Player nodes use complete
ordered selection enumeration, including multi-card selections, capped at
4,096 actions with fail-closed behavior.

The selected hidden prior is intentionally simple. It uses the same six local
candidate decks as Phase 6H and is not calibrated. If a candidate conflicts
with already-public cards, the completion builder records the minimum number of
composition adjustments as an abstraction/OOD signal.

## Live evidence

The checked-in run starts from a genuine native agent observation using the
explicit Watchtower compatibility deck. It creates six simultaneous native
search roots, verifies alternative and repeated branches at the root, then
drives one complete priority-policy path per hidden candidate. Midgame
counterfactual branching is independently checked in every path.

| Candidate | Terminal depth | Player 0 / 1 decisions | Coin chance nodes | Maximum complete actions | Distinct information sets |
|---|---:|---:|---:|---:|---:|
| Abomasnow | 165 | 115 / 49 | 0 | 41 | 164 |
| Alakazam | 186 | 104 / 81 | 0 | 90 | 185 |
| Starmie | 149 | 96 / 52 | 0 | 30 | 148 |
| Dragapult | 81 | 55 / 25 | 0 | 30 | 80 |
| Crustle | 192 | 113 / 70 | 8 | 50 | 183 |
| Rocket Mewtwo mirror | 74 | 30 / 43 | 0 | 23 | 73 |

Across the run:

- all six paths reach native terminal states;
- both players act in every path;
- all terminal utilities are exactly zero-sum;
- root and midgame branching pass;
- repeating the same root transition is exactly deterministic;
- altered action metadata and altered chance labels are rejected;
- 862 native transitions create 868 native states; and
- no private-zone card IDs are persisted.

Priority policies exist here solely to exercise the boundary end to end. Their
terminal results are not strength evidence, an equilibrium evaluation, or a
candidate submission.

The native battle API exposes no match-seed control. The live observation token,
paths, evidence/game fingerprints, and file hashes can change on regeneration;
the artifact is intentionally not claimed byte-deterministic.

Checked-in identity for this live run:

- runtime fingerprint:
  `9a62073d859273c86926a63c30a9a28dc16c698ad759ade3d1a7563b05af1634`;
- game fingerprint:
  `5bf330172540c1b80def82ba90ccf559803a76bc10d7711868b49c0ada288a13`;
- evidence fingerprint:
  `f02f0ad7ca088acf42130cfd4631beea3c4db48e0b65079b1475ece54a1d56b4`;
- evidence JSON SHA-256:
  `1a26d691b9595a3b7a748aba161571a620d5edf752c84357a20fa76d4f44fcbc`;
- evidence Markdown SHA-256:
  `3fad2f0920524d792e39348fbbb987231a4a9159c57d2bd7d1b30acc341dfeeb`;
- hidden-completion manifest SHA-256:
  `a1aefbfda9fe7db9451b073b297f3e2a22e94cae6949df1534c6c9794f355b17`.

## Interfaces

- `HiddenZoneCompletion` — opaque joint hidden particle with a redacted
  persistence manifest.
- `build_hidden_zone_completion` — deterministic completion consistent with
  visible zones and explicit adjustment counts.
- `OfficialSearchGameState` — opaque state binding native search ID, truth
  particle, public-history chain, and agent-visible observation.
- `OfficialNativeSearchGame` — chance, information-set, legal-action,
  transition, and terminal-utility implementation.
- `complete_official_actions` — shared complete ordered action enumeration.
- `run_phase6_native_search_game` — live Phase 6J evidence harness.
- `scripts/phase6_native_game.py` — durable artifact generator.

## Remaining game-theoretic gate

Phase 6J supplies the missing branchable rule transition boundary. It does not
make exhaustive traversal feasible: Pokémon's hidden-state and action spaces
remain enormous, and the six-particle prior is only an engineering seed. The
next gate is sampled information-set training—such as external-sampling neural
CFR or another explicitly measured equilibrium-oriented method—using this
native game interface and Phase 6I initialization. Phase 6K now provides a
restorable two-round sampled-regret candidate and mixed held-out diagnostics;
it deliberately does not claim reach-corrected CFR or convergence. The scaled
trainer must still establish defensible regret/exploitability proxies under
held-out particles, calibrate full-game opponent counterfactual-value bounds,
and retain all-or-blueprint fallback before any official strategy control or
opponent exploitation is enabled. See
`docs/PHASE6K_NATIVE_REGRET_BLUEPRINT.md`.
