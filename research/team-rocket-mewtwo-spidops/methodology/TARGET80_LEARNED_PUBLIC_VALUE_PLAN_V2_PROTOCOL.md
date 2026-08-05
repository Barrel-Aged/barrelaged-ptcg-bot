# Target-80 learned public-value plan V2 protocol

Frozen: 2026-08-05, before training metrics, candidate construction, or match outcomes

Status: **prospective research hypothesis; no package or submission authorized**

## Strategic hypothesis

SafePlan V1's official-engine public-belief resolver was mechanically healthy
and gained two aggregate development wins, but its hand-weighted leaf failed to
generalize across the two Starmie policies. V2 replaces that manual leaf with
a matchup-conditioned public-state outcome model learned from the latest exact
1,025-game V5 arena archive. Search remains constrained by immutable V5 as a
baseline-safe blueprint and uses only public/own-visible information plus a
publicly inferred opponent-deck range.

The deck substrate is the zero-Wobbuffet, three-Murkrow list from
`work/target80_candidates/murkrow_three_v1/`. Its deck-only screen did not pass
the strength gate, but it was the first tested Wobbuffet-free list that did not
collapse and is reused here only under a materially new planning hypothesis.
No forced `Torment`, seed-derived tactic, or hand-authored matchup action is
allowed.

## Frozen source data and split

Training source:
`work/arena_v5_pull_20260805_0144/`.

- exact files: 1,025 arena game JSONL files;
- immutable record: 563 wins, 462 losses, zero draws;
- tree digest: SHA-256 of sorted UTF-8
  `<filename>\t<file_sha256>\n` records:
  `e3897738ea62f6e231dc4d5e2256290408ad0c01ddbe96002d597252e1ec1fcf`;
- no local candidate, smoke, development, or Grimer/Galvantula/Murkrow screen
  enters training.

Games are split before feature extraction by
`int(sha256(filename)[:8], 16) % 20`: buckets 0-13 train, 14-16 validation,
and 17-19 test. Every state from a game remains in one split. One example is
the first focal-player `MAIN` decision in each game turn. The label is the
focal V5 game's final binary outcome. Loss weights are normalized within each
game so long games cannot dominate. The opponent's archived deck list is used
only as the training-time archetype/range anchor; no opponent hand, prize, or
deck order is a feature.

## Frozen value model

`public-value/v2` uses deterministic public features:

- turn, seat, prizes, hand/deck counts, action flags, statuses, HP, Energy,
  board size, Rule Box/stage/ability counts, and ready-attacker proxies;
- exact own public board and own-visible hand summaries;
- opponent public board/discard/stadium summaries;
- fixed hashed count vectors for the public opponent cards, own hand, stadium,
  and the inferred opponent-deck range.

It is a train-standardized MLP `D -> 64 -> 32 -> 1` with ReLU activations and
0.10 training dropout. Freeze seed `805001`, AdamW learning rate `0.001`,
weight decay `0.0001`, batch size `512`, and exactly 40 epochs. The sole
eligible artifact is epoch 40; no checkpoint, threshold, seed, feature,
architecture, or temperature is selected from validation/test results.

Offline admission requires:

1. exact source digest, split, example, finite-value, save/reload, and feature
   dimension checks;
2. validation and test ROC AUC each at least 0.62;
3. validation and test turn-at-most-eight ROC AUC each at least 0.57;
4. validation and test Brier score each strictly below the constant predictor
   using that split's empirical win rate; and
5. no major opponent family with both labels and at least 100 examples below
   ROC AUC 0.52.

Failure stops before candidate match play.

## Frozen resolver

If the offline gates pass, the candidate uses:

- immutable V5 neural policy plus EarlyPad and lethal-only Brave Bangle as the
  blueprint;
- `3` public-consistent hidden-state particles;
- up to `4` semantically distinct blueprint-ranked root actions;
- at most `48` official-engine rollout decisions, stopping when play returns
  to the root player two turns later;
- the learned public win probability as the nonterminal leaf;
- a `5.0` second search budget, at most `3` searches through turn `8`;
- activation only when starting second or after public Marnie, Starmie,
  Archaludon, or Cinderace evidence; and
- a change only when paired mean gain is at least `0.08`, every particle is no
  worse than `-0.03`, and at least 75% of particles improve.

The blueprint action is returned on uncertainty, incomplete particles,
timeout, exception, or a failed safety bound. Telemetry must distinguish real
official-engine rollout execution from the learned public abstraction.

## Match admission

1. Require unit/serialization checks and one terminal official-engine smoke.
2. Run a sealed, side-swapped development panel against exact archived
   opponents: 12 games versus `AshV178MarnieFullBest` and 8 games versus
   `SparkStarmieFroslassCFV13`, for both V2 and the same three-Murkrow deck with
   the immutable V5 policy. Do not inspect outcomes until all 40 games finish.
3. Advance only if all games terminate normally, V2 wins at least 15/20,
   beats the same-deck control by at least four wins, is not worse in either
   opponent row, completes at least 20 searches across at least eight games,
   changes at least four blueprint actions, and records zero resolver errors.

Passing authorizes a fresh untouched broad panel, not submission. Broad
confirmation must include Marnie, both Starmies, Rocket mirror, Alakazam,
Archaludon/Cinderace, both seats, and immutable V5. Only a separately frozen
passing confirmation may authorize one package and one new internal-arena
identity. Arena completion remains at least 160 wins in that identity's own
first 200 rated games; draws are non-wins and identities are never pooled.

## Evidence boundary

The official engine supplies legal transitions and rollout execution. Hidden
zones are sampled from public counts and opponent-range priors, not read from
the real game. The learned leaf is an empirical public-state approximation,
not an official-engine result, an equilibrium solution, or hidden-hand truth.
No Kaggle submission or unverified performance claim is authorized.
