# Public-range safe router V13 protocol

Frozen: 2026-08-07, before V13 match outcomes.

## Motivation

The frozen V10 epoch-4 learner reproduced a strong 70/120 current-meta
confirmation, passing Dragapult, Marnie, Alakazam, Starmie, and aggregate
floors. It missed only direct V5 by one game, 11/24 versus 12/24. The learned
direction is therefore useful but not baseline-safe in an unidentified Rocket
mirror.

V13 implements the documented game-theoretic pivot at the smallest genuine
boundary: V5 is the safe blueprint strategy and the V49-adapted model is an
exploitation overlay exposed only after a public opponent range identifies a
validated family. This is not a new heuristic action guard and does not use
sampled rollout search.

## Frozen router

- `model_safe.pt` is byte-exact V5.
- `model.pt` is byte-exact V10 retention-AWR epoch 4.
- Both `NNAgent` instances consume every non-guard public observation from
  game start, keeping independent belief and autoregressive state aligned.
- The safe V5 action is returned by default.
- The adapted action is latched only after active, Bench, or discard reveals:
  Dreepy/Drakloak/Dragapult ex; Marnie's Impidimp/Morgrem/Grimmsnarl ex; or
  Staryu/Mega Starmie ex.
- Opponent hand, deck, prizes, name, seed, and replay metadata cannot activate
  the overlay.
- Rocket mirror and Alakazam therefore remain on V5 throughout.
- V5's exact 60-card deck, early-Spidops search, exact-lethal Bangle rule,
  official adapter, and legality handling are unchanged.

## Gates

1. Deterministic tests must verify both model hashes, exact deck integrity,
   private-card nonactivation, public-card latching, and Rocket/Alakazam safe
   routing.
2. Replay at least 500 official observations. Every returned action must be an
   offered legal index. Before range activation, V13 must reproduce packaged
   V5 exactly; after activation, it must reproduce packaged epoch 4 exactly.
3. Run a fresh 120-game official-engine panel: 24 side-swapped games each
   against generic Dragapult, current Marnie, V5 direct, the Alakazam threat
   proxy, and flat Starmie.
4. Advance only with zero execution failures and all of:
   - at least 65/120 overall;
   - Dragapult at least 17/24;
   - Marnie at least 11/24;
   - V5 direct at least 12/24;
   - Alakazam at least 20/24; and
   - Starmie at least 4/24.
5. A passing router receives an immutable archive audit and exactly one
   internal-arena submission. No Kaggle submission or live-V49 strength claim
   is authorized.

V13's safety claim is behavioral and public-state scoped: it preserves the V5
blueprint in unrecognized games. It is not a solved-game or equilibrium claim.
