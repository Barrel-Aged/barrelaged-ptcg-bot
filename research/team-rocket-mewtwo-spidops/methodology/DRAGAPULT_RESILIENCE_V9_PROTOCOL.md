# Dragapult resilience V9 protocol

Frozen: 2026-08-07, before local candidate outcomes.

## Motivation and evidence boundary

The mature `HelbrosRocketSpidopsBC2RecoveryV8` pull contains 2,453 official
arena games at 1,257-1,196 (51.24%). Its largest new failure cell is the
Soldier Dragapult policy: 62-121 (33.88%), including 22-69 when starting
second. The unchanged mature V5 identity remains the stronger baseline at
1,411-1,113 (55.90%). In the V8 Dragapult slice, games that put Articuno into
play were 59-103, while games that never did so were 3-18. This is an
association, not a causal estimate, but the card's public Repelling Veil text
directly blocks Phantom Dive's damage-counter effect on Basic Team Rocket
Pokemon.

V9 therefore starts from the byte-exact V5 deck/model/runtime. It does not use
the weaker V8 wrapper as its baseline and does not inspect opponent-private
cards, arena identity, seed, or replay metadata.

## Frozen hypotheses

Two arms may enter development:

1. `dragapult_resilience_v9` keeps V5's exact 60 cards and model. It extends
   the admitted Poke Pad continuity target through turn six but stops after two
   Spidops are in play or reserved. After Dreepy, Drakloak, or Dragapult ex is
   public, it searches and plays Articuno promptly and withholds only redundant
   fifth-slot Basic plays until Articuno is present.
2. `dragapult_resilience_3articuno_v9` applies the same policy to a one-card
   structural arm: the second Murkrow becomes a third Articuno. Wobbuffet is
   retained because its Dragapult opening was 11-10 in the mature V8 slice;
   this experiment does not repeat the rejected zero-Wobbuffet swap.

Both arms preserve V5's lethal-only Brave Bangle rule, greedy neural blueprint,
belief tracker, attack-safe legality handling, and exact official-engine
adapter. Search over sampled rollouts is not introduced.

## Development sequence and gates

1. Deterministic tests must prove 60-card integrity, model equality, legal
   returned indices, public-only Dragapult routing, two-Spidops search cutoff,
   Articuno acquisition/play, and last-slot masking.
2. Run 24 side-swapped official-engine games for V5 control and each arm
   against the frozen `soldier_drag_marl_v4_proxy`. To advance, an arm must
   score at least 13/24 and at least three wins more than the control. If both
   pass, select by wins, then starting-second wins, then fewer deck changes.
3. Run the sole selected arm in fresh side-swapped official-engine cells:
   32 Dragapult-proxy games and 16 games each against V5, current Marnie,
   Alakazam, and flat Starmie packages. All games must terminate without agent
   or legality errors. The arm must score at least 18/32 against Dragapult,
   at least 9/16 against V5, and at least 6/16 in every broad cell.
4. Replay at least 500 official observations through the frozen V5 action and
   V9 action paths. Non-trigger states must reproduce V5 exactly; every changed
   action must be an offered legal option and match one disclosed guard.
5. Only after these gates may one immutable archive be built, audited, and
   submitted to the internal arena. No Kaggle submission or unverified strength
   claim is authorized.

The proxy is genuine official-engine execution but not the live Soldier V49
package. A passing local result is therefore only an admission gate; the new
immutable arena identity remains the independent strength estimate.
