# BCv2 recovery: Murkrow / Articuno V8 protocol

Date frozen: 2026-08-06  
Intended arena identity: `HelbrosRocketSpidopsBC2RecoveryV8`

## Question

Why did `HelbrosRocketSpidopsScaleBCV2` retain materially better Kaggle
strength than the later Wobbuffet-free builds, and can one final internal-arena
iteration preserve the useful BCv2 substrate while honoring the zero-
Wobbuffet deck direction and the queued Articuno bench-slot fix?

## Evidence available before candidate testing

- ScaleBCV2's audited Kaggle run was 13-6 in public results (14-6 including
  validation) with a score of 876.0 at the frozen cutoff. Its deployed actions
  reproduced on all 1,227 replay decisions, with 1,225 byte-exact matches and
  two order-only multi-select differences.
- Wobbuffet made only two of 103 attacks in those games. Both were Headbutt
  Bounce choices in one loss; this does not show that Wobbuffet's *deck slot*
  was valueless because it also supplied a Basic Team Rocket body.
- V5 preserved BCv2's exact deck and exact model and is currently 1,411-1,113
  at rating 822.7 in the read-only arena snapshot.
- V7's zero-Wobbuffet Hancang rebuild is 2,162-1,979 in the freshly pulled
  4,141-game archive (52.21%). It is 240-408 against Alakazam and 241-458
  against Starmie. In 80.14% of its losses it never forms a second ready
  attacker.
- V7's Mimikyu opening is 483-645 (42.82%). Because Mimikyu is not a Team
  Rocket Pokemon in this engine, it also lowers Rocket Rush damage relative to
  a Rocket replacement.
- The archived 519-game V5 slice was 42-19 against Alakazam, and its losses
  lacked a second ready attacker 62.34% of the time. Murkrow openings were
  64-58 (52.46%) and supplied the zero-Energy Deceit setup attack.

These observations support a substrate diagnosis: the later variants changed
the deck distribution and model together, removed both Murkrow copies, added
low-performing non-Rocket openings, and reduced Rocket-count density. They do
not support the claim that one heuristic alone caused the full regression.

## Frozen V8 hypothesis

Start from the exact V5/BCv2 runtime and make only these disclosed changes:

1. Replace the single Team Rocket's Wobbuffet with a third Team Rocket's
   Murkrow. The final list remains at 60 cards, contains zero Wobbuffet, and
   preserves 15 Team Rocket Pokemon.
2. Preserve the byte-exact BCv2 model, deck library, official-engine adapter,
   legality handling, early Poke Pad-to-missing-Spidops guard, and exact-lethal
   Brave Bangle guard.
3. Import the V7 public-information Articuno reservation: at four occupied
   bench slots, after a publicly revealed attack-effect family, play Articuno
   if offered; otherwise mask only redundant Basic plays. On this three-
   Murkrow list, a third Murkrow is redundant when two are already in play.

No hidden hand is read. No sampled rollout, heuristic value planner, new
training claim, Wobbuffet exit rule, Erasure Ball discard rule, or Kaggle
submission is part of V8. Rejected deck packages are not silently combined.

## Preregistered health and regression checks

Before the single arena submission:

1. Verify 60 deck rows, zero card 432, three card 463, 15 Team Rocket Pokemon,
   and byte equality of `model.pt` with ScaleBCV2/V5.
2. Extend the deterministic Articuno guard checks to cover the third-Murkrow
   reservation and verify every forced/masked result remains a legal offered
   option.
3. Run at least one terminal packaged official-engine smoke, as required after
   decision-affecting code changes.
4. Run a side-swapped local recovery screen:
   - 24 games directly against immutable V5;
   - 24 games directly against immutable V7;
   - four games each against Marnie, Froslass/Starmie, Alakazam, and Soldier.
5. Stop for repair if any game is nonterminal/illegal, V8 scores fewer than 10
   wins in either 24-game direct cell, or it goes 0-4 in any broad cell.

Local games are health filters, not performance proof. The one immutable V8
arena identity is the external strength estimate. V5 and V7 results must not
be pooled with V8.

## Terminal action

If integrity and health checks pass, package and submit exactly one immutable
internal-arena identity named `HelbrosRocketSpidopsBC2RecoveryV8`. Do not
submit to Kaggle. Record archive hashes, checks, arena acknowledgement, and
the uncertainty that the arena has not yet accumulated a mature V8 sample.
