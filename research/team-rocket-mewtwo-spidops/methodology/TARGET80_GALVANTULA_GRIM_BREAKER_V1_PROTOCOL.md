# Target-80 Galvantula Grim-breaker V1 protocol

Frozen: 2026-08-04, before the final smoke and development outcomes

Status: **development hypothesis only; no packaging or submission authorized**

## Strategic hypothesis

The current Marnie opponent is built around a 320-HP, Grass-weak Marnie's
Grimmsnarl ex with the `Punk Up` Ability.  The one-prize Galvantula (engine ID
126) attacks for 50, gains 50 from `Compound Eyes`, gains another 80 when
Lightning is attached, and then applies Weakness.  Its resulting 360 damage is
an exact one-hit knockout on that two-prize target.

Joltik (160) can use the zero-cost `Jolting Charge` to attach up to two Basic
Grass and two Basic Lightning Energy from the deck.  The candidate therefore
tests a real one-prize prize-map counter, not another Wobbuffet attack guard.

## Frozen 60-card package

Starting from immutable V5, the candidate makes these paired changes:

- Wobbuffet -> first Joltik;
- one Articuno -> first Galvantula;
- two Basic Psychic -> two Basic Lightning;
- one Bug Catching Set -> Switch;
- one Bug Catching Set -> second Joltik;
- one Factory -> second Galvantula.

The resulting deck has 60 cards, zero Wobbuffet, two Joltik, two Galvantula,
two Lightning, one Psychic, one Switch, thirteen Team Rocket's Pokemon, and a
valid official-engine deck.  It retains the four-four Spidops line, both
Mewtwo ex, V5's neural model, EarlyPad rule, and lethal-only Brave Bangle rule.

Frozen identities:

- deck SHA-256:
  `9cf2696783d99b3468391125dbc070b85adf692d3e2f40909533a48db53cd95b`;
- candidate `main.py` SHA-256:
  `34542a1d10e3819155ed16ae85f2aa475afd1321a893251eac75736464a92e91`;
- deck-identical control `main.py` SHA-256:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`.

The candidate activates only after Marnie's Impidimp, Morgrem, Grimmsnarl ex,
or Spikemuth Gym becomes public.  Its sticky matchup flag is therefore derived
from public turn history, not opponent identity or hidden cards.  It can:

- search and bench the missing line with Poke Pad;
- evolve before a hand reset;
- reserve Grass and Lightning attachments for that line;
- use Transceiver -> Petrel -> Energy Search or Switch when required;
- recover a missing line piece with Night Stretcher;
- switch only a fully paid Galvantula into an Active Grimmsnarl ex; and
- replace an attempted End with legal `Jolting Charge` or lethal
  `Shocking Web`.

Every unhandled decision remains with the frozen V5 neural policy.

## Admission sequence

1. Re-run one terminal official-engine smoke after the final two-two package
   and Petrel bridge are frozen.  Any crash, illegal action, timeout, deck
   rejection, or nonterminal result rejects V1.
2. Run twelve side-swapped official-engine games against the exact
   `AshV178MarnieFullBest` bundle for each of:
   the frozen candidate, the deck-identical original-policy control, and
   immutable V5.  Do not inspect outcomes until all three arms finish.
3. Advance only if the candidate:
   - wins at least 9/12;
   - beats the deck-identical control by at least two wins;
   - beats immutable V5 by at least two wins;
   - has all games terminate normally; and
   - demonstrates at least three genuine `Shocking Web` attacks into
     Grimmsnarl ex, with at least two of those attacks producing a knockout.
4. A pass authorizes only a fresh broad development panel, including both
   Starmies, Rocket, Alakazam, Crustle, and seat rows.  It does not authorize
   packaging, submission, or an 80% claim.

## Evidence boundary

Official-engine games establish legal execution and observed outcomes only.
This package is a tactical public-state counter layered over the preserved
neural blueprint; it is not the final game-theoretic planner described in the
project pivot, and no hidden-hand belief claim is made.
