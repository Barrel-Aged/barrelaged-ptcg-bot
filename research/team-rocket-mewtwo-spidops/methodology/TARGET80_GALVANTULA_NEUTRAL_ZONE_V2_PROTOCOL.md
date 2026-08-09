# Target-80 Galvantula / Neutralization Zone V2 protocol

Frozen: 2026-08-04, before official smoke or outcome inspection

Status: **one final Galvantula-family development test; no submission authorized**

## Causal motivation

V1 genuinely knocked out Grimmsnarl ex for 360 damage ten times, but eight of
those games were losses.  In seven loss traces, the responding attacker was a
new Grimmsnarl ex using `Shadow Bullet` to immediately knock out the 100-HP
Galvantula.  The two V1 wins were the only games in which Galvantula survived
to attack three times.

V2 tests that identified continuation failure.  It replaces Hero's Cape with
the ACE SPEC stadium Neutralization Zone.  While the stadium remains in play,
opposing Pokemon ex cannot damage Galvantula, Spidops, or another non-Rule-Box
Pokemon.  Marnie can counter it with Spikemuth Gym, so this is a falsifiable
one-turn protection hypothesis rather than a claimed hard lock.

## Frozen arms

1. `galvantula_neutral_zone_v2`: V1's exact two-two Joltik/Galvantula package,
   Hero's Cape -> Neutralization Zone, plus a public-state route that plays the
   stadium before an Active, fully paid Galvantula attacks Grimmsnarl.  It may
   use Transceiver -> Petrel -> Neutralization Zone when those cards are
   currently legal, and it promotes a ready Galvantula after a knockout.
2. `galvantula_zone_control_v2`: the identical 60 cards with V1's policy and no
   Zone-sequencing overlay.
3. immutable V5: original deck and policy.

All hidden-state inference remains prohibited.  The route activates only from
public Marnie cards and own-visible hand, board, discard, and stadium state.

Frozen hashes:

- shared V2 deck:
  `a6bf17f81ae92c516d5e607fe56679dbbdbc5eb98afbf3752d38255ac5164ba0`;
- V2 candidate `main.py`:
  `542fec1877a3a8fe52570054aaced0e79fd83301f62ef46cd5927dcb2d3c66d1`;
- deck-identical control `main.py`:
  `34542a1d10e3819155ed16ae85f2aa475afd1321a893251eac75736464a92e91`.

## Admission gate

1. Require one terminal official-engine smoke after the frozen edit.
2. Then run twelve side-swapped games per arm against the exact
   `AshV178MarnieFullBest` bundle.  Do not inspect any outcomes until all 36
   games finish.
3. Advance only at 9/12 or better, at least two wins above each control, all
   terminal, with at least three attacks where Neutralization Zone was played
   immediately before `Shocking Web`, and at least two documented opponent
   turns where the protected Galvantula remained in play after an ex attack
   would otherwise have been legal.

Failure closes the complete Galvantula family for this campaign.  Passing
authorizes only a broad untouched panel, not packaging or arena submission.

Evidence boundary: all match evidence must come from the official engine.  The
overlay is a narrow tactical safety mechanism and is not represented as the
project's final game-theoretic planner.
