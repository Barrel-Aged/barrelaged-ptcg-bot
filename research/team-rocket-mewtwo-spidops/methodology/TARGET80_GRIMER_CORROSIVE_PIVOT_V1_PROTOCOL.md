# Target-80 Grimer corrosive pivot V1 protocol

Frozen: 2026-08-04, before official smoke and development outcomes

Status: **development hypothesis only; no package or submission authorized**

## Hypothesis

Team Rocket's Grimer is a deck-native Wobbuffet replacement.  One Team
Rocket's Energy supplies the Darkness-plus-Colorless cost of `Corrosive
Sludge`.  After that attack, Grimmsnarl must retreat and discard two Energy or
be discarded with all cards attached at the end of its controller's turn.

Unlike Galvantula, Grimer preserves the Team Rocket body count and can be
pivoted in with Team Rocket's Giovanni while Giovanni also gusts Grimmsnarl.
The intended value is action-space restriction and resource denial, followed
by a ready Spidops—not direct damage from another off-engine attacker.

## Frozen package and controller

The 60-card candidate replaces Wobbuffet and one of two Articuno with two Team
Rocket's Grimer.  Every Trainer, Energy, neural weight, EarlyPad decision, and
lethal-only Brave Bangle rule is otherwise unchanged.

The overlay activates only after a Marnie core card or Spikemuth Gym becomes
public.  It can:

- search one Grimer with Poke Pad only after the EarlyPad Spidops need is
  satisfied;
- recover it with Night Stretcher;
- attach one Team Rocket's Energy;
- use Giovanni to pivot the paid Grimer into a public Grimmsnarl when the
  current Spidops or Mewtwo has no certified knockout;
- target Grimmsnarl on Giovanni's opponent-side prompt; and
- replace an attempted End with legal `Corrosive Sludge`.

All other decisions remain with immutable V5.  The deck-identical control uses
the same two-Grimer list with the original V5 policy.

Frozen hashes:

- shared deck:
  `f36993058b31acd9e5224621456e8eb367474d74bd3a03685332df542091708b`;
- candidate `main.py`:
  `cd06472425717ea18592f796d5104084a3125142e5fe998b9bbe9d2eeb7bef00`;
- deck-identical control `main.py`:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`.

## Admission gate

1. Require a terminal official-engine smoke after the frozen edit.
2. Run twelve side-swapped games per arm against the exact
   `AshV178MarnieFullBest`: candidate, deck-identical control, and immutable
   V5.  Do not inspect any outcomes until all 36 finish.
3. Advance only if the candidate wins at least 9/12, beats each control by at
   least two wins, all games terminate normally, `Corrosive Sludge` is used in
   at least four games, and at least two attacked Grimmsnarl produce a public
   two-Energy retreat or end-turn discard.

Passing authorizes only an untouched broad panel.  Failure closes this exact
Grimer package and does not authorize tuning from individual seed outcomes.

Evidence boundary: the official engine can prove card interaction, legality,
and observed outcomes.  This is a narrow public-state tactical overlay, not
the final game-theoretic architecture and not a hidden-information model.
