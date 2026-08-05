# Target-80 Kangaskhan backup V1 protocol

Frozen: 2026-08-05, before official smoke or development outcomes

Status: **development hypothesis only; no package or submission authorized**

## Hypothesis

Replace the required-to-remove Wobbuffet with one Team Rocket's Kangaskhan ex.
This preserves a Basic Team Rocket body for Mewtwo's Power Saver and Proton,
but turns the slot into a credible backup attacker. One Team Rocket's Energy
plus any other attached Energy pays `Wicked Impact`; a Team Rocket Supporter
then raises the attack from 120 to 220 damage.

This is intentionally a deck-only test. The exact immutable V5 neural policy,
EarlyPad rule, lethal-only Brave Bangle rule, Trainers, Energy, and every other
Pokemon count are unchanged. No Kangaskhan-specific routing is allowed until
the unassisted card substitution proves that the model can use the line safely.

## Frozen identity

- candidate directory: `work/target80_candidates/kangaskhan_backup_v1/`;
- deck: 60 cards, zero Wobbuffet, one Team Rocket's Kangaskhan ex;
- deck SHA-256:
  `7fd55561804fc9ed0e85ed15d651e2268ea44a485b43c39fcb5cd591a72d10c3`;
- `main.py` SHA-256:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`;
- `model.pt` SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.

The control is immutable V5 from
`work/target80_candidates/earlypad_bangle_lethal_v1/`.

## Admission gate

1. Require one terminal official-engine smoke from the frozen candidate, with
   no crash, timeout, fallback exception, or illegal action.
2. Run twelve side-swapped games per arm against the exact
   `AshV178MarnieFullBest`: Kangaskhan candidate and immutable V5. Do not inspect
   any outcomes until both 12-game arms finish.
3. Advance only if the candidate wins at least 9/12, beats V5 by at least two
   wins, all games terminate normally, Kangaskhan reaches play in at least four
   games, and it uses an attack in at least two games.

Passing authorizes only an untouched broad panel including both Starmie
variants, Rocket mirror, Alakazam, Archaludon/Cinderace, Marnie, and both seats.
Failure closes this exact deck-only substitution and does not authorize tuning
from individual seed outcomes.

Evidence boundary: official-engine execution can establish legality, public
card interactions, and match outcomes. This deck-only screen is not the final
neural/game-theoretic planner and makes no hidden-information or strength claim.
