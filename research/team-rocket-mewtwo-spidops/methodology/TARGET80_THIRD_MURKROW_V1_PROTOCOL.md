# Target-80 third-Murkrow V1 protocol

Frozen: 2026-08-05, before official smoke or development outcomes

Status: **development hypothesis only; no package or submission authorized**

## Hypothesis

Replace the required-to-remove Wobbuffet with a third Team Rocket's Murkrow.
Unlike an unfamiliar tech attacker, Murkrow is already part of the frozen V5
deck and policy support. The extra copy preserves Basic Team Rocket density,
improves the one-retreat-cost opening pool, and makes `Deceit` Supporter access
or `Torment` disruption available more often without changing the deck's
Energy or evolution requirements.

This is a deck-only test. The exact V5 neural model, EarlyPad rule,
lethal-only Brave Bangle rule, and all remaining card counts are unchanged.
No forced Murkrow search, attachment, promotion, or attack rule is present.

## Frozen identity

- candidate directory: `work/target80_candidates/murkrow_three_v1/`;
- deck: 60 cards, zero Wobbuffet, three Team Rocket's Murkrow;
- deck SHA-256:
  `3778b1c3bcd8c74230af90ecddd849123490a945679ef67b75e2b9a5dac57193`;
- `main.py` SHA-256:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`;
- `model.pt` SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.

The control is immutable V5 from
`work/target80_candidates/earlypad_bangle_lethal_v1/`.

## Admission gate

1. Require one terminal official-engine candidate smoke with no crash,
   timeout, illegal action, or fallback exception.
2. Run twelve side-swapped games per arm against the exact
   `AshV178MarnieFullBest`: third-Murkrow candidate and immutable V5. Do not
   inspect outcomes until both arms finish.
3. Advance only if the candidate wins at least 9/12, beats V5 by at least two
   wins, all games terminate normally, Murkrow enters play in at least six
   games, and at least one `Deceit` or `Torment` is executed.

Passing authorizes only a fresh broad panel including both Starmie variants,
Rocket mirror, Alakazam, Archaludon/Cinderace, Marnie, and both seats. Failure
closes this exact deck-only substitution; it does not authorize adding a
post-hoc Torment script from individual seed outcomes.

Evidence boundary: official-engine execution can establish legality, public
card use, and match outcomes. This deck-only experiment is not the final
neural/game-theoretic planner and makes no hidden-information claim.
