# Target-80 Cornerstone Wall V1 protocol

Status: **frozen before development outcomes**  
Frozen: 2026-08-04

## Hypothesis

Arbok V1 improved the exact Marnie development cell from 6-14 to 11-9, but
its seat-one timing ceiling prevented a lock before Grimmsnarl's first
evolution window. Cornerstone Mask Ogerpon ex is a materially different
fallback: it is Basic and its Cornerstone Stance prevents attack damage from
Marnie's Grimmsnarl ex, Froslass, and Munkidori after they are already in play.

The candidate is a complete Wobbuffet-free package. Relative to untouched V5,
it removes Wobbuffet, both Articuno, all three Basic Psychic Energy, one Bug
Catching Set, and Brave Bangle; it adds three Cornerstone Mask Ogerpon ex,
three Basic Fighting Energy, and two Switch. It retains Hero's Cape and the
exact frozen V5 neural model. The public pilot searches, benches, fuels,
promotes, and retains Cornerstone against public Marnie cards, while yielding
to an immediate active Spidops/Mewtwo knockout. If Cornerstone is offered as a
setup Active before the opponent is revealed, it is selected so the mechanism
can operate from either seat.

## Frozen identities

| Artifact | SHA-256 |
|---|---|
| untouched V5 deck | `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852` |
| untouched V5 main | `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6` |
| shared neural model | `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b` |
| Cornerstone deck | `61ef3d9b28cbdf9a477235b4aebd202b54d9b3296a163160f0210edcf0793ff3` |
| Cornerstone main | `fc11043d88020d2544dc96406a91a8b15e9bbe8ce92a5afde4f6dea7e938cd71` |

The exact opponent remains `AshV178MarnieFullBest` with main/deck/model hashes
`0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b`,
`92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d`,
and `98072b2a75197ad9d073af3c7ba5ff66a6daadd266b86b5463fef2ff34910757`.

## Development and gate

Run a fresh 20-game automatically side-swapped panel for untouched V5 and the
candidate, `max_steps=300`, `action_timeout_s=10`, with decision traces. Do not
inspect either outcome until both 20-game files seal. The earlier one-game
terminal smoke is excluded from all totals.

The candidate advances only if all games are terminal/correct, it wins at
least 16/20, beats control by at least four wins, wins at least 7/10 from each
seat, makes Cornerstone Active in at least 12 games, and has no zero-attack
step-limit or timeout game. Failure closes this exact deck and pilot.

If admitted, use the same broad-health and untouched-confirmation gates frozen
in `TARGET80_ARBOK_LOCK_V1_PROTOCOL.md`, followed by one self-contained package
and one immutable arena successor. Arena success still requires at least 160
wins in that identity's own first 200 rated games; draws are non-wins and
identities are never pooled.

## Claim boundary

Cornerstone targeting uses only public official-engine state. The immunity is
an official card effect; the policy around it is a deterministic abstraction,
not hidden-hand access or equilibrium solving. No Kaggle submission is
authorized and no local result alone proves arena strength.
