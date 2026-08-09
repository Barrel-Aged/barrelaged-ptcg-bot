# Target-80 Mimikyu Marnie Spine V1 protocol

Status: **frozen before development outcomes**  
Frozen: 2026-08-04

## Hypothesis

The dominant arena loss cell is Marnie's Grimmsnarl. A six-Rocket board lets
single-Prize Team Rocket's Spidops use Grass weakness to knock out a fresh
320-HP Grimmsnarl ex in one attack, while Mewtwo is itself weak to Darkness.
The candidate therefore replaces Wobbuffet with one free-retreat Team
Rocket's Mimikyu and preserves the deployed V5 neural policy as its safe
default. Only after an opponent publicly reveals Marnie's Impidimp, Morgrem,
or Grimmsnarl ex does it route to the project's independently implemented
Rocket tactical spine, which prioritizes the one-Prize Spidops campaign and
suppresses early Mewtwo exposure under public Dark pressure. The existing
EarlyPad and exact lethal-Bangle guards remain above either policy.

This is a diagnostic safe-blueprint/exploitation bridge. Its Marnie route is
still deterministic and is not represented as the project's finished
game-theoretic planner. It receives no opponent private cards or arena name.

## Frozen identities

| Artifact | SHA-256 |
|---|---|
| untouched V5 deck | `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852` |
| untouched V5 main | `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6` |
| Wobbuffet-free Mimikyu deck | `dc1b1d8709187aec04f6b89532bf9121ec7e48ec4cec2aca8ce98de02c6eccde` |
| Wobbuffet-free deck-only control main | `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6` |
| candidate main | `a0721517a3d286b4a0a89a8c973c38ee49b76d6f2e0982bcee914cb728460ea5` |
| candidate Rocket spine | `6c3ec73e68acda1fa18cbf1e060d8a584233e9457b06e2971802d9e21ac5d4e0` |
| candidate router policy | `51fb186c197aea703d818c05567d0e4b3f94a8890e1dfc7868aa0e2b2df01151` |
| shared V5 neural model | `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b` |
| frozen specialist ranker | `d64cf440e6afa3323dbf729a45766652e7012b45a9eb684dffc359205d16be93` |

The exact opponent is the archived `AshV178MarnieFullBest` bundle with
main/deck/model hashes
`0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b`,
`92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d`,
and `98072b2a75197ad9d073af3c7ba5ff66a6daadd266b86b5463fef2ff34910757`.

The one-game terminal smoke in `work/target80_mimikyu_spine_smoke_v1/` is
excluded from every strength total.

## Locked development cell

Run three fresh, automatically side-swapped 20-game arms concurrently against
the exact opponent, with `max_steps=300`, `action_timeout_s=10`, and decision
traces:

1. untouched Wobbuffet V5;
2. Wobbuffet-free Mimikyu deck with the unchanged V5 policy and guards;
3. Mimikyu Marnie Spine V1.

Do not inspect any outcome until all three files seal. The candidate advances
only if every game is terminal/correct and all of these conditions hold:

- at least 16/20 wins overall;
- at least four more wins than each contemporaneous control;
- at least 7/10 wins from each seat;
- a unique public Marnie identifier is visible in at least 18/20 games; and
- replaying candidate observations through the frozen deck-only control shows
  at least 20 non-forced action deviations spread across at least 12 games.

Failure closes this exact deck, router, and specialist. It may not be renamed,
packaged, submitted, or silently pooled with another candidate.

## If admitted

The candidate next faces the broad-health and untouched-confirmation gates in
`TARGET80_ARBOK_LOCK_V1_PROTOCOL.md`, including both Starmie variants, Rocket,
Alakazam, and resource-wall opponents. No broad aggregate may hide a losing
Marnie or second-seat row. Only one package can advance to one immutable
internal-arena identity.

Arena completion still means at least 160 wins in that identity's own first
200 rated games, with draws counted as non-wins and no pooling. No Kaggle
submission is authorized.

## Claim boundary

All routing evidence is public official-engine state. Official-engine games,
legality, and card effects are genuine; the specialist's strategic values are
hand-engineered abstractions, not equilibrium values. Passing this cell would
justify broader testing only, not an 80% or superhuman claim.
