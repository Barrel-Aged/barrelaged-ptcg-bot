# Target-80 Mimikyu Poffin Spidops V1 protocol

Status: **frozen before development outcomes**  
Frozen: 2026-08-04

## Hypothesis

The fresh Wobbuffet-free Mimikyu control went 10-10 against the exact Marnie
opponent. All ten wins reached a full six-Pokemon board and attacked with
Spidops; none attacked with Mewtwo. Six of ten losses attacked with Mewtwo,
and 22 of 28 observed Mewtwo Energy attachments had a legal Tarountula or
Spidops alternative. This motivates a coherent setup/attacker rebuild rather
than replacing the neural policy wholesale.

Relative to the Wobbuffet-free Mimikyu control, the candidate removes two
Basic Psychic Energy and one Basic Grass Energy and adds three Buddy-Buddy
Poffin. Its 11-Energy count matches the supplied professional deck's energy
density. Poffin directly establishes 70-HP-or-less Team Rocket basics,
prioritizing three Tarountula/Spidops lines and the free-retreat Mimikyu. The
V5 neural policy remains the blueprint. Once a unique Marnie evolution is
public, narrow guards put Hero's Cape on a Tarountula/Spidops line, redirect a
neural Mewtwo attachment only when the same Energy can complete a legal
Spidops line, and prefer an already-ready Spidops over Mewtwo at promotion.
The existing EarlyPad and exact lethal-Bangle guards remain intact.

No opponent private zone, opponent name, rollout search, or hidden-hand label
is used. The guards are bounded mechanics around a frozen neural baseline;
they are not represented as equilibrium solving.

## Frozen identities

| Artifact | SHA-256 |
|---|---|
| untouched V5 deck | `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852` |
| untouched V5 main | `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6` |
| Mimikyu deck-only control | `dc1b1d8709187aec04f6b89532bf9121ec7e48ec4cec2aca8ce98de02c6eccde` |
| Poffin candidate deck | `4acdc0d75effc2a25fba3feb819d8290a4ca437c4256118530aa61ee9131a9db` |
| Poffin candidate main | `44296e77d82d7370cb98b92399346d525c26ce9149eb614a6b1d49e9d08f96fb` |
| shared V5 neural model | `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b` |

The exact opponent remains `AshV178MarnieFullBest` with main/deck/model hashes
`0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b`,
`92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d`,
and `98072b2a75197ad9d073af3c7ba5ff66a6daadd266b86b5463fef2ff34910757`.

The terminal one-game smoke in `work/target80_mimikyu_poffin_smoke_v1/` is
excluded. It established only runtime viability: Poffin resolved, the board
filled, Cape reached a Spidops line, and Spidops attacked four times.

## Locked development cell

Run three fresh, concurrently sealed, automatically side-swapped 20-game arms
against the exact opponent with `max_steps=300`, `action_timeout_s=10`, and
decision traces:

1. untouched Wobbuffet V5;
2. Wobbuffet-free Mimikyu deck with unchanged V5 policy/guards;
3. Mimikyu Poffin Spidops V1.

No outcome may be inspected until all three arms finish. The candidate
advances only if every game is terminal/correct and it satisfies every gate:

- at least 16/20 wins;
- at least four more wins than each contemporaneous control;
- at least 7/10 wins from each seat;
- Poffin resolves in at least 10 games;
- a full six-Pokemon board is observed in at least 16 games;
- Spidops attacks in at least 16 games; and
- Mewtwo attacks in no more than four games.

Failure permanently closes this exact deck and guard package. It may not be
renamed, submitted, or pooled with another identity.

## If admitted

Use the broad-health and untouched-confirmation gates frozen in
`TARGET80_ARBOK_LOCK_V1_PROTOCOL.md`. Both Starmie variants, Rocket, Alakazam,
resource-wall opponents, and both seats remain mandatory. A broad aggregate
cannot hide a losing Marnie or second-seat row. Only one independently
confirmed package may enter the internal arena.

The success contract remains at least 160 wins in one immutable successor's
own first 200 rated games, with draws as non-wins and no identity pooling. No
Kaggle submission is authorized.
