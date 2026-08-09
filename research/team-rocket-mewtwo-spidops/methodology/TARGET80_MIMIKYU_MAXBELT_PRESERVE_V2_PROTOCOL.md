# Target-80 Mimikyu / Maximum Belt Preserve V2 — Locked Protocol

Status: preregistered before confirmation outcomes are generated or inspected.

Date: 2026-08-05 (America/Denver)

## Hypothesis

Maximum Belt adds 50 damage before Weakness and Resistance. Four Team Rocket
Pokémon therefore let Spidops's Rocket Rush deal `(4 × 30 + 50) × 2 = 340`
damage to a Grass-weak Pokémon ex, crossing Grimmsnarl ex's 320-HP threshold.
The frozen neural policy was trained with Hero's Cape, not Maximum Belt. In
development it repeatedly spent Belt on Mewtwo, Murkrow, Articuno, Mimikyu, or
Tarountula. V2 masks only those public-Marnie Belt attachments whose target is
not Spidops, then selects the next frozen neural logit. It also retains the
existing early-Spidops and lethal damage-tool guards.

This is a bounded public-observation exploitation overlay on the frozen neural
blueprint. It is not hidden-hand access, an equilibrium claim, or the project's
final game-theoretic planner.

## Frozen identities

Candidate directory:
`work/target80_candidates/mimikyu_maxbelt_preserve_v2`

- deck: `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`
- main: `d755e8c9cbe4414af6bb85944be48b0a8e60ece36e32a0c6620b3c875efe06d9`
- model: `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`

Deck-identical neural control directory:
`work/target80_candidates/mimikyu_maxbelt_neural_control_v1`

- deck: `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`
- main: `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`
- model: `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`

Candidate deck requirements: exactly 60 cards; zero Wobbuffet (432); one
Mimikyu (434); one Maximum Belt (1158); zero Hero's Cape (1159); and one Brave
Bangle (1175).

No candidate or control file may change during this protocol. Any edit creates
a new identity and requires a new protocol.

## Opponents

Ash Marnie FullBest:

- deck: `92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d`
- main: `0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b`
- model: `98072b2a75197ad9d073af3c7ba5ff66a6daadd266b86b5463fef2ff34910757`

Marnie Recovery V4:

- deck: `8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b`
- main: `d077ff28d8a267bf3f2c28fb26e0ae798dedeac5c1d9274627cc8c9e45243533`
- model: `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c`

## Stage A — locked Marnie confirmation

Launch all four official-engine arms before reading any outcome:

| Policy | Opponent | Games |
|---|---|---:|
| Candidate | Ash FullBest | 40 |
| Control | Ash FullBest | 40 |
| Candidate | Recovery V4 | 40 |
| Control | Recovery V4 | 40 |

Every arm alternates seats and captures decision traces. Development games and
the one-game smoke are excluded.

Stage A advances only if all gates pass:

1. 160/160 normal terminal records, zero draws, engine errors, agent
   exceptions, or illegal fallbacks.
2. Candidate at least 60/80 combined (75%).
3. Candidate at least 28/40 (70%) against each opponent.
4. Candidate at least ten wins ahead of the control over the 80-game combined
   comparison and not worse than control in either opponent cell.
5. Candidate at least 28/40 from each seat across the two opponents.
6. Maximum Belt is attached to Spidops in at least 20 candidate games and to a
   non-Spidops target in zero candidate games after a Marnie core marker is
   public.
7. Frozen hashes and deck integrity reproduce exactly after the panel.

Failure stops this identity. Do not run Stage B, package, or submit it.

## Stage B — broad non-regression, only after Stage A passes

Run candidate and control for 20 alternating-seat games each against Starmie
CF13, Violet Alakazam V93, and the frozen live V5 Rocket policy: 120 additional
official-engine games.

Advance only if all games terminate normally, candidate trails control by no
more than three wins over the combined 60 games, and no opponent cell is more
than two wins worse than control. The Marnie-specific mask must never activate
without a unique public Marnie core marker.

## Submission and persistent target

Passing both stages permits packaging and one immutable internal-arena
submission. It does not prove the persistent goal. Completion requires that
single submitted identity to record at least 160 wins in its own first 200
rated arena games; draws are non-wins and identities may not be pooled.

## Excluded development evidence

In an exploratory 12-game Ash cell, the candidate went 9-3 while its
deck-identical control went 3-9. Candidate damage-tool attachments all targeted
Spidops; games with Maximum Belt on Spidops went 5-1. The sample motivated this
protocol but is not confirmation evidence.

