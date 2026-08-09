# Target-80 Arbok / Chingling Bridge V2 protocol

Status: **frozen before development outcomes**  
Frozen: 2026-08-04

## Hypothesis

The closed Arbok Lock V1 established a real but seat-asymmetric mechanism:
Arbok became Active before Grimmsnarl ex in 14 games and those games went
10-4, but the lock generally arrived after Marnie's first evolution window
when Rocket played second. V2 does not retune that exact package. It removes
the remaining Articuno, both Murkrow, and one Mewtwo, then adds two Team
Rocket's Chingling and two Switch.

The public Marnie route has two complementary plans:

1. going first, prefer an Ekans opener and establish Potent Glare before the
   first Grimmsnarl ex evolution;
2. going second, prefer zero-retreat Chingling and repeatedly use its
   zero-Energy Chiming Commotion while building the delayed Arbok line.

The bridge is intentionally a single-Prize, all-Team-Rocket package. It keeps
the Spidops line, Mewtwo fallback, V5 neural blueprint, EarlyPad guard, and
lethal Bangle rule. Public board/discard cards identify the Marnie shell; no
opponent private card, arena name, or hidden engine state is used.

This is a bounded matchup experiment, not a claim that a deterministic rule
is the project's final neural/game-theoretic planner.

## Frozen identities

| Artifact | SHA-256 |
|---|---|
| Arbok/Chingling V2 deck | `448bcb4519dfb0a8be06b3ae5bc49e7888ca8bd207f1ed47cfbf2a014ad7cf65` |
| Arbok/Chingling V2 main | `7e9a1c2c7bfba00ae9a9e8d2296b1f7e6ab8d9bf4af0284d57b4c799d8fc94eb` |
| shared frozen neural model | `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b` |
| closed Arbok V1 deck | `7131d009a45586b008fd33ffaa0cbe1107f9e3c0c25ff09116e5cb02b2934749` |
| closed Arbok V1 main | `294e3e42041befc5823452c7d1cb4c09b862cd2982da388eb651397d2dc3facc` |
| untouched V5 deck | `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852` |
| untouched V5 main | `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6` |

The exact archived opponent is `AshV178MarnieFullBest_extracted`. The two
one-game terminal smokes are excluded from every strength total. They proved
only runtime legality and the intended two-seat mechanisms: first-seat Arbok
was Active on turn three before Grimmsnarl became public on turn six; the
second-seat win used 17 Chiming Commotion attacks.

## Locked development panel

Run three automatically side-swapped, fresh 20-game official-engine arms
without reading any outcome until all 60 games seal:

1. untouched Wobbuffet V5 control;
2. closed Arbok Lock V1 mechanism control;
3. frozen Wobbuffet-free Arbok/Chingling Bridge V2.

Every arm must produce 20 terminal rows with no exception, invalid action,
timeout, or step-limit result. V2 advances only if all of these hold:

- at least 16/20 wins overall;
- at least 7/10 wins from each seat;
- at least four more wins than untouched V5;
- at least three more wins than closed Arbok V1;
- Chiming Commotion occurs in at least five of the ten second-seat games;
- Arbok is observed Active before the first public Grimmsnarl ex in at least
  six of the ten first-seat games; and
- the public Marnie signature is visible in all 20 games.

Any failed gate closes this exact deck and policy. Do not rescale or rename it
from development outcomes.

## If admitted

Run a broad, side-swapped health panel against both Starmie variants, Rocket
mirror, Alakazam, and the Archaludon/Cinderace resource-wall family. Then run
an untouched confirmation with new seeds/opponent processes. Package and
submit no more than one successor, and only after exact deck/model/runtime
parity and archive smoke.

Arena completion remains one immutable Wobbuffet-free identity with at least
200 rated games and at least 80% wins in that identity alone. Draws are
non-wins and identities may not be pooled. No Kaggle submission is authorized.

