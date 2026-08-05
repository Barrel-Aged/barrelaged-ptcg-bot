# Target-80 Mimikyu / Maximum Belt Preserve V2 — Locked Result

Status: **rejected after Stage A; Stage B, packaging, and submission are forbidden**

Date: 2026-08-05 (America/Denver)

This document closes the preregistered protocol in
`TARGET80_MIMIKYU_MAXBELT_PRESERVE_V2_PROTOCOL.md`. All four Stage-A arms were
launched before any result was inspected. Development games and the one-game
official-engine smoke are excluded.

## Frozen identity verification

The candidate still reproduces the preregistered identity exactly:

- `deck.csv`: `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`
- `main.py`: `d755e8c9cbe4414af6bb85944be48b0a8e60ece36e32a0c6620b3c875efe06d9`
- `model.pt`: `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`

`deck.csv` contains exactly 60 cards, including zero Wobbuffet (432), one
Mimikyu (434), one Maximum Belt (1158), zero Hero's Cape (1159), and one Brave
Bangle (1175).

## Official-engine Stage-A result

| Opponent | Preserve V2 | Deck-identical neural control | Delta |
|---|---:|---:|---:|
| Ash Marnie FullBest | 18-22 | 15-25 | +3 wins |
| Marnie Recovery V4 | 18-22 | 18-22 | 0 wins |
| Combined | **36-44 (45.0%)** | **33-47 (41.25%)** | **+3 wins** |

Candidate seat split across both opponents:

- seat 0: 15/40 wins (37.5%)
- seat 1: 21/40 wins (52.5%)

All 160 records ended with the official engine's normal `terminal` reason.
There were zero draws, engine errors, agent exceptions, or observed illegal
fallbacks.

## Maximum Belt mechanism audit

Across the 80 candidate games, Maximum Belt was attached in 49 games:

| Opponent | Belt games | To Spidops | To another Pokémon | Wins in Belt games |
|---|---:|---:|---:|---:|
| Ash Marnie FullBest | 26 | 23 | 3 | 14 |
| Marnie Recovery V4 | 23 | 18 | 5 | 11 |
| Combined | 49 | 41 | 8 | 25 |

The eight non-Spidops attachments all occurred before a unique Marnie core
marker was public, where the preregistered Marnie-specific mask was deliberately
inactive. After a Marnie core marker was public, all 39 Belt attachments targeted
Spidops and zero targeted another Pokémon. The intervention therefore passed
its scoped mechanism check.

## Gate decision

| Preregistered Stage-A gate | Required | Observed | Decision |
|---|---:|---:|---|
| Normal terminal records | 160/160, zero draws/errors | 160/160 | Pass |
| Candidate combined wins | at least 60/80 | 36/80 | **Fail** |
| Candidate wins per opponent | at least 28/40 each | 18/40, 18/40 | **Fail** |
| Candidate advantage over control | at least +10, no cell worse | +3, no cell worse | **Fail** |
| Candidate wins per seat | at least 28/40 each | 15/40, 21/40 | **Fail** |
| Public-Marnie Belt routing | at least 20 Spidops games; zero other targets | 39 Spidops; 0 other | Pass |
| Frozen hashes and deck | exact | exact | Pass |

The small exploratory 9-3 versus 3-9 Ash result did not reproduce. It was seed
noise. Preserve V2 produced a bounded tactical improvement over its
deck-identical control, concentrated in the Ash second-seat cell (11/20 versus
6/20), but that gain was far below both the admission threshold and the
persistent 80% arena objective.

## Required disposition

Per the locked protocol, this identity is closed. Do not run Stage B, package
it, submit it to the internal arena, or combine its games with another identity.
The live V5 bot remains unchanged.

