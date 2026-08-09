# Internal-arena recovery winner

Date: 2026-07-28  
Status: objective met; exact active rank-1 archive frozen and audited

## Outcome

`HelbrosMarnieGeneralRecoveryV4` is the current internal-arena leader. It
crossed rank 1, remained rank 1 through fresh rated games, and then widened its
lead:

| Snapshot | Record | Rating | Rank | Margin over second |
|---|---:|---:|---:|---:|
| First crossing | 31-8-0 | 1064.7 | 1 | +1.8 |
| First archive audit | 32-9-0 | 1063.7 | 1 | +0.8 |
| Frozen final audit | 36-9-0 | 1091.2 | 1 | +28.3 |

At the frozen audit, second place was `AshMarnieGeneralV110` at 1062.9. V4
was active in the live 47-bot roster. This was not declared from the earlier
transient V1 snapshots; V4 survived additional rated blocks before completion.

## What won

V4 is a conservative neural-policy composition built from the strongest
published team package and the independent residual checkpoints available in
the team corpus:

1. The frozen Ash V151 Marnie's Grimmsnarl transformer-style policy supplies
   the tactical base.
2. Seed-1 and seed-2 public-history recurrent residuals run in parallel.
3. Their output logits are averaged for the validated ensemble branch.
4. A public-information matchup router chooses the ensemble for Marnie and
   Alakazam, and the proven seed-1 policy for Rocket Spidops and unknown
   archetypes.
5. Both recurrent memories consume every observation before routing, so a
   later public reveal never switches to stale state.
6. The existing autoregressive multi-select behavior and attack-safe legality
   guard are preserved.

The router reads only card IDs already visible in the opponent's public active,
bench, discard, and stadium zones. It does not inspect hidden hands, prizes,
opponent names, private deck lists, or simulator-private state.

## Why the route was selected

All comparisons below used the packaged official engine and ended normally:

| Comparison | Result |
|---|---:|
| Two-seed ensemble vs published V151 | 56-44 |
| Seed-2 alone vs published V151 | 46-54 |
| Weight soup vs published V151 | 44-56 |
| Ensemble vs Alakazam | 66-14 |
| Seed-1 vs Alakazam | 64-16 |
| Ensemble vs Rocket Spidops | 47-33 |
| Seed-1 vs Rocket Spidops | 57-23 |

The results expose a nontransitive policy cycle rather than a universally best
checkpoint. Routing preserves seed-1's ten-win advantage in the 80-game
Spidops control, takes the ensemble's smaller Alakazam edge, and uses the
ensemble's 56-44 mirror advantage. A direct V4 smoke gauntlet added 30/30
normal terminal games with zero runtime or legality failures.

## Exact winning artifact

- Archive: `artifacts/submissions/HelbrosMarnieGeneralRecoveryV4.tar.gz`
- Bytes: `18,820,579`
- SHA-256: `8798eca1ff3bfea1c7e656c1225275229476f9e1d69e59957a448b3dc35f85c5`
- Members: 26 regular files
- Safety: no links, duplicate members, traversal paths, or unexpected members
- Runtime: packaged deck returned exactly; official validator returned `ok`

Key component hashes:

| Component | SHA-256 |
|---|---|
| `model.pt` | `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c` |
| `history_residual.pt` | `25ee55a36d35d771997de61e61f71395f882178668317b8b5ffee4ccd4ff181c` |
| `history_residual_seed2.pt` | `9947415b659ad6b43c42cbd1b7142e5249b2955b388c80fd0c01f2942cb0fe4a` |
| `soldier/history_residual_ensemble.py` | `2c3f04bf0e2e09d7aae9ed3f4112c39813f90c399a931a1f4a3762e40d720396` |
| `soldier/public_matchup_router.py` | `c9d1433888b279a9534e87826e49b83cbd046ece44a208d60be1d64b98151e64` |
| `main.py` | `d077ff28d8a267bf3f2c28fb26e0ae798dedeac5c1d9274627cc8c9e45243533` |
| `deck.csv` | `8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b` |

The machine-readable audit is
`artifacts/recovery/WINNING_ARCHIVE_AUDIT.json`; every gate is true, including
active-roster status and live rank 1.

## Winning deck

Pokemon (18): 4 Marnie's Impidimp, 3 Marnie's Morgrem, 3 Marnie's Grimmsnarl
ex, 4 Munkidori, 2 Snorunt, 2 Froslass.

Trainers (32): 4 Buddy-Buddy Poffin, 4 Poke Pad, 4 Lillie's Determination,
4 Team Rocket's Petrel, 4 Spikemuth Gym, 3 Rare Candy, 3 Night Stretcher,
2 Boss's Orders, 2 Handheld Fan, 1 Unfair Stamp, 1 Dawn.

Energy (10): 10 Basic Darkness Energy.

## Scope boundary

This is not a claim that the original Team Rocket's Mewtwo ex / Spidops deck
became strong. Its two submitted policies remained near the bottom of the
arena, so the authorized recovery required a deck-and-policy pivot. V4 is also
not the completed six-phase belief-aware/game-theoretic research architecture;
it is a strong, legal arena policy assembled from published neural components
with a narrow public-state routing improvement.

No Kaggle submission was made and no hidden-information or private-engine
shortcut is used. The original Mewtwo research scaffold, phase plan,
official-engine adapter, telemetry, replays, and future game-theoretic planning
interfaces remain preserved separately.
