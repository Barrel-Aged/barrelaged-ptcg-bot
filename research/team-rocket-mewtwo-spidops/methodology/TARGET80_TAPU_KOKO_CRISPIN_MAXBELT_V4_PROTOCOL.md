# Target-80 Tapu Koko / Crispin / Maximum Belt V4 protocol

Status: **completed; exact V4 route rejected**  
Locked: 2026-08-05 (America/Denver)

## Hypothesis

Mega Starmie ex is Lightning-weak and has 330 HP. Tapu Koko ex's Linked
Lightning does 60 damage plus 20 for each own Benched Pokemon. With at least
three Benched Pokemon and Maximum Belt, the official weakness calculation
reaches at least 340 damage and can take a three-Prize knockout in one attack.

V4 is a coherent zero-Wobbuffet package rather than a one-card swap. It uses
four Crispin and a 4 Grass / 4 Lightning / 4 Rocket Energy split to ready Tapu
Koko, two Team Rocket's Petrel plus four Team Rocket's Transceiver to find the
single legal Maximum Belt, and Poke Pad / Ultra Ball / Night Stretcher routing
to maintain the bench width Linked Lightning requires.

This deterministic public-state route is a mechanism probe and prospective
trajectory teacher. It is not eligible to become the final strategic policy.

## Development history excluded from admission

All V1, V1R, V2, V2R, V3, and V4 smoke games are open implementation data and
are excluded from every gate. The corrected V2R prompt replay proved that the
official Crispin sequence is context 7 (Energy to hand), context 22 (Energy to
attach), and context 21 (Pokemon target). V3 established a ready Koko, Belt,
and Linked Lightning in 4/4 open games but went 2-2. V4's one-game smoke won;
its separate four-game smoke went 1-3. The collector's two-game smoke was also
excluded. None of these games may be pooled with the locked panel.

## Frozen arms

Both arms use the exact same 60 cards and exact-deck model checkpoint:

- deck SHA-256:
  `0a8a93a59ad69801482da5a45d85493e12f4f2f3f6bfd001c6e119ca7cad70dc`;
- model SHA-256:
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`;
- Wobbuffet (`432`) count: zero;
- key counts: 2 Tapu Koko ex, 4 Crispin, 2 Team Rocket's Petrel,
  4 Team Rocket's Transceiver, 1 Maximum Belt, 4 Grass Energy,
  4 Lightning Energy, and 4 Team Rocket's Energy.

Control: `work/target80_candidates/tapu_koko_crispin_maxbelt_deck_control_v4`

- semantic tree:
  `88e433beef6d733dfc9b31e9dbe4b4d74368cf87bc87a33a20d4d42a435fd407`;
- main:
  `3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341`.

Candidate: `work/target80_candidates/tapu_koko_crispin_maxbelt_route_v4`

- semantic tree:
  `df106c3eb4df80a0e2a52ff6848c1f05bbf965cd24d0d6f03cb858f8e4ace046`;
- main:
  `99ac929e46b2df439f57b71dc65d8edfe564e46c49fe7392f0deb09891c165cb`.

## Frozen opponents and collector

| Opponent | Semantic tree | Deck |
|---|---|---|
| `SparkStarmieMistFlatV20_20260801_extracted` | `25690bea08bc73dcd93207b1e3ed77428ed09f1a15d6137251f2e06b6bd7dd79` | `64ef6e2c0f5fadc12f96426aa80a07e1402501ed00d1778a0badd25d171e4901` |
| `SparkStarmieFroslassCFV13_extracted` | `c693adec6062422241f2b52caf6beba6518913c2c1ed66581b369eba15b92f76` | `60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61` |

Collector: `scripts/collect_target80_deck_matched_bc_v1_panel.py`

- SHA-256:
  `cab2ad79dae7a58b2f1e7458025fe3d60344410193f74339e68944bf9fff6b26`;
- evaluator SHA-256:
  `d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303`;
- genuine official `cg` legality and transitions;
- both packaged runtimes execute in isolated official-evaluator workers;
- candidate is always engine seat 0, matching the internal-arena convention;
- randomized interleaving of eight candidate and eight control games per
  opponent;
- no hidden opponent hand, visualizer truth, or realized hidden allocation;
- no output or manifest may exist before launch; and
- neither result file may be opened, summarized, or inspected until both
  collectors have terminated.

Locked outputs and seeds:

- MistFlat: seed `806511`,
  `work/target80_tapu_koko_v4_locked/mist_route_vs_control.jsonl`;
- Froslass: seed `806512`,
  `work/target80_tapu_koko_v4_locked/fros_route_vs_control.jsonl`.

## Admission gates

All gates are required:

1. Exactly 32 terminal games, eight per arm/opponent cell, unchanged semantic
   identities, and zero engine, worker, legality, timeout, or fallback errors.
2. Candidate at least 12 wins in 16 games (75%).
3. Candidate at least four more wins than the deck-identical control.
4. Candidate at least six wins in each eight-game opponent family.
5. Candidate Koko seen in at least 14 games, ready in at least 12, Active in at
   least 12, and using Linked Lightning in at least 12.
6. Maximum Belt observed on Koko in at least 10 games, a Belted Linked
   Lightning used in at least eight games, and at least six telemetry-confirmed
   full-health Mega Starmie lethal attacks.
7. At least 20 Linked Lightning attacks, at least 14 with three or more Benched
   Pokemon, and mean first Linked Lightning no later than global turn 5 among
   games where the attack occurs.

Failure closes this exact V4 deck/model/wrapper and the current Tapu Koko /
Crispin / Petrel package. It forbids packaging, arena submission, and learning
from these wrapper trajectories. Passing every gate authorizes only a
separately frozen exact-deck learned-policy experiment and a later broad,
side-swapped regression panel; it does not authorize deployment by itself.

## Claim boundary

Games, legality, actions, transitions, prizes, and terminal outcomes are
genuine official-engine execution. The named route, its damage telemetry, and
the admission thresholds are research abstractions. V4 is not CFR,
public-belief search, safe re-solving, exploitability evidence, or an arena
strength claim.

## Result

Both sealed collectors terminated before either output was analyzed. All 32
games used the genuine official engine, exact frozen bundle identities, the
locked first-seat convention, and strict packaged-worker legality. There were
zero engine, worker, legality, timeout, or fallback errors.

| Opponent | Candidate | Deck-identical control |
|---|---:|---:|
| MistFlat Starmie | 1-7 | 2-6 |
| Froslass Starmie | 4-4 | 2-5-1 |
| Total | **5-11** | **4-11-1** |

The route genuinely executed, but not reliably or strongly enough:

- Tapu Koko appeared and became Active in 16/16 candidate games;
- it became ready in 11/16 and used Linked Lightning in 11/16;
- Maximum Belt was observed on Koko in 10/16, but a Belted Linked Lightning
  occurred in only 6/16;
- only four full-health Mega Starmie lethal attacks were recorded;
- Linked Lightning was used 32 times, including 25 attacks with at least three
  Benched Pokemon; and
- mean first Linked Lightning was global turn 4.64 across the 11 attack games.

Thus integrity passed; the 32-attack, bench-width, Koko-seen/Active, Belt-seen,
and mean-first-attack metrics also passed. Every decisive strength gate failed:
5/16 wins rather than 12/16, only +1 win over control rather than +4, 1/8 and
4/8 family rows rather than 6/8 each, 11 rather than 12 ready/attack games, six
rather than eight Belted attack games, and four rather than six full-health
lethals.

The exact V4 deck/model/wrapper and the current Tapu Koko / Crispin / Petrel
package are closed. These wrapper trajectories are not authorized as learning
supervision. No broad panel, training run, package, arena submission, Kaggle
submission, or strength claim was created from V4.

Evidence:

- Mist output SHA-256:
  `9624bc85bba73d5bf19137caf6c926743c27f62114088f5b0dd15084c3852fc9`;
- Froslass output SHA-256:
  `03f65a586b92d03764dafb2b20391e63a456cf4b5ec92ea521566003b7b89170`;
- Mist manifest SHA-256:
  `28cd19d621c204444ac1b4084659cc61cc6f0e68fc5d520de7c290f9dcd4f4be`;
- Froslass manifest SHA-256:
  `5b238b8da1da4659da5a0ef3b65d61dc7288341f14a26eb5bb0d497481256d7d`.
