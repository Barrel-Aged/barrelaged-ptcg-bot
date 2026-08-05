# Target-80 Maximum Belt Mewtwo/Starmie V1 result

Status: **closed after the locked focused panel; no submission**  
Recorded: 2026-08-05

## Result

The Maximum Belt candidate produced the preregistered eight-win relative
improvement, but remained far below every absolute-strength gate:

| Arm | W-L-D | Win rate | Starts first | Starts second |
|---|---:|---:|---:|---:|
| Maximum Belt candidate | 21-59-0 | 26.25% | 16/40 | 5/40 |
| Hero's Cape control | 13-67-0 | 16.25% | 11/40 | 2/40 |

The candidate gained exactly eight wins (`+10.00` percentage points), but
missed the locked `56/80` absolute gate by 35 wins and missed both `24/40`
start-order gates. Starting second remained the largest failure.

## Matchups and mechanism

| Opponent | Candidate W/40 | Control W/40 |
|---|---:|---:|
| Froslass Starmie | 10 | 8 |
| MistFlat Starmie | 11 | 5 |

The candidate recorded 69 Mewtwo Erasure Ball attacks against Mega Starmie,
12 Belt-aware Starmie discard prompts, and 17 Energy cards discarded in those
prompts. The exact Belt lethal attachment route occurred in only two games,
below the required four. The control recorded 71 corresponding Erasure Ball
attacks and no Belt mechanism events.

This establishes that the Cape-to-Belt package can help this weak cell, but it
does not establish that the narrow lethal guard caused all eight additional
wins. The mechanism was too rare and the absolute performance too weak for
that claim.

## Integrity

All four locked rows completed before outcomes were opened: 160 genuine
bundled official-engine terminal games, 80 per arm, both opponents, and both
fixed start orders. There were zero illegal actions, exception fallbacks, or
nonterminal rows. Both decks contained zero Wobbuffet.

Locked JSONL files:

- `work/target80_maxbelt_mewtwo_starmie_v1/froslass_first.jsonl` —
  `f1788b180a96749a40490ef488f741be504873148a94ebb0bc613c6af89887e1`
- `work/target80_maxbelt_mewtwo_starmie_v1/froslass_second.jsonl` —
  `9820d11e5b8ba54ca5a76e588989c2df086ed0c55a6cf0fe8cea42093c592fec`
- `work/target80_maxbelt_mewtwo_starmie_v1/mist_first.jsonl` —
  `b8b8035a167603d640e16b8954d88dbb84758918f54f996e9166a1c78cd91ad8`
- `work/target80_maxbelt_mewtwo_starmie_v1/mist_second.jsonl` —
  `89fbdae3b012719b27db4ba1f4a8b9ec699cb37fefdf7c10e90943133e99aa08`

## Decision

Close this exact Cape-to-Belt plus narrow Mewtwo lethal hypothesis. Do not run
the broad panel, package it, submit it, or relax the gates. Maximum Belt may be
reconsidered only inside a materially rebuilt, exact-deck-trained policy; it
is not a validated one-card solution. V5 and V6 remain immutable, and no
Kaggle submission was made.
