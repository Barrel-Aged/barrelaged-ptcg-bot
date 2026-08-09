# Target-80 Mimikyu Marnie Spine V1 result

Status: **closed; failed the frozen development gate**  
Evaluated: 2026-08-04

## Locked result

All three 20-game arms ran concurrently against the exact archived
`AshV178MarnieFullBest` package. No outcome was inspected until all three
files had sealed, and the earlier one-game smoke stayed excluded.

| Arm | Seat 0 | Seat 1 | Total | Terminal |
|---|---:|---:|---:|---:|
| untouched Wobbuffet V5 | 7-3 | 2-8 | 9-11 | 20/20 |
| Wobbuffet-free Mimikyu control | 6-4 | 4-6 | 10-10 | 20/20 |
| Mimikyu Marnie Spine V1 | 2-8 | 1-9 | 3-17 | 20/20 |

The candidate required at least 16/20, at least four more wins than both
controls, and at least 7/10 from each seat. It failed every strength gate and
was seven wins worse than the same-deck neural control.

## Mechanism audit

This was not an inactive or misrouted test:

- a unique public Marnie identifier was visible in 20/20 games;
- replaying all candidate observations through the frozen same-deck control
  produced 1,110 non-forced comparisons;
- the specialist differed from the neural control 688 times across 20/20
  games; and
- all games ended normally without agent or legality failures.

The older deterministic tactical spine is therefore an unsafe replacement for
the current neural blueprint in this matchup, despite encoding the intended
single-Prize Spidops plan. This exact deck/router/specialist combination is
permanently closed and must not be renamed, packaged, submitted, or reused as
a deployment branch.

## Residual information

The pure Wobbuffet-to-Mimikyu deck control was directionally useful in this
fresh cell: it scored 10-10 versus V5's 9-11 and improved the second-seat row
from 2-8 to 4-6, while losing one first-seat win. This does not reopen the
previously rejected direct-swap package or authorize submission, but it
supports retaining Mimikyu as the lowest-risk Wobbuffet-free baseline for a
future learned or tightly bounded intervention.

## Evidence

- Protocol: `TARGET80_MIMIKYU_MARNIE_SPINE_V1_PROTOCOL.md`
- Sealed panel: `work/target80_mimikyu_spine_development_v1/`
- Candidate deck SHA-256:
  `dc1b1d8709187aec04f6b89532bf9121ec7e48ec4cec2aca8ce98de02c6eccde`
- Candidate main SHA-256:
  `a0721517a3d286b4a0a89a8c973c38ee49b76d6f2e0982bcee914cb728460ea5`

No Kaggle or internal-arena submission was made from this result.
