# ScaleStopV3 2,846-game arena audit

Status: **complete historical audit; not a current-rank completion claim**  
Completed: 2026-08-01

## Identity and boundary

- bot: `HelbrosRocketSpidopsScaleStopV3`;
- frozen qualifying deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- model SHA-256:
  `c6b808e3e50b5fffcfd44b59ad2a541c6661e8d22915b1cbb5ade5a57464d8f2`;
- source: 2,846 pulled official internal-arena JSONL games;
- execution: genuine official-engine games, not abstract rollouts;
- result: 1,635 wins, 1,192 losses, 19 draws, or 57.45% raw wins.

These historical games establish a large strength and failure-analysis sample.
They do not get added to the current post-reset rating row, and they do not by
themselves satisfy the requirement to be current rank 1 after 300 games.

## Matchup result

| Opponent | Games | W-L-D | Raw win rate |
|---|---:|---:|---:|
| Hops Trevenant V257 | 685 | 521-164-0 | 76.06% |
| Arch Energy Wall V11 | 668 | 338-330-0 | 50.60% |
| Crustle Kang ADS V1 | 599 | 384-215-0 | 64.11% |
| Starmie Froslass CFV13 | 685 | 230-436-19 | 33.58% |
| Starmie MistFlat V20 | 14 | 2-12-0 | 14.29% |

The remaining 195 games cover smaller opponent samples. The complete machine
summary is `work/arena_scalestop_reset_latest/summary.json`.

## Strategic failure signal

The combined Starmie slice is 232-448-19 over 699 games (33.19% raw wins).
Among the 448 decisive losses:

- 328 (73.2%) never developed a second ready attacker;
- 176 (39.3%) never attacked with Mewtwo;
- only 29 (6.5%) failed to reach four Team Rocket Pokemon in play;
- zero ended a main phase while a legal attack was available.

Starmie wins averaged 2.02 maximum Spidops in play and 2.62 manual Energy
attachments to Spidops. Losses averaged 1.34 maximum Spidops and 1.61 such
attachments. This is an association, not a causal policy target. It indicates
a board-development and prize-race weakness rather than a legality or
packaging failure.

## Reproducible analysis

`scripts/autopsy_scale_live_losses.py` now supports an exact `--bot`, repeated
`--opponent-contains` filters, and `--summary-only`. Its tests cover draw
retention and custom bot identity. The four frozen filtered reports are under:

- `work/arena_scalestop_reset_latest/autopsy_starmie_v4/`;
- `work/arena_scalestop_reset_latest/autopsy_arch/`;
- `work/arena_scalestop_reset_latest/autopsy_crustle/`;
- `work/arena_scalestop_reset_latest/autopsy_trevenant/`.

No policy mutation or deployment is authorized by this audit.
