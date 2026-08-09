# N15 Prospective Terminal Consolidation Result

Status: **complete; offline pass, development-panel failure, no confirmation or
deployment authority**  
Protocol: `rocket-n15-prospective-terminal-consolidation/v1`  
Completed: 2026-08-01

## Outcome

N15 exactly reproduced the full N14 optimizer trajectory and saved its sole
preregistered step-2,000 terminal model. That model passed every offline gate
and completed the locked 960-game official-engine development panel. It showed
a useful positive signal against V110, but failed the mandatory RecoveryV4 and
uncertainty gates. It does not advance.

The live bot and exact Team Rocket deck remain unchanged. N15 supplies no
deployment authority and no verified strength-improvement claim.

## Exact replay and offline admission

At N14 steps 62, 90, 149, and 286, all 63 model state tensors in the N15 replay
were exactly equal to their archived N14 checkpoint tensors. Correction counts
also exactly reproduced 12, 24, 37, and 48. This establishes that the terminal
state came from the frozen N14 path rather than a modified rerun.

The sole terminal model has SHA-256
`354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7`.
Its offline results were:

| Gate | Result | Requirement |
|---|---:|---:|
| strict counterfactual corrections | 48/48 | 48/48 |
| N6R non-target top-1 retention | 99.831% | at least 98.0% |
| N6R mean forward KL | 0.010005 | at most 0.05 |
| N9 overall top-1 retention | 99.126% | at least 99.0% |
| N9 V110 top-1 retention | 99.045% | at least 98.5% |
| N9 Recovery top-1 retention | 99.211% | at least 98.5% |
| N9 mean forward KL | 0.003970 | at most 0.02 |
| cached/full-runtime mismatch | 0/7,392 | zero |
| serialization mismatch | 0 tensors | zero |

Only `logit_adapter.*` tensors changed from the zero-residual substrate. The
training report SHA-256 is
`b508381af55d3f86654ab527622bec333696fa7ab37b05fb06f5b71933d92ce1`.

## Packaged-path audit

The two-arm bundle retained the exact deck and residual-capable official
adapter. Across 16 official-engine audit games, the strict in-process actor and
packaged `NNAgent` agreed on 1,049/1,049 Rocket decisions with zero mismatches.
The parity report SHA-256 is
`95f6d3f74883baf73395a2504772ac014607659bf5a9fe0a9060b37b57cf8ca8`.

This verifies execution equivalence, not playing strength.

## Locked official-engine development panel

All four shards sealed before outcomes were opened. Each opponent received 240
control and 240 candidate games with Rocket in seat 1, for 960 games total.
Correctness passed: exact provenance, complete counts, official results, second
seat in every game, and zero exception fallbacks.

| Opponent | Control | Candidate | Difference |
|---|---:|---:|---:|
| V110 | 94/240 (39.167%) | 109/240 (45.417%) | **+6.250 pp** |
| RecoveryV4 | 92/240 (38.333%) | 91/240 (37.917%) | **-0.417 pp** |
| Equal-opponent aggregate | -- | -- | **+2.917 pp** |

The stratified 95% bootstrap interval was **[-3.333, +9.172] percentage
points**. The analysis report SHA-256 is
`cd1ae652e72de635898638b4c4a07b1735e563a80f1296de04f82ff4a6d01631`.

Advancement required at least +2.0 aggregate points, strictly positive
improvement against both opponents, and a bootstrap lower bound above -0.5
points. N15 met only the aggregate point-estimate gate. It failed the
RecoveryV4 and interval gates, so no fresh confirmation is authorized.

## Interpretation and next admissible hypothesis

N15 is the clearest positive local signal after N11, but it is not conclusive.
Its V110 gain was 15 wins while its Recovery difference was only one loss; the
panel therefore suggests a useful intervention mixed with residual uncertainty
or a small amount of non-target behavior. Repeating or extending this selected
panel would violate the lock.

A distinct successor can be defined without using match outcomes to choose an
arm: hold the trained residual direction fixed and analytically choose the
smallest global adapter scale that preserves all 48 teacher-best actions at a
preregistered positive margin. This projection uses only the frozen
counterfactual and preservation supports. It should minimize incidental policy
changes while retaining the entire targeted correction set. It needs a new
protocol, immutable model, unchanged offline gates, new official seeds, and a
fresh development panel. N15 itself is closed.
