# Target-80 NAIC Watchtower Proxy Neural V1 result

Completed: 2026-08-05

Status: **rejected; exact proxy closed; no training, package, or submission**

## Execution integrity

The candidate retained every hash in
`TARGET80_NAIC_WATCHTOWER_PROXY_NEURAL_V1_BUILD_LOCK.md`. Its one-game smoke
ended normally. All six panel arms were launched before outcomes were opened.

The initial proxy/Alakazam arm was interrupted at 6/8 by the outer 600-second
orchestration limit. Per
`TARGET80_NAIC_WATCHTOWER_PROXY_V1_EXECUTION_AMENDMENT.md`, those six rows are
excluded, retained only as operational diagnostics, and the exact arm was
rerun from game 1. The isolated eight-game rerun completed normally. No result
was inspected before the recovery rule and replacement arm were frozen.

## Eligible locked result

| Opponent | Watchtower proxy | V5 control | Difference |
|---|---:|---:|---:|
| AshV178MarnieFullBest | 4-8 | 3-9 | +1 win |
| SparkStarmieFroslassCFV13 | 1-7 | 2-6 | -1 win |
| VioletAlakazamMKV93 | 6-2 | 8-0* | -2 wins |
| Total | **11-17 (39.29%)** | **13-15 (46.43%)** | **-2 wins** |

Candidate seat split was 7/14 from seat 0 and 4/14 from seat 1.

`*` One V5/Alakazam row ended as a V5 forfeit win because the Alakazam worker
closed stdout. The other 55 eligible rows ended by the official terminal
condition. This operational error independently fails the frozen correctness
gate. Treating that row as unavailable would reduce control to 12 valid wins;
it would not rescue the candidate's 11/28 absolute result or any row/seat gate.

## Gate decision

- at least 20/28 candidate wins: failed with 11;
- at least five wins above V5: failed, trailing by two as recorded and by at
  least one after excluding the control forfeit;
- row floors of 8/12 Marnie and 5/8 each other row: failed on Marnie and
  Starmie;
- no row worse than V5: failed on Starmie and Alakazam; and
- at least 10/14 wins from each seat: failed from both seats.

## Interpretation and closure

The coherent tournament-style card package is not a viable substrate for the
frozen V5 policy in this engine. Its two available-card Watchtower substitutes
are not Prism Tower, and its Ultra Ball/11-Energy/Lucky Helmet engine is far
outside the action distribution on which the policy was trained. The 6-2
Alakazam row shows matchup value, but the 1-7 Starmie collapse and severe
second-seat weakness preclude counterfactual training or submission.

Do not rerun this exact proxy/model combination, reinterpret it as the exact
NAIC deck, train on the interrupted prefix, package it, or submit it. The
result does not prove the real Prism Tower list weak; that exact deck remains
unrepresentable in the current engine.

Eligible evidence: `work/target80_naic_watchtower_proxy_v1_dev/`, using
`proxy_alakazam_8_rerun.jsonl` and excluding the interrupted
`proxy_alakazam_8.jsonl` prefix.

No Kaggle or internal-arena submission was made.
