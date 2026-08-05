# Target-80 NAIC Watchtower Proxy Neural V1 protocol

Frozen: 2026-08-05, before any candidate game outcome

Status: **prospective Wobbuffet-free deck-substrate screen; no submission authority**

## Hypothesis and evidence boundary

The supplied Juho Kallama NAIC 2026 list is the strongest coherent competitive
deck-level alternative preserved in this project, but its two Prism Tower CRI
80 cards do not exist in the current official TEF-POR engine pool. This screen
therefore uses the project's long-standing explicit compatibility
substitution: two Team Rocket's Watchtower for two Prism Tower. It is a
**Watchtower proxy**, not the exact tournament deck.

Unlike the rejected one-card and attacker-only swaps, every available slot is
moved together: the tournament Pokemon package, 11-Energy package, Ultra Ball,
Night Stretcher, Lucky Helmet, Watchtower/Factory stadium mix, Energy Switch,
and Maximum Belt. Wobbuffet, Murkrow, Poke Pad, Energy Search, Hero's Cape,
Brave Bangle, and Team Rocket's Archer are absent. The frozen V5 neural policy
is deliberately left unchanged so this first experiment isolates whether the
coherent deck substrate is strong enough to justify new policy learning.

## Frozen identity

Candidate:
`work/target80_candidates/naic_watchtower_proxy_neural_v1/`

Requirements:

- exactly 60 legal engine card IDs;
- zero Wobbuffet (`432`) and zero Murkrow (`463`);
- two Mimikyu (`434`) and one Lillie's Clefairy ex (`272`);
- two Watchtower (`1256`), one Factory (`1257`), one Maximum Belt (`1158`);
- exact copied V5 model and neural entry point; and
- no matchup override, sampled rollout, learned-value resolver, or hidden-zone
  feature.

Comparator: immutable live V5 candidate directory
`work/target80_candidates/earlypad_bangle_lethal_v1/`.

Record SHA-256 hashes and pass a terminal official-engine smoke before the
sealed panel. Any candidate code or deck edit after the hash lock creates a new
hypothesis.

## Sealed development panel

Launch all six arms before inspecting outcomes, with default side swapping and
decision traces:

| Arm | Opponent | Games |
|---|---|---:|
| proxy | `AshV178MarnieFullBest` | 12 |
| V5 | `AshV178MarnieFullBest` | 12 |
| proxy | `SparkStarmieFroslassCFV13` | 8 |
| V5 | `SparkStarmieFroslassCFV13` | 8 |
| proxy | `VioletAlakazamMKV93` | 8 |
| V5 | `VioletAlakazamMKV93` | 8 |

All 56 records must terminate normally. Draws are non-wins. Advance the deck
substrate only if the proxy:

1. wins at least 20/28 games;
2. wins at least five more games than V5 overall;
3. wins at least 8/12 against Marnie and at least 5/8 in each other row;
4. is not worse than V5 in any opponent row; and
5. has at least 10/14 wins from each seat across the panel.

Failure closes this exact Watchtower proxy under the frozen V5 policy and
stops before counterfactual data generation. Passing would authorize a new
paired-intervention training protocol and untouched confirmation, not
packaging or submission.

No Kaggle submission is authorized. Arena completion still requires a single
new immutable identity to earn at least 160 wins in its own first 200 rated
games; identities and development games are never pooled.
