# Target-80 Hancang Starmie specialist BC V1 protocol

Status: **completed; exact specialist rejected**  
Locked: 2026-08-05 (America/Denver)

## Hypothesis

The balanced exact-deck elite-BC run improved the Wobbuffet-free Hancang
recovery list from 129 to 136 wins in 210 official first-seat games, but its
selected checkpoint remained only 12/30 against MistFlat Starmie and 11/30
against Froslass Starmie. Its stochastic training batches retained only one to
three Starmie wins per 28 rollouts and then diluted those decisions to equal
mass with five easier opponent families.

V1 tests a materially different learning allocation: keep the exact deck and
selected exact-deck checkpoint, collect near-greedy winning trajectories only
against the two Starmie policies, and train a Starmie specialist. The specialist
is not a general replacement policy. If it passes, a later candidate may route
to it only after definitive Staryu or Mega Starmie public evidence; unseen and
unclassified opponents remain on the frozen blueprint.

This is a learned opponent-exploitation component behind a prospective
baseline-safe public router. It is not hidden-hand modeling, CFR, equilibrium
training, counterfactual-value safety, or the final strategic architecture.

## Frozen Rocket identity

Bundle:
`work/target80_candidates/hancang_elitebc_iter3_control_v1`

- exact 60-card deck SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`;
- zero Wobbuffet;
- frozen exact-deck starting model SHA-256:
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`;
- unchanged runtime `main.py` SHA-256:
  `3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341`.

The list contains 4 Tarountula, 4 Spidops, 2 Mewtwo ex, 2 Articuno,
2 Mimikyu, 8 Grass Energy, 4 Team Rocket's Energy, Poffin, three Bug Catching
Set, two Night Stretcher, four Poke Pad, four Transceiver, Hero's Cape, two
Brave Bangle, and three Team Rocket's Factory.

## Frozen opponents

| Opponent | Deck SHA-256 | Main SHA-256 |
|---|---|---|
| `SparkStarmieMistFlatV20_20260801_extracted` | `64ef6e2c0f5fadc12f96426aa80a07e1402501ed00d1778a0badd25d171e4901` | `cd20fa6a69b70d7da77bb251d13db784a8029b858badc42ab47e00d8ce9f3aed` |
| `SparkStarmieFroslassCFV13_extracted` | `60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61` | `3d570642a9294bca0abea9587786b4a9059a42d1c777c462631e131cde24e87e` |

## Frozen trainer and run

Trainer: `scripts/train_target80_exact_deck_elite_bc_v1.py`  
Trainer SHA-256:
`da7b1e790b7ab129b1ac44e125b6e32f4fd2011e32097f9e7ef176d84d67633f`

Output, which must not exist at launch:
`work/target80_hancang_starmie_specialist_bc_v1_seed806601`

Locked parameters:

- seed `806601`;
- Rocket always engine seat 0, matching the internal-arena convention;
- four iterations;
- 128 near-greedy stochastic rollout games per opponent per iteration;
- temperature `0.60`;
- retain only multi-option decisions from Rocket wins and balance retained
  mass across the two opponent policies;
- train only the action head and final relational-transformer block;
- two epochs per iteration, batch size 96, AdamW learning rate `1e-5`, weight
  decay `1e-5`, label smoothing `0.01`, frozen-reference KL `0.15`, and
  entropy coefficient `0.0005`;
- independently evaluate the start model and every iteration greedily for 60
  games against each opponent; and
- select lexicographically by 120-game wins, then minimum opponent wins.

The exact command is:

```text
python scripts/train_target80_exact_deck_elite_bc_v1.py
  --candidate work/target80_candidates/hancang_elitebc_iter3_control_v1
  --opponent work/arena_bundles/SparkStarmieMistFlatV20_20260801_extracted
  --opponent work/arena_bundles/SparkStarmieFroslassCFV13_extracted
  --output work/target80_hancang_starmie_specialist_bc_v1_seed806601
  --iterations 4
  --rollout-games-per-opponent 128
  --eval-games-per-opponent 60
  --epochs 2
  --batch-size 96
  --learning-rate 0.00001
  --weight-decay 0.00001
  --kl-coef 0.15
  --entropy-coef 0.0005
  --label-smoothing 0.01
  --temperature 0.60
  --seed 806601
  --threads 4
  --timeout 30
  --max-steps 1000
  --seat first
  --train-last-block
  --expected-deck-sha256 0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8
  --expected-model-sha256 b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f
```

## Admission gates

Every gate is required before building a runtime router:

1. All 1,624 official-engine games complete with zero engine, worker,
   legality, timeout, or numerical failure and byte-stable input identities.
2. Exact frozen 60-card deck and zero Wobbuffet.
3. Every iteration retains winning trajectories from both opponent policies.
4. The selected specialist wins at least 90/120 evaluation games (75%).
5. It wins at least 42/60 (70%) against each Starmie policy.
6. It improves by at least 24 wins over the fresh 120-game frozen-start
   baseline.

Failure closes this training configuration without post-hoc checkpoint,
temperature, KL, or threshold selection. Passing authorizes only construction
of a public-evidence router plus a fresh, side-swapped broad development panel.
It does not authorize packaging or arena submission.

## Later deployment boundary

Any routed successor must prove exact baseline parity before Starmie is
publicly identified, exact specialist parity afterward, zero illegal/fallback
actions, and broad non-regression. A separate untouched 200-game official-
engine confirmation must reach at least 160 wins before one arena identity is
submitted. The arena identity must then independently reach at least 160 wins
in its first 200 rated games. No prior identity or local game may be pooled.

The official `cg` engine supplies genuine legality, transitions, outcomes, and
training/evaluation games. Winning-trajectory selection, behavioral cloning,
the public opponent-family router, and every threshold above are research
abstractions. Kaggle submission is out of scope.

## Result

The trainer completed the full locked workload and atomically wrote its report.
PowerShell returned status 1 only because the ordinary PyTorch
`enable_nested_tensor` warning was promoted from native stderr; the Python
trainer itself reached its `complete: true` record. All 1,624 required games,
four updates, five evaluation matrices, checkpoint writes, and the semantic
before/after identity check completed. No engine, worker, legality, timeout, or
numerical error was recorded.

| Arm | MistFlat | Froslass | Total |
|---|---:|---:|---:|
| Frozen start | 23-37 | 11-46-3 | **34-83-3** |
| Iteration 1 / selected | 22-38 | 19-40-1 | **41-78-1** |
| Iteration 2 | 18-42 | 18-38-4 | **36-80-4** |
| Iteration 3 | 17-43 | 19-38-3 | **36-81-3** |
| Iteration 4 | 16-44 | 20-35-5 | **36-79-5** |

All four iterations retained wins from both opponents. Iteration 1 was selected
lexicographically at 41 wins and a 19-win opponent floor. It improved by seven
wins over the fresh baseline, not the required 24, reached 41/120 rather than
90/120, and reached only 22/60 and 19/60 rather than 42/60 in each family.
Gates 1-3 passed; gates 4-6 failed.

The exact specialist, its checkpoints, and a Starmie public router are closed.
No runtime candidate, broad panel, package, internal-arena submission, Kaggle
submission, or strength claim is authorized from this experiment.

Evidence:

- training report SHA-256:
  `f6173839773b91347908ec307e8809c1e5bea71421e5b1743327eb0780fc7541`;
- selected `best_model.pt` SHA-256:
  `95a5c945cc845fc622aba58f446b278e7c1ff3f948ca31e5cd6cc14fb4accdc2`;
- selected iteration-1 SHA-256:
  `eb8acd9dcdd00162e31e1b92f7ad9ae62887fdc80daf5a9a1541342ec76a3533`;
- complete stdout SHA-256:
  `5d90095d5e50b2f1b592176694ac510572c1c1da62b564b5973c47193142f006`;
- warning-only stderr SHA-256:
  `f3714a31810779fc48149a5a99d23854126a227c872deea1d50ae87ff0de6977`.
