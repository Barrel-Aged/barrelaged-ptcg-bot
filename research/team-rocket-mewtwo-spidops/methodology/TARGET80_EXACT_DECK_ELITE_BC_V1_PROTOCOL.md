# Target-80 exact-deck elite BC V1 protocol

Status at preregistration: **locked; no experiment outcome inspected**

Final status: **infrastructure failure; no checkpoint admitted**

Preregistered: 2026-08-05 (America/Denver)

## Purpose

Test whether conservative, exact-deck learning can turn the strongest known
zero-Wobbuffet structural shell into a materially stronger first-seat policy.
The hypothesis is that successful stochastic lines from this exact deck teach
deck-specific setup, second-attacker readiness, Mewtwo conversion, and prize
sequencing that the transferred Scale policy does not represent well.

This experiment is a development gate. It does not authorize an internal-arena
submission, does not use Kaggle as a write target, and does not claim to be a
game-theoretic solver.

## Exact Rocket deck (60 cards)

- Energy: 8 Basic Grass Energy; 4 Team Rocket's Energy.
- Pokemon: 4 Team Rocket's Tarountula; 4 Team Rocket's Spidops; 2 Team
  Rocket's Articuno; 2 Team Rocket's Mewtwo ex; 2 Team Rocket's Mimikyu.
- Trainers: 1 Buddy-Buddy Poffin; 3 Bug Catching Set; 2 Night Stretcher; 1
  Ultra Ball; 4 Team Rocket's Transceiver; 4 Poke Pad; 1 Hero's Cape; 2 Brave
  Bangle; 4 Team Rocket's Ariana; 2 Team Rocket's Giovanni; 1 Team Rocket's
  Petrel; 2 Team Rocket's Proton; 4 Lillie's Determination; 3 Team Rocket's
  Factory.

The hard constraint is zero copies of Team Rocket's Wobbuffet (engine card ID
432). No deck mutation is permitted inside this experiment.

## Frozen candidate identity

Bundle:
`work/target80_candidates/hancang_wobbfree_scale_control_v1`

- `deck.csv` SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`
- `model.pt` SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`

The starting model remains frozen as the KL reference. Only the candidate
copy in the new experiment output may be updated.

## Locked opponent panel

All games place the Rocket candidate in seat 0, matching the current internal
arena convention. Opponents execute as isolated workers through the official
evaluator.

| Opponent bundle | `deck.csv` SHA-256 | Primary runtime/model SHA-256 |
|---|---|---|
| `AshV178MarnieFullBest_20260801_extracted` | `92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d` | `model.pt` `98072b2a75197ad9d073af3c7ba5ff66a6daadd266b86b5463fef2ff34910757` |
| `SparkStarmieMistFlatV20_20260801_extracted` | `64ef6e2c0f5fadc12f96426aa80a07e1402501ed00d1778a0badd25d171e4901` | `main.py` `cd20fa6a69b70d7da77bb251d13db784a8029b858badc42ab47e00d8ce9f3aed` |
| `SparkStarmieFroslassCFV13_extracted` | `60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61` | `main.py` `3d570642a9294bca0abea9587786b4a9059a42d1c777c462631e131cde24e87e`; counterfactual model `a49fbedbd4aa8de3949b2dc94ce493ad78376427fbb1e6554ff42f8cec9cffe5` |
| `SparkRocketSpidopsV1_extracted` | `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852` | `main.py` `24c62917c58ed02bfa913ddd54a21db1846a42e2acadb53ae6b63257edf4c3a1` |
| `SoldierPureRLV45_extracted` | `80798a3cae4e541da01acaa9a58c493f3d5010571e4dbabb10bf7dc701c94281` | `model.pt` `9735dd7ac36d2043209857c5941305dc19768907b1682717e4326ffd7afa6efe` |
| `VioletAlakazamMKV93_extracted` | `4e56a20249fad14097d06aaee2a241f355b90ef9bef19307f95b47798d0a2a3d` | `main.py` `c324c943d2dfe62d28c9a404e264e6d25eabeea296e37f9575ee1231a76bbec4` |
| `SparkTrueCrustleCrispinGalvV71_extracted` | `599393da0203019c8cc3b2192c54f91ea5213d8c6e9631070df14fce1944d79c` | `main.py` `77c6e2b94aeb4eac54b04eaf40b75342960174a163638874c3f05e067c303798` |

The two Starmie bundles are load-bearing adversarial gates. Marnie represents
the mature ladder leader; the remaining decks prevent a narrow Starmie-only
overfit.

## Locked learning method and command

Trainer:
`scripts/train_target80_exact_deck_elite_bc_v1.py`

Trainer SHA-256 at lock:
`aed1eefc8c12b6cb42e990b6fc98f70c87859a68e8757862fff43c57b230eb00`

Output (must not exist when the run starts):
`work/target80_exact_deck_elite_bc_v1_seed806050`

Locked parameters:

- seed `806050`;
- 3 learning iterations;
- 28 stochastic rollout games per opponent per iteration (196 per iteration,
  588 total);
- temperature `1.15`;
- retain only multi-option decisions from games the Rocket policy wins;
- balance retained training mass equally across opponents with at least one
  winning trajectory;
- train the action logit head and final relational-transformer block only;
- 2 epochs per iteration, batch size 96, AdamW learning rate `1e-5`, weight
  decay `1e-5`;
- legal-option label smoothing `0.01`, frozen-blueprint forward-KL coefficient
  `0.10`, entropy coefficient `0.0005`;
- never train the value head;
- baseline and each iteration use 30 greedy games against each locked opponent
  (210 games per evaluation);
- select the best checkpoint lexicographically by total wins, then the minimum
  wins against any single opponent.

The exact command is:

```powershell
& 'C:\Users\jhos4\AppData\Local\Programs\Python\Python312\python.exe' `
  scripts\train_target80_exact_deck_elite_bc_v1.py `
  --candidate work\target80_candidates\hancang_wobbfree_scale_control_v1 `
  --opponent work\arena_bundles\AshV178MarnieFullBest_20260801_extracted `
  --opponent work\arena_bundles\SparkStarmieMistFlatV20_20260801_extracted `
  --opponent work\arena_bundles\SparkStarmieFroslassCFV13_extracted `
  --opponent work\arena_bundles\SparkRocketSpidopsV1_extracted `
  --opponent work\arena_bundles_20260804\SoldierPureRLV45_extracted `
  --opponent work\arena_bundles_20260804\VioletAlakazamMKV93_extracted `
  --opponent work\arena_bundles_20260804\SparkTrueCrustleCrispinGalvV71_extracted `
  --evaluator 'C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py' `
  --output work\target80_exact_deck_elite_bc_v1_seed806050 `
  --iterations 3 --rollout-games-per-opponent 28 `
  --eval-games-per-opponent 30 --epochs 2 --batch-size 96 `
  --learning-rate 1e-5 --weight-decay 1e-5 --kl-coef 0.10 `
  --entropy-coef 0.0005 --label-smoothing 0.01 `
  --temperature 1.15 --seed 806050 --threads 4 --timeout 30 `
  --max-steps 1000 --seat first --train-last-block
```

## What is official and what is abstract

Genuine official-engine execution:

- deck legality and game transitions;
- every stochastic training trajectory;
- every greedy baseline and checkpoint evaluation game;
- opponent decisions through their packaged runtime;
- seat order, terminal outcome, and legal selection boundary.

Research abstraction:

- treating a winning trajectory as an elite supervised target;
- equalizing training mass across opponent families;
- the frozen-reference KL safety anchor;
- checkpoint selection from this development panel.

No hidden opponent hand is supplied to the actor or trainer. This method is
not CFR, safe re-solving, public-belief search, or a Nash/robustness guarantee.

## Admission gates

The selected checkpoint may advance only if every gate passes:

1. The run completes without worker, legality, timeout, or engine failures.
2. Candidate and opponent semantic input identities are byte-stable before
   and after the run.
3. The candidate still uses the exact locked 60-card deck and contains zero
   Wobbuffet.
4. Each opponent contributes at least one winning training trajectory across
   the complete three-iteration run.
5. The selected checkpoint wins at least 147 of 210 development games (70%).
6. It improves by at least 14 wins over the independently sampled 210-game
   frozen baseline evaluation.
7. It wins at least 18 of 30 games against each Starmie opponent.
8. It wins at least 15 of 30 games against every other opponent.

Passing authorizes only a newly preregistered, sealed broad confirmation panel.
That successor panel must be large enough to test whether the policy is
plausibly near the 80% goal before packaging. Only after that confirmation may
an immutable internal-arena successor be considered. Kaggle submission is out
of scope.

If any gate fails, this exact seed/hyperparameter route is rejected. The
trainer remains reusable research infrastructure, but the checkpoint is not
packaged or submitted and no threshold is relaxed after observing outcomes.

## Result

The seed-806050 run completed its 210-game baseline and all 196 first-iteration
stochastic rollouts, then failed before the first post-update evaluation game.

Baseline: **118-90-2 (56.19%)**. Per-opponent wins out of 30 were Marnie 12,
MistFlat Starmie 14, Froslass Starmie 7 (plus two draws), Rocket mirror 17,
PureRL 20, Alakazam 25, and Crustle 23.

The first rollout block was 45-150-1. Every opponent supplied winning
trajectories: Marnie 5, MistFlat Starmie 2, Froslass Starmie 6, Rocket mirror
4, PureRL 2, Alakazam 15, and Crustle 11.

The update's KL/entropy calculation consumed `-inf` logits for padded options.
Although the forward result masked those entries, backward propagation created
non-finite parameters. The first subsequent policy decision raised on
`tensor([nan, nan])`. This is an implementation-validity failure, not a
checkpoint strength result.

No iteration checkpoint, best model, package, arena submission, or Kaggle
submission was produced. The designated output directory exists but is empty.
The source candidate's deck and model hashes remained exactly locked. A
finite-mask correction must use a new trainer hash, fresh seed, new output, and
separate preregistration; seed 806050 will not be reused.
