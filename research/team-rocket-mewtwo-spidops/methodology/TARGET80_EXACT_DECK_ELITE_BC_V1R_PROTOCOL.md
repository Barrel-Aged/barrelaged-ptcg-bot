# Target-80 exact-deck elite BC V1R protocol

Status at preregistration: **locked; no V1R outcome inspected**

Final status: **completed; rejected by strength gates**

Preregistered: 2026-08-05 (America/Denver)

## Relationship to V1

This is a defect-corrected, fresh-seed replication of
`TARGET80_EXACT_DECK_ELITE_BC_V1_PROTOCOL.md`. V1 never produced a checkpoint:
its first optimizer update propagated non-finite gradients from padded
`-inf` option logits and crashed before post-update game 1.

V1R changes only the numerical implementation and run identity:

- padded illegal logits are replaced by a finite sentinel before softmax in
  cross-entropy, reference-KL, and entropy calculations;
- the trainer aborts immediately if a loss or trainable parameter is
  non-finite;
- seed changes from `806050` to fresh seed `806051`;
- output changes to a new directory.

The finite-mask unit check passed with finite loss, finite gradients, and zero
gradient on illegal padded entries. No V1 checkpoint, package, or submission
exists. The V1R hypothesis, opponents, sample sizes, hyperparameters, and gates
are not relaxed in response to the failed run.

## Frozen Rocket identity

Bundle:
`work/target80_candidates/hancang_wobbfree_scale_control_v1`

- exact 60-card `deck.csv` SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`
- frozen blueprint `model.pt` SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`
- Team Rocket's Wobbuffet engine ID `432`: required count zero.

The deck is the same zero-Wobbuffet Hancang recovery shell enumerated in the V1
protocol: 8 Grass plus 4 Rocket Energy; 4 Tarountula, 4 Spidops, 2 Articuno, 2
Mewtwo ex, 2 Mimikyu; and the locked 34-card Poffin/Bug Set/recovery/Rocket
Supporter/Factory trainer engine. No deck mutation is permitted.

## Frozen first-seat opponent panel

| Opponent bundle | `deck.csv` SHA-256 |
|---|---|
| `work/arena_bundles/AshV178MarnieFullBest_20260801_extracted` | `92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d` |
| `work/arena_bundles/SparkStarmieMistFlatV20_20260801_extracted` | `64ef6e2c0f5fadc12f96426aa80a07e1402501ed00d1778a0badd25d171e4901` |
| `work/arena_bundles/SparkStarmieFroslassCFV13_extracted` | `60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61` |
| `work/arena_bundles/SparkRocketSpidopsV1_extracted` | `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852` |
| `work/arena_bundles_20260804/SoldierPureRLV45_extracted` | `80798a3cae4e541da01acaa9a58c493f3d5010571e4dbabb10bf7dc701c94281` |
| `work/arena_bundles_20260804/VioletAlakazamMKV93_extracted` | `4e56a20249fad14097d06aaee2a241f355b90ef9bef19307f95b47798d0a2a3d` |
| `work/arena_bundles_20260804/SparkTrueCrustleCrispinGalvV71_extracted` | `599393da0203019c8cc3b2192c54f91ea5213d8c6e9631070df14fce1944d79c` |

The individual primary runtime/model hashes remain those in the V1 protocol.
The trainer records semantic bundle digests before and after execution and
requires equality.

## Locked trainer and execution

Trainer:
`scripts/train_target80_exact_deck_elite_bc_v1.py`

Corrected trainer SHA-256:
`da7b1e790b7ab129b1ac44e125b6e32f4fd2011e32097f9e7ef176d84d67633f`

New output (must not exist at launch):
`work/target80_exact_deck_elite_bc_v1r_seed806051`

Locked parameters:

- seed `806051`;
- Rocket in seat 0 for every game;
- 3 iterations;
- per iteration, 28 stochastic games against each of seven opponents (196;
  588 total);
- retain only multi-option decisions from Rocket wins and equalize retained
  mass across opponent families;
- action-head plus final-transformer-block training; value head frozen;
- 2 epochs, batch size 96, AdamW learning rate `1e-5`, weight decay `1e-5`;
- temperature `1.15`, label smoothing `0.01`, frozen-reference forward-KL
  coefficient `0.10`, entropy coefficient `0.0005`;
- 30 greedy games per opponent for the baseline and after each iteration (210
  per evaluation);
- select lexicographically by total wins and then minimum single-opponent wins.

The exact command is identical to V1 except:

```text
--output work\target80_exact_deck_elite_bc_v1r_seed806051
--seed 806051
```

All other arguments are exactly the V1 locked command.

## Official execution boundary

The official `cg` engine controls legality, transitions, seat order, terminal
outcomes, and all training/evaluation games. Opponents execute their packaged
runtimes in isolated official-evaluator workers. The actor/trainer receives no
hidden opponent hand.

Elite-trajectory selection, supervised updating, balancing, the KL anchor, and
checkpoint selection are research abstractions. V1R is not CFR, public-belief
search, safe re-solving, or a game-theoretic exploitability guarantee.

## Unchanged admission gates

Every gate is required:

1. Complete with zero engine, worker, legality, timeout, or numerical failures.
2. Byte-stable semantic input identities before and after execution.
3. Exact locked deck and zero Wobbuffet.
4. At least one winning training trajectory from every opponent across the
   three iterations.
5. Selected checkpoint at least 147 wins in 210 games (70%).
6. At least 14 more wins than the fresh independently sampled 210-game V1R
   baseline.
7. At least 18 wins in 30 against each Starmie opponent.
8. At least 15 wins in 30 against every other opponent.

Passing authorizes only a separately preregistered sealed broad confirmation.
It does not authorize packaging or arena submission by itself. Failure rejects
this corrected configuration without threshold relaxation. Kaggle submission
is out of scope.

## Result

V1R completed all 1,428 official-engine games, all three optimizer updates, and
the before/after semantic identity comparison without an engine, worker,
legality, timeout, or numerical failure. The exact deck remained Wobbuffet-free
and every opponent supplied winning training trajectories.

The fresh baseline was **129-80-1 (61.43%)**. Iteration results were:

| Iteration | Result | Win rate | Minimum opponent wins (of 30) |
|---|---:|---:|---:|
| 1 | 123-87-0 | 58.57% | 7 |
| 2 | 126-82-2 | 60.00% | 6 |
| 3 / selected | 136-73-1 | 64.76% | 11 |

The selected third checkpoint improved by seven wins over the independently
sampled baseline, not the required 14, and reached 136 wins rather than the
required 147. Its per-opponent wins were Marnie 16, MistFlat Starmie 12,
Froslass Starmie 11 (plus one draw), Rocket mirror 23, PureRL 20, Alakazam 27,
and Crustle 27. Both Starmie floors therefore failed decisively.

Gates 1-4 and the non-Starmie floors passed. Gates 5, 6, and 7 failed. The
checkpoint is rejected and must not be packaged or submitted.

Artifacts:

- `work/target80_exact_deck_elite_bc_v1r_seed806051/training_report.json`
- selected `best_model.pt` SHA-256:
  `522220d4b5cc2ef8937aa454a0504872ec7ca18f8b21537bcd422b452a7aa265`
- selected iteration checkpoint SHA-256:
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`

No bot package, internal-arena submission, or Kaggle submission was created.
