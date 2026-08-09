# N7 Arena-Distribution Marnie-Second AWR Protocol

Status: preregistered before training or official-engine development evaluation.

Date: 2026-07-28 (America/Denver). Arena replay filenames use the arena host clock and extend into `20260729`.

Deployment authority: **none**. N7 is a development experiment. It cannot replace or update the live arena bot unless every advancement, validation, router, packaging, and integrity gate below passes.

## 1. Motivation and bounded hypothesis

The live `HelbrosRocketSpidopsScaleBCV2` policy is already the strongest known Team Rocket arena entry by more than 100 rating points, but it is not a rank-one generalist. Its full 4,880-game replay corpus isolates the dominant weakness:

- overall: 2,724 wins, 2,154 losses, 2 draws (55.82% decisive win rate);
- going first: 1,867-1,070-1 (63.55%);
- going second: 857-1,084-1 (44.13%);
- versus Marnie: 873-1,315-2 (39.86%);
- losses with no second ready attacker: 1,234/2,154 (57.29%), versus 27.72% in wins.

N6R exact-feature belief-rollout distillation was rejected after its locked 2,400-game official-engine panel: no arm satisfied the aggregate, worst-opponent, and confidence gates. N7 therefore tests a materially different and narrower hypothesis:

> Game-balanced advantage-weighted regression on exact live-policy trajectories can improve the policy specifically when going second into a publicly identified Marnie line, while a public router leaves every other state on the byte-frozen live policy.

This is conservative offline policy improvement, not hidden-hand belief modeling, tree search, counterfactual re-solving, or a claim of game-theoretic optimality. The longer-term public-belief blueprint and safe re-solving architecture remains documented in `PIVOT_TO_NEURAL_GAME_THEORETIC_PLANNING.md`.

## 2. Immutable inputs

### Live reference bundle

- candidate directory: `artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2`
- `model.pt`: `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`
- `deck.csv`: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`
- `main.py`: `8c87db1a011acf3cc5a614159d1ac6b6bf56aa7286f0f8a7148bb8e2aa20a6d1`
- `deck_library.json`: `ed56d55cec6e2a87415db422b72f16baa8ab552e87c625de75cd0d8cb7e8d4e0`

The deck hash is the frozen exact 60-card Team Rocket's Mewtwo ex / Spidops list. N7 may not alter a card, card count, deck ordering, or deck file byte.

### Replay corpus

- directory: `work/arena_scale_v2_all_20260728`
- official arena replay files: 4,880 JSONL games
- deterministic tree SHA-256: `0cf23a8276395aed0c53e98812eac3faa1857440b72b1b934d2ad421fb3c2381`
- first filename: `20260728-193917_g0000_HelbrosRocketSpidopsScaleBCV2_vs_AshMarnieGeneralV151.jsonl`
- last filename: `20260729-223607_g0003_HelbrosRocketSpidopsScaleBCV2_vs_SoldierPureRLV38.jsonl`

The corpus is read-only. Opponent names are permitted only as offline family labels. A deployed router may not inspect opponent names, bundle metadata, private cards, or hidden engine state.

### Exact-agent parity manifest

- report: `artifacts/team_rocket_recovery/n7_marnie_second_awr/exact_live_agent_parity_v2.json`
- report SHA-256: `0c3b2ed26dd06d2df3c9ac490371d108c06c2d2c3b654b2e072d7ab57357e80a`
- audit script: `scripts/audit_exact_live_agent_parity.py`
- audit script SHA-256: `e61efa4ec9559c8dc014215b617487e044eddb5fad8c2e9d4745db88c139c6ea`

The exact submitted `NNAgent` was executed serially with one PyTorch thread over all eligible Marnie-second replays. After excluding one draw, it audited 1,458 games and 85,928 decisions:

- 85,918 exact or engine-equivalent actions;
- 34 differences only in the serialization order of a forced full selection set;
- 10 games with exactly one genuine action mismatch each.

All ten genuinely mismatching games are excluded in full because their later states are conditioned on an archived action the frozen local reference does not reproduce. The trainer must consume the signed report, require exactly those ten exclusions, preserve their original split membership, and reject any report/hash/count/filter change. The exclusion cap is therefore frozen at exactly 10/1,458 games (0.686%).

### Training implementation

- trainer: `scripts/train_live_outcome_awr.py`
- trainer SHA-256 at preregistration: `4f233d40e625c33711ab87438388f8ba68f65f2e103c563bf621c790c9a657e7`
- tests: `tests/test_live_outcome_awr.py`
- tests SHA-256: `49840ed3b51196ed97042ed784cd15a239a9a01754bca3901d5093628cfba883`
- test status at preregistration: 7/7 relevant tests passed, including the draw, chronological split, single-select attack guard, multi-select non-guard, live-collation, exact-manifest, and replay-autopsy cases.

The trainer fixes a previously discovered historical-collator error: real but already-selected options remain visible as Transformer tokens, exactly as in deployed single-state inference, while only their final logits are masked unavailable. Forced full-set prompts are excluded because they contain no deployable set choice. The attack-safe END mask is applied after inference and only in the live agent's single-selection branch.

## 3. Dataset and split

Eligibility is fixed before extraction:

1. replay contains `HelbrosRocketSpidopsScaleBCV2`;
2. opponent family label contains `marnie`;
3. the Rocket bot started second according to public `firstPlayer` state;
4. game has a decisive winner;
5. game is not one of the ten signed exact-agent mismatch games.

The 1,458 decisive eligible games are sorted lexicographically by timestamped filename, split 70%/15%/15%, and only then are the ten signed games removed without moving any boundary:

| Split | Before exclusion | Excluded | Retained | Retained wins |
|---|---:|---:|---:|---:|
| Train | 1,020 | 7 | 1,013 | 390 |
| Validation | 218 | 1 | 217 | 76 |
| Test | 220 | 2 | 218 | 80 |

Validation and test trajectories are diagnostic holdouts only. Neither may select an epoch, hyperparameter, threshold, router rule, or official-engine opponent/seed.

Sequential feature extraction must process every acted observation, including trivial prompts, so public-belief state matches deployment history. Training examples include only supported nontrivial choices that emit the runtime policy's full selection count. Unsupported optional early stops are reported and excluded. Every retained split must reproduce the frozen reference action with zero batched mismatches before training starts.

## 4. Fixed learning rule

One training run produces four immutable epoch arms. Hyperparameters are fixed:

- seed: `794101`;
- PyTorch threads: `1`;
- trainable scope: last Transformer block plus action head (`mode=last`);
- epochs: `4`;
- batch size: `128`;
- learning rate: `3e-6`;
- AdamW weight decay: `1e-4`;
- outcome temperature: `0.5`;
- reference KL coefficient: `0.25`;
- gradient-norm clip: `1.0`;
- source policy: frozen live model above.

Each game receives equal base mass regardless of trajectory length. Within the single Marnie-second training stratum, each action's advantage is `outcome - shrunken_train_win_rate`; its AWR multiplier is `exp(advantage / 0.5)`. The objective is weighted negative log likelihood plus `0.25 * KL(reference || student)`.

The four development arms are exactly `epoch_01.pt`, `epoch_02.pt`, `epoch_03.pt`, and `epoch_04.pt`, plus the byte-frozen control. Offline validation metrics are recorded but have no selection authority. `model.pt` is only an explicit final-epoch alias and is not the development winner.

## 5. Pre-engine integrity gates

Training stops without official-engine evaluation unless all conditions hold:

1. live model, deck, replay tree, parity report, trainer, and protocol hashes match;
2. exact-agent manifest contains exactly ten excluded games and the frozen filters;
3. post-exclusion train/validation/test action parity is 100% with zero mismatches;
4. every epoch checkpoint loads under the unmodified official adapter;
5. deck hash remains exact for control and every arm;
6. runtime parity smoke executes at least 2,000 official acted decisions across arms with zero exceptions, illegal outputs, fallbacks, or adapter drift;
7. immutable arm manifest records every model and bundle hash before game collection.

Failure is terminal for this N7 execution; it may be diagnosed only under a separately documented amendment created before any replacement panel.

## 6. Locked official-engine development panel

Only after the integrity gates pass, run the existing official engine with strict no-fallback execution. Rocket is always seat 1 (going second). Opponents are the frozen bundles:

- `AshMarnieGeneralV110_extracted`;
- `HelbrosMarnieGeneralRecoveryV4_extracted`.

Arms: control, epoch 1, epoch 2, epoch 3, epoch 4. Each shard runs 120 games per arm (600 games). Seeds are fixed:

- V110 shard 1: `795101`;
- V110 shard 2: `795102`;
- RecoveryV4 shard 1: `795201`;
- RecoveryV4 shard 2: `795202`.

Total development panel: 2,400 official-engine games, 480 per arm, 240 per opponent. All arms share the same shard schedules. Manifests must assert official-engine execution, own seat 1, exact hashes, strict no-fallback behavior, and zero fallbacks.

For each epoch arm compute its win-rate delta from control separately against each opponent and in aggregate. Use the pre-existing deterministic game-level bootstrap implementation with 20,000 replicates for the aggregate 95% interval.

An arm advances only if all three conditions hold:

1. aggregate delta is at least **+2.0 percentage points**;
2. its worse of the two opponent-specific deltas is strictly above **0.0 percentage points**;
3. the lower endpoint of its aggregate 95% bootstrap interval is above **-0.5 percentage points**.

If no arm passes, N7 is rejected and the live bot remains unchanged. If more than one passes, choose the arm with the largest aggregate delta; exact ties choose the earlier epoch. No post-panel hyperparameter change, epoch interpolation, opponent removal, seed extension, or threshold relaxation is allowed.

## 7. Router and independent validation

The adapted model is not a global replacement. After exactly one arm advances, build a dual-agent public router:

- first seat: always use the byte-frozen base policy;
- second seat: use the base policy until public observation reveals a Marnie-line card ID in `{646, 647, 648}`;
- after the public trigger: route actions to the adapted arm;
- advance both base and adapted public-belief agents on every observation so switching cannot reset history;
- never inspect opponent identity, bundle name, private hand, prize identities, random seed, or replay label.

Corpus feasibility facts are frozen but are not strength claims: the Marnie-line trigger is visible before the first main-phase action in 1,376/1,459 (94.31%) Marnie-second games, eventually in 1,457/1,459, and had 0/483 false positives in non-Marnie second-seat games.

Before submission, the routed candidate must pass:

1. deterministic router unit tests and replay tests with zero non-Marnie false positives;
2. exact base-policy action parity on first-seat and untriggered second-seat fixtures;
3. independent official-engine Marnie validation using new, preregistered seeds, control versus the single selected routed candidate, with no development-arm reselection;
4. non-Marnie/first-seat official-engine regression shards against frozen Alakazam, Soldier, and Rocket opponents;
5. package audit proving the exact deck, base model, adapted model, router, adapter, and manifests are the tested bytes;
6. zero exceptions, illegal actions, or fallbacks.

The independent validation must retain positive aggregate and worst-opponent Marnie deltas and show no material regression in non-routed shards. Exact validation seeds and sample sizes will be frozen in a separate validation protocol before any validation games, after the single development winner is immutable.

## 8. Submission rule

No Kaggle submission is authorized or implied. An internal-arena update is permitted only after all gates above pass. The submitted bot must retain the exact Team Rocket deck and official adapter. No unverified performance, rank, or superhuman claim may be made.

If submitted, success still requires a post-submission audit of the returned immutable archive, an active rated entry, at least 300 rated games on that exact archive/deck, and observed rank #1. Until then, the project goal remains in progress.
