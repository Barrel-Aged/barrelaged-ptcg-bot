# Neural Plan Residual Training Protocol

Status: preregistered before campaign N1 training  
Protocol: `rocket-neural-plan-residual/v1`  
Frozen Team Rocket deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`  
Frozen blueprint model SHA-256: `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`

## Motivation

Randomized turn-2 reranking, turn-4 reranking, and a 4,000-game locked persistent semantic overlay all failed. The fixed overlay changed decisions in 64.2% of games but changed win probability by only +0.05 points. The next experiment must learn long-horizon credit from complete outcomes rather than encode more manual preferences.

## Architecture

The 4.3M-parameter relational public-state blueprint remains frozen. A small trainable residual receives the blueprint's public global token and option tokens:

- a GRU maintains a persistent latent plan state across every acting decision in the game;
- a learned low-rank projection maps option tokens into a 64-dimensional residual space;
- the latent plan emits an option query that adds bounded residual logits to the frozen tactical logits, including conditional autoregressive picks;
- six strategic value heads estimate setup, tempo, resource efficiency, prize control, board control, and optionality from the persistent state;
- a state-conditioned meta head combines those six estimates into the overall outcome value used for advantage estimation.

The residual starts near zero, so the initial policy is the frozen blueprint. Legal options, the attack-safe END guard, public belief tracker, and exception fallback are unchanged. The acting policy receives only the ordinary redacted observation and its own visible private zones. It never receives opponent hidden cards, visualizer truth, a cloned branch, or future information.

## Genuine official execution versus abstractions

Every rollout is a complete game executed by the local official competition engine. Opponents are the immutable extracted arena policies, hosted through the same isolated worker boundary used by the benchmark harness. Terminal wins/losses come from the official engine. The recurrent plan state, strategic targets, PPO loss, and KL constraint are research abstractions, not official primitives or an equilibrium certificate.

## N1 training campaign

Train four independent seeded residuals from the same zero-residual initialization.

- Seeds: `781101`, `781102`, `781103`, `781104`.
- Opponent mixture: 50% `AshMarnieGeneralV110_extracted`, 50% `HelbrosMarnieGeneralRecoveryV4_extracted`, assigned in seeded shuffled blocks.
- Seat: Team Rocket seat 1; seat 0 is the official first player.
- Games per seed: 1,024 (16 iterations × 64 complete rollout games).
- Total N1 outcomes: 4,096 complete training games.
- PPO epochs per batch: 2.
- Residual memory: 128; option width: 64.
- Learning rate: `1e-4`; gradient clip: `1.0`; PPO clip: `0.15`.
- Discount: `0.997`; GAE lambda: `0.95`.
- Entropy coefficient: `0.003`; frozen-blueprint KL coefficient: `0.10`.
- Auxiliary strategic-value coefficient: `0.15`; residual L2 coefficient: `1e-4`.
- Maximum residual option bias: `2.5` logits.

Training checkpoints are diagnostic only. The final iteration is the seed artifact; no tiny in-training evaluation chooses a lucky checkpoint.

## Public auxiliary targets

Auxiliary targets are computed only from the trajectory's public/own-visible state and do not alter terminal reward:

- setup: whether the four-Rocket board threshold was established by own turn 3;
- tempo: normalized inverse first-attack turn;
- resource efficiency: whether two attackers became ready without an already-ready active receiving avoidable manual Energy;
- prize control: fraction of own prize cards taken at terminal;
- board control: maximum number of ready attackers, clipped to two;
- optionality: whether Mewtwo was online while a distinct ready attacker remained.

These heads regularize the persistent representation. PPO policy advantages still use only the official terminal game outcome and the learned meta value; the auxiliary targets are not shaped rewards.

## N1 development selection

After all four seeds finish, build final-seed candidates and parameter soups only where state dictionaries are aligned. Use a fresh development panel, not training outcomes, to choose at most two candidates. The development panel must contain at least 240 genuine-second-seat games per candidate, equally divided between both opponents, plus an interleaved frozen control.

Advance only a candidate whose equal-opponent point estimate is positive, whose intervention rate is at least 5%, and whose worst per-opponent regression is no more than 2 points. Development results are not an arena claim.

## N1 locked validation and test

Before validation, freeze one candidate, runtime, seeds, and sample size. Validation must use at least 2,000 fresh games total, interleaving candidate and control equally across both opponents. Pass only if:

- point improvement is at least +3 pp;
- the stratified 95% bootstrap lower bound is greater than 0;
- neither opponent regresses by more than 2 pp;
- zero correctness or hidden-information violations occur.

Only a passing validation may authorize an untouched test, full non-Marnie regression gates, packaging, and arena submission. No Kaggle submission is part of this protocol.

## Stop rule

Stop N1 without test or submission if training is numerically unstable, residuals saturate at the bias cap, correctness fails, all seed candidates regress in development, or locked validation fails. A failed N1 justifies scaling/changing the learned architecture; it does not justify returning to unvalidated heuristics.
