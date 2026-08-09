# N5 Rocket Blueprint Fine-Tune Protocol

Status: preregistered before N5 training  
Protocol: `rocket-greedy-compatible-ppo/v1`

## Motivation

N1-N4 show that policy overlays cannot rescue a blueprint trained primarily for Marnie. N5 adapts the blueprint itself to the frozen Team Rocket deck. It corrects N1's largest train/deploy mismatch: behavior rollouts use temperature 0.35 rather than 1.10, so exploration is concentrated among near-tied legal actions and the learned policy is evaluated greedily.

## Frozen substrate

- Deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Initial model SHA-256: `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`.
- Exact official engine, second-seat placement, public belief features, legality mask, and attack-safe END guard.
- Opponent mixture: 50% immutable `AshMarnieGeneralV110`, 50% immutable `HelbrosMarnieGeneralRecoveryV4`.
- No hidden opponent cards, visualizer truth, branch clone, future information, or shaped outcome reward enters actor or critic.

## Trainable scope

- relational action logit head;
- detached public-state value head; and
- final transformer encoder block.

All embeddings, feature projections, earlier transformer blocks, and belief machinery remain frozen. The original model is an immutable KL reference.

## Campaign

- Independent seeds: 788101, 788102, 788103, 788104.
- 32 iterations x 64 complete games = 2,048 games per seed.
- Total: 8,192 complete official-engine games.
- Final iteration only; no in-training win-rate checkpoint selection.
- PPO epochs 2; episode action batch 128.
- Behavior/policy temperature 0.35.
- PPO clip 0.10; gamma 0.997; GAE lambda 0.95.
- Final-block learning rate 0.00002; head/value learning rate 0.0001.
- Weight decay 0.00001; gradient clip 1.0.
- Value coefficient 0.30; entropy coefficient 0.001; immutable-reference KL coefficient 0.20.

Terminal win/draw/loss is the only policy reward. Training telemetry is diagnostic and cannot establish strength.

## Development and validation

After all seeds finish, compare exactly the four final models and one aligned equal-weight parameter soup against an interleaved frozen control. Development requires at least 480 games per arm, equally split between both opponents. At most one model advances only if:

- equal-opponent improvement is at least +3 points;
- stratified 95% bootstrap lower bound is greater than zero;
- neither opponent regresses below -2 points; and
- correctness and fallback violations are zero.

A separate locked validation of at least 3,000 fresh games is required before broader regression, packaging, test, or arena submission. Failure does not authorize an intermediate checkpoint or post-hoc hyperparameter change.
