# N2 Neural Plan Ensemble Protocol

Status: preregistered before N2 arms were executed in any fresh game  
Protocol: `rocket-neural-plan-ensemble/v1`

## Motivation

N1 failed locked validation by -3.0 percentage points. Its selected residual intervened in 95.6% of games and independent seeds disagreed on most held-trace changes. N2 tests whether independent-seed agreement can retain learned long-horizon changes while rejecting idiosyncratic terminal-reward noise. N1 training, development, and validation outcomes are declared development history; none is reused as fresh N2 evidence.

## Frozen substrate

- Team Rocket deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Tactical blueprint SHA-256: `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`.
- Source residuals: final iteration 16 from seeds 781101, 781102, 781103, and 781104 in that order.
- Each source retains its own 128-wide recurrent memory. All sources observe the same legal public decision sequence.
- The official legal-option boundary, attack-safe END guard, public belief tracker, and random-legal exception fallback remain unchanged.

## Locked arms

1. `zero_control`: ignore all residual logits; exact frozen blueprint behavior.
2. `mean_logits_025`: add 0.25 times the mean of four source residual-logit vectors.
3. `mean_logits_050`: add 0.50 times the mean residual vector.
4. `mean_logits_100`: add the full mean residual vector.
5. `consensus_3of4`: choose a non-blueprint option only when at least three source policies agree on the same option; otherwise choose the blueprint option.
6. `consensus_4of4`: choose a non-blueprint option only when all four source policies agree; otherwise choose the blueprint option.

The rule is applied autoregressively at each pick using only official legal options. No arm uses hidden truth, cloned branches, future observations, development outcomes, opponent identity, or a post-hoc semantic filter.

## Development

- Opponents: immutable `AshMarnieGeneralV110_extracted` and `HelbrosMarnieGeneralRecoveryV4_extracted`.
- Team Rocket is seat 1 and must start second.
- Two shards per opponent; 120 games per arm per shard.
- Total per arm: 480 games, split 240/240 by opponent.
- Total development sample: 2,880 complete official-engine games.
- Shard seeds: V110 785101/785102; RecoveryV4 785201/785202.
- Bootstrap: opponent-stratified, 50,000 replicates, seed 785901.

An arm is eligible only if its candidate-minus-control equal-opponent point difference is at least +3 points, its 95% bootstrap lower bound is strictly above zero, neither opponent difference is below -2 points, intervention-game rate is at least 5%, and correctness/fallback violations are zero. At most one arm advances, ordered by point difference, lower bound, then name.

## Locked validation if development passes

Before collection, freeze the single advancing arm, artifact hashes, runtime hashes, seeds, and sample. Validation will contain at least 3,000 new interleaved candidate/control games, equally split across both opponents. It must pass all of:

- point improvement at least +3 points;
- stratified 95% lower bound greater than zero;
- neither opponent regression below -2 points;
- zero correctness, legality, fallback, or information-boundary violations.

Failure stops N2 without test, packaging, or arena submission. N2 is a variance-controlled ensemble experiment, not an equilibrium or safe-resolving certificate.
