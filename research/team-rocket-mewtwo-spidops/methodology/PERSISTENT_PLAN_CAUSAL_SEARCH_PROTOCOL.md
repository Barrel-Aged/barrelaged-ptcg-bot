# Persistent Plan Causal Search Protocol

Status: preregistered before the first persistent-plan outcome collection  
Protocol version: `rocket-persistent-plan-causal/v1`  
Frozen Team Rocket deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`  
Frozen action-blueprint model SHA-256: `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`

## Why this experiment exists

The live Rocket bot is legal, stable, and competitive with the lower arena field, but its measured weakness is strategic and concentrated in genuine-second-seat games against the two leading Marnie agents. In the 610-game live replay autopsy, losses were strongly associated with failing to establish a second ready attacker and with Mewtwo never attacking. Two preregistered randomized full-match experiments at individual turn-2 and turn-4 roots did not validate: the blueprint's first action was usually locally preferable, and learned single-root rerankers did not produce a positive confidence bound.

This protocol therefore changes the unit of intervention from one action to a persistent public-state plan held across a complete game. It is the first experiment in this recovery campaign that tests the original persistent-latent-plan hypothesis directly.

## Claim boundary

The matches in this protocol are genuine executions of the local official competition engine: legal observations are supplied by that engine, both agents return ordinary legal action indices, and the terminal result is produced by the engine. The plan axes, semantic option features, and linear logit overlay are research abstractions. They are not an official-engine search primitive, an equilibrium solver, or evidence of superhuman strength. No hidden opponent cards, visualizer truth, cloned search branches, or future outcomes are available to either acting policy.

## Frozen elements

- Deck: the exact 60-card qualifying Team Rocket target documented in `TEAM_ROCKET_300_GAME_TARGET.md`; it must not be edited during this experiment.
- Tactical blueprint: `artifacts/team_rocket_recovery/candidates/scale_value_calibrated_v1`.
- Opponents:
  - `AshMarnieGeneralV110_extracted`
  - `HelbrosMarnieGeneralRecoveryV4_extracted`
- Matchup focus: Rocket occupies seat 1, and a game is accepted only when the official observation declares seat 0 the first player. This yields genuine-second-seat Rocket games.
- Safety behavior: the existing legal-action construction, exception fallback, and END-when-ATTACK guard remain active.

## Persistent plan representation

A plan is a fixed vector over the following public, interpretable strategic axes. The vector is sampled or selected once for the game and remains unchanged until the game ends.

1. `board_setup`: reach the four-Team-Rocket board threshold without wasting late bench slots.
2. `mewtwo_access`: find and bench a first Mewtwo early when it is absent.
3. `spidops_engine`: preserve Tarountula-to-Spidops evolution depth.
4. `mewtwo_charge`: allocate compatible Energy toward a credible Mewtwo attack.
5. `second_attacker`: allocate Energy toward a distinct backup attacker rather than an already-ready active.
6. `charging_fuel`: use Spidops's Charging Up to create useful attack capacity or benched Erasure Ball fuel.
7. `draw_engine`: sequence Factory, draw supporters, and public search resources during setup.
8. `tool_assignment`: place Hero's Cape and Brave Bangle on strategically compatible targets.
9. `attack_tempo`: preserve the option to cash in a legal attack instead of endlessly improving the board.
10. `bench_discipline`: avoid low-value extra liabilities after the required board is established.
11. `fuel_preservation`: when Erasure Ball discards benched Energy, prefer surplus Energy over disabling the only backup attacker.

For single-select roots, the frozen neural logits are adjusted by the bounded dot product between this plan vector and public semantic option features. For autoregressive multi-select roots, the same bounded adjustment is applied at every conditional pick. A zero vector must be action-identical to the frozen greedy blueprint on the same observation history. The overlay never invents an option; it only reranks indices the official engine supplied.

## Randomization and causal estimand

- An arm is assigned before game play by a deterministic seeded, uniformly shuffled block schedule.
- A first-player rejection repeats the pending arm, so every completed shard has equal accepted counts per arm.
- Assignment does not use cards, observations, outcomes, or hidden information.
- Propensity is exactly `1 / number_of_arms` within a stage.
- The primary causal estimand is the stratified difference in win probability between a persistent plan and the zero-plan blueprint, averaged equally across the two frozen Marnie opponents in genuine-second-seat games.
- Secondary estimands are the per-opponent win-rate differences, intervention frequency, Mewtwo-board/attack telemetry where observable, second-ready-attacker telemetry where observable, legality failures, exceptions, and match length.

## Intended sequencing

### D1 — semantic-axis discovery

Use the immutable `persistent_plan_axis_screen_v1.json` arm library. It contains the zero blueprint, individual axis probes, and a small number of preregistered coordinated plans. Collect at least 80 accepted games per arm per opponent, in independent shards. This split is discovery-only; it may be used to select axes and strengths.

Advance at most four nonzero plans whose pooled point estimate exceeds the zero arm and that are not more than 5 percentage points worse on either opponent. Do not interpret discovery ranking as a strength claim.

### D2 — combination development

Construct a new, durably recorded library from only the axes admitted by D1. Use seeded bounded coordinate/CEM proposals plus the zero arm. Collect at least 160 accepted games per arm per opponent. This split may select exactly one candidate and its fixed vector.

### V1 — locked validation

Before collecting validation, freeze the selected vector, implementation hashes, opponents, seeds, and sample size. Randomize only between the candidate and zero blueprint. Collect at least 500 accepted games per arm in total, with equal target counts against each opponent.

The candidate clears validation only if all of the following hold:

- pooled stratified point improvement is at least 3 percentage points;
- the stratified 95% bootstrap lower bound on improvement is greater than 0;
- neither opponent's point estimate is worse than zero by more than 2 percentage points;
- at least 10% of games contain a genuine action intervention;
- there are zero illegal actions, zero engine/agent exceptions attributable to the overlay, and zero hidden-information audit failures.

If validation fails, the untouched test is never opened and no candidate from this family is packaged or submitted.

### T1 — sealed untouched test

Test collection may occur ahead of time, but its outcome rows and summaries must remain unopened until V1 passes. The sealed test uses new seeds and at least 600 accepted games per arm total, balanced across the two opponents. Passing requires a positive pooled point estimate, a positive 95% lower confidence bound, no opponent regression worse than 2 percentage points, and zero correctness errors.

### G1 — full official gates

Only after T1 passes:

- run fresh balanced-seat and genuine-second-seat official panels;
- verify deck bytes and SHA-256, model hash, plan hash, importability, deterministic zero-plan parity, legal shapes, latency, telemetry, and archive contents;
- require improvement over the current Rocket submission on the locked Marnie gates and no material regression on the established non-Marnie benchmark panel;
- package an immutable archive and run the archive smoke test.

An arena submission is authorized only after these gates pass. Arena performance is the only basis for an arena strength claim. No Kaggle submission is part of this protocol.

## Data separation and audit rules

- Every JSONL and manifest records protocol version, split, seed, plan-library hash, deck hash, blueprint hash, opponent path/hash, accepted/attempted counts, and the official-engine claim boundary.
- Discovery and development rows may be inspected.
- Validation may be inspected only after the vector is locked.
- A sealed test file is never read, summarized, grepped, counted by outcome, or used for decisions before validation passes.
- Existing sealed tests from the rejected turn-2 and turn-4 experiments remain permanently sealed and are unrelated to this protocol.
- All acting features must be derivable from the ordinary redacted observation. Offline deck metadata may be used only for immutable card identities and rules.

## Stop conditions

Stop this family without opening T1 if D1 shows no credible nonzero arm, D2 cannot reproduce a gain, V1 fails, intervention frequency is negligible, or any correctness/hidden-information violation occurs. A failed family is useful evidence that bounded semantic planning cannot repair the matchup and that the next investment should be a substantially trained plan-conditioned policy rather than stronger hand-authored biases.
