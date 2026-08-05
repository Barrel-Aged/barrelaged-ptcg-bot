# N7 Arena Marnie-Second AWR Result

Status: **rejected at the locked development gate; live bot unchanged**  
Protocol: `N7_ARENA_MARNIE_SECOND_AWR_PROTOCOL.md`  
Completed: 2026-07-30

## Question tested

N7 asked whether outcome-weighted behavioral cloning from the exact live arena
policy's public-observation decisions could repair the Rocket bot's weak
second-seat Marnie matchup. The experiment changed only the final Transformer
block and action-logit head. It did not add runtime search, use hidden opponent
information, change the frozen 60-card deck, or alter the first-seat policy.

## Integrity and parity

- Frozen deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Frozen live base-model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- The replay audit covered 1,458 decisive, second-seat Marnie games and 85,928
  policy decisions.
- Ten games contained one genuine replay/action mismatch each and were excluded
  in full. Thirty-four forced-set ordering differences were proven equivalent.
- After exclusion, exact batched parity was 57,139/57,139 training decisions,
  11,988/11,988 validation decisions, and 12,263/12,263 test decisions.
- The outer command timed out after 3,604 seconds while its original trainer
  process continued. All four atomically written epoch files were subsequently
  accepted under the pre-panel recovery amendment. The auditor confirmed that
  every model loads, all tensors are finite, timestamps and hashes are
  monotonic, every epoch advanced, and only the 14 preregistered tensors changed.
- Packaged-style runtime parity was exact on 2,456/2,456 official-engine
  decisions across 40 games.
- The locked development panel executed 2,400 fresh official-engine games: 240
  games per arm against each of the two frozen Marnie opponents. There were no
  engine errors, illegal fallbacks, agent fallbacks, or protocol shortfalls.

## Locked development result

| Arm | V110 win rate | Change | Recovery V4 win rate | Change | Stratified change | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|---:|
| Control | 37.50% | — | 47.92% | — | — | — |
| Epoch 1 | 46.67% | +9.17 pp | 37.50% | -10.42 pp | -0.63 pp | [-6.88, +5.63] pp |
| Epoch 2 | 44.58% | +7.08 pp | 36.25% | -11.67 pp | -2.29 pp | [-8.54, +3.96] pp |
| Epoch 3 | 41.67% | +4.17 pp | 42.92% | -5.00 pp | -0.42 pp | [-6.67, +6.04] pp |
| Epoch 4 | 42.50% | +5.00 pp | 39.58% | -8.33 pp | -1.67 pp | [-7.71, +4.58] pp |

No trained epoch satisfied the preregistered aggregate-improvement,
worst-opponent, bootstrap, and correctness gates. Validation, seat routing,
packaging, and submission were therefore forbidden by protocol.

## Strategic interpretation

The experiment found a real but unsafe policy specialization. Every checkpoint
improved against `AshMarnieGeneralV110`, while every checkpoint regressed against
`HelbrosMarnieGeneralRecoveryV4`. Treating “Marnie” as one stationary opponent
distribution is therefore wrong: public-identical deck families can implement
meaningfully different policies, and the current rank-leading Recovery policy
punishes the V110 exploit.

This result explains why a generic offline imitation update cannot safely make
the arena bot rank one. It also supplies a concrete next hypothesis: preserve
the live policy as a baseline-safe blueprint and expose the epoch-1 exploit only
when a public-history opponent-plan model identifies V110-like behavior with a
strict, preregistered false-activation cap. If public evidence cannot separate
the policies early enough, the exploit must remain inactive. A second path is
robust multi-opponent training on newly captured exact official-engine
trajectories rather than conditioning on one aggregate Marnie label.

## Artifact record

- Development analysis:
  `artifacts/team_rocket_recovery/n7_marnie_second_awr/development_panel/analysis.json`.
- Exact live-agent parity:
  `artifacts/team_rocket_recovery/n7_marnie_second_awr/exact_live_agent_parity_v2.json`.
- Recovered-training audit:
  `artifacts/team_rocket_recovery/n7_marnie_second_awr/training_v1_recovery_manifest.json`.
- Development-arm manifest:
  `artifacts/team_rocket_recovery/n7_marnie_second_awr/development_arms/arms.json`.
- Runtime parity:
  `artifacts/team_rocket_recovery/n7_marnie_second_awr/development_arms/runtime_parity.json`.

These are development artifacts only. They carry no deployment authority and
support no unverified performance claim. `HelbrosRocketSpidopsScaleBCV2`
remains the submitted bot.
