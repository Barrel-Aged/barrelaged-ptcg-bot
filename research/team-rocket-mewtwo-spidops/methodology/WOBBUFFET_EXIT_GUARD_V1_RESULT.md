# Wobbuffet exit guard V1 result

Status: **runtime-safe, behaviorally inactive in development; not admitted or submitted**  
Completed: 2026-08-01

## Hypothesis

Preserve the byte-frozen `HelbrosRocketSpidopsScaleBCV2` neural blueprint, but
replace a selected Wobbuffet `Headbutt Bounce` with an offered Retreat when a
Benched Spidops is already able to use `Rocket Rush` for at least 90 damage.
The guard never intercepts `Rocket Mirror`, reads only public in-play state,
and returns only an option offered by the official engine.

The hypothesis was motivated by the exact Kaggle replay audit. Across all 20
Kaggle episodes, the frozen policy attacked with Wobbuffet only twice. Both
were in episode 89414044, and both selected Headbutt over an offered retreat to
a ready Spidops. Replaying all 1,227 recorded decisions changed exactly those
two actions and no others. That audit proves the intended action surface, not
the outcome of the unexecuted alternative branch.

## Official-engine development

All games used the packaged official engine with decision traces enabled.

| Opponent | Games | Candidate W-L-D | Terminal games | Guard activations |
|---|---:|---:|---:|---:|
| frozen ScaleBCV2 mirror | 20 | 12-8-0 | 20/20 | 0 |
| AshMarnieGeneralV110 | 20 | 8-12-0 | 20/20 | 0 |
| SparkStarmieFroslassCFV13 | 20 | 2-16-2 | 20/20 | 0 |

An additional one-game mirror smoke also ended normally. The trace replayer
reproduced the candidate exactly on 4,044/4,044 decisions across the 60-game
panel, with zero mismatches. Every one of those decisions matched the frozen
blueprint because the narrow guard condition did not occur.

The 12-8 mirror score is therefore not evidence of guard efficacy: the two
policies behaved identically on every recorded candidate decision, so ordinary
game/chance variation generated the result. The same limitation applies to the
other matchup scores.

## Decision

V1 passes legality and runtime-health checks but supplies no official-engine
causal or behavioral evidence of improved strength. It is retained only as an
isolated research candidate. It is not packaged, submitted, or allowed to
replace either immutable live Team Rocket identity. The live model and deck
remain byte-unchanged.

Any successor must either obtain an official-engine state distribution where
the intervention genuinely activates or test a broader, separately disclosed
mechanism. The two Kaggle counterfactuals alone are insufficient deployment
evidence.

## Durable artifacts

- candidate: `work/wobbuffet_spidops_exit_guard_v1/`;
- Kaggle action-surface audit:
  `work/kaggle_scalebcv2_55172263_20260801/wobbuffet_exit_guard_replay_audit.json`;
- official results and traces: `work/wobbuffet_exit_guard_v1_eval/`;
- official trace auditor:
  `scripts/audit_wobbuffet_exit_guard_official_trace.py`.

