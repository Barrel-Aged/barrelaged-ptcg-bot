# N18 New-Leader Evidence Audit

Status: **sealed before N18 protocol selection**  
Audit schema: `rocket-n18-new-leader-evidence-audit/v1`

## Purpose

N1-N17 did not admit a replacement for the live exact-deck Team Rocket bot.
N18 therefore starts from genuinely new official-engine evidence rather than
resampling or retuning the rejected N6R localized-residual family.  This audit
fingerprints the current arena leaders, validates their public replay actions,
measures duplicate evidence, and checks whether their policy representation is
compatible with the unchanged live Rocket model.

This document is an evidence record, not an experiment protocol or deployment
authorization.

## Frozen live control

- bot: `HelbrosRocketSpidopsScaleBCV2`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- downloaded live archive SHA-256:
  `27dff66b6d39d780e56a745c34039305ce5e76e0a17811253ed4315698f2c6f9`;
- deck length: exactly 60 cards;
- live submission was not edited, repackaged, or replaced during this audit.

## Fresh official-engine evidence

The authenticated internal-arena data endpoint returned:

- 612 games for current rank-one `AshMarnieGeneralV177MultiFailure`:
  405 wins, 205 losses, and 2 draws;
- 510 games for current rank-two `SparkZoroarkArenaV2`:
  332 wins, 177 losses, and 1 draw;
- 102,920 recorded decisions across the two downloaded views;
- 102,920/102,920 actions legal for their same-row official-engine
  observations;
- 138 replay payloads present in both views, leaving 984 deduplicated games.

The rank-one corpus includes 178 direct games against the exact live Rocket
bot.  Rank one won 117 and lost 61 in those games.  That is relevant evidence
that its executed strategy is strong against the live bot, but the result is
confounded with a different deck and does not establish that its weights will
transfer to Team Rocket.

## Leader-bundle identity result

The two apparent leaders are not independent strategic teachers:

- their `model.pt` files are byte-identical, SHA-256
  `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c`;
- their `history_residual.pt` files are byte-identical, SHA-256
  `5df9bd6e9f2d733092b0653765b4d1d3f082a488cfe68fe31f23c94f3aa43f4f`;
- their 60-card deck files are byte-identical, SHA-256
  `92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d`;
- that deck is not the required Team Rocket deck;
- their entry-point files differ, but both load the same model, residual, and
  deck strategy.

The archives remain separately frozen:

- rank-one archive SHA-256:
  `cf8ae2122a6cfedff0a82e8d3f368fcf029dbc4d17c5c8898f355ec44049b9cf`;
- rank-two archive SHA-256:
  `e33d640cefb1fe594122e9a5155ce5767b52035ceebcc484fb486194f2efe6bc`.

Treating these bots as two independent replications would be invalid.  The
deduplicated corpus is one strategy observed under two submission names.

## Neural compatibility

The leader and live models have the same V10 relational architecture, exactly
the same configuration, all 59 state-dictionary keys, and identical tensor
shapes.  Each has 4,298,205 parameters.  Relative to the live model:

- 45 tensors are byte-exact;
- 14 tensors differ;
- the differing tensors contain exactly 1,184,257 parameters;
- those are the same last-transformer-block plus action-head parameter family
  that produced the live scale-BC policy.

The leader history residual is a separate 222,273-parameter recurrent module
with schema `history-residual-policy/v1`, global width 384, option width 64,
memory width 128, and projection shape 384 by 64.  It declares raw card-belief
representation and no learned card-belief checkpoint.

This establishes mechanical architecture compatibility.  It does not
establish that the Marnie-trained last block, action head, or recurrent
residual is beneficial when the official engine supplies the Team Rocket deck.

## N18 consequence

N18 may legitimately test a small, frozen cross-deck transfer family:

1. the unchanged live Rocket model as control;
2. the leader base model with the exact Rocket deck;
3. the leader base plus its history residual with the exact Rocket deck;
4. the live base plus that history residual with the exact Rocket deck.

Any such test must freeze packages, runtime parity, seeds, opponents, seats,
sample sizes, analysis, multiplicity handling, and admission gates before
official-engine outcomes are collected.  The direct leader transfer and both
residual arms have zero deployment authority until they pass development,
fresh independent confirmation, and broad both-seat regression.

## Durable artifacts

- machine-readable audit:
  `artifacts/team_rocket_recovery/n18_new_leader_evidence/N18_NEW_LEADER_EVIDENCE_AUDIT.json`;
- audit JSON SHA-256:
  `f0441e0e602421c0ccbb57447e9b5857af442c93e7b6a02224cd4f09c24e9a0c`;
- audit script:
  `scripts/audit_n18_new_leader_evidence.py`;
- audit script SHA-256:
  `5f0b31b566edf95e17f70a423caa9c4be55b7ee32c6c304c36b1abc67cac97ef`.

## Claim boundary

The replay files are genuine internal-arena official-engine execution, and the
legality checks are exact for the recorded row-local observations.  Model
transfer, residual application, candidate evaluation, and any later
counterfactual interpretation are abstractions until exercised by a locked
official-engine panel.  No new Team Rocket strength, causal improvement,
leaderboard outcome, or deployment claim is made here.
