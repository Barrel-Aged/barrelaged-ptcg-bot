# Target-80 EarlyPad Lethal V5 submission

Date: 2026-08-04  
Arena identity: `HelbrosRocketSpidopsEarlyPadLethalV5`  
Status: submitted to the internal arena; 200-game target not yet adjudicated

## Frozen success contract

One newly submitted, immutable Team Rocket identity must complete at least 200
rated arena games with at least 160 wins. Draws count as non-wins. Games from
older Team Rocket identities are not pooled into this record.

## Candidate selection

The restart evaluated the queued deck and policy hypotheses without altering
the live V4 or its TrainingPin.

- Replacing Wobbuffet with Mimikyu or Lillie's Clefairy ex scored 26-14 in the
  frozen current-opponent panel; replacing Wobbuffet plus both Murkrow with the
  NAIC attacker package scored 29-11. The untouched EarlyPad control scored
  31-9, so all three Wobbuffet-free decks were rejected.
- Extending unconditional EarlyPad search through turn six scored 11-13 in a
  Marnie/Soldier development panel. Capping the extension at two Spidops scored
  14-10, tying the control, and an independent Marnie-conditioned confirmation
  tied its control 9-11. These routes were rejected.
- The archived broad backup-Spidops Bangle rule had regressed to 21-27 against
  the old rank-one control and remained rejected. The archived lethal-only
  Bangle rule had scored 24-24 versus a 23-25 baseline, so it was the sole
  prize-conversion rule combined with EarlyPad.

The frozen combined candidate scored 32-8 (80.0%) across eight side-swapped
official-engine games against each exact development bundle:

| Opponent | Record |
|---|---:|
| AshV178MarnieFullBest | 4-4 |
| LotusPX025IntentV2Control | 7-1 |
| SoldierPureRLV45 | 7-1 |
| SparkTrueCrustleCrispinGalvV71 | 6-2 |
| VioletAlakazamMKV93 | 8-0 |

All 40 games ended normally. Seat rows were 17-3 as seat 0 and 15-5 as seat 1.
This development result selected the candidate; it is not arena proof.

## Behavioral and package audit

The candidate preserves the exact deployed EarlyPad deck and model:

- deck rows: 60
- deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`
- model SHA-256: `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`
- main SHA-256: `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`

The trace audit reproduced 2,748/2,748 candidate decisions with zero
mismatches. EarlyPad fired 20 times. The lethal-Bangle rule fired seven times:
four against Marnie and three against Soldier. Five of the seven games with a
lethal-Bangle activation were wins. This establishes that V5 changes real
official-engine behavior, not that each activation caused the observed result.

Archive:
`artifacts/submissions/HelbrosRocketSpidopsEarlyPadLethalV5.tar.gz`

- bytes: 17,920,734
- members: 26 regular files
- extracted/source hash mismatches: 0
- archive SHA-256:
  `dddfe37f38df6aa5189c3e11a85cde7364ec9ad1742e61d5544cbf9f43300675`
- extracted archive completed a fresh terminal official-engine smoke

## Submission

The internal arena accepted the bundle into its inbox under
`HelbrosRocketSpidopsEarlyPadLethalV5`, from `Helbros`, for promotion on the
next worker job. No Kaggle submission or upload was made.

The submission remains unchanged while its rated record is collected. The
first full replay pull is due after roughly 50 completed games. The target is
complete only if this same immutable identity reaches at least 160 wins after
at least 200 rated games; otherwise the exact submitted replays must drive a
materially different successor identity.

## First immutable arena pull

The first exact pull captured 52 completed rated games and 3,803 focal
decisions. V5 was 35-17 (67.3%) and temporarily rank 1 by rating. Rank is not
the success metric; at this checkpoint the same identity would need to win at
least 125 of its next 148 games (84.5%) to reach 160-40 after 200.

| Public opponent family | Record |
|---|---:|
| Marnie | 12-11 |
| Rocket mirror | 7-2 |
| Alakazam | 8-0 |
| Starmie | 1-3 |
| Soldier | 2-0 |
| Crustle | 1-0 |
| Gardevoir | 1-0 |
| Trevenant | 1-0 |
| Other / Hariyama | 2-1 |

The sample was 31-12 when starting first and 4-5 when starting second. Twelve
of 17 losses never built a second ready attacker, six never attacked with
Mewtwo, and only one failed to reach four Rocket Pokemon in play. Opening
Wobbuffet went 6-0, while opening Articuno went 3-5; this is fresh evidence
against removing Wobbuffet as a blind one-card swap.

Live replay of the submitted runtime found 29 EarlyPad activations and five
lethal-Bangle activations. The five Bangle games went 4-1. The Windows replay
matched 3,799 decisions byte-exactly, one additional multi-select as a set,
and differed on three of 3,803 decisions, consistent with small cross-platform
neural tie/numerical drift. All five direct Bangle actions reproduced exactly.

Artifacts:

- replay pull: `work/arena_v5_pull_first50/replays/`
- ingested decisions: `work/arena_v5_pull_first50/training.jsonl`
- live autopsy: `work/arena_v5_pull_first50/autopsy/autopsy.json`
- guard replay audit: `work/arena_v5_pull_first50/v5_live_decision_audit.json`

V5 remains immutable and active toward 200 games. Any successor research must
target the demonstrated second-attacker, Marnie, Starmie, and second-player
deficits and must not revive the already rejected Wobbuffet swaps or earlier
Marnie routes without new evidence.
