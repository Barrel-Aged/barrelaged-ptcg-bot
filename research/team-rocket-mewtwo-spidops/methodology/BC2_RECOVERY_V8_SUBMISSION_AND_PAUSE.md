# BCv2 recovery V8 submission and pause

Completed: 2026-08-06  
Status: **accepted by the internal arena; development paused**

## Outcome

Exactly one post-V7 identity was submitted:

- arena identity: `HelbrosRocketSpidopsBC2RecoveryV8`;
- arena acknowledgement: accepted, 17,923,962 bytes, queued for promotion on
  the next worker job;
- archive: `work/HelbrosRocketSpidopsBC2RecoveryV8.tar.gz`;
- archive SHA-256:
  `876ac11a1878267251844fad2fcae11fcad96f32a8f5b403ab16521e2c335e12`;
- deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- main SHA-256:
  `e9c2b0c59116845e6c11e520e8d47ca2f1476a32a69e6dad9fcd8da371dd4c9e`.

The submission is internal-arena only. No Kaggle upload or submission was made.

The first read-only post-promotion snapshot was 8-3 at rating 778.4. Eleven
games are enough to verify live scheduling, not enough for a stable strength
estimate; preserve the identity unchanged if a mature estimate is requested.

## Why BCv2 outperformed the recent variants

The evidence does not support a one-card explanation.

1. **Exact distribution alignment mattered.** V5 preserved BCv2's exact deck
   and model and reached 1,411-1,113 at rating 822.7 in the pre-submission
   read-only arena snapshot. The later variants changed deck composition and
   model together.
2. **The Hancang/V7 rebuild lost Rocket density.** Its Mimikyu openings were
   483-645 (42.82%) in the fresh 4,141-game pull, and non-Rocket Mimikyu also
   reduced Rocket Rush's board multiplier.
3. **Setup continuity regressed.** V7 was 2,162-1,979 (52.21%); 80.14% of its
   losses never formed a second ready attacker. The archived V5 loss rate for
   that bucket was 62.34%.
4. **The matchup pools differed.** BCv2's 20 Kaggle games were 8-0 across
   Alakazam and Mega Lucario, while the mature V7 arena pull contained 648
   Alakazam and 699 Starmie games. V7 was only 37.04% and 34.48% in those two
   cells. A Kaggle score and arena rating therefore are not interchangeable.
5. **Wobbuffet was a bad primary attacker but not proven to be a bad slot.** It
   made two of 103 Kaggle attacks, both bad Headbutt selections in one loss.
   Yet it was also a searchable Basic Team Rocket body. Every completed direct
   or package-level removal experiment introduced larger setup tradeoffs.

The strongest defensible conclusion is that BCv2 benefited from a coherent,
deck-matched policy and dense Rocket setup. The data do not show that
Wobbuffet itself created the strength.

## Rejected zero-Wobbuffet recovery arm

The preregistered first V8 arm replaced Wobbuffet with a third Murkrow and
retained the Articuno reservation. It was runtime healthy and scored 14-10
against V5 and 14-10 against V7, but failed the broad gate at 0-4 against
Marnie and 0-4 against Starmie (4-0 against Alakazam). It was not packaged or
submitted.

That result supersedes the earlier assumption that the next deployable build
must remove Wobbuffet. Zero-Wobbuffet research remains reasonable, but it now
requires a genuinely deck-matched policy and a package that passes Marnie and
Starmie gates; it should not be treated as an already validated fix.

## Submitted V8

V8 restores the exact BCv2/V5 deck, model, deck library, official-engine
adapter, and legality path. It combines only disclosed public-state guards:

- early Poke Pad search for a missing Spidops through turn four;
- Brave Bangle only when it converts Rocket Rush into an exact ex knockout;
- matchup-conditioned Articuno fifth-bench-slot reservation, including
  withholding the lone Wobbuffet from that last slot;
- redirect selected Headbutt Bounce to an offered retreat only when a Benched
  Spidops is already ready. Rocket Mirror is never intercepted.

The Wobbuffet exit guard has exact archived action-surface evidence but no
fresh causal win proof. It changed the two audited Kaggle Headbutt decisions
and activated zero times in the fresh final screen.

## Verification

- deterministic deck/model and five guard cases: passed;
- source official-engine smoke: terminal;
- final direct V5 cell: 9-7, 16/16 terminal;
- final direct V7 cell: 9-7, 16/16 terminal;
- official trace replay: 2,510/2,510 exact decisions, zero mismatches;
- observed final-screen guard activations: 14 early-Spidops, one forced
  Articuno, zero lethal-Bangle, zero Articuno-mask, zero Wobbuffet-exit;
- archive safety: 27 files, zero unsafe members, zero source/extracted hash
  mismatches;
- extracted-archive official-engine smoke: terminal.

The 9-7 cells are health filters, not verified strength gains. A mature arena
record did not exist at submission time and V8 games must not be pooled with
BCv2, V5, V6, or V7.

## Pause boundary

Development is paused after this submission and publication handoff. Do not
start another local experiment, arena identity, monitor, training run, Kaggle
submission, or deck mutation without a new user request. The arena may continue
to schedule the immutable V8 identity.

## GitHub handoff

The publication-safe source and methodology snapshot was updated on branch
`agent/publish-mewtwo-spidops-research` in the existing private draft PR:
<https://github.com/Barrel-Aged/barrelaged-ptcg-bot/pull/1>. The PR now contains
748 changed files relative to its empty research base. Sixteen final/new remote
blobs were compared with the local Git object hashes and matched exactly. The
official engine, packaged third-party runtime, model binary, opponent priors,
private telemetry/replays, credentials, and submitted archive remain excluded
under `PUBLICATION_BOUNDARY.md`.
