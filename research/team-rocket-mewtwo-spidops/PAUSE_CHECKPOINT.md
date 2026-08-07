# Development pause checkpoint

Paused: 2026-08-06 (America/Denver)

## Final action

`HelbrosRocketSpidopsBC2RecoveryV8` was accepted into the internal arena inbox
and promoted to the live ladder. Exactly one post-V7 successor was submitted.

- Package bytes: `17,923,962`
- Package SHA-256:
  `876ac11a1878267251844fad2fcae11fcad96f32a8f5b403ab16521e2c335e12`
- Deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`
- Model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`
- Main SHA-256:
  `e9c2b0c59116845e6c11e520e8d47ca2f1476a32a69e6dad9fcd8da371dd4c9e`

The source candidate, extracted package, deterministic guard checks, and two
fresh official-engine terminal smokes passed. The final direct cells were 9-7
against V5 and 9-7 against V7; all 32 games were terminal. A 2,510-decision
replay audit had zero mismatches and observed one forced Articuno play. These
are health checks, not a mature strength estimate.

## Admitted logic

- Exact 60-card ScaleBCV2/V5 deck and frozen ScaleBC checkpoint.
- Early Poke Pad preference for a missing Spidops evolution through turn four.
- Brave Bangle attachment only when it converts legal Rocket Rush into an ex
  knockout.
- Matchup-conditioned fifth-bench-slot reservation for Team Rocket's Articuno,
  using only own hand and public opponent information.
- Headbutt Bounce exit to an offered retreat only when a Benched Spidops is
  already ready; Rocket Mirror is never intercepted.

## Rejected or deferred

- The final zero-Wobbuffet third-Murkrow arm went 14-10 against both V5 and V7
  but failed 0-4 against Marnie and 0-4 against Starmie, so it was not
  submitted.
- Joltik/Galvantula and Rocket Zapdos/Maximum Belt deck packages failed locked
  screens.
- The raw-action sampled teacher failed preregistered integrity, stability, and
  second-seat gates; no learned rollout change was admitted.
- Full public-belief solving, blueprint training, safe subgame re-solving, and
  constrained exploitation remain architecture/research interfaces rather
  than the final live controller.

The first read-only live snapshot was 8-3 at rating 778.4. That is useful
confirmation that the exact package is executing, but far too small for a
stable performance claim. If development resumes, start with
`methodology/BC2_RECOVERY_V8_SUBMISSION_AND_PAUSE.md` and the detailed
continuation note; do not assume either retaining or removing Wobbuffet is
already proven optimal.

## GitHub publication

The updated snapshot is published in the existing private draft PR
<https://github.com/Barrel-Aged/barrelaged-ptcg-bot/pull/1> on branch
`agent/publish-mewtwo-spidops-research`. It contains 748 changed files relative
to the repository base. Sixteen final/new remote blobs were compared with the
local Git object hashes and matched exactly; the documented private and
third-party exclusions remain absent.
