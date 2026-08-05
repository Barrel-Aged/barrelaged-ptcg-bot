# Development pause checkpoint

Paused: 2026-08-05 (America/Denver)

## Final action

`HelbrosRocketSpidopsArticunoV7` was accepted into the internal arena inbox for
the next ladder job. Exactly one successor was submitted in this final pass.

- Package bytes: `17,923,621`
- Package SHA-256:
  `8858324f993656691caa59700b968b88390e3edce94068df8dfb5cf28e455289`
- Deck SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`
- Model SHA-256:
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`
- Main SHA-256:
  `d690d9f316180ebd87f5648257deacbaaf8a9bb7bd50a5231ed8b128f6be2484`

The source candidate, extracted package, and deterministic Articuno unit checks
all passed. Two one-game official-engine terminal smokes against
`AshV178MarnieFullBest` also completed. These are health checks, not a strength
estimate.

## Admitted logic

- Exact 60-card zero-Wobbuffet deck and frozen elite ScaleBC checkpoint.
- Early Poke Pad preference for a missing Spidops evolution through turn four.
- Brave Bangle attachment only when it converts legal Rocket Rush into an ex
  knockout.
- Matchup-conditioned fifth-bench-slot reservation for Team Rocket's Articuno,
  using only own hand and public opponent information.

## Rejected or deferred

- Joltik/Galvantula and Rocket Zapdos/Maximum Belt deck packages failed locked
  screens.
- The raw-action sampled teacher failed preregistered integrity, stability, and
  second-seat gates; no learned rollout change was admitted.
- Full public-belief solving, blueprint training, safe subgame re-solving, and
  constrained exploitation remain architecture/research interfaces rather
  than the final live controller.

If development resumes, start with the detailed
`methodology/PAUSED_DECK_REWORK_REVISIT_NOTE.md`, wait for a mature arena sample,
and continue the Wobbuffet-free deck search rather than restoring Wobbuffet.

## GitHub publication

The 742-file, 8,138,811-byte publication snapshot was posted to the existing
private competition repository as draft PR
<https://github.com/Barrel-Aged/barrelaged-ptcg-bot/pull/1> on branch
`agent/publish-mewtwo-spidops-research`. The initial publication commit was
`cb4c0e3aa459c2cc6c906faef39740010940c521`; a follow-up checkpoint-only commit
records this link. Critical remote blobs were compared with local Git object
hashes and matched byte-for-byte.
