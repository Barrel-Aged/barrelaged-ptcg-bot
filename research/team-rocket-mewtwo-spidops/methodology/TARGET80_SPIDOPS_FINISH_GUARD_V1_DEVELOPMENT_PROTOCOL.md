# Target-80 Spidops-finish guard V1 development protocol

Status: **prospectively locked 200-game direct development match; no deployment authority**  
Locked: 2026-08-05, before the development match.

## Hypothesis

The broad compact-rule V5 test failed and is closed. Its paired evidence still
identified a narrower public action pattern with unusually controlled downside:
when baseline would attach to a zero-energy Mewtwo or Spidops, redirecting the
engine plan's action to finish a different one-energy Spidops was positive on
24/25 openly analyzed roots and averaged about +0.31 at the end-of-own-turn
milestone. This is development evidence, not prospective confirmation.

V1 implements only that narrow MAIN action guard. It does not deploy the failed
broad V5 rule or make sampled rollout search the final strategy. The exact
frozen neural blueprint remains the policy everywhere else.

## Frozen arms

Both arms use the same Wobbuffet-free 60-card deck and same neural checkpoint:

- deck SHA-256:
  `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`;
- model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.

The control is the unchanged guarded package:

- tree:
  `30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f`;
- main:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`.

The candidate is:

- tree:
  `952fc72d0cb2eb39c122b0f29e3aee3dac98c3cc17dbc97238c22a7ca3eae70c`;
- main:
  `f7df55b3a09b5adbe27918bc91007848065d0bd19778a501404480cd97c11437`;
- frozen semantic runtime:
  `f6c3e2056eb143b6ad4bf537095aeb5b78703b0c9f4ffd789587279e8e1b85af`.

Early Poke-Pad-to-Spidops and lethal-only Brave-Bangle guards keep priority.
For an ordinary single-choice MAIN prompt, the candidate reproduces the frozen
greedy logits and uses the `engine_resilience` semantic overlay only to propose
an alternative. It changes the action exactly when:

1. baseline targets a zero-energy Mewtwo or Spidops;
2. the engine action attaches Grass, Psychic, or Team Rocket's Energy to a
   different Spidops with exactly one energy; and
3. the engine and baseline actions differ.

The guard records one stderr telemetry event per intervention. It reads only
the normal redacted observation, own visible hand, public board, legal options,
and the existing public/own-visible belief state.

## Excluded smoke

The completed four-game runtime smoke at
`work/target80_spidops_finish_guard_v1_smoke/smoke_abs.jsonl` ended 4-0 with
terminal results and no agent error. It contained zero guard interventions and
therefore supports legality/import only. It is excluded from every development,
confirmation, submission, and strength total.

## Locked development match

The official local evaluator runs exactly 200 direct candidate-versus-control
games, swapping sides every game:

- candidate name: `finish_guard`;
- control name: `control`;
- games: 200;
- maximum steps: 2,000;
- action timeout: 15 seconds;
- trace every decision: yes;
- output:
  `work/target80_spidops_finish_guard_v1_development/head_to_head_200.jsonl`.

The evaluator SHA-256 is
`d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303`.
It runs each bot in its own worker process while the parent owns the genuine
official `cg` engine. All 200 games must finish before outcomes or telemetry are
analyzed. Draws count as non-wins.

## Locked gates

V1 advances only if all gates hold:

1. exactly 200 games complete, with the candidate in each engine side exactly
   100 times;
2. candidate wins at least 110/200 (55%);
3. candidate win rate is at least 50% when starting first and at least 50% when
   starting second, as recovered from the decision trace;
4. at least five genuine guard interventions occur;
5. every game ends normally with zero agent, step-limit, or traceback errors;
6. record accounting is exact and draws remain non-wins.

Passing authorizes only a separately locked broad-opponent development panel.
Failure closes this exact guard. Neither result alone authorizes untouched
confirmation, packaging, arena submission, or a strength claim.

## Claim boundary

The match executes genuine official-engine games and legality. The semantic
overlay and action guard are research abstractions, and the direct match is a
development comparison against one identical-deck policy—not evidence of broad
matchup strength, equilibrium safety, or an 80% arena win rate.

## Frozen analyzer

The analyzer/gate SHA-256 is
`cf002e05c885b75979a45e88f5c9ccc7699738ef6ebecf8d142ab3430b0cac99`.
The analyzer checks live tree/deck/model/evaluator identities, exact side
alternation, trace-derived start order, intervention telemetry, execution
errors, and every gate above. It writes only after the full match seals.

Post-run mechanical amendment: see
`TARGET80_SPIDOPS_FINISH_GUARD_V1_ANALYZER_AMENDMENT.md`. The corrected
tree-hash implementation has SHA-256
`c5fab30abe08e14202ea11ff5336d523d61222b1f3bba5b73d88852cd6bae5b6`;
no scoring or gate logic changed.

The same amendment records a second pre-output trace-schema correction. The
final analyzer SHA-256 is
`ec99128a2b7ab310224e124acf41195e136db0c8e16ed06c9129eaed2a336623`.
