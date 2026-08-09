# Final extracted-archive audit

Audit date: 2026-07-27

Audit scope: the submitted `HelbrosMewtwoSpidopsGTWTV1.tar.gz` archive extracted
under `work/final_archive_audit_20260727_v1`, not merely the source checkout.
“Independent” here means an archive-level verification independent of the build
tree and training process; it does not imply review by a separate person.

## Verdict

**PASS — faithful submission; FAIL — superhuman strength.**

The archive executes the requested belief-aware neural plan controller and the
game-theoretic safety interfaces on the official action path. It is not a
heuristic-only or sampled-rollout search bot. All integrity, legality,
serialization, reset, deterministic-replay, runtime-parity, and arena-execution
checks pass. Therefore the automatic “restart if the submission is not faithful”
condition is not triggered.

The arena result is poor and the learned strategic outputs are highly
collapsed. The audit specifically rejects any claim that the bot is superhuman,
leaderboard-competitive, equilibrium-derived, or an exact Prism Tower deck.

## Artifact identity and reproducibility

| Check | Observed result | Verdict |
|---|---|---|
| Submitted archive size | 2,924,898 bytes | Pass |
| Submitted archive SHA-256 | `01c02185df68ae161d5883bbca2cb32431023e721e8064673f83200b548dc315` | Pass |
| Clean rebuild SHA-256 | Same `01c02185…dc315` | Pass |
| Archive paths | 18 members; no absolute or `..` paths | Pass |
| Manifest hashes | 17/17 payload hashes match | Pass |
| Frozen model fingerprint | `de6db2bc87a17ddaabe04ac626a4694ef69e69f8753e0f6b53d10c6dcbcafe4d` | Pass |
| Frozen model SHA-256 | `72d41b534610601ae8cdecc1db96f322cc639d12f71eaedbf5ad2f157d941b31` | Pass |
| Text secret/path scan | No user path, arena token, API key, authorization, or bearer pattern found | Pass |

The post-evaluation model fingerprint differs from the one recorded by the
240-game evaluation only because the rejected exploitation report was appended
as non-gameplay evidence. Removing precisely that evidence entry reconstructs
the evaluated fingerprint
`0a8c2d7a0958b9f965716e5f471f827ca258582bf8d4a6877a96a3f7ef96427c`.
All gameplay-bearing fields are unchanged; see
`artifacts/integrated_evaluation/deployed_artifact_binding_report.json`.

## Entry-point and strategy-provider trace

The extracted `main.py` performs this chain:

1. resolve the 60-card official compatibility deck;
2. instantiate the bundled ProV3 policy as `TACTICAL_PROVIDER`;
3. load `integrated_planner.json` into `IntegratedBeliefPlannerAgent`; and
4. dispatch every observation through `BOT.choose`.

Inside `BOT.choose`, the live path:

1. updates the public history and persistent plan memory;
2. creates relational state/card tokens;
3. evaluates six-archetype, six-intent, and ten-resource beliefs;
4. constructs multiple hidden-resource alternatives;
5. computes five plan priors/values, six strategic values, meta-weights,
   aggregate value, uncertainty, and the selective-search gate;
6. invokes the fail-closed bounded resolver when admitted;
7. scores legal actions autoregressively from neural state, plan, belief, token,
   and prefix representations; and
8. validates the ordered result against the official selection contract.

The tactical provider may replace only a low-confidence exact action sequence
under the admitted 0.68255 threshold. It does not choose beliefs, plans, values,
meta-weights, gate decisions, or the safe-resolve mixture. The manifest says
`teacher_is_strategy_provider: false`. The sampled-rollout controller is absent
from the packaged final path.

Verdict: **faithful neural strategy trace passes**.

## Deck audit

The deck query, `DECK.card_ids`, and `deck.csv` are identical and contain 60
cards. The research source remains the user-specified 15 Pokémon, 34 Trainer,
11 Energy list. The official engine cannot resolve Prism Tower, so the archive
contains the documented substitution:

- 2 Prism Tower → 2 Team Rocket's Watchtower.

The manifest marks `exact: false`, `playable: true`, and records both the exact
source fingerprint and compatibility fingerprint. The archive and bot name use
the `GTWT` compatibility label.

Verdict: **integrity and disclosure pass; exact-deck claim prohibited**.

## Reset, serialization, and deterministic behavior

- After one extracted-archive decision, the summary reported decision index 1.
- Calling `main.agent({"select": null})` returned all 60 deck IDs and reset the
  decision index to 0.
- Model summaries, plan memory, belief ranges, hidden hypotheses, and telemetry
  JSON-serialize.
- The model rejects a tampered layer fingerprint.
- PyTorch training math and the dependency-free runtime match within
  `3.55e-15` maximum absolute error.
- All 1,382 decisions in 30 pulled arena replays reproduce exactly from the
  independently extracted archive.

Verdict: **pass**.

## Official legality and runtime execution

The extracted archive completed 12 deterministic official-engine smoke games:

- 12 games, 8 wins against the smoke baselines;
- zero engine errors;
- zero runner illegal fallbacks;
- zero controller exceptions;
- 17 manifest hashes verified; and
- eight hidden-resource hypotheses retained by the final smoke game.

The broader frozen-artifact evaluation contains 240 official games, again with
zero illegal fallbacks, controller exceptions, or engine errors. It recorded all
six requested evaluation categories and 12 full replays.

The internal arena independently executed the submitted archive for at least 30
completed games. Arena replays contain the expected compatibility deck and
match the local deterministic controller decision-for-decision.

Verdict: **genuine official-engine execution passes**.

## Test and canary audit

`python scripts/run_tests.py` completed **225 tests in 125.629 seconds: OK**.
The suite covers deck integrity, selection legality, state/history round trips,
public/private redaction, hidden-zone canaries, belief normalization, persistent
plans, all six value heads, meta aggregation, action masking, information-set
provenance, counterfactual calibration, fail-closed resolving, exploit support
constraints, artifact tamper rejection, and archive manifest/entry-point checks.

Verdict: **pass**.

## Game-theoretic safety audit

- No equilibrium is claimed.
- Official counterfactual bounds are a scoped empirical split-conformal guard,
  not a theorem about exploitability.
- Untouched marginal coverage is 95.56%; worst listed archetype coverage is
  83.33%; radius is 0.3824.
- The selective gate passed its fresh concentration protocol only after two
  rejected iterations.
- In the local component sample, full safe resolving beat `no_safe_resolve`
  16–4; the sample is small and not a general claim.
- The exploitation overlay candidate failed 23–37 and was correctly disabled at
  weight zero.

Verdict: **safe admission/rejection behavior passes within its stated empirical
scope; equilibrium and universal safety remain unproven**.

## Strength and loss audit

The internal-arena snapshot is 4–26 at rating 355.2. Its 13.33% observed win
rate has a 95% Wilson interval of approximately 5.31%–29.68%. Replay review
shows:

- 24/26 losses ended behind in the prize race;
- 9 losses contained no attack;
- 1,362/1,382 decisions named tempo as the dominant strategic axis;
- 1,235/1,382 selected `board_setup`;
- only 15/620 safe-resolve attempts changed the plan mixture; and
- zero runtime or legality faults explain the losses.

Verdict: **strength fails**. The failure is learned strategic/tactical quality
and distribution generalization, not the submission pipeline.

## Official versus abstract boundary

Genuine official behavior:

- card pool and deck validation;
- native game transitions and outcomes;
- legal option generation and ordered selection execution;
- local and arena match records; and
- captured public observations and replays.

Research abstractions:

- the six-archetype and six-intent opponent range;
- hidden-resource category hypotheses;
- five strategic plans and six strategic value labels;
- learned plan/action value surfaces;
- empirical counterfactual intervals; and
- the selective safe-resolve mixture.

No abstract output is represented as a native engine fact.

## Final pass/restart decision

The submitted archive is faithful to the plan's intended controller and safety
pivot, so the mandated automatic restart condition is **not met**. The completed
run should be frozen as a negative but useful benchmark. A successor should be
a new, preregistered training cycle that must beat this arena snapshot and
demonstrate non-collapsed plan/meta behavior before promotion.
