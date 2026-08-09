# Majkel Lucario Public-BC V3 Safe-Hybrid Protocol

Frozen: 2026-08-08, before V3 calibration or official-engine outcomes were inspected.

## Motivation

V122 materially improved public-leader imitation and beat the two local same-deck controls, but its first 31 internal-arena games were only 15-16 (654.5 rating). Losses clustered against `LotusPX026MajkelBCV17` (2-6) and Team Rocket ScaleBCv2 (1-3). The fully fine-tuned V2 checkpoint failed its preregistered held-out top-1 gate and is permanently rejected.

V3 therefore preserves the admitted V122 neural model and exact Majkel deck. It adds the already-live V121 deterministic Lucario clone as a separately evaluated safety baseline. This is a policy-composition experiment, not a new training run and not a way to revive V2.

## Frozen candidate

- Neural policy: exact V122 model, SHA-256 `e80b691f50b82fbaecd61d3a9354cd727737a4e302d9a243fa5f91fcfffb8eea`.
- Safety policy: exact extracted V121 `LucarioPromotionClonePolicyV17` runtime.
- Deck: exact current public Majkel1337 60-card Mega Lucario deck, unchanged from V122.
- Both policies receive every decision observation and keep independent per-game state.
- If the neural policy fails to produce a confidence value or legal action, use the safety action.
- Otherwise use the safety action only when neural top-action probability is below a frozen scalar threshold.
- Keep V122's attack-safe guard that suppresses `END` when a legal attack exists.

## Threshold selection

Reuse V1's frozen whole-game split without modification. On the 17 validation games only:

1. Replay the uniquely identified Majkel1337 seat with Kaggle's required one-step action alignment.
2. Record the complete neural action, complete safety action, and neural top-action probability for each supported non-trivial decision.
3. Scan the observed confidence cut points plus the no-fallback endpoints.
4. Select the threshold with the highest complete-action validation accuracy; break ties by fewer safety interventions, then by the lower threshold.

The 18 frozen test games remain untouched until the threshold is selected.

## Offline admission gate

V3 advances only if all conditions hold:

- validation complete-action accuracy is strictly higher than V122 neural alone;
- frozen-test complete-action accuracy is no lower than V122 neural alone;
- frozen-test first-action accuracy is no lower than V122 neural alone;
- the selected threshold causes at least one neural/safety disagreement to route to safety;
- all actions are legal and deterministic on repeat.

If the validation optimum is the no-fallback policy, V3 is rejected as non-material.

## Official-engine admission gate

Only after offline admission, run side-swapped official-engine panels with fixed seeds and terminal-result checks:

- V3 versus V122 parent: at least 7-5 over 12 games;
- V3 versus V121 safety baseline: at least 7-5 over 12 games;
- V3 versus Team Rocket ScaleBCv2: no worse than V122's structural-counter result by more than one game over a matched 12-game panel;
- V3 versus Starmie CFV13: at least 6-6 over 12 games;
- archive-extracted smoke: two terminal games, no load error, no illegal action, no timeout.

Failure rejects V3. Gates will not be weakened after outcomes are known.

## Publication rule

An admitted V3 may replace the weaker active Helbros Lucario slot in the internal arena. It is not authorized for direct Kaggle submission. The user must manually promote it after sufficient internal evidence. Regardless of admission, preserve the full code, frozen protocols, data provenance, and honest results in the final GitHub handoff; do not claim 1000 Elo or leader equivalence without verification.
