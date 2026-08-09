# Helbros Mega Lucario V123 public-BC safe hybrid

This directory preserves every project-authored part of the final competition
candidate. The executable arena archive also contained copied inference,
baseline-policy, and official-engine runtimes; those are intentionally not
republished here. See `../PUBLICATION_BOUNDARY.md`.

## Method

1. Use the exact current public Majkel1337 Mega Lucario deck.
2. Start from the admitted ScaleBCv2 neural checkpoint and fine-tune the last
   relational-transformer block plus autoregressive action head on 172 public
   Majkel trajectories, with a frozen whole-game 80/10/10 split.
3. Keep V122's guard that suppresses `END` when a legal attack exists.
4. Run the live V121 deterministic Lucario clone in parallel as a safety
   provider.
5. Route to V121 only when the neural top-action probability is below
   `0.35228917002677923`, confidence is unavailable, or the neural action is
   illegal. Both policies receive every observation and reset per game.

The threshold was selected on the frozen validation games only. It improved
complete-action validation accuracy from 60.84% to 61.88% and untouched-test
accuracy from 57.52% to 58.27%. First-action test accuracy improved from 60.51%
to 61.47%. This is behavioral-imitation evidence, not an Elo measurement.

## Files

- `main.py`: exact authored arena entrypoint.
- `hybrid_policy.py`: exact authored confidence router and legality fallback.
- `nn_agent_confidence.patch`: authored change exposing calibrated confidence
  from the copied neural runtime.
- `deck.csv`: exact ordered 60-card deck IDs.
- `CALIBRATION_SUMMARY.json`: sanitized immutable calibration result.
- `PUBLIC_BC_TRAINING_SUMMARY.json`: exact V1 training/audit summary.
- `HELBROS_NOTES.md`: complete V122/V123 methodology and local results.

## Integrity

- `main.py`: `f70595002932dc43991195c99f07feb2f437a9dde50f373873ad7670ecd61959`
- `hybrid_policy.py`: `e993b970f0e283603a1f9c3d5ce33e296ef3fc49698733bfde405d333c291cdd`
- `deck.csv`: `5ddb7ca2790518e3c1eac6e2ff8b7fdb6ff0a817bf888536349a090ec7582a9f`
- excluded `model.pt`: `e80b691f50b82fbaecd61d3a9354cd727737a4e302d9a243fa5f91fcfffb8eea`
- excluded `deck_library.json`: `ed56d55cec6e2a87415db422b72f16baa8ab552e87c625de75cd0d8cb7e8d4e0`
- internal-arena archive: `1e9f32a1aa8c4725c424cfea42a4cc73915489a3555c9028797221d07bf2a975`

All four preregistered official-engine panels passed, and the extracted archive
completed two terminal smoke games. Internal arena acceptance is execution
evidence only; no direct Kaggle submission or verified 1000-Elo claim is made.
