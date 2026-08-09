# Publication boundary

This snapshot is intended to preserve all project-authored bot code and
methodology without publishing credentials, private evaluation data, or code
whose redistribution rights were not established.

## Included

- The complete `rocketbot` research package under `src/rocketbot/`.
- All project tests, scripts, design documents, preregistrations, execution
  locks, analyzers, and result reports.
- The final authored arena entrypoint, exact 60-card deck ID file, and training
  and structural provenance records.
- The final authored Lucario confidence router, entrypoint, exact deck, frozen
  calibration summary, and public-BC training summary.
- The complete six-phase plan and neural game-theoretic pivot specification.

## Excluded

- `cg/`: official competition engine/runtime.
- `soldier/` and V121 `bot/`: packaged inference and deterministic baseline
  runtimes copied from prior bundles; redistribution rights were not
  established in this task. The authored V3 changes around those runtimes are
  preserved in `competition_bot_v123/`.
- Team Rocket `model.pt`: SHA-256
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Lucario V123 `model.pt`: SHA-256
  `e80b691f50b82fbaecd61d3a9354cd727737a4e302d9a243fa5f91fcfffb8eea`.
- `deck_library.json`: frozen opponent-prior artifact derived from local arena
  materials. Its SHA-256 is
  `ed56d55cec6e2a87415db422b72f16baa8ab552e87c625de75cd0d8cb7e8d4e0`.
- Official-engine packages, pulled opponent bundles, replay corpora, training
  examples, local caches, work directories, Kaggle downloads, and internal
  arena endpoints or tokens.
- The submitted `.tar.gz`; archive SHA-256:
  `876ac11a1878267251844fad2fcae11fcad96f32a8f5b403ab16521e2c335e12`.
- The V123 submitted internal-arena archive; SHA-256:
  `1e9f32a1aa8c4725c424cfea42a4cc73915489a3555c9028797221d07bf2a975`.

The exclusions mean `arena_bot/main.py` and `competition_bot_v123/main.py` are
auditable source snapshots rather than standalone arena bundles. The research
package itself remains installable and its dependency-free/core demos remain
runnable with PyTorch.

## Claim boundary

Official-engine smoke validates execution and legality, not strength. Internal
arena records are empirical and opponent-pool dependent. Abstract games,
public-belief interfaces, counterfactual-value bounds, and re-solving APIs are
research components; they are not evidence that the submitted bot implements
a solved Pokemon TCG equilibrium strategy.
