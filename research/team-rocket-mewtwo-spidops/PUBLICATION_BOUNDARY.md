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
- The complete six-phase plan and neural game-theoretic pivot specification.

## Excluded

- `cg/`: official competition engine/runtime.
- `soldier/`: packaged inference support copied with the frozen source bot;
  redistribution rights were not established in this task.
- `model.pt`: binary frozen checkpoint. Its SHA-256 is
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`.
- `deck_library.json`: frozen opponent-prior artifact derived from local arena
  materials. Its SHA-256 is
  `ed56d55cec6e2a87415db422b72f16baa8ab552e87c625de75cd0d8cb7e8d4e0`.
- Official-engine packages, pulled opponent bundles, replay corpora, training
  examples, local caches, work directories, Kaggle downloads, and internal
  arena endpoints or tokens.
- The submitted `.tar.gz`; archive SHA-256:
  `8858324f993656691caa59700b968b88390e3edce94068df8dfb5cf28e455289`.

The exclusions mean `arena_bot/main.py` is an auditable source snapshot rather
than a standalone executable. The research package itself remains installable
and its dependency-free/core demos remain runnable with PyTorch.

## Claim boundary

Official-engine smoke validates execution and legality, not strength. Internal
arena records are empirical and opponent-pool dependent. Abstract games,
public-belief interfaces, counterfactual-value bounds, and re-solving APIs are
research components; they are not evidence that the submitted bot implements
a solved Pokemon TCG equilibrium strategy.

