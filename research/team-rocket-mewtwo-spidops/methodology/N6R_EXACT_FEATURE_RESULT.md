# N6R exact-feature belief-rollout result

Status: complete negative development result, 2026-07-28. No N6R student is
authorized for validation, packaging, or arena submission.

## What was valid

- Frozen Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Frozen base model SHA-256:
  `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`.
- Training: 399 exact actor-feature roots, 9,576 matched rollout sets, 28,728
  terminal continuations, 399/399 baseline reproduction, zero rollout errors.
- Independent selection: 240 roots, 5,760 matched sets, 17 admissible hard
  changes (7.08%), 42.08% split-half agreement, 240/240 baseline reproduction,
  zero rollout errors.
- The initial distillation invocation failed closed before producing a model.
  Its deployment-parity repair is recorded in
  `N6R_DISTILLATION_PARITY_REPAIR.md`.
- Four repaired students encoded all 399 training and 240 selection roots.
- Runtime parity: 2,041/2,041 strict-versus-packaged decisions matched across
  32 official-engine games, with zero mismatches.
- Development: 2,400 strict official-engine games, always second seat, 480 per
  arm, 240 per opponent, zero fallbacks, and four valid output manifests.

## Locked development result

| Arm | V110 delta | Recovery delta | Stratified delta | Bootstrap 95% | Pass |
| --- | ---: | ---: | ---: | ---: | :---: |
| A | -4.17 pp | +0.42 pp | -1.88 pp | [-8.13, +4.38] pp | No |
| B | -0.83 pp | -5.00 pp | -2.92 pp | [-9.17, +3.33] pp | No |
| C | +2.92 pp | -0.42 pp | +1.25 pp | [-5.00, +7.50] pp | No |
| D | -4.58 pp | -1.25 pp | -2.92 pp | [-9.17, +3.33] pp | No |

The control won 107/240 against V110 (44.58%) and 98/240 against Recovery
(40.83%). Arm C was the only positive aggregate arm, but it failed all three
strength conditions that matter here: at least +1.5 percentage points
aggregate, a strictly positive worst-opponent estimate, and a 95% lower bound
above -0.5 points.

The immutable analyzer output is
`artifacts/team_rocket_recovery/n6r_exact_feature/development_panel/analysis.json`.
It records `advances_exactly_one_or_none: []` and has no deployment authority.

## Interpretation and decision

Exact persistent state fixed the invalid N6 data boundary, but the resulting
sampled-rollout teacher still did not create a robust match-level improvement.
The V110/Recovery sign split in arm C is evidence that the learned correction
is opponent-policy sensitive, while the wide interval shows the effect is too
uncertain to route or submit safely.

No post-panel temperature, KL, epoch, checkpoint, opponent router, or hybrid is
selected from these outcomes. N6R artifacts are retained as negative research
evidence and excluded from all deployment claims. The current arena bot remains
the immutable control.

The next recovery experiment must be materially different: it should learn
from substantially more official match trajectories and direct match outcomes,
with opponent-policy diversity and a frozen independent evaluation design,
rather than trying another small distillation of the same 639 early-turn roots.

