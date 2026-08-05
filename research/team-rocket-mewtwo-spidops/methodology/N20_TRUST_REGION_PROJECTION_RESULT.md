# N20 Trust-Region Projection Result

Status: **offline rejected; no selected model; no deployment**  
Protocol: `rocket-n20-trust-region-projection/v1`

## Decision

No preregistered scalar passed all teacher and exact-Rocket safety gates.  N20
ends offline.  No candidate model was serialized, and no official-engine panel,
confirmation, packaging, or submission is authorized.  The live bot remains
unchanged.

## Execution integrity

The evaluator recomputed both diagnostic endpoints and all seven eligible
alphas.  Every point used:

- the frozen live source and rejected N19 epoch-4 terminal;
- the unchanged disjoint N19 teacher and fresh Rocket splits;
- full relational forward passes on 469 frozen N6 unchanged anchors;
- full relational forward passes on all 6,753 frozen N9 examples;
- both N9 opponent groups separately;
- exact model, deck, feature, corpus, and support-script hash validation.

No cached norm2-only shortcut, new grid point, tensor-specific scale,
checkpoint, training update, or official-match outcome entered selection.

## Fixed-grid frontier

| Alpha | Teacher validation top-1 | Gain vs live | Minimum fresh retention | N6 retention | N9 retention | Pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.000 diagnostic | 76.30% | 0.00 pp | 100.00% | 100.00% | 100.00% | ineligible |
| 0.125 | 77.81% | +1.51 pp | 99.58% | 100.00% | 99.70% | no |
| 0.250 | 78.81% | +2.51 pp | 99.16% | 100.00% | 99.30% | no |
| 0.375 | 79.82% | +3.52 pp | 98.88% | 100.00% | 99.02% | no |
| 0.500 | 80.65% | +4.36 pp | 98.60% | 99.79% | 98.80% | no |
| 0.625 | 81.99% | +5.69 pp | 98.00% | 99.79% | 98.47% | no |
| 0.750 | 83.08% | +6.78 pp | 97.88% | 99.36% | 98.25% | no |
| 0.875 | 83.84% | +7.54 pp | 97.19% | 99.15% | 97.93% | no |
| 1.000 diagnostic | 84.42% | +8.12 pp | 97.00% | 98.51% | 97.53% | ineligible |

The two boundary points explain the rejection precisely:

- `alpha=0.125` passed fresh retention, N6, and N9.  Fresh retention was
  99.79% train, 99.58% validation, and 100.00% test.  Teacher validation/test
  gains were only +1.51/+1.41 pp, so the locked +2.0 pp validation gate failed.
- `alpha=0.250` passed the teacher gates with validation/test gains of
  +2.51/+2.90 pp.  Fresh retention was 99.50% train, 99.16% validation, and
  99.38% test; validation and test missed the 99.5% top-choice gate, and both
  recorded-action retention gates also failed.

No point on the frozen grid simultaneously clears the teacher and retention
frontiers.  The largest-safe-step rule therefore selects nothing.

## Interpretation and closure

The N19 direction is smooth and useful for teacher imitation, but its global
weight-space projection has an unfavorable tradeoff near the required safety
boundary.  Moving far enough to achieve the preregistered held-out teacher gain
changes too many exact-Rocket top actions; moving little enough to preserve
those actions does not learn enough teacher behavior.

N20 may not be rerun with a finer scalar grid, relaxed threshold, selected
endpoint, or post-hoc interpolation.  The source-terminal global projection
family is closed.

A future experiment requires a qualitatively different constraint, such as a
top-action margin objective that directly protects the live winner on exact
Rocket states while allowing orthogonal teacher updates.  Such work must be a
new preregistered training run with its own held-out and N6/N9 gates; it cannot
reinterpret N19 or N20 as a pass.

## Durable artifacts

- protocol SHA-256:
  `8f4628bf07f5a1abf21ba132bc9c7a7705031652a9d74560adbb509e1a9e952a`;
- execution lock SHA-256:
  `557c8855526be9eb5134b530a52ee7a10f2ebcd9e3f363441047a584a90bcbfe`;
- projection evaluator SHA-256:
  `d612058c08f6c2dfbffd0f444300d95bd57e5d1d82c4718c881eab79ed0ac6f1`;
- projection report SHA-256:
  `83f8094ecfacbeb6c73caaf899af2fb7cbbfc9779ce5f4ec9d77bb23218319e0`;
- projection provenance SHA-256:
  `b10826e8db43d8df29a0241e249d69003f66a3be1b5423861175536e41d098fd`.

## Claim boundary

N20 is an offline fixed-grid retention result only.  It provides no
official-match, strength, leaderboard, rank, confirmation, superhuman, or
deployment evidence.
