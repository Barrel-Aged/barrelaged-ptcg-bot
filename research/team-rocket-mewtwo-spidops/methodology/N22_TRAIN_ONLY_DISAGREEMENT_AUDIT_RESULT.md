# N22 Train-Only Disagreement Audit Result

Status: **feasibility gate passed; no development or deployment authority**  
Protocol: `rocket-n22-train-only-disagreement-audit/v1`

## Integrity and boundary

- protocol SHA-256:
  `337d2a37dda170e0c856532a398bf7d26ad7be4e7d94a3adee31ece11edf22d3`;
- execution-lock SHA-256:
  `5113a88ede0df9522b592e2a909c9b0b57a5d577e656b5f9fa2dda21a4c073cc`;
- audit script SHA-256:
  `e5377d671fabc1141e28af05cea4b8ef7f418fad843822c5b0f61d988c1d8c56`;
- audit JSON SHA-256:
  `1dd3904da69914c322fe6d4b0f4337d1a4ed6146533519353431c9f70c999dd8`.

The sole audit loaded N19 `teacher/train` and `retention/train` only, plus the
historical N9 preservation support. It loaded zero N19 validation examples and
zero N19 test examples. It performed no training, model mutation, official
match, build, packaging, or submission.

## Disagreement evidence

| Training evidence | Decisions | Live/N21 disagreements | Candidate-only recorded corrections | Live-only recorded corrections | Neither |
|---|---:|---:|---:|---:|---:|
| Fresh leader teacher | 9,483 | 646 (6.81%) | **503** | 69 | 74 |
| Exact-Rocket retention | 6,240 | 48 (0.77%) | 2 | **46** | 0 |
| Historical N9 Rocket | 6,753 | 130 (1.93%) | 10 | **116** | 4 |

The key separation was N21 confidence on its replacement action:

- teacher-disagreement candidate margin: median **0.2017**, p75 **0.4539**;
- exact-Rocket retention: median **0.0435**, p75 **0.0709**;
- N9 Rocket: median **0.0227**, p75 **0.0654**.

At the fixed candidate-margin thresholds:

| Minimum N21 margin | Teacher candidate-only / live-only | Exact retention overwrites | N9 overwrites |
|---:|---:|---:|---:|
| 0.10 | 366 / 39 | 8 | 19 |
| 0.25 | 237 / 16 | 2 | 1 |
| **0.50** | **124 / 2** | **0** | **0** |
| 1.00 | 35 / 0 | 0 | 0 |

The preregistered feasibility gates passed: there were at least 100
candidate-only teacher corrections, and multiple fixed public buckets exceeded
25 corrections and 70% correction precision while also occurring in Rocket
disagreement evidence. The result is not evidence that those states are
valuable for the Team Rocket deck.

## Decision

One qualitatively new, deterministic composition may be preregistered: retain
the exact live action unless N21 disagrees and its deployed top-versus-runner-up
margin is at least `0.50`. This changes the mechanism from global model transfer
or loss-weight tuning to a state-conditioned all-or-live fallback.

The threshold must be frozen before opening N19 validation/test states. The
held-out experiment must require both leader-signal transfer and nonzero
exact-Rocket relevance; a gate that merely detects the leader's different deck
is not eligible for official-engine testing. `development_authority=false` and
`deployment_authority=false` remain unchanged.
