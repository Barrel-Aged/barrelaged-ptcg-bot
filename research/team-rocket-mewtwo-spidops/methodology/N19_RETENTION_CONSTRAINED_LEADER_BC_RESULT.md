# N19 Retention-Constrained Leader BC Result

Status: **offline rejected; no match panel; no deployment**  
Protocol: `rocket-n19-retention-constrained-leader-bc/v1`

## Decision

The sole terminal epoch-4 model failed the preregistered exact-Rocket retention
gates.  N19 stops offline.  The N6/N9 anchor audit, official-engine development
panel, confirmation, packaging, and submission are not authorized.

The live bot remains unchanged:

- bot: `HelbrosRocketSpidopsScaleBCV2`;
- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

## Execution integrity

The one locked run completed exactly four epochs and 396 optimizer steps under
seed `819001`.  Only the final transformer block and action head were trained:
14 tensors and 1,184,257 parameters.  The other 45 tensors remained byte-exact
to live.  The serialized model reloaded exactly, all values were finite, and
the source corpus, deck, model, scripts, and protocol passed their frozen hash
checks.

No checkpoint, seed, loss coefficient, learning rate, or interpolation factor
was selected after observing results.

## Teacher learning

The model learned a substantial, independently held-out portion of the fresh
leader behavior:

| Split | Live top-1 | N19 top-1 | Change | Live loss | N19 loss |
|---|---:|---:|---:|---:|---:|
| train | 77.12% | 85.61% | +8.49 pp | 0.6649 | 0.4639 |
| validation | 76.30% | 84.42% | +8.12 pp | 0.7132 | 0.5021 |
| test | 77.59% | 86.29% | +8.70 pp | 0.6617 | 0.4698 |

All four preregistered teacher improvement gates passed.  This confirms that
the fresh recorded leader actions contain a learnable signal that generalizes
to the frozen teacher validation and test games.

## Retention failure

The trust-region KL values remained numerically small, but the top legal action
changed too often:

| Fresh exact-Rocket split | Top-1 agreement with live | Mean KL | Required top-1 |
|---|---:|---:|---:|
| train | 97.84% | 0.00229 | at least 99.5% |
| validation | 97.05% | 0.00282 | at least 99.5% |
| test | 97.00% | 0.00344 | at least 99.5% |

Recorded-action retention also regressed:

- validation fell from 96.63% to 93.69%;
- test fell from 96.88% to 94.38%.

Consequently, the base-policy top-1 gate and both recorded-action retention
gates failed.  A small average KL is not a sufficient safety proxy near action
boundaries: modest distribution movement can flip the selected action in about
3% of exact-Rocket decisions.

## Interpretation and closure

N19 separates two facts:

1. current-leader public actions are learnable by the live relational model;
2. the fixed four-epoch update is not safe enough for the exact Rocket policy.

The terminal model may not be match-tested, checkpoint-selected, rescaled,
seed-tuned, or deployed under N19.  Epochs 1-3 were diagnostic history only;
the protocol made epoch 4 the sole eligible model, and none of those earlier
states was serialized as an eligible candidate.

Any future experiment must be a new preregistered hypothesis.  A defensible
next question is whether a source-to-terminal trust-region projection can keep
the independently held-out teacher gain while restoring the required 99.5%
exact-Rocket top-choice agreement.  Such a projection must freeze its scalar
grid, selection rule, held-out gates, N6/N9 audit, and single downstream
candidate before examining official-match outcomes.  It cannot retrospectively
turn N19 into a pass.

## Durable artifacts

- protocol SHA-256:
  `006fd34be9d3770ed8025c2db2a5c0f21c6aa5f2e4dbd4baa9cde50ad37e153e`;
- execution lock SHA-256:
  `5128ad4bb4dc7b0f47ceeb40c9e1cc98fffe6cd3c50ebef044e87e0fd1dafdc9`;
- trainer SHA-256:
  `3928abaf17457d05e02c7d9734f69f819705601051154bc78236a981868de7b7`;
- rejected terminal model SHA-256:
  `b0f0e72f7eafbf2bd5e0e4eca1920398534ddef516b57750a3c1f58dbb7d467f`;
- training report SHA-256:
  `ed88d68c63988e79b74e0c33966bcdfe219cabfb78d9b0fe19f1482df7ebbe43`;
- training provenance SHA-256:
  `9a0e452c30ef6675f6c6635933b819119ca107e14a1539c3adac54027dd0861d`.

## Claim boundary

N19 provides supervised offline evidence only.  It proves neither that the
teacher changes are strategically beneficial for Team Rocket nor that a
projected model will be stronger.  No official-match, leaderboard, rank,
superhuman, confirmation, or deployment claim is made.
