# N23 High-Margin Gate Result

Status: **rejected offline; no runtime build, official panel, or deployment**  
Protocol: `rocket-n23-heldout-high-margin-gate/v1`

## Integrity correction

The original N23 execution is quarantined. Its evaluator used an unstable
NumPy tie order instead of the deployed first-index `torch.argmax` rule, and
its preregistered live-reproduction gate correctly failed. Quarantined output
SHA-256:
`873efb37a8652ceb03935af461883bb9abc50f1bb360a9199d295011559966fb`.

The correction was frozen before rerun:

- tie-correction amendment SHA-256:
  `99afca5165703ad1b60882486cba67021a776a160025a5a70305535f5c4d59c7`;
- corrected train-only audit SHA-256:
  `4a33b1226829e760231c48477f227c8c78429b7d4d9cbc41e90377f4357238e4`;
- corrected held-out execution-lock SHA-256:
  `89d02a8f151efe9efe6d7e01c14f032f8ac28101dcea329c021dc70b951d6d52`;
- corrected evaluator SHA-256:
  `ea263cffbbf8320c5f193569caa1d36329031d78281871ee2d12e85ea77f6519`;
- corrected evaluation SHA-256:
  `64015c346fd133f196898ad37eff344a4a4f674efed230525643f025f4c5dc15`.

The corrected evaluator exactly reproduced every frozen live top-1 and loss
metric. The threshold remained `0.50`, all admission gates remained unchanged,
and no held-out result was used to modify the rule.

## Held-out signal

The high-margin rule transferred the cross-deck teacher signal cleanly:

| Teacher split | Live top-1 | Gated top-1 | Change | Replacements | Candidate-only / live-only / neither |
|---|---:|---:|---:|---:|---:|
| Validation | 76.30% | **77.72%** | **+1.42 pp** | 18 | 17 / 0 / 1 |
| Test | 77.59% | **79.44%** | **+1.85 pp** | 21 | 21 / 0 / 0 |

Both losses decreased, both replacement-count gates passed, and candidate
precision versus live-only errors was 100% on both held-out splits.

## Exact-Rocket relevance failure

The rule made **zero** deployed top-action replacements on every exact-Rocket
support:

- N19 retention train: 0 / 6,240;
- N19 retention validation: 0 / 713;
- N19 retention test: 0 / 801;
- N6 unchanged anchors: 0 / 469;
- N9 V110: 0 / 3,456;
- N9 RecoveryV4: 0 / 3,297.

Consequently, the preregistered requirement of at least two held-out Rocket
replacements failed. The gate distinguishes high-confidence leader-deck states
from Rocket states; it does not provide evidence of a behaviorally active Team
Rocket improvement. Testing a behaviorally inert gate in official matches would
not be a strength experiment.

## Decision

`offline_admission=false`, `development_panel_authority=false`, and
`deployment_authority=false`. No dual-policy runtime, parity audit, official
match panel, confirmation, packaging, submission, or live mutation is
authorized. The live model remains
`3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.

N23 closes state-conditioned high-margin composition of the N21 cross-deck
candidate. Together N18-N23 show that the fresh arena leader contains
learnable strategic signal, but that signal is deck-specific and does not
activate safely on the exact Team Rocket policy. A successor must obtain
Team-Rocket-native value or outcome supervision rather than further filtering
or distilling the leader model.
