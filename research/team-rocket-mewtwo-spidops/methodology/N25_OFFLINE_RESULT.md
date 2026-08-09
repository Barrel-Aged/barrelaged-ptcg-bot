# N25 Offline Composition Result

Status: **all frozen offline gates passed; packaging only**  
Protocol: `rocket-n25-complementary-safe-composition/v1`

The sole N25 candidate was constructed as the frozen N15 residual model plus
the exact N8B action-head delta. The source-base correction proves that the
N8B control and live action heads are identical, the N8B delta changes exactly
the two logit-head tensors, and no N8B value tensor enters N25. Serialization
was tensor-exact.

The candidate model SHA-256 is
`e1d6567dfefee9c7b408b9438153e75e75f613864d7c0a2da021f7f5c63ebf04`.
The build report SHA-256 is
`b94d281479b4be1cb29b7352f27c3b91ec36f7503edd3f5b3a5ede556b7a81ce`.

Offline results:

- all 48/48 strict correction targets are top-1; minimum margin `0.375283`;
- N6R anchors retain 588/591 (`99.4924%`), including `98.9691%` against V110
  and `100%` against RecoveryV4; mean forward KL `0.011061`;
- N9 exact-Rocket choices retain 6,648/6,753 (`98.4451%`), including
  `98.3218%` against V110 and `98.5745%` against RecoveryV4; mean forward KL
  `0.005245`;
- changed tensors are restricted to `logit_adapter.*` and the two logit-head
  tensors.

These are frozen-corpus preservation checks, not official candidate-game
evidence or a strength claim. They authorize packaging and runtime parity only.
The live bot remains unchanged.
