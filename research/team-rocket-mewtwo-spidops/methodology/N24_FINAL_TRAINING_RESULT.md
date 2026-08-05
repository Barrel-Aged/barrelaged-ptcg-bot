# N24 Final Training Result

Status: **locked terminal checkpoint; held-out and deployment authority remain false**  
Protocol: `rocket-native-conservative-iql/v1`

The sole final N24 training run completed under
`N24_TRAINING_EXECUTION_LOCK.md`. All non-timing epoch metrics reproduced the
train-only feasibility run exactly.

- terminal checkpoint SHA-256:
  `fab66f115d132f578768ef83cc4901c5981a825b79dd427615fb7203e165c6cf`;
- training report SHA-256:
  `0b622f1df00d684940e00d20abca899335f0608a29dcee91243ce78d263af347`;
- trainer SHA-256:
  `1b372ea9e01f5c6a2d45befbaae8b1a2ab060dc746ec00fa12336488582d6be0`;
- train cache SHA-256:
  `e6e83143773462e52a0142e718a65c75ec93cc4abe0461f9437dcc42a332c3b0`;
- selected train-only overlay margin: `0.5132653117179871`;
- opponent-blind outcome calibration: scale `2.040775409806487`, bias
  `0.7523584194392771`;
- exact train reference parity: 55,666/55,666 choices;
- train overlay rate: 278/55,666 (0.4994%).

The checkpoint is now immutable. It may be used only by the preregistered
held-out evaluator and, if all offline gates pass, a later runtime-parity build.
It has not executed as a candidate in the official engine. The live bot remains
unchanged.

