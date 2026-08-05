# N24 Held-Out Cache Lock

Status: **frozen before the first N24 held-out metric**  
Protocol: `rocket-native-conservative-iql/v1`

The validation and test token caches were generated in parallel with the
frozen N24 encoder. Cache generation emitted only provenance, shape, extraction,
and exact reference-parity facts; no value-head or outcome metric was run.

## Frozen artifacts

- terminal Q/V checkpoint SHA-256:
  `fab66f115d132f578768ef83cc4901c5981a825b79dd427615fb7203e165c6cf`;
- validation cache SHA-256:
  `0d64c0d2a716a55869ae1e61ffdca1ad7e1f559cedd7271f6e78f93a9c2690b5`;
- validation manifest SHA-256:
  `b33d441516985a38c0b38543dda4a86c969f0b8c3d23c580b9912a14e6aaff02`;
- test cache SHA-256:
  `49fae7b0897ab91b89e6c7c3628b1835c0710700ebf3b532f8c27195eb1821ac`;
- test manifest SHA-256:
  `526aff934a486eb570b42e0a6ed858f58c8e0b9fa3a91d2d3abeb7b55e04c8fd`;
- evaluator SHA-256:
  `1b372ea9e01f5c6a2d45befbaae8b1a2ab060dc746ec00fa12336488582d6be0`.

Validation contains exactly 120 games, 6,709 supported choices, and zero
reference-action mismatches. Test contains exactly 119 decisive games, 6,753
supported choices, and zero reference-action mismatches.

Run validation once. Do not run test unless validation passes every gate in
`N24_CONSERVATIVE_NATIVE_IQL_PROTOCOL.md`. The frozen test cache must not be
loaded for analysis before that decision. The live bot remains unchanged.

