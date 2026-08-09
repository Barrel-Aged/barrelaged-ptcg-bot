# N17 Runtime Parity Lock

Status: **frozen before official-engine package parity execution**  
Protocol: `rocket-n17-replicated-target-residual/v1`

- verifier SHA-256:
  `99a1650ad5e214d2c1d9881f829cf1353462932559f5f1442459662e8457f5ca`;
- arm manifest SHA-256:
  `28ac3ef5c0748f9b28faef4e9d071f809855c9d78fcdd951ea1f4aa38132b8bc`;
- candidate bundle tree SHA-256:
  `a7d02c5bc39428f570317c704893a103c6f1225491f1b72c84312b980ef50dd3`;
- candidate model SHA-256:
  `381d6cb87a51fa1c8061f5e3ae441e446b0331bdd9c1f99e9118d4227e038772`;
- live control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- residual runtime SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`;
- official helper SHA-256:
  `f3cb21219efbf82c94a1bebe1d90b712fcd02cb28037397a02176d2ce5047f6b`;
- opponent-worker SHA-256:
  `a8eb4808fd29109ebd851b9802b72245b3353b0d64f97384a1aebd1948b80b85`;
- evaluator SHA-256:
  `d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303`;
- V110 opponent tree SHA-256:
  `3bb5e40bb1583e31470eee263ed615c9207427c7b5d2199e80fbe3571e2cbb2b`;
- RecoveryV4 opponent tree SHA-256:
  `caf361fd297ab63b24338650b6e396a296b961bb6ba61d2519799a90dff738f4`.

The output did not exist at lock time. Under seed `817000`, the verifier must
run four genuine official-engine games per opponent per arm and require at
least 600 Rocket decisions. For every Rocket decision, the direct strict agent
and the packaged `NNAgent` must select the identical legal action. It observes
no match outcome for model selection and has no strength or deployment
authority.

```powershell
python scripts/verify_n17_runtime_parity.py
```
