# N17 Replicated-Target Residual Result

Status: **closed; official-engine development gate failed; no confirmation,
deployment, or live-bot change**  
Protocol: `rocket-n17-replicated-target-residual/v1`

## Candidate and execution evidence

- sole terminal candidate model SHA-256:
  `381d6cb87a51fa1c8061f5e3ae441e446b0331bdd9c1f99e9118d4227e038772`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- offline result SHA-256:
  `0ffc6b8ecc6872059835552ff497433b663f8b63a9ee51abefec39389206413c`;
- development execution lock SHA-256:
  `a68f57b937c583e050f3dc3fc6638ba4d9b11c16b765b530bc00da71114bd6d2`;
- arm manifest SHA-256:
  `28ac3ef5c0748f9b28faef4e9d071f809855c9d78fcdd951ea1f4aa38132b8bc`;
- candidate bundle tree SHA-256:
  `a7d02c5bc39428f570317c704893a103c6f1225491f1b72c84312b980ef50dd3`;
- packaged runtime parity SHA-256:
  `c913449822c8ed573fe78b371ca237b564e6402424c4486a156bdd4069f719cf`;
- development analysis SHA-256:
  `9b588af4de787b6df66bda6d429e1066fa137d940b46c764bd03f30b611d6ace`.

The package audit matched 1,082/1,082 genuine official-engine decisions with
zero mismatches. All four preregistered shards sealed at 240 rows, giving 960
total games, 240 per arm/opponent cell. Strict correctness passed: every Rocket
game started second, all results were valid, the frozen arm hashes matched, and
there were zero exception fallbacks.

## Frozen development result

| Opponent | Control | N17 candidate | Difference |
|---|---:|---:|---:|
| V110 | 97/240 (`40.417%`) | 96/240 (`40.000%`) | `-0.417 pp` |
| RecoveryV4 | 86/240 (`35.833%`) | 85/240 (`35.417%`) | `-0.417 pp` |

- equally stratified difference: `-0.417 pp`;
- stratified bootstrap 95% interval: `[-6.458 pp, +5.625 pp]`;
- worst-opponent difference: `-0.417 pp`;
- required: at least `+2.0 pp`, positive against each opponent, and 95% lower
  bound above `-0.5 pp`;
- development gate: **failed**;
- advancing candidates: none.

## Interpretation and authority boundary

N17 proved that the rank-32 residual could memorize all 170 replicated/strict
counterfactual targets while preserving more than 99% of the old N9 action
support. That offline signal did not translate to fresh match strength: the
candidate was slightly negative against both development opponents. Together
with N15 and N16, this closes the current family of localized residuals trained
on the old N6R counterfactual target set; further rank, scale, seed, checkpoint,
or target-threshold selection on these outcomes would be post-selection and is
not authorized.

No independent confirmation was run. The candidate has no strength,
submission, or deployment authority. The submitted/live bot remains
`HelbrosRocketSpidopsScaleBCV2`, model SHA-256
`3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`,
with the exact deck above.
