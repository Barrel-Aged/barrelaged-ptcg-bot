# N17 Replicated-Target Residual Offline Result

Status: **all preregistered offline gates passed; fresh development panel
authorized; no strength or deployment claim**  
Protocol: `rocket-n17-replicated-target-residual/v1`

## Frozen execution

- Protocol SHA-256:
  `0a3348c41fd5fc821fbfda70280ee516d25c0425b45556a742afc56c9ef381b4`.
- Target-audit SHA-256:
  `c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c`.
- Trainer SHA-256:
  `5501bb3e7e65a2382159160a71258f0274db4035500c8fa933e02a571fc5e6e9`.
- Training-execution lock SHA-256:
  `c07fcfcc8dbd79ffbba4a9463312ecf97dffdc04ab9c5650d07ddd2a6c21fce3`.
- Exact-zero base model SHA-256:
  `de4679c136af57af9dc6818b3100c1e10d74ac5c32d615ae02019c897ec07a6c`.
- Exact-zero parity report SHA-256:
  `6268b01da89e73ee3e1fbf21d9141100b86bd7a503a09eeeaf1f4ecdcea692f2`.
- Sole step-4,000 terminal model SHA-256:
  `381d6cb87a51fa1c8061f5e3ae441e446b0331bdd9c1f99e9118d4227e038772`.
- Training report SHA-256:
  `50aaac5e3531699489133790bdcbf0d566aea7acec23775ddd5b91d144fc02dd`.
- Training provenance SHA-256:
  `1b14b12eef429f5019816d905dbd9c3e4c8d0e1da31265af8832b8423fca5789`.
- Exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Residual-capable runtime SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`.

The locked trainer ran exactly 4,000 steps. It created no early-stop,
best-loss, first-hit, alternate-rank, alternate-seed, or alternate-hyperparameter
model.

## Terminal offline audit

- strict targets: 48/48 top-1;
- expanded-only replicated targets: 122/122 top-1;
- expanded-only per-cell fit: 34/34, 21/21, 35/35, and 32/32;
- N6R non-target anchors: 469/469 retained, mean forward KL
  `0.009773081247404372`;
- N9 anchors: 6,697/6,753 retained (`0.9917073893084555`), with
  V110 `0.9921875` and RecoveryV4 `0.9912041249620868` retention;
- N9 mean forward KL: `0.009000004862021018`;
- only the four `logit_adapter.*` tensors changed;
- fresh serialization is tensor-exact across all 63 state tensors;
- all 25 copied runtime/deck assets are byte-exact;
- cached/full-runtime parity: zero top-action mismatches and `0.0` maximum
  finite relative-logit error on all 639 N6R roots and 6,753 N9 actions;
- all 13 offline admission gates passed;
- all three independent N17 implementation tests passed again after training.

## Authority boundary

These results establish faithful optimization and preservation on frozen
offline support only. They do not establish match strength. The terminal model
is authorized solely for the preregistered 960-game fresh official-engine
development panel against V110 and RecoveryV4. The live submitted bot remains
unchanged, and no Kaggle submission or deployment is authorized by this result.
