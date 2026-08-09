# N24 Conservative Team-Rocket-Native IQL Result

Status: **rejected on validation; test unopened; no runtime build or official-engine candidate games**  
Protocol: `rocket-native-conservative-iql/v1`

## Decision

N24 failed multiple preregistered validation gates and is rejected. The test
cache was not evaluated. The terminal checkpoint, threshold, and calibration
must not be retuned under the N24 identifier.

- validation report SHA-256:
  `4696209f21ac0cb172f4a49a5b556bb0b623b052b0bd26193a170e118a8ddb5b`;
- exact reference parity: 6,709/6,709 choices;
- replacements: 41/6,709 (0.6111%);
- minimum replacement reference support: 2.3535%;
- overall first-choice AUC: 0.5535, below the 0.56 gate;
- worst-opponent AUC: 0.5413, above the 0.53 gate;
- overall calibrated Brier: 0.21946 versus constant 0.21688, gate failed;
- per-opponent Brier gate: failed in both opponents;
- logged-action versus mean-option target-MAE ratio: 1.0727, above the 0.98
  maximum;
- V110 ratio: 0.9906, failed;
- RecoveryV4 ratio: 1.1647, failed.

The intervention-count, support, exact-parity, and worst-opponent-AUC gates
passed. That is insufficient: the value model's train action-specific signal
did not transfer, particularly against RecoveryV4, and its train-only
calibration did not generalize.

## Boundary

The official engine produced the source replays, but N24 validation was cached
model evaluation, not new official-engine candidate execution. No N24 runtime
bundle was built, no fresh panel was run, no arena/Kaggle submission occurred,
and no strength claim is made. The live
`HelbrosRocketSpidopsScaleBCV2` model and qualifying Team Rocket deck remain
unchanged.

This specific frozen-token Monte-Carlo/CQL overlay is closed. A future native
value effort would need new information—such as randomized action support or
fresh intervention data—rather than threshold, calibration, or architecture
retuning on the same deterministic behavior corpus.

