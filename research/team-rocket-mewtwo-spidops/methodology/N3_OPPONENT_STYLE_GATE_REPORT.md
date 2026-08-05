# N3 Opponent-Style Gate Report

Status: calibration failure; untouched validation remains unopened  
Protocol: `rocket-opponent-style-gate/v1`

N3 collected 600 training games and 200 calibration games through the exact zero-control official actor. These yielded 47,388 frozen public/own-visible global tokens with zero fallbacks. A preregistered 64-wide GRU trained for all 30 epochs.

Calibration rejected the classifier:

- fitted temperature reached the allowed maximum, 20.0;
- final-step balanced accuracy was 43.5%;
- the threshold satisfying the 1% V110 false-activation cap was 0.5790379;
- V110 activation was 1/100 games (1%);
- RecoveryV4 activation was only 2/100 games (2%);
- median Recovery activation decision was 37.

The protocol required at least 50% Recovery activation on untouched validation and intended an informative calibrated classifier. Calibration performance is already incompatible with that purpose, so N3 stops without opening validation, building a gated actor, running a strength panel, or submitting.

Artifacts:

- classifier SHA-256: `5c39b8cc604925a0c7fdfc4c06afc7b999de7df183655468e224520bffa3755a`;
- training report SHA-256: `ba43c151d28f7a2665a7d771e98c092a3dd050d710ac8476e4e77260366ee579`;
- unopened V110 validation SHA-256: `d19b0783b091199d4fa7bbde950184e21dbfebb974be1988e095e46922f6193b`;
- unopened RecoveryV4 validation SHA-256: `574fa91d495170d060109ff80492b66776d7c43edd5b5d94eb7773a907cb9d1e`.

The frozen blueprint global token is not a reliable carrier of exact opponent-policy implementation identity. A later opponent model must encode explicit public action/log events rather than infer style indirectly from state embeddings. The unopened validation artifacts remain sealed and must not be reused as training or development data without an explicit reclassification document.
