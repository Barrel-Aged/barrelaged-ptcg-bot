# N17 Development Panel Execution Lock

Status: **frozen before match collection**  
Protocol: `rocket-n17-replicated-target-residual/v1`

## Immutable artifacts

- candidate model SHA-256:
  `381d6cb87a51fa1c8061f5e3ae441e446b0331bdd9c1f99e9118d4227e038772`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- arm manifest SHA-256:
  `28ac3ef5c0748f9b28faef4e9d071f809855c9d78fcdd951ea1f4aa38132b8bc`;
- candidate bundle tree SHA-256:
  `a7d02c5bc39428f570317c704893a103c6f1225491f1b72c84312b980ef50dd3`;
- runtime parity SHA-256:
  `c913449822c8ed573fe78b371ca237b564e6402424c4486a156bdd4069f719cf`;
- collector SHA-256:
  `1e8057c885e31a0ee9dffaab54dd57dc40c7d05a1016c13113d5768ac7d79144`;
- analyzer SHA-256:
  `c428ca44f652781fb6217c6aba432c4692142b8473aaa6bdbbfd65aa34a12a0f`;
- official-engine helper SHA-256:
  `f3cb21219efbf82c94a1bebe1d90b712fcd02cb28037397a02176d2ce5047f6b`;
- opponent-worker SHA-256:
  `a8eb4808fd29109ebd851b9802b72245b3353b0d64f97384a1aebd1948b80b85`;
- evaluator SHA-256:
  `d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303`;
- stratified-bootstrap dependency SHA-256:
  `cabf1026ab21429ec11fb37b8e98760be3ef354a3303316c9f09d72340281df8`.

The packaged direct/`NNAgent` audit matched 1,082/1,082 genuine
official-engine Rocket decisions with zero mismatches. Both opponent bundles
and every engine/helper, model, deck, runtime, arm, and parity artifact are
hash-frozen by the collector. The panel directory did not exist at lock time.

## Locked collection

Each shard contains 120 games per arm and 240 rows, Rocket seat 1:

```powershell
python scripts/collect_n17_development_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 817101 --output artifacts/team_rocket_recovery/n17_replicated_targets/development_panel/v110_01.jsonl
python scripts/collect_n17_development_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 817102 --output artifacts/team_rocket_recovery/n17_replicated_targets/development_panel/v110_02.jsonl
python scripts/collect_n17_development_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 817201 --output artifacts/team_rocket_recovery/n17_replicated_targets/development_panel/recovery_01.jsonl
python scripts/collect_n17_development_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 817202 --output artifacts/team_rocket_recovery/n17_replicated_targets/development_panel/recovery_02.jsonl
```

The shards may run concurrently. Until all four files reach exactly 240 rows
and all four manifests seal, only process health, file existence, and row
counts may be observed. Do not inspect rows, wins, or partial manifests.
Preserve an interrupted run unopened; no restart is allowed without a recovery
lock.

## Locked analysis

After all shards seal:

```powershell
python scripts/analyze_n17_development_panel.py
```

The analyzer uses 20,000 stratified bootstrap replicates under seed `817001`.
Advancement requires strict correctness, at least +2.0 percentage points
equally stratified, strictly positive improvement against both opponents, and a
95% lower bound greater than -0.5 percentage points. A pass permits only at
least 3,200 fresh confirmation games plus broad both-seat regression. Neither
outcome permits direct deployment, submission, or a verified strength claim.
