# N25 Development Panel Execution Lock

Status: **frozen before match collection**  
Protocol: `rocket-n25-complementary-safe-composition/v1`

## Immutable artifacts

- candidate model SHA-256:
  `e1d6567dfefee9c7b408b9438153e75e75f613864d7c0a2da021f7f5c63ebf04`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- arm manifest SHA-256:
  `defd9b9a02bbb221a8ef6d22b496d9dd37931870c8bb93001c32fdeaefe176a1`;
- candidate bundle tree SHA-256:
  `38ce5eaed5074544bf89fb723789ca39258dd9d734bf714f9803f5235e60dcbd`;
- runtime parity SHA-256:
  `43556e15d5e783a0e2a1f14c2eb033b13ce0e7943281ecbd7222855d4341a602`;
- collector SHA-256:
  `e73612c3dda5fd784218ddfad365dd37daef57fd1eefdf2b16e4c7b01684401c`;
- analyzer SHA-256:
  `9c24889d2c2156487cf358453b12aae22c74f92ccc4680b69101004757522367`.

The package audit matched 968/968 official-engine Team Rocket decisions with
zero mismatches. The collector freezes both opponent bundles and all
engine/helper, model, deck, runtime, arm, and parity hashes. The panel
directory did not exist at lock time.

## Locked collection

Each shard contains 120 games per arm and 240 rows, Team Rocket seat 1:

```powershell
python scripts/collect_n25_development_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 825101 --output artifacts/team_rocket_recovery/n25_complementary_safe_composition/development_panel/v110_01.jsonl
python scripts/collect_n25_development_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 825102 --output artifacts/team_rocket_recovery/n25_complementary_safe_composition/development_panel/v110_02.jsonl
python scripts/collect_n25_development_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 825201 --output artifacts/team_rocket_recovery/n25_complementary_safe_composition/development_panel/recovery_01.jsonl
python scripts/collect_n25_development_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 825202 --output artifacts/team_rocket_recovery/n25_complementary_safe_composition/development_panel/recovery_02.jsonl
```

The shards may run concurrently. Until all four files reach exactly 240 rows
and all four manifests seal, only process health, file existence, and row
counts may be observed. Do not inspect rows, wins, or partial manifests.
Preserve an interrupted run unopened; no restart is allowed without a recovery
lock.

## Locked analysis

After all shards seal:

```powershell
python scripts/analyze_n25_development_panel.py
```

The analyzer uses 20,000 bootstrap replicates under seed `825001` and requires
at least +3.0 points equally stratified, positive improvement against each
opponent, a lower bound above -0.5 points, and strict correctness. A pass
permits only a separately preregistered fresh confirmation of at least 3,200
games; neither outcome permits direct deployment.
