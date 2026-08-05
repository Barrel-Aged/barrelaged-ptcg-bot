# N16 Development Panel Execution Lock

Status: **frozen before match collection**  
Protocol: `rocket-n16-minimum-scale-residual-projection/v1`

## Immutable artifacts

- candidate model SHA-256:
  `b4f35c7816097800b0c7265691f5f650bc9b467a4258612930d0fef07afcbcf7`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- arm manifest SHA-256:
  `3cb430f6e281004b1671d0df3941e6a14386278d789c2e153b51221fe6e4bf72`;
- candidate bundle tree SHA-256:
  `d4732ea468b9046abffd808937935963cba4091f180ae72704d03e66a6bfcf02`;
- runtime parity SHA-256:
  `0a8bad5cc2e6f231ecd5eba66d8b9b3880c43ab22680cebade2f7630f1068a34`;
- collector SHA-256:
  `6895b48af08c825354efae65c826e0b8c96e0f493e4bd89e602990dfc9068dd7`;
- analyzer SHA-256:
  `fc1208551a280131a57f37f860008fa133c01c2506d5486f2ee6873910e6c603`.

The package audit matched 964/964 official-engine Rocket decisions with zero
mismatches. The collector freezes both opponent bundles and all engine/helper,
model, deck, runtime, arm, and parity hashes. The panel directory did not exist
at lock time.

## Locked collection

Each shard contains 120 games per arm and 240 rows, Rocket seat 1:

```powershell
python scripts/collect_n16_development_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 816101 --output artifacts/team_rocket_recovery/n16_minimum_scale_projection/development_panel/v110_01.jsonl
python scripts/collect_n16_development_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 816102 --output artifacts/team_rocket_recovery/n16_minimum_scale_projection/development_panel/v110_02.jsonl
python scripts/collect_n16_development_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 816201 --output artifacts/team_rocket_recovery/n16_minimum_scale_projection/development_panel/recovery_01.jsonl
python scripts/collect_n16_development_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 816202 --output artifacts/team_rocket_recovery/n16_minimum_scale_projection/development_panel/recovery_02.jsonl
```

The shards may run concurrently. Until all four files reach exactly 240 rows
and all four manifests seal, only process health, file existence, and row counts
may be observed. Do not inspect rows, wins, or partial manifests. Preserve an
interrupted run unopened; no restart is allowed without a recovery lock.

## Locked analysis

After all shards seal:

```powershell
python scripts/analyze_n16_development_panel.py
```

The analyzer uses 20,000 bootstrap replicates under seed `816001` and the same
gates as N15: at least +2.0 points equally stratified, positive against each
opponent, lower bound above -0.5 points, and strict correctness. A pass permits
only fresh confirmation; neither outcome permits direct deployment.
