# N9 Development Panel Execution Lock

Status: **frozen before development match execution**  
Protocol: `rocket-balanced-exact-pcgrad-awr/v1`

## Immutable artifacts

- Protocol SHA-256:
  `157dc3d168d7b868b9d40f6aa008ab5303a1cd849c9f8a88d32af18f7dc38dbd`.
- Training lock SHA-256:
  `285dd1a98b60a9db53699ff4ac6a1b0d376a53d517d042351b5c0e5115f02779`.
- Trainer SHA-256:
  `62182583002847e7ed0e18011050f35319ad5661b73f3fb2d83c2564286ed1ad`.
- Arm builder SHA-256:
  `5b686b8023bac0f3c6d1ec02f199a8190addc7c30400231fd9eb01b022b21f52`.
- Arm manifest SHA-256:
  `eb6efcd1a839ab494ccd6afd6a1e7854dec67bed0717ebaf6037c68f40824b23`.
- Runtime verifier SHA-256:
  `54f0cadf4317be3f71afd81b6e2b843f5918dff67d0f73f61002175f418a08ed`.
- Runtime parity report SHA-256:
  `6c11d589ad7d7cb1235734a1ecf87f00f40159a31e2a90ece5f57900af04e535`.
- Collector SHA-256:
  `376101b680be8a9e4d886c0741c2ea9783cd1de0f5c629e5093dbd18fef9af78`.
- Analyzer SHA-256:
  `34d88553d2a7196aa2b2b7e3f77b05a3d560f93ba504fc33f8c6619ed044825e`.

The runtime audit matched 2,296/2,296 official decisions across 40 games with
zero mismatches.

## Locked collection

Run 120 games per arm in each command:

```powershell
python scripts/collect_n9_development_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 798101 --output artifacts/team_rocket_recovery/n9_balanced_pcgrad_awr/development_panel/v110_01.jsonl
python scripts/collect_n9_development_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 798102 --output artifacts/team_rocket_recovery/n9_balanced_pcgrad_awr/development_panel/v110_02.jsonl
python scripts/collect_n9_development_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 798201 --output artifacts/team_rocket_recovery/n9_balanced_pcgrad_awr/development_panel/recovery_01.jsonl
python scripts/collect_n9_development_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 798202 --output artifacts/team_rocket_recovery/n9_balanced_pcgrad_awr/development_panel/recovery_02.jsonl
```

Then run only the frozen analyzer with seed 798001 and 20,000 bootstrap
replicates. Partial outcomes cannot alter arms, counts, seeds, gates, or
tie-breaks. These are development data with no deployment authority.
