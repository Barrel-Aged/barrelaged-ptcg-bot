# N8B Development Panel Execution Lock

Status: **frozen before development match execution**  
Protocol: `rocket-recovery-logit-delta-transfer/v1`  
Frozen: 2026-07-30

## Immutable inputs and executables

- Protocol SHA-256:
  `737332df23fc199b9367931b11795ad89a42a4658551b8dda80845850f02b008`.
- Arm builder SHA-256:
  `f15cac396fbd400b95d1823ea224c792ff593c8afba037df803f7cf9f44d4854`.
- Arm manifest SHA-256:
  `3eccbc2b4d10138dac66391e5ac83e18a4e3d951c920adef0fbfb7d14aebccd2`.
- Runtime verifier SHA-256:
  `972905b4e4bda7f8ac3166c9355cf5b5c343001b574c32796b6f095becf74d80`.
- Runtime parity report SHA-256:
  `4986cc6a2b2c65895710d1e65932659666df569283d4fa9aa4012abc415f3ba3`.
- Panel collector SHA-256:
  `2ed31a223c129b508e77649d8ce05f36dab02f1fbc062319559a2de4d196f0c8`.
- Panel analyzer SHA-256:
  `0ee0d78f101808184a8793eb4ac2cd6fc091917c4b9c50d1f1183f92310d58c3`.

The runtime audit completed 40 official-engine games and matched 2,505/2,505
strict versus packaged decisions, with zero mismatches. All five model hashes
are now fixed by the arm manifest.

## Locked collection commands

Each command writes a new file and refuses overwrite. The exact opponent,
seed, and 120-games-per-arm count are enforced in the collector.

```powershell
python scripts/collect_n8b_logit_delta_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 796101 --output artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_panel/v110_01.jsonl
python scripts/collect_n8b_logit_delta_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 796102 --output artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_panel/v110_02.jsonl
python scripts/collect_n8b_logit_delta_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 796201 --output artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_panel/recovery_01.jsonl
python scripts/collect_n8b_logit_delta_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 796202 --output artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_panel/recovery_02.jsonl
```

After all four strict shards complete, the only permitted development analysis
is:

```powershell
python scripts/analyze_n8b_logit_delta_panel.py artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_panel/v110_01.jsonl artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_panel/v110_02.jsonl artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_panel/recovery_01.jsonl artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_panel/recovery_02.jsonl --output artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_panel/analysis.json
```

No partial result may alter the schedule or gate. These files are development
evidence only and carry no submission authority.
