# N9 Trajectory Collection Lock

Status: **frozen before trajectory execution**  
Protocol: `rocket-balanced-exact-pcgrad-awr/v1`

- Protocol SHA-256:
  `157dc3d168d7b868b9d40f6aa008ab5303a1cd849c9f8a88d32af18f7dc38dbd`.
- Collector SHA-256:
  `076a141ebf6e7618d1812334cf2127eeaaf95d8557580937e1f1fd8b5a3929b7`.
- Games: 300 per shard, 1,200 total.
- Focal seat: 1; official first player: 0.
- Recorded bot: `N9RocketControl`.

Locked commands:

```powershell
python scripts/collect_n9_balanced_replays.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 797101 --output work/n9_balanced_exact_replays/v110_01
python scripts/collect_n9_balanced_replays.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 797102 --output work/n9_balanced_exact_replays/v110_02
python scripts/collect_n9_balanced_replays.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 797201 --output work/n9_balanced_exact_replays/recovery_01
python scripts/collect_n9_balanced_replays.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 797202 --output work/n9_balanced_exact_replays/recovery_02
```

All outputs are development-only. Partial results cannot change the remaining
schedule, split, optimizer, or gates.
