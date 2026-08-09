# N18 Development Panel Execution Lock

Status: **frozen before match collection**  
Protocol: `rocket-n18-cross-deck-leader-transfer/v1`

## Immutable inputs

- protocol SHA-256:
  `b64bd56fb0c1593f2310b636723c83c7c08596f5dbb63c3d9e848015b3c25805`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- arms manifest SHA-256:
  `57ab74eaaf05aa2f4e36a09f1555aef4eadf1d84dd0415c59376de036bfbed18`;
- packaged-runtime audit SHA-256:
  `93c2e64764108ee78415b13ce3de3bf6e0c7bc8256b44804a002081e802f0b93`;
- collector SHA-256:
  `2891a7c6ec14e7c6de68f755b66e5ede606bcf302ac23e01254b4bcc02844625`;
- analyzer SHA-256:
  `5b9f8b0f9379ebf31306b2b8099b48456a5486313231e4b19dd9e83bcca68ca1`;
- packaged-worker helper SHA-256:
  `a8eb4808fd29109ebd851b9802b72245b3353b0d64f97384a1aebd1948b80b85`;
- prior official-engine helper SHA-256:
  `f3cb21219efbf82c94a1bebe1d90b712fcd02cb28037397a02176d2ce5047f6b`;
- arena evaluator SHA-256:
  `d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303`.

The collector imports the native official engine only from the exact live
`control` bundle.  Candidate and opponent bundles execute only as isolated
packaged-agent workers.  Every worker handshakes its deck at game reset, and
every returned action is checked against the official selection before it is
applied.

The panel directory did not exist when this lock was written.  Neither the
collector nor analyzer has been run on fresh N18 match outcomes.

## Frozen arm allocation

The only match-tested arm order is:

1. `control`;
2. `live_residual`;
3. `leader_base`;
4. `leader_residual`.

Each shard contains exactly 120 games per arm.  `bridge_base` is excluded and
has no advancement authority.

## Frozen collection commands

```powershell
python scripts/collect_n18_development_panel.py --seed 818101
python scripts/collect_n18_development_panel.py --seed 818102
python scripts/collect_n18_development_panel.py --seed 818201
python scripts/collect_n18_development_panel.py --seed 818202
python scripts/collect_n18_development_panel.py --seed 818301
python scripts/collect_n18_development_panel.py --seed 818302
```

The seed fixes opponent, Rocket seat, output path, and shuffled arm schedule:

- `818101`: current rank one, Rocket seat 0,
  `leader_seat0.jsonl`;
- `818102`: current rank one, Rocket seat 1,
  `leader_seat1.jsonl`;
- `818201`: V110, Rocket seat 0, `v110_seat0.jsonl`;
- `818202`: V110, Rocket seat 1, `v110_seat1.jsonl`;
- `818301`: RecoveryV4, Rocket seat 0, `recovery_seat0.jsonl`;
- `818302`: RecoveryV4, Rocket seat 1, `recovery_seat1.jsonl`.

All output paths are below
`artifacts/team_rocket_recovery/n18_new_leader_evidence/development_panel`.
Each JSONL must seal at 480 rows and receive its matching manifest.  The final
panel contains exactly 2,880 games.

The shards may run concurrently.  Until all six JSONL files and all six
manifests seal, only process health, file existence, and row counts may be
observed.  Do not inspect row contents, wins, draws, action telemetry, partial
manifests, or outcome-dependent process output.  An interrupted shard is
preserved and may not be restarted without a durable recovery lock.

## Frozen analysis

After and only after all six shards seal:

```powershell
python scripts/analyze_n18_development_panel.py
```

The analyzer performs the preregistered 20,000-replicate equal-cell stratified
bootstrap under seed `818001`, three-comparison Bonferroni correction, gates,
and fixed tie resolution.  A pass grants only fresh 3,200-game confirmation
authority.  It never grants deployment or submission authority.
