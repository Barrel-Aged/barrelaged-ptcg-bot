# N10 Independent Epoch-4 Confirmation Result

Status: **rejected as operationally invalid; outcome analysis forbidden; live bot unchanged**  
Protocol: `rocket-n10-independent-epoch4-confirmation/v1`  
Closed: 2026-08-01

## What completed

The two V110 shards completed before the user-requested host power interruption
and remained hash-identical afterward. The two interrupted Recovery V4 files
were archived without inspecting outcomes. On resume, all frozen hashes passed
and only the original Recovery seeds 799201 and 799202 were rerun from game 1
with the original Python 3.12 runtime, collector, models, deck, opponent, and
400-games-per-arm schedule.

Both restarted Recovery shards reached 800 rows and wrote manifests:

| Shard | Rows | JSONL SHA-256 | Manifest SHA-256 |
|---|---:|---|---|
| `recovery_01` | 800 | `7b3105754693f0374df4ff7b2e070fcce52eab971044ad3cf6e11dd0f86bcd21` | `55eab8f11f1731c0e8de1e51e769b868ae20a3579f08bbfe0a0c422d6b8ae9ab` |
| `recovery_02` | 800 | `5d8d1b7f7f3136d09479b7f1fe6b5dcbe9145e20e34b9ea3973dde5e9c7260dd` | `f40108e28291a056c35332bc009a23d49cd9d090ae43203b9f418a423021e8f9` |

This physically completed the intended 3,200 rows, but physical completion is
not statistical admissibility.

## Locked resume gate failed

`N10_POWER_INTERRUPTION_AND_RESUME_LOCK.md` required all 551 and 558 archived
Recovery rows to be byte-identical to the corresponding prefixes of the
restarted shards. The frozen verifier
`scripts/verify_n10_resumed_prefixes.py` stopped at the first comparison:

- `recovery_01` mismatch at line 1;
- archived first-line SHA-256:
  `7cb3a5080eaf748c67ff0e707cd66d41da284e2890c4a61b2e9ff089d3ebb0a7`;
- restarted first-line SHA-256:
  `585b338626cc9c692fde73acd12e78d674dd09182ef6d68580bc42b47f4ccd3b`.

The checked non-outcome provenance fields—protocol, schema, game index,
schedule seed, arm and model hash, opponent and bundle fingerprints, seat, and
fallback count—were identical. No game outcome, arm win count, or matchup
comparison was inspected.

The lock states that one prefix mismatch operationally fails N10 and prohibits
the N10 analyzer. Therefore:

- `scripts/analyze_n10_confirmation_panel.py` was not run;
- no N10 strength estimate or admission result exists;
- the restarted Recovery files are diagnostic-only and excluded from all
  selection, validation, packaging, submission, and strength claims;
- epoch 4 remains an unconfirmed N9 development artifact; and
- `HelbrosRocketSpidopsScaleBCV2` remains the unchanged live bot.

## Consequence

The interrupted experiment cannot be repaired by examining these outcomes,
extending its sample, or selecting replacement seeds post hoc. Any subsequent
confirmation must be preregistered as a fully fresh panel, use new seeds for
all opponents and arms, and treat every N10 row as unavailable for training or
selection. The new collection should also avoid a resumability requirement
that assumes the official engine and external opponent worker are replay-
deterministic across process restarts.

