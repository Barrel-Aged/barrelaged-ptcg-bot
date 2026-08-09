# Target-80 randomized-support advantage V1 build lock

Locked: 2026-08-05  
Status: **smoke passed; full corpus authorized**  
Submission authority: none

## Immutable implementation

- prospective protocol SHA-256:
  `044720b2c874fce7527a19d45877fa579fcc897db808af67de40876e9ef6af84`
- collector SHA-256:
  `d6e72dc79f3a92b49b718ae88e0685227f223d82641ce7aee10bc4bc53b09ece`
- independent auditor SHA-256:
  `09f1ae62378565cbc3945ede7e5a9d9e2621ceeec5d8ae9d03ba87e82d6d5603`
- Rocket bundle tree SHA-256:
  `30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f`
- Rocket deck SHA-256:
  `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`
- Rocket model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`
- Rocket main SHA-256:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`

The collector refuses altered candidate/opponent bytes, altered split sizes,
non-Marnie smoke, existing output paths, native-engine seeded-execution claims,
illegal actions, hidden-hand boundary violations, missing behavior records,
timeouts, fallbacks, or nonterminal games.

## Smoke evidence

The four preregistered Marnie smoke games completed through the official
engine. Win/loss outcomes are excluded from every gate and dataset.

| Rocket seat | Games | Total decisions | Rocket decisions | Eligible | Explored | Packaged guards | Replay tree SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2 | 295 | 136 | 99 | 19 | 0 | `14b74ec98ab8dde392ba6578a92eb70f31782b27c7642aa6b61e90a6aecb0964` |
| 1 | 2 | 339 | 153 | 120 | 18 | 3 | `ca567ab5c7e0908593b170192a7a11188040b83b70e7fcdc5729e5053620a31f` |
| total | 4 | 634 | 289 | 219 | 37 | 3 | separate immutable trees above |

Both shards passed the independent audit with the smoke gates enabled:

- 4/4 terminal official-engine games;
- 289/289 Rocket decisions passed the opponent-hidden-hand check;
- applied actions matched logged choices and were legal;
- all reference and behavior probability vectors were finite, normalized, and
  consistent with their recorded chosen propensities;
- every supported alternative met the frozen 0.03 reference threshold;
- all ATTACK/END safety checks passed;
- both seats contained eligible decisions and realized interventions;
- all replay file and tree digests matched their manifests.

The auditor initially expected a terminal field named `decisions`, while the
collector correctly records that field as `steps`. This verifier-only defect
was fixed before the successful independent audit. No replay, collector,
protocol, behavior rule, or smoke output was changed.

## Authorized next action

Collect exactly the 18 frozen train/validation/test shards in the prospective
protocol. Do not inspect or summarize their outcomes until all 1,200 games are
terminal and every shard passes the independent audit. Any material collector
change after this lock requires a new prospective protocol version.

