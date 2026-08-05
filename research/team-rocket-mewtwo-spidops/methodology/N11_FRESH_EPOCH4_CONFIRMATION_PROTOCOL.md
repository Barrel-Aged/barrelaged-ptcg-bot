# N11 Fresh Epoch-4 Confirmation Protocol

Status: **preregistered before any N11 game is collected**  
Protocol: `rocket-n11-fresh-epoch4-confirmation/v1`  
Preregistered: 2026-08-01

## Purpose and independence

N9 selected epoch 4 in development but did not admit it because its aggregate
bootstrap lower bound narrowly missed the locked threshold. N10 attempted a
single-candidate confirmation, but a host power interruption and failed
byte-prefix reproduction made the entire N10 panel operationally invalid.
No N10 outcome, win count, arm comparison, or strength estimate was inspected.

N11 is a fully fresh replacement. It keeps the one candidate selected before
N10, changes no model or hyperparameter, uses new seeds for every opponent
cell, and excludes every N10 row from training, selection, estimation, and
claims.

## Frozen substrate

- Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Live control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Fixed N9 epoch-4 candidate SHA-256:
  `bdafc1ce36f7e572554930a1a9b1beff541a351ece09f934d0416b4ce269a020`.
- Two-arm manifest SHA-256:
  `d60ab7cbd53a49cc9a924f8a9b8d7445937f7db4b9c80d90ef7a04c007bdf443`.
- V110 opponent model SHA-256:
  `552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b`.
- Recovery V4 opponent model SHA-256:
  `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c`.
- Both opponents use deck SHA-256:
  `8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b`.
- Existing packaged-runtime parity: 389/389 exact control decisions and
  461/461 exact candidate decisions, zero mismatches.

Candidate and control differ only in `logit_head.weight` and
`logit_head.bias`. The official adapter, legal-action handling, public belief
tracker, greedy/autoregressive semantics, deck, and every other model tensor
remain frozen. Opponent identity and hidden opponent cards are unavailable to
the acting policy.

## Fresh panel

Collect exactly 3,200 strict official-engine games, with Rocket always seat 1:

- 800 `control` games and 800 `candidate` games against V110;
- 800 `control` games and 800 `candidate` games against Recovery V4.

Four shards contain 400 games per arm:

| Shard | Opponent | Seed |
|---|---|---:|
| `v110_01` | `AshMarnieGeneralV110_extracted` | 800101 |
| `v110_02` | `AshMarnieGeneralV110_extracted` | 800102 |
| `recovery_01` | `HelbrosMarnieGeneralRecoveryV4_extracted` | 800201 |
| `recovery_02` | `HelbrosMarnieGeneralRecoveryV4_extracted` | 800202 |

Each shard randomizes its arm schedule before play. Files are written under
`artifacts/team_rocket_recovery/n11_fresh_epoch4_confirmation/panel/`. Only
line counts, manifests' existence, and process health may be monitored during
collection. Outcome fields and win counts stay unopened until all four shards
contain 800 rows and all four manifests close.

Any interruption, engine error, timeout, illegal action, exception fallback,
wrong hash, wrong seat, incomplete shard, or missing manifest fails N11. N11
will not be resumed or restarted. A later attempt would require a separately
preregistered protocol and entirely new seeds.

## Locked confirmation gate

Analyze only the four complete N11 shards using 50,000 equal-opponent
stratified-bootstrap replicates with seed 800001. N11 passes only if:

- strict correctness passes across all 3,200 rows with zero fallbacks;
- every arm/opponent cell contains exactly 800 games;
- equal-opponent improvement is at least +2.0 percentage points;
- the two-sided 95% stratified-bootstrap lower bound is strictly above zero;
- the candidate point estimate is strictly better than control against each
  opponent.

There is one fixed candidate, no checkpoint choice, no subgroup rescue, no
sample extension, and no seed replacement.

## Authority boundary

A pass admits the candidate only to broad both-seat regression against the
frozen Rocket, Alakazam, and Soldier families. Submission still requires that
regression, exact packaged-runtime and deck audits, an immutable archive, and
then live arena evidence. The project completes only if the exact Team Rocket
submission is active, has at least 300 rated games, and is current rank one.

