# Internal arena submission and preliminary strength estimate

Snapshot date: 2026-07-27

## Submitted artifact

- Bot: `HelbrosMewtwoSpidopsGTWTV1`
- Arena response: accepted into the inbox and promoted to the live ladder
- Bundle: `artifacts/submissions/HelbrosMewtwoSpidopsGTWTV1.tar.gz`
- Bundle size: 2,924,898 bytes
- Bundle SHA-256: `01c02185df68ae161d5883bbca2cb32431023e721e8064673f83200b548dc315`
- Frozen model fingerprint: `de6db2bc87a17ddaabe04ac626a4694ef69e69f8753e0f6b53d10c6dcbcafe4d`
- Frozen model SHA-256: `72d41b534610601ae8cdecc1db96f322cc639d12f71eaedbf5ad2f157d941b31`

The arena bundle is the explicit Watchtower compatibility build. It replaces
the research list's two Prism Tower cards with two Team Rocket's Watchtower
cards because Prism Tower is absent from the official engine card pool. It is
not an exact-deck result.

No Kaggle submission was made.

## Preliminary arena snapshot

At the recorded snapshot the bot had:

- rating: **355.2**;
- record: **4 wins, 26 losses, 0 draws**;
- observed win rate: **13.33%**;
- 95% Wilson interval for that win rate: **5.31%–29.68%**; and
- displayed ladder position: **563** in that ratings snapshot.

For context only, not as a controlled comparison, the same snapshot showed:

- `HelbrosMewtwoSpidopsPhase6V1`: rating 397.6, 78–256; and
- `HelbrosRocketSpidopsProV3`: rating 864.7, 945–785–2.

The bots differ in controller, deck details, opponent mix, and sample size, so
those figures are orientation rather than a causal ablation. The defensible
strength estimate is simply: **the submitted bot is currently weak in the
internal arena and is not close to the superhuman target**.

## Replay audit

Thirty arena games were pulled and replayed through the independently extracted
archive:

- 1,382 of 1,382 recorded decisions reproduced exactly;
- zero controller exceptions;
- 415 low-confidence tactical-provider deferrals (30.03%);
- 620 safe-resolve attempts and 15 accepted changes;
- 109 recorded actions differed from the tactical provider, including 68 main
  actions;
- 24 of 26 losses ended with more prizes remaining than the opponent;
- 9 losses contained no attack; and
- average bot compute recorded by the arena was 0.566 seconds per game.

The replayed plan/meta outputs were highly collapsed:

- `board_setup`: 1,235 of 1,382 selected plans;
- `energy_engine`: 143;
- `control`: 4;
- `prize_push` and `recovery`: 0;
- `tempo`: 1,362 of 1,382 dominant strategic objectives.

This rules out an import, legality, nondeterminism, or packaging failure. It
points to a learned policy/value/generalization failure under the arena field.
The machine-readable evidence is
`artifacts/internal_arena/HelbrosMewtwoSpidopsGTWTV1_replay_analysis.json`.

## Claim boundary

This is a preliminary, opponent-schedule-dependent strength estimate. It is not
a Kaggle result, a proof of equilibrium play, or evidence of superhuman
performance. Its value is that it tests the exact submitted artifact and
identifies the next learning failure to address.
