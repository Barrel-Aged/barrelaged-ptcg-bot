# Target-80 NAIC Watchtower Proxy Neural V1 build lock

Frozen: 2026-08-05, before the first official-engine candidate game

Candidate: `work/target80_candidates/naic_watchtower_proxy_neural_v1/`

| Artifact | SHA-256 |
|---|---|
| `deck.csv` | `325e493f2f7021ddb7cf79be0744fea1262a45efcf8d8129a1392f42e6766e1a` |
| `main.py` | `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6` |
| `model.pt` | `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b` |
| `deck_library.json` | `ed56d55cec6e2a87415db422b72f16baa8ab552e87c625de75cd0d8cb7e8d4e0` |

The deck parser returned 60 cards and all 23 unique IDs resolve in the bundled
official card metadata. It contains zero Wobbuffet, zero Murkrow, two Mimikyu,
one Lillie's Clefairy ex, two Team Rocket's Watchtower, one Team Rocket's
Factory, and one Maximum Belt. `main.py` and `model.pt` reproduce the frozen V5
neural control hashes exactly.

This lock authorizes only the smoke and sealed development screen in
`TARGET80_NAIC_WATCHTOWER_PROXY_NEURAL_V1_PROTOCOL.md`. It does not describe
the exact NAIC list, because Prism Tower is unavailable, and it does not
authorize training, packaging, or submission.
