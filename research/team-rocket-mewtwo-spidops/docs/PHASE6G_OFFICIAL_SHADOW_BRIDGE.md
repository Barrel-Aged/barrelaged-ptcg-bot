# Phase 6G official shadow bridge

Date: 2026-07-26  
Status: one-step official boundary validated; full-game planner not deployed  
Evidence fingerprint: `99218701a62bc2c9711fdb54868e2e27d04ab951125ae48da8dd60feb808c10e`

## Outcome

Phase 6G adds a production-disabled observation/action bridge around the
preserved competition agent. In the audit configuration, the official baseline
chooses first, then the bridge:

1. separates allow-listed public fields from the acting player's private
   observation;
2. creates a public-belief state and a deliberately coarse opponent range;
3. enumerates every legal ordered selection allowed by the native
   `SelectionContract`;
4. lifts the baseline's exact native selection into the abstract action space
   and back; and
5. returns the baseline's original list to the engine unchanged.

The bridge therefore cannot improve or damage play in this evidence run. It is
an integration and information-boundary audit, not a strength experiment.

## What genuinely executed

Twelve matches genuinely ran in the staged official `cg` engine: six legal
opponent profiles, both target seats, and the explicit Team Rocket's Watchtower
compatibility profile. The native engine supplied every observation and legal
option, validated each returned selection, and advanced every game. The target
agent made 566 audited decisions with:

| Metric | Result |
|---|---:|
| Exact native-to-abstract-to-native round trips | 566/566 |
| Bridge errors | 0 |
| Invalid abstract actions | 0 |
| Unknown native option fields | 0 |
| Hidden-canary feature changes | 0/566 |
| Own-private sensitivity passes | 557/557 |
| Native illegal actions | 0 |
| Native runner/policy fallbacks | 0 |
| Native soft timeouts | 0 |
| Shadow action changes | 0 |
| Maximum native options at one decision | 37 |
| Maximum complete ordered action count | 586 |
| Mean bridge audit latency | 1.391 ms |
| Maximum bridge audit latency | 11.700 ms |

The runtime fingerprint is
`9a62073d859273c86926a63c30a9a28dc16c698ad759ade3d1a7563b05af1634`;
the bridge fingerprint is
`8cbecb624f70ffa99e23cc069953258d2d6505954fc61e3450c3fe2ba75405d4`;
and the compatibility-deck fingerprint is
`af3bf9c3d2690c82f2326d2dfb900389ab769a58ee24255bab967e06ec7de315`.

The official engine exposes no match-seed control. Match outcomes, shuffled
observations, timing fields, and therefore this live bundle are not claimed to
be byte-deterministic. Structural checks are enforced on every observed
decision, and latency is omitted from the separate decision-content
fingerprint.

## Information boundary

`sanitize_official_public_state()` uses an allow list. It excludes log text,
serial/search identifiers, the opponent's hidden hand and Prize identities,
and all acting-player private identities. `official_private_observation_payload()`
fingerprints the acting player's own private cards separately. Option payloads
are private agent-visible decision context; every known field is preserved for
action lifting, while an unknown field fails the audit.

The opponent range created here is intentionally only a public aggregate
placeholder. It is not a learned hidden-hand posterior. The hidden-canary test
mutates forbidden opponent identities and requires identical public features;
the private-sensitivity test mutates visible own-private identity fields and
requires the private fingerprint to change when such fields exist.

## Exact claim boundary

This evidence validates only a one-decision official observation/action
boundary and complete lifting for all actions observed in the 12-match panel.
It does **not** implement or validate:

- a Markov/full-game abstract transition model;
- a full-game neural equilibrium blueprint;
- replay-derived full-game counterfactual-value calibration;
- official endgame or continual safe re-solving;
- an official opponent model or exploitation overlay;
- matched-compute blueprint/resolver/exploitation ablations;
- the exact NAIC deck, because Prism Tower remains absent; or
- a Kaggle submission, leaderboard gain, expert endorsement, or superhuman
  strength.

The Phase 6C-F neural, bound, safe-resolve, and exploitation evidence remains
restricted to the exact non-official `RocketPrizeDuel` microgame. Its game
fingerprints and provenance do not authorize it to control this official
bridge.

## Artifacts and reproduction

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `bridge_decisions.json` | 533,871 | `694254254ed054e59dab6f30bfe663b6f958ccdebcc60c369328a55e8c38d1af` |
| `phase6g_evidence.json` | 8,989 | `7711168b8787c662ee07965eefda94217744ca3dfc4c10a5e83d41cd91e0082f` |
| `phase6g_evidence.md` | 2,844 | `44ed97a4a733df5d7f2ed1e0ca2b1af62c24179492d175918ab423d51fe92063` |

From the project root, with the licensed runtime staged under
`work/official_kaggle/sample_submission`:

```powershell
python scripts/phase6_bridge.py
python -m unittest tests.test_official_bridge tests.test_phase6_bridge -v
python -m unittest discover -s tests -q
```

At the Phase 6G checkpoint, the complete suite passed 82 tests and all 360
then-current public exports resolved uniquely. Current repository-wide totals
are recorded in the Phase 6H evidence.
