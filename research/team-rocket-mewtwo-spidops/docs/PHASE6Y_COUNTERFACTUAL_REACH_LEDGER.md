# Phase 6Y: exact counterfactual-reach audit and native ledger

Phase 6Y adds the probability accounting that Phase 6X intentionally lacked.
It validates the counterfactual importance ratio on the exact microgame, then
recollects the native five-plan action surfaces while attaching a redacted
root-relative reach trace to every surface.

This is a reach-semantics milestone, not calibrated native counterfactual-value
evidence.

## Estimator convention

For target player `i` at information set `I`, the prefix contribution uses

```text
counterfactual reach = chance reach × reach of player (1 - i)
importance weight    = counterfactual reach / sampling reach
```

The target player's own reach is absent from the numerator. Multiplying a
sampled return by this ratio gives an unbiased contribution to the
unnormalized counterfactual action value when the sampling support is valid.
The corresponding weighted reach-mass contribution permits a conditional,
self-normalized estimate.

## Exact expectation audit

The enumerable `RocketPrizeDuel` audit uses a full-support uniform profile and
visits every state in every information set. For each of 24 information-set /
action rows, it compares:

- the direct chance-and-opponent-reach sum;
- the exhaustively enumerated sampling-probability × importance-weight sum;
- the corresponding normalized conditional action values.

Maximum absolute error is exactly `0.0`; the maximum exercised importance
weight is `2.0`. This proves the implementation on the non-official microgame,
not on the Pokémon engine.

## Native collection result

The fresh native collection contains:

| Quantity | Result |
|---|---:|
| Supplied hidden-allocation surfaces | 12 |
| Forced plan-action targets | 60 |
| Terminal continuations | 600 |
| Genuine native transitions | 70,741 |
| Unique public histories | 12 |
| Maximum histories per public information set | 1 |
| Native importance-weight range | `1.0`–`1.0` |
| Exact reach-semantics gate | pass |
| Native reach-ledger gate | pass |
| Native counterfactual-value gate | **fail** |

All 12 macro-game roots were already at player 1's first plan decision, so
their persisted strategic reach traces contain zero prefix steps. Player 1's
own pre-decision reach is therefore structurally `1.0`; because target and
sampling profiles match, the recorded counterfactual importance weight is also
`1.0`. This is a useful audit result: Phase 6X's values did not need a missing
self-reach correction at this particular boundary, but they still lack the
coverage required for a calibrated counterfactual-value provider.

## Why the value gate remains closed

Four blockers are explicit and machine-enforced:

1. Hidden completions and their uniform prior are supplied research particles,
   not an official or calibrated public posterior.
2. The ledger is root-relative and does not certify all upstream official
   chance and behavior reach.
3. Every sampled public information set appears only once, so no within-root
   hidden-state range is estimated.
4. No hypothesis-conditioned model or collection-disjoint interval calibration
   exists yet.

Consequently Phase 6Y cannot produce `CounterfactualValueBounds`, enter the
safe resolver, enable the exploitation overlay, or control official actions.

## Official execution versus abstraction

Genuine official-engine evidence:

- all 600 terminal continuations execute through the licensed native search
  API;
- all 70,741 descendant transitions are native engine transitions;
- each tactical selection crosses the existing legality boundary.

Research abstractions:

- hidden completions and their prior;
- the five turn-level plan actions and deterministic tactical lifting;
- the blueprint rollout profile, root selection, reach estimator, exact
  microgame, and admission gates.

The native descendants are genuine; the information-set game and probability
model are not official game semantics.

## Durable artifacts

Evidence fingerprint:
`d57936de3a85e10b3704470e984a46c1461c9b38db3f0b5d31a9bfcf3d974f3c`  
Admission fingerprint:
`d11f45cc12b95ddefd1ba4da578200d67cf61fc7f68b508185708103fcf8b38c`  
Native dataset fingerprint:
`2e786f3fe0690ed28e04244276a88f53adfc53d8cf3495b9a369ac067d8853e3`  
Exact audit fingerprint:
`be10998405ee7019f0ea997c22e5cd5da935a53150e7124ff519e3ff05219237`  
Source collection fingerprint:
`ed16778ad6e960aa195599c9d2fc49019a5476cf0fa7823aee1abd82e4714f4d`  
Source calibration fingerprint:
`c01102e18a194ccb142e883b7c5f48d78ef83996a110b3aa6ea3427158f5038f`

| File | Bytes | SHA-256 |
|---|---:|---|
| `source_opponent_deviation_dataset.json` | 172,947 | `e65d630de701bc333bda7651b9ab0856bd2cb99eb7cc5bf009f9901461fcc33b` |
| `source_opponent_deviation_calibration.json` | 2,329 | `a0fa9567069de56854b9bab3f5cf777a194600e0ccd3fffd11ee4563ffdeae74` |
| `exact_reach_audit.json` | 15,718 | `5e21383c3824e3804442246c099b7d320e1b786d34d69f8351a1f5a990ec6496` |
| `native_counterfactual_reach_dataset.json` | 52,763 | `5616186301fed837a670e91e87fee619834adca8952e7c40446b272ab4c53156` |
| `native_counterfactual_reach_admission.json` | 1,110 | `6aeea9cd749b86ce99308295835e856536c2d69d7afd8b91f2127dca2442794c` |
| `phase6y_evidence.json` | 1,796 | `592f0dca4b6b3c9695571e77895da21eaf0bd028799935e4db6a30baa1aebb9d` |
| `phase6y_evidence.md` | 1,140 | `e1ba74aeeeb3dc1b044827fb49f2b2986159eb179dd6ef9ba6361fa927c05cef` |

Reproduce the expensive native run with:

```powershell
python scripts/phase6_native_counterfactual_reach.py
```

Run artifact-only checks with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_native_counterfactual_reach.py" -v
```

The competition runtime is unseeded, so independent executions are not claimed
byte-deterministic. Phase 6Z subsequently constructs repeated, public-root-
aligned particle families at later decisions and reserves complete collections
for model fitting, interval calibration, and untouched validation. Its ridge
diagnostic is rejected and its exact observations still fix the acting-player
hand; preserving upstream match-root reach and calibrating a genuine public
range remain pending.
