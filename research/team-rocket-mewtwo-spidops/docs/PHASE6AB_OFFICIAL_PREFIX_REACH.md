# Phase 6AB: redacted official-prefix reach ledger

Phase 6AB isolates the upstream-reach limitation exposed by Phases 6Y and 6Z.
It starts a fresh genuine official battle for each split/candidate pair and
records every player selection from `battle_start` to player 1's second MAIN
turn. It does not reuse or modify the Phase 6Z value roots.

The result is deliberately partial. The selected actions have exact
probability one under the declared deterministic `PriorityBaseline` behavior
and sampling policies. The live competition adapter does not expose the
likelihood of automatic shuffle, Prize, draw, or coin outcomes, so official
chance reach, full sampling reach, and counterfactual importance weights are
serialized as `null`.

## Collection design

The collection mirrors the Phase 6Z split/style grid while executing new
battles:

| Split | Style | Candidate roots | Target root |
|---|---|---:|---|
| Train | `tempo` | 6 | Player 1's second MAIN turn |
| Calibration | `attack_rush` | 6 | Player 1's second MAIN turn |
| Validation | `control` | 6 | Player 1's second MAIN turn |

All 18 checked-in captures succeeded on attempt 1 and reached turn 4. The
candidate strata are Abomasnow, Alakazam, Crustle, Dragapult, Rocket Mewtwo,
and Starmie. These labels identify the declared collection conditions; they are
not inferred opponent-range predictions.

## Redacted step contract

Each `OfficialPrefixDecisionStep` contains:

- actor, turn, selection type/context, and legal min/max/option counts;
- SHA-256 digests of the allow-listed public state and conservative public
  history before and after the selection;
- a digest of the selection contract and a digest of the chosen option-index
  sequence, without the raw sequence or option payload;
- declared behavior and sampling probabilities of `1.0`; and
- explicit provenance flags for genuine `battle_select` execution and missing
  official chance probabilities.

Trace validation requires contiguous step indices, an unbroken before/after
public-state and public-history chain, and exact termination at the declared
root. Raw hands, decks, Prize cards, selected indices, option payloads, search
inputs, and private card identities are not serialized.

## Reach semantics

For a trace ending before player 1's later decision:

```text
declared player-0 behavior reach = product of player-0 policy factors = 1
declared player-1 behavior reach = product of player-1 policy factors = 1
declared player-only sampling reach = product of all policy factors = 1

official chance reach = unavailable
full sampling reach = unavailable
counterfactual importance weight = unavailable
```

The target player's own reach would be excluded from a player-1
counterfactual numerator, but that convention does not repair the missing
chance term. Treating unobserved chance likelihoods as one would be an invalid
promotion, so the dataclasses reject non-null values for all three unavailable
quantities.

## Result

| Metric | Result |
|---|---:|
| Later official roots | 18 |
| Candidate strata | 6 |
| Genuine `battle_select` executions | 512 |
| Player 0 decisions | 358 |
| Player 1 decisions | 154 |
| Steps per trace, min / max | 21 / 39 |
| Complete behavior-action ledger gate | **pass** |
| Declared deterministic behavior-reach gate | **pass** |
| Official chance-probability coverage | `0.0` |
| Official chance reach available | **false** |
| Full upstream reach available | **false** |
| Official counterfactual weights available | **false** |
| Phase 6Z value-root alignment available | **false** |
| Safe resolver / exploitation / official control | **false / false / false** |

## Official execution versus abstraction

Genuine official-engine execution:

- the 18 battles start through the licensed competition runtime;
- all 512 stored selections satisfy the live `SelectionContract` and execute
  through `battle_select`;
- the official engine advances its real internal rules and automatic chance
  events between observations; and
- runtime and compatibility-deck fingerprints match the frozen Phase 6Z source
  metadata.

Research declarations and abstractions:

- `PriorityBaseline` is a project policy, not an official strategy;
- its deterministic probability-one behavior/sampling factors are declared by
  the collector rather than reported by the engine;
- public-history allow-listing and all digests are project schemas; and
- the train/calibration/validation names are collection strata, not a fitted
  counterfactual estimator.

The adapter genuinely executes internal chance events but reports neither a
chance-outcome catalog nor likelihoods. Execution provenance is therefore not
probability provenance.

## Why the result stays fail-closed

The prefix cannot be joined to the separately collected Phase 6Z conditional
value surfaces because no shared episode/root token or joint collection exists.
Even a joined root would still need official or otherwise justified chance
likelihoods. The following remain disabled and machine-enforced:

1. complete upstream reach;
2. counterfactual importance weighting and value calibration;
3. counterfactual-value bounds;
4. safe subgame/endgame re-solving;
5. opponent-model exploitation; and
6. official competition policy control.

Phase 6AC implements the first half of that path in a fresh collection: each
behavior prefix, transient realized official truth sample, and five-plan value
surface shares the same root, with exact native root and forced-next-draw replay
parity. Because target and sampling use the same opaque official chance law, its
likelihood cancels symbolically without supplying absolute chance reach. Each
information set still has only one truth sample, however, so a calibrated range
and counterfactual-value model remain unavailable. See
`docs/PHASE6AC_JOINT_OFFICIAL_VALUE_SAMPLES.md`.

The next admissible path now requires repeated official truth samples behind a
defensible shared public information-set abstraction, complete opponent/player
reach accounting, collection-disjoint calibration, and lower-noise values.
Only then can calibrated bounds and the resolver admission audit be tested.

## Durable artifacts

Evidence fingerprint: `804eb749005aef537cb1cbfb96861369a76cc079e37c7896a2447cd0e6c97cdb`  
Dataset fingerprint: `130260bba10822829ebc38757ed6d5f4e6e0e89fef373f09c0a3c04b54075cf5`

| File | Bytes | SHA-256 |
|---|---:|---|
| `official_prefix_reach_dataset.json` | 748,248 | `18a8cec1e0bf2eb3d74ce06ec4783edec75c4b130764e991f5530752334d45a3` |
| `phase6ab_evidence.json` | 1,698 | `819d4b7bae57234c0bc2b8f47d3f9a5979c21a7d0130d6db55f777336e8b4888` |
| `phase6ab_evidence.md` | 1,350 | `05644832ac1514329169256b9ff2244f3dfd3d4c0024cae7329f580b4868bbb2` |

Recollect from the live official runtime with:

```powershell
python scripts/phase6_official_prefix_reach.py
```

Artifact-only validation:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_official_prefix_reach.py" -v
```

The competition runtime does not expose match seeding, so independent live
runs are not claimed byte-deterministic. No Kaggle submission or Pokemon TCG
performance claim is made.
