# Phase 6H official-replay trajectory substrate

Date: 2026-07-26  
Status: deterministic logged training substrate validated; branchable game absent  
Evidence fingerprint: `16a70f2ed8597b5c6f0709a74dd0398cb3fe6456482b7737b98d8370664fb645`

## Outcome

Phase 6H converts the corrected, perspective-isolated official truth replays
into typed information-set trajectories for future full-game representation,
belief, value, and action-model training. Each acting-player decision contains:

- a common public-state and causal public-history fingerprint;
- a public-only range over six cataloged opponent deck hypotheses;
- a separate hash of the acting player's private observation;
- every legal ordered selection represented as an `AbstractAction`;
- the exact official behavior selection, when recorded;
- the next observed acting-player public-belief fingerprint;
- elapsed public/opponent frames and public event counts;
- terminal outcome supervision; and
- indices of later-reveal labels that may supervise only prefixes before reveal.

This is an offline semi-Markov corpus. It contains only transitions that
actually occurred, not counterfactual branches, and therefore is not an
`AbstractInformationSetGame`, a full-game simulator, or a safe resolver.

## Corrected official visualizer semantics

The Phase 6H contract audit found and fixed a Phase 6A alignment bug. In native
visualizer output, `selected` on frame N records the action that advanced frame
N-1. The previous converter attached it to frame N's new prompt and actor. On
the six source replays:

- all 610 recorded selections validate against the previous/source prompt;
- only 441 of 610 validate against the incorrectly paired current prompt; and
- 111 of those transitions cross to a different current actor.

Schema 2 now shifts each label back to its source prompt, suppresses terminal
phantom selections, validates count/distinctness/range against that prompt, and
excludes the behavior label from observation and public-history fingerprints.
It also removes all event card identities— including the observer's private
draw identities—from the common public-history chain.

The corrected public-replay dataset contains 1,232 isolated frames and 610
acting-observer decisions, all with contract-valid behavior labels. Its
fingerprint is
`3a9e8d1b740c0b79f986e10369823a668de57e9b1453c79384dbd1baacce4df7`.
The regenerated Phase 6A-B evidence fingerprint is
`60e3c49b3d6881dc2f868b8ac41e0b2e82db587b308a72741ab90a7bc521bcd8`.

## Trajectory evidence

| Metric | Result |
|---|---:|
| Official source matches | 6 |
| Isolated player episodes | 12 |
| Acting-observer steps | 610 |
| Contract-valid behavior labels | 610/610 |
| Observed transition links | 610 |
| Terminal links | 12 |
| Train / calibration / validation steps | 385 / 86 / 139 |
| Complete deck-range candidates | 6 |
| Maximum complete ordered action count | 260 |
| Public / private / action feature widths | 111 / 75 / 11 |
| Causal later-reveal supervision links | 8,794 |
| Paired common-public frames | 616 |
| Public state mismatches | 0 |
| Public history mismatches | 0 |
| Raw private identities persisted in steps | 0 |
| Public range NLL | 0.035248 |
| Uniform range NLL | 1.791759 |
| Public range Brier | 0.016393 |
| Uniform range Brier | 0.833333 |
| Public range top-1 accuracy | 98.03% |

The range updater uses only high-water counts of publicly visible opponent
cards and an equal prior over the six known local deck profiles. It never
receives the true opponent-deck fingerprint or future reveal labels as input.
These in-corpus metrics are a deterministic representation check, not held-out
population calibration, hidden-hand inference, or playing-strength evidence.

Public board/stadium and acting-player hand/search identities are encoded into
fixed-width deterministic card-hash buckets; raw private card IDs are not
persisted in steps. The trajectory dataset fingerprint is
`de42c3abb2ab32e334a7547189391be08b98c2148666572204421919a4d4203a`.
JSON restoration reproduces it exactly.

## Provenance cascade

The public-replay correction changes evidence bound to the replay fingerprint.
The deterministic Phase 6C-F chain was regenerated instead of retaining stale
manifests:

| Phase | New evidence fingerprint | Invariant result |
|---|---|---|
| 6C neural microgame | `ba58c83e4b83f0f072dd53cb8fdd3f08aa714a80fbc7058f4040950bf39a9be8` | selected checkpoint remains `2dab9a2d…`; NashConv remains `0.00004637` |
| 6D calibrated bounds | `829dc9714e55779b93f6189bbed0a1a75ae6ea9e341423e912ac827c37dafc50` | validation coverage remains 1.0; false-safe certificates remain 0 |
| 6E safe resolving | `369c083fc221933be4702dc9e86c43359b50233a42799369f8dc1708794955c3` | five endgame changes; strict non-regression remains |
| 6F exploitation | `c2c32fb61c4274383840a6071ae2ed979585628bf88553f46b3cdb1dbba43c13` | six enabled roots; max degradation remains `0.00999815` |

## Artifact hashes

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `official_trajectory_dataset.json` | 16,667,247 | `f3ac0dfcbf0146eca13e9e30053cea3f9d94d53db240a8771a80cf91c8325dcb` |
| `phase6h_evidence.json` | 2,366 | `b4fd8ea5e8910852498badab25ccb8069e6ef34425f4e237c408fad4e935da3b` |
| `phase6h_evidence.md` | 2,120 | `fce5735cf1cbe433466cda2fa481de6e624838f0013da469b32f87f192a06463` |

Reproduce from the project root:

```powershell
python scripts/phase6_foundation.py
python scripts/phase6_neural.py
python scripts/phase6_bounds.py
python scripts/phase6_resolving.py
python scripts/phase6_exploitation.py
python scripts/phase6_trajectory.py
python scripts/phase6_trajectory_neural.py
python -m unittest tests.test_public_replay tests.test_official_trajectory -v
```

## Remaining Phase 6 boundary

This milestone supplies causally aligned offline training data and a public-only
range baseline. Phase 6I now validates a split-safe neural representation and
logged-behavior bootstrap over it; see
`docs/PHASE6I_LOGGED_NEURAL_BOOTSTRAP.md`. Phase 6J later supplies a branchable
native-search transition boundary under a supplied hidden-state prior; see
`docs/PHASE6J_BRANCHABLE_NATIVE_SEARCH_GAME.md`. Phase 6K later connects both
with a small sampled-regret neural candidate, but does not validate equilibrium
convergence or improved play. These milestones still do not
provide a calibrated neural public-belief updater, equilibrium blueprint
trained on the full game, replay/full-game
counterfactual-value calibration, official safe
resolving, official exploitation, matched-compute neural ablations, human
expert review, the missing Prism Tower cards, or submission approval.
