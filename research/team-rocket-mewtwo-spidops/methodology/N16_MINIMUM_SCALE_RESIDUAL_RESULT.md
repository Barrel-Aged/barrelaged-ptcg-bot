# N16 Minimum-Scale Residual Projection Result

Status: **complete; offline pass, development-panel failure, no confirmation or
deployment authority**  
Protocol: `rocket-n16-minimum-scale-residual-projection/v1`  
Completed: 2026-08-01

## Outcome

The outcome-independent constraint solver selected the sole permitted residual
scale `0.859655`. The resulting model retained all 48 corrections, improved
offline preservation, passed packaged execution, and was positive against both
development opponents. Its effect was nevertheless too small and uncertain for
the locked advancement gate. N16 does not advance, and the live bot remains
unchanged.

## Analytic construction and offline gates

Across 570 pairwise legal-option constraints, the maximum lower bound was
`0.8596539908266627`; the frozen guard and upward rounding produced `0.859655`.
The model SHA-256 is
`b4f35c7816097800b0c7265691f5f650bc9b467a4258612930d0fef07afcbcf7`.

Only the N15 final adapter projection was scaled; all other N15 tensors and the
runtime configuration remained exact. The fresh model retained 48/48 teacher
actions with minimum margin `0.0200083`.

| Gate | Result | Requirement |
|---|---:|---:|
| N6R non-target top-1 retention | 99.831% | at least 98.0% |
| N6R mean forward KL | 0.007383 | at most 0.05 |
| N9 overall top-1 retention | 99.230% | at least 99.0% |
| N9 V110 top-1 retention | 99.132% | at least 98.5% |
| N9 Recovery top-1 retention | 99.333% | at least 98.5% |
| N9 mean forward KL | 0.002934 | at most 0.02 |
| cached/full-runtime mismatch | 0/7,392 | zero |
| serialization mismatch | 0 tensors | zero |

The projection report SHA-256 is
`cb8b8f970ab913362180f89a396d19adca7a300bf65be3ed64f4990d08ee525a`.
The packaged audit then matched 964/964 official-engine Rocket decisions with
zero mismatch.

## Fresh official-engine development panel

All four new shards sealed before outcome analysis. The panel contained 240
control and 240 candidate games per opponent, 960 games total, with complete
provenance, Rocket always second, and zero exception fallbacks.

| Opponent | Control | Candidate | Difference |
|---|---:|---:|---:|
| V110 | 98/240 (40.833%) | 104/240 (43.333%) | **+2.500 pp** |
| RecoveryV4 | 102/240 (42.500%) | 104/240 (43.333%) | **+0.833 pp** |
| Equal-opponent aggregate | -- | -- | **+1.667 pp** |

The stratified bootstrap 95% interval was **[-4.583, +7.917] percentage
points**. The analysis report SHA-256 is
`540017e905985d7267e5f48be04e03493b23b94520e20ba40b5e7a34d7aabc1b`.

Advancement required aggregate improvement at least +2.0 points, strictly
positive improvement against both opponents, and lower bound above -0.5
points. N16 passed the per-opponent point-estimate gate but failed the aggregate
and uncertainty gates. A selected sample extension or confirmation is not
authorized.

## Interpretation

Reducing residual magnitude moved RecoveryV4 from N15's one-game negative to a
two-game positive while retaining a smaller V110 gain. Across two distinct
fresh panels, this residual family looks directionally useful but does not show
an effect large enough for the strength target. Additional sampling of the same
selected family would estimate a small local gain, not create one.

The next credible experiment should increase intervention coverage using the
existing official-descendant counterfactual evidence while keeping the
localized residual and preservation constraints. In particular, the frozen
N6R corpus contains utility estimates for 639 roots, whereas N14-N16 trained
only the 48 strict hard-change roots. A new protocol may define a broader target
set or counterfactual utility-ranking objective entirely from that corpus,
before training or match outcomes, with unchanged safety gates and new seeds.
N16 itself is closed.
