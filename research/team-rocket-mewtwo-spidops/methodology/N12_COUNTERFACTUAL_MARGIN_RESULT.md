# N12 Counterfactual-Margin Result

Status: **complete negative offline result, 2026-08-01**. No N12 model exists
or is authorized for official-engine match testing, packaging, validation, or
arena submission.

## Valid evidence

- Frozen Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Frozen live/control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Transfer audit: all 639 N6R exact-feature roots reproduce the current live
  action; all 48 hard-admissible counterfactual corrections remain new and in
  the current policy's three-action shortlist.
- Broad preservation support: 6,753 exact actions from 119 decisive N9 games,
  balanced across the two frozen opponent policies.
- Final v3 cache parity: 639/639 N6R and 6,753/6,753 N9 actions match, with
  exactly zero finite relative-logit error.
- V3 trainable scope: exactly 1,153 parameters in final `norm2` plus the action
  head; all other model and execution state stayed frozen.

The initial full-final-block implementations were operationally invalidated
before a delivered training record. V1 produced an empty directory. V2 wrote
one partial file during termination; that directory remains opaque,
uninspected, and excluded from every input and claim. V3 is the only analyzed
N12 training result.

## Locked v3 result

V3 completed all 400 fixed steps in 116.625 optimization seconds. The
correction loss moved from `1.9556686` to `1.9403193`, while preservation drift
remained very small (last-step N6 KL `0.0000358`, N9 KL `0.0000152`). However,
the policy selected the teacher-best action on 0/48 correction roots at step 1
and still 0/48 at step 400, in every opponent/turn cell.

Therefore none of the preregistered fit12/fit24/fit36/fit48 thresholds was ever
reached. The output contains only `training_report.json`, no checkpoint model,
and `admitted: []`. No official-engine development panel is permitted.

Immutable report:

- `artifacts/team_rocket_recovery/n12_counterfactual_margin_adapter/training_v3_norm2/training_report.json`;
- SHA-256:
  `4a32577c7dbfaec481bf3993de3d89985f0043a7c3483baca6249fb52a5b4947`.

## Interpretation and next boundary

N12 established that the exact counterfactual corpus transfers losslessly to
the current live model and that a post-attention context adapter can be trained
with essentially perfect preservation. Its fixed optimization strength was
far too small to cross even the weakest behavioral threshold: the un-clipped
gradient norm stayed near `4.4`, the cap was `0.25`, learning rate was `5e-7`,
and relative parameter displacement ended at only `1.11e-7`.

This is not match-strength evidence for or against the 48 corrections. A next
experiment may use the observed offline learning curve to preregister one
materially stronger, behavior-indexed optimization path, but it may not inspect
or reuse the invalid v2 partial and may not tune on official-engine outcomes.
Any resulting checkpoint still has to pass the same broad preservation gates,
locked balanced development, and independent confirmation. The live bot is
unchanged.

