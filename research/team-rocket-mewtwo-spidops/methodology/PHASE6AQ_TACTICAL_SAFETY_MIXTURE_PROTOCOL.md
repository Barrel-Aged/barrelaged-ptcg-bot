# Phase 6AQ: Calibrated Tactical Safety Mixture

Status: locked before calculating the validation confidence threshold or test
mixture metrics.

## Motivation and boundary

The expanded neural controller is strategically integrated and achieves 82.7%
held-out action top-1 accuracy, but its remaining tactical errors are costly in
complete games. Phase 6AQ does not replace the public-belief blueprint with a
heuristic strategy. It adds a bounded tactical safety provider: the frozen
ProV3 pilot may return its exact legal sequence only when the neural
autoregressive decoder's minimum prefix confidence is below a calibrated
threshold.

The neural controller always computes and updates beliefs, persistent plan,
strategic heads, meta-value, selective gate, and safe resolver. It controls all
decisions above the threshold. The tactical provider never changes the deck,
beliefs, plan distribution, value estimates, search gate, or legality rules.
Every deferral is logged. The archived policy's arena rating is not inherited
or claimed by the mixture.

## Frozen partitions and threshold

The complete replay-file split from Phase 6AN remains unchanged. Only
validation files select the threshold. Candidate thresholds are the distinct
observed minimum-prefix confidences. The chosen threshold is the smallest one
that reaches at least 95% exact sequence agreement with the frozen teacher
while leaving at least 50% of nontrivial decisions under neural control. If no
threshold satisfies both, this candidate is rejected.

The already untouched test files are then opened once. Admission requires:

- at least 93% exact sequence agreement overall;
- at least 90% agreement on nontrivial decisions;
- at least 50% of nontrivial decisions remain neural-controlled;
- zero invalid teacher sequences; and
- deterministic runtime serialization and parity remain within `1e-6`.

Complete official games against the legacy heuristic and tactical teacher are
fresh outcome evaluation, not threshold-selection data. If the mixture remains
materially weaker, the completion cycle restarts again.
