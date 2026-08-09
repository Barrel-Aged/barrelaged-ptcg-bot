# Target-80 randomized advantage V1 cache-parity amendment

Recorded: 2026-08-05  
Scope: verifier representation only; no data, model, target, configuration, or
admission-gate change

The first train-cache integration stopped on the first eligible decision,
before a cache was written and before any fitting. The logged JSON behavior
probability and chosen propensity were both exactly `0.85`. The verifier first
cast the probability vector to float32, yielding `0.8500000238418579`, then
incorrectly compared that representation with the original double at `1e-8`.

The cache builder now compares the original parsed JSON doubles at `1e-12`
before casting the stored tensor to float32. This strengthens source parity and
does not alter the stored float32 training representation.

- pre-amendment learner SHA-256:
  `14859418a975bf6305c421b34a06191a102de8bafd5f6cb95a6f2858cce27182`
- amended learner SHA-256:
  `91ec5a06adb5a3e135027899b124bbd92aee56f4d78fb764c4b0a878b86b45ab`
- failed cache output exists: **false**
- outcomes used to choose amendment: **none**

A second cache-only integration condition appeared in validation: one terminal
game contained zero eligible supported-alternative decisions. Such a game is
valid in the admitted 1,200-game corpus but supplies no action-identification
row. The builder now counts it as `zero_eligible_games`, preserves it in the
source-corpus audit, and omits only that empty game from the token cache. This
does not select by outcome and does not drop any eligible decision. To keep
builder provenance uniform, all fitted V1 models must use newly generated
`*_v2.pt` caches; the earlier train cache and any detached first-attempt held-out
cache are superseded and have no training or evaluation authority.

- final amended learner SHA-256:
  `a2fb92ab855e89a5c71ca3f32ec982f8b65dfec8d9c9f2721e1559fc5b71f946`
- configuration, seed, target, propensity, and validation/test changes: **none**

All frozen architectures, configurations, seeds, propensity weights,
thresholds, validation rules, and test gates in the learning lock remain
unchanged.
