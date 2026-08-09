# Target-80 randomized advantage V1 result

Completed: 2026-08-05  
Decision: **closed at validation; no runtime candidate, package, or submission**

## Useful outcome

V1 successfully produced the first large, audited randomized-action dataset for
this project:

- 1,200 official-engine games in 18 locked opponent/seat/split shards;
- 57,741 known-support decisions and 8,620 realized interventions;
- 74,694/74,694 Rocket observations passed the hidden-hand audit;
- no illegal action, fallback, duplicate replay, duplicate digest, or truncated
  game;
- exact frozen-token reconstruction held within `2e-5` on all cached choices.

This repairs N24's primary data defect: actions are no longer purely
deterministic. The artifact is reusable evidence, even though this first
learner did not qualify.

## Cache evidence

Only the final `*_v2.pt` caches have authority:

| Split | Source games | Cached games | Eligible choices | Exploratory choices | Cache SHA-256 |
|---|---:|---:|---:|---:|---|
| train | 720 | 720 | 34,402 | 5,141 | `b2bc46ce2d46bf5e9733051ca09c1e15c35fca7a24efeb0cb5801b168d1d038a` |
| validation | 240 | 239 | 11,997 | 1,763 | `fa6e4d6d8966d35f254edff772010080b1470ce43f333bed8f403c06ee4ed47d` |
| test | 240 | 240 | 11,342 | 1,716 | `886973ccb85bb98ffd971bfb487240b9a862f3df79b6f84aa65d943be0db34a7` |

One validation game had no eligible supported alternative, so it supplied no
action-identification row. It remains in the admitted source corpus and was
not removed based on outcome. The parity-only amendments are documented in
`TARGET80_RANDOMIZED_ADVANTAGE_V1_CACHE_PARITY_AMENDMENT.md`.

## Frozen validation result

All three predeclared configurations were fitted on train only and evaluated
together on validation.

| Configuration | Exploratory Q AUC | Reference AUC | State-only AUC | Q Brier | Reference Brier | First-choice Q–reference AUC gain | Valid frozen margin | Eligible |
|---|---:|---:|---:|---:|---:|---:|---|---|
| low regularization | 0.815 | 0.498 | 0.832 | 0.172 | 0.251 | +0.100 | none | no |
| balanced | 0.823 | 0.498 | 0.832 | 0.172 | 0.251 | +0.108 | 0.10, 0.15, 0.20 | no |
| high regularization | 0.828 | 0.498 | 0.834 | 0.168 | 0.251 | +0.097 | none | no |

The outcome heads clearly predict game results better than reference logits,
but that is not sufficient evidence of action advantage. In every
configuration, exploratory-action Q AUC remained below the corresponding
state-only AUC. Much of the apparent signal is therefore public-state strength,
not proof that the selected action caused a better result.

More decisively, every configuration failed the preregistered effective-sample
gate. Overall validation ESS ratio was 0.242, below 0.25. The minimum
opponent/seat ratio was 0.210 in Alakazam seat 1; Starmie seat 1 was also below
0.25. ESS is fixed by the logged propensities and weighting scheme, so no V1
model can repair this gate. Low/high regularization also lacked a frozen margin
with a validation change rate between 0.5% and 5%.

## Test handling

No V1 checkpoint was evaluated on the test cache, as required when no
configuration qualifies. Cache construction did expose aggregate test outcome
counts before validation selection; this means that cache cannot serve as a
pristine final admission set for a later model. It may be used as development
validation in a separately preregistered V2, followed by a newly collected,
untouched confirmation split.

## Next justified route

V2 should separate state prediction from action-effect estimation:

1. fit and freeze a cross-fitted state-only outcome baseline on train;
2. residualize terminal outcomes against that baseline;
3. train twin behavior-centered action-advantage heads only on the residual;
4. use a lower-variance, prospectively frozen overlap/stabilized weighting rule
   rather than V1's clipped `1/p` weights;
5. require Q to improve over the state-only baseline on exploratory actions,
   not merely over raw reference logits;
6. use the existing test cache only as V2 development validation, then collect
   a new seed-locked confirmation set after the model and margin are frozen.

The live V5 identity and all submitted bytes remain unchanged.

