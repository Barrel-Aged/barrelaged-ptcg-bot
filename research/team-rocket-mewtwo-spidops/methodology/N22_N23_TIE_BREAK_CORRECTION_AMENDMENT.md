# N22/N23 Tie-Break Correction Amendment

Status: **frozen after invalid N23 execution and before any corrective rerun**  
Amendment: `rocket-n22-n23-first-index-tie-correction/v1`

## Invalidated execution

The first N23 evaluator completed and wrote
`artifacts/team_rocket_recovery/n23_high_margin_gate/EVALUATION.json` with
SHA-256
`873efb37a8652ceb03935af461883bb9abc50f1bb360a9199d295011559966fb`.
Its preregistered `integrity_and_live_reproduction` gate was false.

The failure is implementation-specific and independently diagnosed: N22 and
N23 selected a top action with `numpy.argsort`, whose ordering of equal values
is not the deployed policy's first-index tie rule. The frozen N19 live losses
reproduced to numerical tolerance, while top-1 counts did not. The official
runtime uses `torch.argmax`, which returns the first maximum index.

The N23 output is quarantined. It has no offline-admission, development-panel,
deployment, strength, or family-rejection authority. Its held-out aggregates
must not be used to change the threshold, rule, gates, corpus, or candidate.

## Sole correction

The only permitted semantic change is:

```text
top = smallest legal index attaining max(logit)
runner = max legal logit after excluding that exact top index
```

This matches deployed `torch.argmax`. All of the following remain unchanged:

- N21 candidate and live model bytes;
- exact Team Rocket deck;
- candidate-margin threshold `0.50`;
- complete-candidate-row versus complete-live-row composition;
- all N23 held-out admission gates;
- N19/N6/N9 corpora and exclusions;
- no official-match or deployment authority.

## Corrective sequence

1. Run a new train-only correction audit to reproduce the frozen N19 live
   train top-1/loss metrics under first-index ties and emit corrected threshold
   counts. It must load no N19 validation or test example.
2. Continue only if teacher/train still has at least 100 candidate-only
   replacements versus at most 5 live-only replacements at margin `>=0.50`,
   retention/train has zero replacements, and live metrics reproduce.
3. Freeze the corrected train counts and a new evaluator hash.
4. Run one corrected held-out evaluation to a new output path. It must retain
   every original N23 gate. No further correction or threshold experiment is
   authorized from this family.

If either corrective execution fails, the high-margin composition family ends
without runtime construction or official-engine matches. The live submission
remains unchanged throughout.
