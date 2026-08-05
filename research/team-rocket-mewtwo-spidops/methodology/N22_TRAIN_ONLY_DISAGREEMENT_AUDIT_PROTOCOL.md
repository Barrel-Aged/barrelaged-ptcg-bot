# N22 Train-Only Disagreement Audit Protocol

Status: **preregistered before analysis**  
Protocol: `rocket-n22-train-only-disagreement-audit/v1`

## Question

N21 improved held-out cross-deck leader imitation but failed exact-Rocket
top-action preservation. Before any state-conditioned composition is designed,
this audit asks a narrower question: on training evidence only, do N21 choices
that correct a recorded leader action have a public, Rocket-relevant signature
that distinguishes them from choices that overwrite the live Rocket policy?

This is a diagnostic, not training, model selection, official-engine match
evidence, or a candidate policy.

## Frozen inputs

- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- rejected N21 terminal model SHA-256:
  `7e98ef2eb0933b1444183f8b9238344df3b17220af638a4b2045e5c6f5aeba2b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- N19 corpus audit SHA-256:
  `d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed`;
- N21 training report SHA-256:
  `8f51c4ad18558ef7af48ca7a08fd9eb2e059a7e5f94290b0eb0a1f345357a0f6`;
- N21 result SHA-256:
  `53683ec3d968e59d4343c5b7294aec00d46b2e8d0a49ed7a6d54d6f4890b1e98`.

The extraction and full-forward dependencies are also frozen by SHA-256 in the
audit implementation.

## Permitted evidence

The audit may reconstruct only:

- the N19 `teacher/train` examples;
- the N19 `retention/train` examples;
- the historical N9 preservation examples, which are treated only as live-top
  overwrite evidence.

It must not reconstruct, score, bucket, inspect, or serialize any N19
validation or test example. It must not use official match outcomes, hidden
opponent information outside the actor observation, N21 validation/test
per-example results, or any post-N21 arena evidence.

## Fixed summaries

For the two N19 training roles, compare live and N21 deployed top actions and
report:

1. total agreement and disagreement counts;
2. on disagreements, candidate-only recorded correctness, live-only recorded
   correctness, and neither-correct counts;
3. logit-margin and recorded-action probability-delta distributions;
4. fixed selection-type, selection-context, top-option-type-pair, and selected
   card-overlap summaries;
5. fixed threshold tables for candidate margin, live margin, and their
   difference.

For N9, report disagreements by frozen opponent group and the same public action
context summaries. Card overlap means membership in the exact Team Rocket deck;
it is not a value or optimality label.

## Decision boundary

The audit may justify preregistering one new state-conditioned gate family only
if training evidence contains both:

- at least 100 candidate-only teacher corrections; and
- at least one fixed public bucket with at least 25 candidate-only corrections,
  correction precision at least 70% against live-only errors, and nonzero
  representation in exact-Rocket retention or N9 evidence.

Passage is feasibility evidence only. Failure closes N21 disagreement gating
without consuming validation/test examples. No outcome grants development,
deployment, packaging, submission, rank, or strength authority.
