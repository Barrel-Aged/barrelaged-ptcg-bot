# N23 Held-Out High-Margin Gate Protocol

Status: **preregistered before opening held-out examples**  
Protocol: `rocket-n23-heldout-high-margin-gate/v1`

## Candidate mechanism

N23 composes the exact live Rocket policy with the rejected N21 terminal using
one deterministic all-or-live rule. At each deployed autoregressive selection
step:

1. compute attack-guarded, legal logits from both policies on the same
   observation and prior selected indices;
2. compute each policy's deployed top action;
3. if the tops differ and N21's top-versus-runner-up logit margin is at least
   `0.50`, use the complete N21 logit row;
4. otherwise use the complete live logit row exactly.

There is no learned gate, threshold grid, interpolation, residual, bucket
filter, checkpoint selection, or official-outcome selection. The threshold
`0.50` was frozen from the N22 training-only table, where it retained 124
candidate-only teacher corrections versus 2 live-only errors and made zero
exact-Rocket retention or N9 replacements.

## Frozen inputs

- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- N21 terminal model SHA-256:
  `7e98ef2eb0933b1444183f8b9238344df3b17220af638a4b2045e5c6f5aeba2b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- N19 corpus audit SHA-256:
  `d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed`;
- N21 result SHA-256:
  `53683ec3d968e59d4343c5b7294aec00d46b2e8d0a49ed7a6d54d6f4890b1e98`;
- N22 audit SHA-256:
  `1dd3904da69914c322fe6d4b0f4337d1a4ed6146533519353431c9f70c999dd8`;
- N22 result SHA-256:
  `df2bbb6343999ad2db77ea3e49256a9cac10cad838d768e353b4787ac6833b5d`.

All extraction, feature, attack-guard, full-relational-forward, and historical
support implementations are frozen by SHA-256 in the evaluator.

## Evaluation evidence

The evaluator may now reconstruct the frozen N19 validation and test splits for
both roles. It also evaluates the 469 unchanged N6 anchors and all 6,753 N9
examples. N19 train is reported only as a reproduction check against N22 and
has no selection authority.

For N19 examples, each autoregressive subchoice is evaluated under its recorded
prior selections. This is an offline action audit, not a full-trajectory or
official-engine claim. Any passing mechanism still requires a runtime build,
exact same-observation parity, and a separately locked both-seat official panel.

## Terminal offline gates

Every gate must pass:

1. exact source/deck/support hashes, legal masking, finite values, extraction
   counts, and deterministic reproduction are valid;
2. teacher validation and test gated top-1 each improve over live by at least
   +0.5 percentage points and gated recorded-action loss decreases;
3. the gate makes at least 10 replacements on each teacher held-out split;
4. candidate-only precision versus live-only errors is at least 80% on each
   teacher held-out split;
5. live-top retention is at least 99.75% on each exact-Rocket retention held-out
   split;
6. recorded-action top-1 on each retention held-out split is no more than 0.25
   percentage points below live;
7. the two retention held-out splits contain at least two replacements in
   aggregate, proving nonzero exact-Rocket relevance rather than deck-identity
   separation;
8. N6 unchanged-anchor and N9 overall top-action retention are exactly 100%,
   with every N9 opponent group exactly 100%;
9. no model, deck, live package, arena submission, or prior artifact is mutated.

Failure closes high-margin N21/live composition without an official panel.
Passage grants only authority to construct and parity-audit a dual-policy
runtime and preregister one both-seat official-engine development panel. It does
not grant confirmation, deployment, submission, rank, or strength authority.
