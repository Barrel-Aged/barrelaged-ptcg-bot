# N4 Public-Event Style Gate Protocol

Status: preregistered before non-smoke N4 collection  
Protocol: `rocket-public-event-style-gate/v1`

N3 showed that the frozen blueprint global token does not reliably preserve exact opponent-policy behavior. N4 replaces that indirect representation with a leakage-audited explicit public history.

## Features

At each Team Rocket decision, N4 accumulates only log records already present in Team Rocket's redacted observation and owned by the opponent. Card serials and opponent hidden-zone identities are never retained. The fixed 644-wide vector contains:

- 512 stable-hash bins for cumulative public opponent event type/field/joint tokens;
- 128 stable-hash bins for the current opponent public active, Bench, discard, attachment, and Stadium cards; and
- four normalized public scalars: turn, opponent hand count, deck count, and Prize count.

Allowed event fields are public engine fields such as event type, revealed card IDs, attack ID, public source/destination area, HP change, and result/reason. Opponent draw identities are absent from Team Rocket's redacted logs. Serial fields are forbidden.

## Split-isolated data

- Train: 200 games per style; seeds V110 787101, RecoveryV4 787201.
- Calibration: 100 games per style; seeds V110 787301, RecoveryV4 787401.
- Untouched validation: 200 games per style; seeds V110 787501, RecoveryV4 787601.

All games use the exact zero-control Team Rocket actor from seat 1 in the official engine. Validation may be sealed in advance but is not loaded before model and threshold freeze.

## Model and activation

- Layer normalization over 644 inputs and one linear binary logit head.
- Fixed seed 787701.
- AdamW, learning rate 0.001, weight decay 0.001, 30 epochs, gradient clip 1.0.
- Per-decision binary cross-entropy normalized per game; balanced game counts.
- One positive calibration temperature.
- At least three Team Rocket decisions and two consecutive probabilities above threshold.
- Calibration threshold maximizes Recovery activation subject to V110 false activation no greater than 1%; ties choose the higher threshold.

## Untouched classifier gates

- V110 false activation no greater than 2%.
- RecoveryV4 activation at least 40%.
- Balanced final-step accuracy at least 70%.
- Zero correctness, fallback, provenance, or information-boundary violations.

Failure stops N4 before strength testing. A pass authorizes a fail-closed gated actor that uses exact blueprint actions before activation and `mean_logits_025` afterward. That actor must then independently pass a fresh mixed-opponent development panel and a locked validation of at least 3,000 games before submission.
