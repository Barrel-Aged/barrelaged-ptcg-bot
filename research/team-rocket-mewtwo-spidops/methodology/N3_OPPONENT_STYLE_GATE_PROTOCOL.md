# N3 Opponent-Style Gate Protocol

Status: preregistered before non-smoke N3 data collection  
Protocol: `rocket-opponent-style-gate/v1`

## Hypothesis

N2 conservative residuals improved against `HelbrosMarnieGeneralRecoveryV4` but regressed against `AshMarnieGeneralV110`, although both opponents use the same deck. N3 tests a separate public-history opponent-model overlay. The baseline strategy remains the exact frozen blueprint. A conservative `mean_logits_025` residual is enabled only after the style model emits consecutive high-confidence RecoveryV4 evidence; unknown or V110-like histories stay on the baseline.

This is an empirical exploitation overlay, not an equilibrium strategy or formal safety certificate.

## Information boundary

Classifier input is the sequence of 384-wide frozen-blueprint global tokens computed at Team Rocket's own decision prompts. Those tokens derive from the ordinary redacted observation, public history/logs, and Team Rocket's own visible private zones. The classifier never receives the opponent's acting observation, hidden hand, deck order, Prize identities, visualizer truth, cloned branches, future observations, source-policy file identity, or outcome.

Every sequence is collected while an exact zero-control actor plays complete official-engine games from seat 1.

## Split-isolated data

Each opponent contributes:

- train: 300 games; seeds V110 786101, RecoveryV4 786201;
- calibration: 100 games; seeds V110 786301, RecoveryV4 786401;
- untouched validation: 200 games; seeds V110 786501, RecoveryV4 786601.

Totals: 600 train, 200 calibration, and 400 validation games. Validation artifacts may be collected and sealed in advance but cannot be loaded before the classifier, temperature, activation rule, and artifact hashes are frozen.

## Classifier

- Layer-normalized 384-wide token input.
- One 64-wide GRU recurrent state.
- Linear binary logit head; label 0 is V110 and label 1 is RecoveryV4.
- Fixed initialization/training seed 786701.
- AdamW, learning rate 0.001, weight decay 0.0001, 30 epochs, gradient clip 1.0.
- Balanced episode batches; binary cross-entropy at every sequence step, with per-game normalization so long games do not dominate.
- No outcome, win/loss, N2 arm result, or hidden-information auxiliary target.

Calibration may fit one positive temperature and choose one activation threshold. Live activation requires at least three own decisions and two consecutive probabilities at or above the threshold. Once activated, the overlay stays active for that game; this prevents its own changed actions from feeding a later classifier decision.

Threshold selection maximizes RecoveryV4 game activation subject to V110 game false activation no greater than 1% on calibration. Ties select the higher threshold.

## Untouched classifier validation gates

All gates must pass on the 400-game untouched split:

- V110 false activation rate no greater than 2%;
- RecoveryV4 activation rate at least 50%;
- balanced final-step classification accuracy at least 80%;
- zero data/provenance, correctness, fallback, or information-boundary violations.

Failure stops N3 before any strength panel. Passing only authorizes construction of the gated policy.

## Gated policy and later strength gates

Before a strength panel, freeze the classifier, temperature, threshold, consecutive/minimum-decision rule, `mean_logits_025` artifact, runtime code, and exact deck/model hashes. The actor must:

1. maintain the classifier and all four residual memories on every own decision;
2. return the exact blueprint action before activation;
3. after activation, use 0.25 times the mean residual logits;
4. remain inside official legal options and the attack-safe END guard; and
5. fail closed to the blueprint if classifier or overlay inference fails.

A fresh mixed-opponent development panel must then show at least +3 points overall, a strictly positive stratified 95% lower bound, and no opponent regression below -2 points. A separate locked validation of at least 3,000 games is required before test, packaging, or arena submission.
