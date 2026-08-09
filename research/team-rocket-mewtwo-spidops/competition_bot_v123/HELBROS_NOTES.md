# Helbros Mega Lucario V123 - calibrated safe hybrid

V123 keeps V122's exact public-leader deck, admitted neural checkpoint, and
attack-safe guard. It runs the live V121 deterministic Lucario clone alongside
the neural policy and uses V121 only when the neural top-action probability is
below the validation-selected threshold `0.35228917002677923`, or when the
neural action is unavailable/illegal. Both policies see every observation and
reset between games.

Frozen-split calibration (complete-action accuracy):

- validation: 60.84% neural -> 61.88% hybrid; 84/766 rows routed;
- untouched test: 57.52% neural -> 58.27% hybrid; 88/937 rows routed;
- untouched test first action: 60.51% -> 61.47%;
- repeat calibration artifact was byte-identical.

Preregistered official-engine panels, all side-swapped and terminal:

- 7-5 versus V122 parent;
- 11-1 versus V121 safety baseline;
- 2-10 versus Team Rocket ScaleBCv2 (known structural Lucario counter; V122
  previously scored 1-11);
- 7-5 versus Starmie CFV13.

This is an admitted internal-arena candidate, not verified 1000-Elo strength
and not authorization for direct Kaggle submission. The user must promote it
manually after arena evidence.

# Helbros Mega Lucario V122 - public-leader neural clone

V122 uses the exact ordered 60-card deck exposed by current public leader
Majkel1337 (submission `55333348`) and a neural policy fine-tuned on that
submission's public Kaggle decisions.  It is a behavioral clone, not copied
source code and not proof of equal Kaggle strength.

## Data and training

- 172 completed public episodes at the frozen cutoff.
- Only the uniquely identified Majkel1337 seat is used, in wins and losses.
- Whole-game deterministic split: 80% train / 10% validation / 10% test.
- Starting checkpoint: exact 876.0-Elo ScaleBCv2 neural policy with the leader
  Lucario deck substituted before training.
- Last transformer block and autoregressive head trained for 15 epochs; best
  validation checkpoint was epoch 14.
- Held-out test top-1: 38.59% source -> 60.35% tuned.
- Held-out test loss: 1.7462 source -> 1.0658 tuned.
- Tuned model SHA-256:
  `e80b691f50b82fbaecd61d3a9354cd727737a4e302d9a243fa5f91fcfffb8eea`.

## Official-engine development

All reported games were side-swapped and terminal:

- 12-0 vs same-deck untouched neural source;
- 11-1 vs local Majkel exact-deck rule V120;
- 11-1 vs live VerifiedClone V121 replacement control;
- 11-1 vs Zoroark Intent proxy;
- 11-1 vs fast Alakazam heuristic proxy;
- 7-5 vs Starmie CFV13;
- 1-11 vs ScaleBCv2 Rocket;
- 2-10 vs Marnie FullBest.

The last two are structural Lucario counter-matchups: local V120 went 0-8 into
ScaleBCv2 and 1-7 into Marnie.  V122 is submitted to the internal arena for
external calibration because it is materially stronger than both existing
same-deck Helbros controls and strong into several current public-leader
families.  It is not authorized for direct Kaggle submission; the user must
promote it manually after arena evidence.
