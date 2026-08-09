# Majkel Lucario Public-BC V1 protocol

Status: **frozen before training outcomes**  
Frozen: 2026-08-08 (America/Denver)

## Objective

Build a legal neural policy for the exact 60-card Mega Lucario deck used by
the current 1,214.1-Elo public leader, using only that leader's public Kaggle
observations and actions.  This is a behavioral-cloning experiment, not access
to the leader's source code and not a claim of strategy equivalence.

The locally available rule-based same-deck agent is not an adequate clone: on
20 game-disjoint sampled public replays it matched 613 of 1,148 decisions
(53.40%) and only 294 of 639 MAIN decisions (46.01%).

## Frozen inputs

- Public submission: `55333348`, team `Majkel1337`.
- Corpus cutoff: 172 completed public episodes available when the full corpus
  manifest was written.
- Replay selection: all 172 episodes; no private logs or hidden source.
- Deck: exact ordered 60-card deck from every focal replay, identical to local
  `helbros_mega_lucario_v120_majkel_exact_deck_built/deck.csv`.
- Starting policy: the exact neural checkpoint from the 876.0-Elo
  `HelbrosRocketSpidopsScaleBCV2` package, with only `deck.csv` replaced by the
  leader's exact Lucario deck before training.
- Split seed: `20260808`.
- Split unit: whole Kaggle games, 80% train / 10% validation / 10% test.
- Focal behavior: every uniquely identified `Majkel1337` action in both wins
  and losses.  Opponent actions are excluded.

## Training surface

Train the last transformer block and autoregressive action head (`mode=last`)
for at most 15 epochs with early stopping:

- batch size 96;
- learning rate `3e-5`;
- weight decay `1e-4`;
- label smoothing `0.01`;
- patience 4;
- CPU threads 6.

Optional empty selections and action sequences shorter than `maxCount` remain
unsupported by the inherited action head and are excluded with explicit
counts.  All fully supported multi-select decisions are unrolled into
conditional next-choice targets.  No validation or test example may enter the
optimization stream.

## Admission gates

The trained checkpoint is eligible for official-engine development only if:

1. training finishes without non-finite values;
2. the source and tuned checkpoints, split game identities, extraction counts,
   and metrics are serialized;
3. tuned held-out test top-1 accuracy improves over the source checkpoint by at
   least 10 absolute percentage points;
4. tuned held-out test cross-entropy is lower than the source;
5. the exact leader deck fingerprint is preserved;
6. a clean package returns only legal actions and completes terminal official-
   engine games.

Official-engine development must then include the local same-deck V120 policy,
the untouched neural source, Rocket ScaleBCv2, Marnie, Starmie, Alakazam, and a
current Zoroark proxy.  Results are matchup evidence, not Kaggle claims.

## Internal arena and Kaggle boundary

If the deterministic and official-engine gates pass, submit one candidate to
the internal arena as `HelbrosMajkelLucarioPublicBCV1`.  Do not submit it to
Kaggle directly.  A Kaggle promotion request is allowed only after the
internal-arena identity is live and has accumulated credible strength evidence.
The user, not this agent, performs any Kaggle promotion.

