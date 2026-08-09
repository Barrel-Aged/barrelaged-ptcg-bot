# Kaggle resource-safe V14 protocol

Frozen: 2026-08-08, before V14 official-engine outcomes.

## Objective and baseline

The active objective is a verified Kaggle score of at least 1000. The strongest
exact team result is `HelbrosRocketSpidopsScaleBCV2` at 876.0. Later V5 and V8
used the same deck/model but added broader guards and scored 748.0 and 719.5,
respectively. V14 therefore restores the exact 876.0 archive as its immutable
baseline rather than stacking more changes on V13.

Frozen baseline fingerprints:

- archive SHA-256:
  `27dff66b6d39d780e56a745c34039305ce5e76e0a17811253ed4315698f2c6f9`;
- deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.

V14 must not contain the V5 early-Spidops or Bangle guards, V8 Articuno bench
reservation, V13 router, a new model, or a deck change.

## Frozen interventions

V14 adds exactly two public-state tactical safety rules:

1. **Erasure Ball resource floor.** During the optional context-26 Energy
   discard, return the legal empty selection when Erasure Ball's 160 base
   damage already knocks out the opposing Active after weakness/resistance.
   Otherwise preserve the ScaleBCV2 neural action byte-for-byte.
2. **Wobbuffet exit.** If the neural action selects Headbutt Bounce, a legal
   retreat is offered, and a Benched Spidops is already able to use Rocket Rush
   for at least 90 damage, return the offered retreat. Never intercept Rocket
   Mirror.

Across the 20 archived ScaleBCV2 Kaggle games, the first rule has 12 observed
redundant-discard prompts and the second changes the two known Headbutt Bounce
mistakes in episode 89414044. These are immediate action-quality claims, not
claims that the unplayed branches win.

## Gates

1. Exact-deck/model hashes and official legality must pass.
2. Replay all 1,227 ScaleBCV2 Kaggle decisions. V14 may differ only on the two
   frozen intervention surfaces; every returned action must be offered/legal.
3. Run side-swapped official-engine control and candidate panels against the
   exact ScaleBCv2 mirror, current Marnie, Alakazam, Dragapult, flat Starmie,
   and a Cinderace tempo proxy. All games must terminate normally.
4. Advance only if V14 trails the exact control by no more than two wins
   overall, does not trail in the Cinderace cell, and has no material
   Alakazam/Dragapult regression. Because these rules are rare, parity is an
   acceptable local health result; local score noise is not strength proof.
5. Run an independently packaged archive smoke and verify the extracted
   artifact's deck/model/main hashes.
6. If every gate passes, consume at most one 2026-08-08 submission in
   `pokemon-tcg-ai-battle`, using the immutable identity
   `HelbrosRocketSpidopsResourceSafeV14`.

The Kaggle submission is the external strength experiment. A score below 1000
keeps the persistent goal active and supplies the next iteration's evidence;
it does not authorize a second submission on the same day.

