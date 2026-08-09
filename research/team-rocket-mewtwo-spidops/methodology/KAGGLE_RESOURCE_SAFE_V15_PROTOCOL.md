# Kaggle Resource-Safe V15 confirmation protocol

Status: **frozen before V15 confirmation and packaging**  
Frozen: 2026-08-08 (America/Denver)

## Purpose

V15 is a confirmation/package identity for the two narrow tactical corrections
implemented in V14.  Its executable bot code, deck, and neural policy must be
byte-identical to the V14 candidate.  The new version number does not claim a
new strategy; it corrects the experimental design used to decide whether the
already-frozen intervention is safe to deploy.

V14's separate candidate and control games produced 41/72 and 42/72 wins.  Its
Cinderace proxy cells were 6/12 and 11/12, so V14 failed its written numerical
gate and is permanently closed.  A decision-trace audit subsequently proved
that the two policies did not share the same chance realizations and that the
Erasure Ball rule activated four times in the candidate Cinderace games.  The
four affected games split 2-2.  The unmatched 6-11 totals therefore cannot
identify the causal effect of the rule, but they also cannot be ignored by
retroactively admitting V14.

## Frozen executable intervention

Starting point: exact `HelbrosRocketSpidopsScaleBCV2`, Kaggle score 876.0.

The deck and `model.pt` remain byte-identical.  Only these public-state,
official-legality overrides are allowed:

1. On Team Rocket's Mewtwo ex's optional Erasure Ball discard prompt, choose
   no Energy when the base 160 damage (including Psychic weakness/resistance)
   already knocks out the opposing Active.
2. When the frozen neural policy selects Wobbuffet's Headbutt Bounce, redirect
   to an offered Retreat only if a Benched Spidops already has at least two
   Energy units including Grass and the board has at least three Team Rocket
   Pokemon.  Never intercept Rocket Mirror.

No Articuno bench reservation, early Spidops reservation, Brave Bangle rule,
public-range router, search layer, deck change, retraining, or other heuristic
is admitted in V15.

## Corrected admission method

The decision is based on deterministic properties of the submitted policy,
not on comparing unrelated random games.

V15 is eligible only if all of the following pass:

- exact ScaleBCV2 deck and model SHA-256 hashes;
- all unit tests and official-engine package smoke;
- exact replay of all 1,227 historical ScaleBCV2 Kaggle decisions;
- every action remains legal;
- all deviations are classified as one of the two frozen rules;
- no Erasure Ball override fires unless selection is optional, the target is
  knocked out by base damage, and every offered discard is the acting player's
  own Benched Energy;
- on the official V14 development traces, exact control replay has no
  unclassified disagreements;
- the four realized Cinderace interventions and their outcomes are reported
  without treating them as paired outcome evidence;
- clean extraction and a terminal official-engine smoke from the final archive.

The Erasure Ball local invariant is deliberately narrow: declining an optional
discard cannot reduce damage below the current Active's remaining HP, while it
preserves one or more Energy cards on the acting player's Bench.  This proves a
strictly safer immediate resource transition under the implemented card
effect.  It is not a claim that any particular replay would have changed its
terminal winner.

## Submission rule

If every deterministic gate passes, package one identity named
`HelbrosRocketSpidopsResourceSafeV15` and use at most the one Kaggle submission
authorized for 2026-08-08.  Do not claim 1,000 Elo until Kaggle reports it.  If
the score is below 1,000, retain the result as the external observation for the
next daily iteration and keep the long-running goal active.

