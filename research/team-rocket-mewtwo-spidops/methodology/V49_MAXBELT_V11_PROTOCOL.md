# V49 Maximum Belt V11 protocol

Frozen: 2026-08-07, before V11 match outcomes.

## Motivation

The V10 epoch-4 learner improved its four-cell development aggregate from
V5's 54/96 to 63/96, including Dragapult 21/24, Marnie 13/24, Alakazam threat
proxy 24/24, and flat Starmie 5/24. It failed V10's preregistered 8/24
absolute Starmie floor and therefore cannot advance under that protocol.

An earlier, different Wobbuffet-free deck established a reproducible structural
direction: Hero's Cape to Maximum Belt improved two Starmie cells from 13/80
to 21/80, although that candidate remained too weak for deployment. Its result
explicitly allowed reconsideration inside a materially rebuilt policy. V11
tests that one-card package with the newly learned exact V5-distribution V49
checkpoint, while retaining Wobbuffet, two Murkrow, four Spidops, and V5's
Rocket density.

## Frozen candidate

`dragapult_v49_maxbelt_epoch4_v11` is byte-equal to the V10 epoch-4 package
except:

- Hero's Cape card 1159 becomes Maximum Belt card 1158; and
- the previously audited public Mega Starmie lethal guard attaches Belt to an
  active Mewtwo only for a legal 330-damage Erasure Ball line and selects the
  minimum offered discard count needed.

The learned model is not retrained or tuned after V10 outcomes. The early
Poke Pad-to-Spidops and exact-lethal Brave Bangle guards remain unchanged. No
opponent identity, private card, replay result, or sampled rollout is available
at runtime.

## Gates

1. Deterministic deck, model, and legal-option tests must pass. The 60-card
   sequence must differ from epoch 4 at exactly one position.
2. Run 24 fresh side-swapped official-engine games against flat Starmie. V11
   must win at least 8/24 with zero failures; otherwise close V11 immediately.
3. If it passes, run 24 games each against Dragapult, Marnie, and the Alakazam
   threat proxy. Require at least 18, 11, and 20 wins respectively, with zero
   failures. These floors permit at most the disclosed small tradeoff from the
   V10 epoch-4 development cells.
4. A passing V11 enters the same fresh 120-game confirmation geometry as the
   V10 protocol. It must win at least 64/120 overall, at least 11/30 in every
   cell, and return only legal terminal actions.
5. Only one confirmed archive may be submitted to the internal arena. No
   Kaggle submission or unverified live-V49 claim is authorized.

V11 is a new composite hypothesis, not a reopening or relabeling of the failed
2026-08-05 Maximum Belt candidate.
