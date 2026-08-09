# Majkel Lucario Public-BC V2 protocol

Status: **frozen before V2 training outcomes**  
Frozen: 2026-08-08 (America/Denver)

## Hypothesis

V1 trained only the final transformer block and action head.  It improved
game-disjoint held-out top-1 accuracy from 38.59% to 60.35% and decisively beat
the existing same-deck controls, but its first 12 internal-arena games were
6-6.  V2 tests whether adapting the complete relational policy trunk learns
the public leader's Lucario-specific representations rather than only remapping
the final action logits.

## Frozen design

- Exact V1 source checkpoint, deck, 172-game public corpus, focal team, and
  whole-game split identities.
- Reuse V1's serialized validation/test split with
  `--frozen-split-summary`; no resampling.
- Training mode: `full`.
- Maximum 12 epochs, patience 3.
- Batch size 96, learning rate `1e-5`, weight decay `1e-4`, label smoothing
  `0.01`, seed `20260808`, CPU threads 6.
- Train examples remain Majkel1337 decisions from both wins and losses.
- Validation and test games remain excluded from optimization.

## Admission gates

V2 proceeds beyond offline evaluation only if:

1. held-out test top-1 is at least 63.35% (three absolute points over V1);
2. held-out test loss is below V1's `1.065826`;
3. validation loss is finite and early-stopping metadata is complete;
4. exact Lucario deck and runtime legality are preserved.

If those pass, V2 must beat V1 in at least 7 of 12 side-swapped official-engine
games and remain terminal/healthy against V121, Zoroark, Starmie, Rocket, and
Marnie probes.  V2 may replace V1 in the internal arena only if this direct
gate passes.  It may never be sent directly to Kaggle; any eventual promotion
is requested from the user after internal-arena evidence.

