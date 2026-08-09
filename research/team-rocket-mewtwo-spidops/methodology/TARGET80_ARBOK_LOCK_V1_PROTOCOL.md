# Target-80 Arbok Lock V1 protocol

Status: **frozen before development outcomes**  
Frozen: 2026-08-04

## Hypothesis

The exact 238-game V5 arena archive is 146-92 (61.3%), but only 70-61
(53.4%) against the Marnie family, which supplied 131/238 games. Team
Rocket's Arbok has a directly relevant public-state effect: while it is
Active, the opponent cannot play Pokemon with Abilities from hand except Team
Rocket's Pokemon. Against the dominant exact Marnie list this can deny new
Marnie's Grimmsnarl ex, Froslass, and Munkidori plays while the Rocket player
develops Spidops or attacks with Arbok.

This is a Wobbuffet-free package redesign, not a one-card substitution. It
removes one Wobbuffet, one Articuno, one Bug Catching Set, one Energy Search,
and one Basic Psychic Energy from V5; adds three Team Rocket's Ekans and two
Team Rocket's Arbok; and converts the remaining two Basic Psychic Energy to
Basic Darkness Energy. The resulting 60-card deck has 12 Basics, zero
Wobbuffet, three Ekans, two Arbok, seven Basic Grass, two Basic Darkness, and
four Team Rocket's Energy.

## Frozen arms

1. `control`: untouched `earlypad_bangle_lethal_v1`.
2. `deck_only`: `arbok_lock_base_v1`, using the unchanged V5 neural policy and
   unchanged EarlyPad/lethal-Bangle guards on the new deck.
3. `public_lock`: `arbok_lock_guard_v1`, using the same deck and model plus a
   deterministic public-information Arbok plan. The plan acquires the line,
   evolves and promotes Arbok, attaches Hero's Cape and Darkness-compatible
   energy, retains Potent Glare, and chooses Spinning Tail when the baseline
   would end or voluntarily release the lock. It yields to an immediately
   lethal active attack.

Frozen SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| V5 control deck | `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852` |
| V5/control main | `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6` |
| shared neural model | `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b` |
| Arbok deck | `7131d009a45586b008fd33ffaa0cbe1107f9e3c0c25ff09116e5cb02b2934749` |
| deck-only main | `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6` |
| public-lock main | `294e3e42041befc5823452c7d1cb4c09b862cd2982da388eb651397d2dc3facc` |

## Frozen development panel

Run all three arms for 20 games each against the exact extracted
`AshV178MarnieFullBest` bundle, with automatic side swapping, `max_steps=300`,
and `action_timeout_s=10`. The opponent identities are:

| Artifact | SHA-256 |
|---|---|
| opponent main | `0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b` |
| opponent deck | `92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d` |
| opponent model | `98072b2a75197ad9d073af3c7ba5ff66a6daadd266b86b5463fef2ff34910757` |

Do not inspect any development outcome until all 60 games finish. The earlier
one-game deck-only smoke is a legality check and is excluded from every
selection or strength total.

## Admission gate

At most one challenger advances. It must:

- finish all 20 games without an invalid action, exception, timeout, or
  nonterminal step-limit result;
- win at least 16/20 games (80%);
- beat the untouched control by at least four wins;
- win at least 7/10 games from each seat; and
- for `public_lock`, demonstrate at least four games in which Arbok became
  Active while a Marnie card was public.

If both challengers pass, select the higher-win arm; break an exact tie by
fewer mean steps, then prefer `deck_only` as the lower-complexity policy. If
neither passes, close both exact packages and do not submit either one.

## Downstream gates

An admitted arm must next pass:

1. a 20-game broad health panel: four games each against untouched V5,
   VioletAlakazamMKV93, StarmieFroslassCFV13, StarmieMistFlatV20, and
   SparkTrueCrustleCrispinGalvV71, requiring at least 14/20 overall, no
   zero-win cell, and no correctness failure;
2. an untouched 60-game confirmation containing 40 exact Marnie games and 20
   broad games, requiring at least 48/60 overall, at least 32/40 against
   Marnie, at least 14/20 broad, and all correctness gates; and
3. compile, deck-integrity, official one-game smoke, self-contained package,
   archive-member, and packaged-runtime checks.

Only one immutable successor identity may be submitted after those gates.
Arena completion remains at least 160 wins in that identity's own first 200
rated games; draws count as non-wins and identities may not be pooled.

## Claim boundary

Local official-engine execution proves legality and controlled matchup
behavior only. Arbok targeting is a deterministic public-state abstraction,
not hidden-hand access, genuine equilibrium solving, or an official-engine
belief API. No Kaggle submission is authorized, and no arena-strength claim
exists until an immutable submitted identity reaches the required record.
