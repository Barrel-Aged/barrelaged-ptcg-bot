# Single semantic sequencing rule: fresh full-match protocol

Status: frozen before the new full-match candidate was evaluated.

## Evidence and fixed rule

Both prior causal model families were rejected: the contextual pairwise head
failed selection, and the empirical three-rule table failed untouched
admission. Their six two-rollout panels are now development evidence. Across
180 official second-seat games, one exact public action pair was uniformly
non-harmful across every source where it occurred:

- blueprint: play Team Rocket's Transceiver
  `(type=7, card=1134, target=0, attack=0)`;
- alternative: play Team Rocket's Ariana
  `(type=7, card=1216, target=0, attack=0)`;
- support: 7 roots in 5 independent engine processes;
- observed terminal result: 5 improvements, 0 harms, 2 ties;
- mean two-rollout terminal delta: `+0.7142857143`.

The candidate contains exactly this one fixed sequencing rule. There is no
threshold or rule sweep. If the blueprint's first eligible early-main choice
of a turn is Transceiver and Ariana is among its semantically distinct top four
legal actions, choose Ariana. Otherwise preserve the blueprint exactly.

The tactical interpretation is narrow: when the draw/disruption Supporter is
already available, use it before spending Transceiver. This does not hard-code
later selections; the neural blueprint resumes immediately.

## Runtime boundary

The overlay is active only while all conditions hold:

- Rocket is going second;
- the opponent is publicly identified as Marnie;
- the selection is a single-choice main-phase root;
- it is the first such root encountered for the turn;
- the global turn is at most 8; and
- Ariana is in the frozen neural blueprint's semantic top-four shortlist.

It receives public observations and legal options only. It is an explicitly
bounded opponent-exploitation/sequencing overlay on the neural safe baseline,
not the project's final game-theoretic architecture, an equilibrium strategy,
or a hidden-state search policy.

## Promotion gates

Before any arena submission:

1. import, exact 60-card deck, deterministic reset, and legal-action smoke;
2. at least 60 fresh official second-seat matches against each of
   `AshMarnieGeneralV110` and `HelbrosMarnieGeneralRecoveryV4`;
3. zero engine errors, illegal actions, or runtime exception fallbacks;
4. at least five observed rule interventions in the combined match panel;
5. combined win rate at least 45.00%, above the equal-opponent historical
   baseline expectation of 42.08%; and
6. no opponent panel more than two wins below its corresponding historical
   60-game baseline expectation (42.50% for V110, 41.67% for RecoveryV4),
   corresponding to minimum panel records of 24-36 and 23-37.

Passing admits a candidate only for the internal arena's non-rated evaluation.
Submission is permitted only if that arena strength estimate is not weaker
than the current live Team Rocket bot. No rank or strength claim follows from
this protocol alone.
