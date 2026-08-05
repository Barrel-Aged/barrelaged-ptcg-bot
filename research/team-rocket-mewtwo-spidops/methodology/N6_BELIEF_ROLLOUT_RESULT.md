# N6 Belief-Rollout Result

Status: diagnostic collection complete; distillation invalidated; no student
trained; no validation or arena submission authorized.

## Valid result

N6 collected 400 training roots and 240 independent selection roots from
official-engine source matches. Every root completed all 24 matched
particle/coin comparison sets across three actions. In total this was 15,360
matched sets and 46,080 terminal action continuations, with zero rollout
errors, no realized hidden-truth access, and no serialized raw hidden
allocations.

The independent signal replicated:

- training: 22/400 admissible changes (5.50%), with 42.5% half-best agreement;
- selection: 14/240 admissible changes (5.83%), with 37.5% half-best agreement;
- selection balance: 7/120 admissible changes against each frozen Marnie
  opponent.

These are valid diagnostic facts about the sampled public-belief rollout
teacher. They are not evidence that a trained policy is stronger.

## Distillation failure

The fail-closed trainer attempted to reconstruct each root feature from its
current observation and a fresh belief tracker. It reproduced only 391/400
training baseline actions; all nine failures were turn-4 roots. The live actor
has legitimate persistent public/own-visible memory, including prior reveal
history, deck-burn samples, and own prize-search knowledge. A current
observation cannot losslessly recreate all of that memory.

Public-log reconstruction did not solve the problem and introduced an
additional mismatch, confirming that the missing state includes information
not recoverable from the current public log alone. Excluding mismatched rows or
training against misaligned features would change the preregistered evidence
after inspection and could teach updates in the wrong state.

Therefore N6 stopped before writing any arm. Its training and selection files
remain analysis-only and are excluded from N6R training, validation,
deployment, and every strength claim.

## Corrective action

N6R recollects with the same official paired terminal-rollout method but
serializes a typed, hashed snapshot of the exact actor-visible policy features
at each root. The collector must prove that the snapshot independently
reproduces the frozen baseline action before evaluating alternatives. A
genuine two-root official-engine smoke passed 2/2 collector-time and 2/2
offline reproductions, including the turn-4 boundary that invalidated N6. That
turn-4 snapshot carried six legal known-prize feature entries; a fresh tracker
at the same main-phase observation would not retain the earlier prize-check
memory.
