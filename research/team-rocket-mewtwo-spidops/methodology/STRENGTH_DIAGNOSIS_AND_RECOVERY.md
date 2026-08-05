# Strength Diagnosis and Recovery

Updated: 2026-07-30

## Current evidence

The strongest submitted Rocket bot is `HelbrosRocketSpidopsScaleBCV2`. At the
latest read-only arena snapshot on 2026-07-29 it was active at rank 23, rating
983.2, with a 2210-1766-2 record after 3,978 rated games. The current leader was
`HelbrosMarnieGeneralRecoveryV4` at 1064.6 after 329 games. The Rocket bot is
useful and legal, but it is not close to rank-one strength; its plateau is not
an artifact of too few rated games.

A public-observation audit of 610 official arena replays found:

- 344-266 overall, 56.4% win rate;
- 258-136 when starting first, 65.5%;
- 86-130 when starting second, 39.8%;
- 113-171 against the Marnie family, 39.8%;
- 71.5% against Rocket, 74.7% against Alakazam, and 64.2% against Soldier;
- 57.9% of losses never established a second ready attacker; and
- Mewtwo never attacked in 49.6% of losses.

Those associations do not prove individual action errors, but the seat and matchup gaps are too large to be sampling noise.

## Why the existing bot plateaued

The tactical backbone is a capable 4.3-million-parameter relational policy, but its policy prior was not learned through a large, deck-specific Rocket campaign. It can represent cards, zones, public belief features, targets, and legal options; that does not mean its weights encode the Mewtwo/Spidops setup, prize map, backup-attacker timing, or Marnie counterplay well.

The frozen provenance makes that limitation concrete. The model descends from
`v110_transfer` and was calibrated on 584 public leaderboard replay files
(38,886 supported choice targets), not a large corpus of expert Rocket games.
Its held-out action-imitation top-1 accuracy improved from 56.6% to 60.7%.
That is a useful generic tactical initialization, but it is not evidence that
the policy learned a strong Mewtwo/Spidops strategy.

The strategic Phase 1-6 work supplied interfaces, research abstractions, telemetry, official-engine integration, and experimental planners. It did not magically create a trained superhuman strategy. Heuristic plan labels and sampled rollouts still inherit the tactical policy's errors. When the prior prefers a poor attachment, evolution, or attack line, shallow search often evaluates variants of the same bad plan rather than discovering expert play.

The completed residual experiments confirmed this:

- N1 recurrent plan residual failed locked validation by 3.0 percentage points versus control.
- N2 ensemble gating had suggestive Recovery-specific development results but no arm cleared the confidence and worst-opponent gates.
- N3 and N4 opponent-style classifiers could not distinguish the two target opponents at a safe activation threshold.

None was submitted as a strength claim.

## N5 result: mechanically sound, strategically insufficient

N5 did change the neural blueprint rather than add another runtime overlay. Four independent policies were fine-tuned in 8,192 complete official-engine games, always from the weak second seat and equally against the current rank-one and rank-two opponents. The immutable-control development panel then ran 2,880 fresh official games.

No arm advanced. The best individual seed scored 212-268 versus the control's 207-273: a point difference of only +1.04 percentage points, with a 95% interval from -5.21 to +7.29 points. The other three seeds and the parameter mean were neutral or worse. This rules out a strength submission from N5; it does not prove that the small positive direction was false.

The likely failure is credit assignment. N5 supplied one terminal result to a long chain of card selections while training only the action/value heads and final relational block. It generated plenty of games but very little information about which early attachment, search, evolution, bench, or prize-map decision caused the result.

## N6 result and active correction: N6R

N6 changes the learning target. At an early Rocket decision it evaluates the blueprint and two semantically distinct alternatives under 12 public-consistent hidden-zone allocations and two shared coin streams. Every alternative is continued through the official engine to a terminal result. The 24 outcomes are paired by allocation and chance, and uncertainty is clustered by allocation.

This directly addresses the earlier rollout teacher's failure. Its two independent three-particle runs agreed on a nominal best action at only 22 of 95 roots and on a jointly declared robust intervention at only four roots. N6 refuses that noise: a hard alternative target needs at least 22 paired results, nonnegative evidence in both allocation halves, and a strictly positive clustered-bootstrap lower bound.

Search remains offline. Its values are distilled into the relational neural policy's final block and action head; no sampled-search module ships in the student. Four preregistered temperature/KL arms will be compared with the frozen control in 2,400 fresh official games. If one advances, it still needs independent validation, broader regression, exact packaged-runtime parity, and a public seat router that preserves the byte-frozen first-seat policy.

The sealed N6 training collection is now complete: 400 decision roots produced
9,600 complete matched particle/coin comparison sets, each evaluating three
actions (28,800 terminal continuations), with zero rollout errors. The two
six-allocation halves agreed on the nominal best action in 42.5% of roots, but
only 22 roots (5.5%) cleared every hard-change condition. All 4,800 sampled
hidden allocations were consistent with the frozen decks without prior-repair
adjustments. This is concrete evidence that most apparent small-search
improvements are too unstable to supervise directly. The independent selection
collection remains separate and cannot authorize deployment.

N6 did not proceed to training. A new fail-closed check found that reconstructing
features from the current observation plus a fresh belief tracker reproduced
only 391/400 frozen training actions. All nine mismatches were turn-4 roots,
where the live actor can carry legitimate prior reveal, deck-burn, or own
prize-search memory. Public logs were insufficient to recreate that state
losslessly. No N6 arm was written, validated, packaged, or submitted.

N6R is the corrective campaign. It recollects with new seeds and serializes a
typed, hashed snapshot of the exact actor-visible feature tensors at every
root. The collector must feed that snapshot back through the frozen model and
reproduce the baseline action before any counterfactual rollout. A two-root
official-engine smoke, including turn 4, passed both collector-time and offline
reproduction 2/2. The original N6 files remain diagnostic-only and cannot enter
N6R training or a strength claim.

N6R then completed with valid exact features, four lossless students, exact
runtime parity, and 2,400 strict development games. No arm advanced. The best
arm gained 1.25 percentage points in aggregate, lost 0.42 points against
Recovery, and had a 95% interval of -5.0 to +7.5 points. The immutable negative
result is recorded in `N6R_EXACT_FEATURE_RESULT.md`; no N6R artifact was
validated, packaged, or submitted.

## N7 result: the Marnie weakness is policy-specific

N7 replayed the exact live arena actor against 1,458 decisive second-seat
Marnie games, proved exact training/runtime feature parity, trained four
outcome-weighted behavioral-cloning epochs, and tested every epoch plus control
in 2,400 fresh official-engine games. No arm advanced.

The split result is the useful finding. Epoch 1 improved from 37.50% to 46.67%
against `AshMarnieGeneralV110` (+9.17 percentage points), but fell from 47.92%
to 37.50% against `HelbrosMarnieGeneralRecoveryV4` (-10.42 points). Every other
epoch showed the same sign pattern: better against V110 and worse against
Recovery. The apparent “Marnie matchup” is not a stationary target; the two
policies require different responses.

This forbids a blanket N7 deployment but makes the next experiment more
specific. N8 should keep the current live policy as the baseline-safe strategy,
infer opponent plan/style only from public history, and expose the V110 exploit
only under a stringent false-activation cap. If the two policies cannot be
distinguished early enough, the overlay stays off. This is the concrete
game-theoretic exploitation pattern described in the project pivot document:
opponent-specific gain constrained by a safe baseline. The immutable N7 result
is recorded in `N7_ARENA_MARNIE_SECOND_AWR_RESULT.md`.

N8A tested that router hypothesis with linear, nonlinear, and recurrent models
using public-event histories. On a 200-game diagnostic holdout, the only safe
models activated in 0% or 2% of V110 games, with the rare activation around
decision 35. The public history does not identify these sibling policies early
enough to support the exploit, so the sealed validation set remains unopened
and the router path is closed.

N8B therefore removes the identity requirement. The two frozen Marnie source
models share an identical deck and differ in exactly the action-head weight and
bias. N8B preregisters a minimal transfer of that Recovery-minus-V110 logit
delta onto the Rocket-adapted policy and demands improvement against both
opponents. This tests a robust shared-policy update rather than an unsafe
opponent-name proxy.

N8B completed 2,400 strict official-engine games. The alpha-1.0 delta improved
V110 by 0.42 points and Recovery by 2.08 points, for +1.25 points stratified,
but its 95% interval was -5.0 to +7.5 points and it missed the locked +2-point
gate. Other scales were neutral or negative. The direction suggests that a
small shared action-head update can be compatible with both opponents, but the
source-policy delta alone is not large enough. N8B was rejected without
validation or submission; the immutable result is in
`N8B_RECOVERY_LOGIT_DELTA_RESULT.md`.

## N9 result: the first broad positive candidate, not yet a pass

N9 collected 1,200 new exact trajectories balanced between V110 and Recovery,
proved all 72,499 focal decisions against the frozen actor, and trained only
the 385-parameter action head with opponent-balanced AWR and PCGrad. Four
epochs plus control then completed a locked 2,400-game official-engine panel.

Epoch 4 was positive against both opponents: +0.42 points against V110 and
+10.42 points against Recovery, for +5.42 points equally stratified. Its 95%
bootstrap interval was [-0.83, +11.67] points. The locked lower-bound gate was
strictly greater than -0.5 points, so epoch 4 missed by 0.33 points and N9 has
no advancing arm. The live bot remains unchanged.

This is materially more promising than the earlier failures because the
candidate preserved the non-Recovery matchup in point estimate while producing
a large Recovery gain, changed only two action-head tensors, and had perfect
runtime parity. It still cannot be submitted on a selected development result.
The next experiment is an untouched, preregistered, candidate-only
confirmation sized to account for development selection, followed—only on a
pass—by broader matchup, seat, packaging, and live-arena gates. The immutable
result is in `N9_BALANCED_EXACT_TRAJECTORY_AWR_RESULT.md`.

## N10 result: confirmation invalidated by restart nondeterminism

N10 froze epoch 4 as the only candidate and began a 3,200-game independent
confirmation. A user-requested host shutdown interrupted only the two Recovery
shards. Their partial files were archived without opening outcomes, and a
resume lock required byte-exact reproduction before any outcome analysis.

Both Recovery shards were rerun from game 1 with their original seeds and
completed 800 rows, but the frozen verifier found a mismatch at the first row
of `recovery_01`. All checked non-outcome provenance fields matched. Under the
resume lock, this operationally invalidates N10: its analyzer was not run, no
win counts were inspected, and no N10 row may inform selection or a strength
claim. The live bot remains unchanged. The durable result is in
`N10_INDEPENDENT_EPOCH4_CONFIRMATION_RESULT.md`.

This is not evidence against epoch 4; it is evidence that the official-engine
plus external-opponent process is not safely assumed to replay byte-for-byte
across process restarts. A valid replacement must use a completely fresh,
uninterrupted panel with new seeds for every cell and exclude all N10 outcomes.

## N11 result: the N9 signal was a development false positive

N11 supplied the valid fresh replacement: the same one preselected epoch-4
candidate, new seeds in all four shards, no N10 rows, and an uninterrupted
3,200-game official-engine panel. Correctness passed with exactly 800 games in
every arm/opponent cell.

The candidate gained only 0.375 points against V110 and lost 0.125 points
against Recovery. Its equal-opponent improvement was +0.125 points with a 95%
bootstrap interval of [-3.25, +3.50]. It therefore failed the locked +2-point,
positive-worst-opponent, and lower-bound-above-zero gates.

This rejects the N9/N10/N11 action-head update family. More sampling or another
epoch selection is not justified: a 1,600-game-per-arm confirmation estimates
the effect near zero. The next credible route needs decision-level
counterfactual credit and a stronger representational update, constrained by
the frozen live policy as a safety baseline. The immutable result is in
`N11_FRESH_EPOCH4_CONFIRMATION_RESULT.md`.

## N12 result: exact counterfactual transfer, underpowered intervention

N12 audited all 639 old N6R exact-feature roots against the current live model.
The live actor reproduced the old baseline on 639/639 roots, remained inside
every frozen three-action shortlist, and had learned none of the 48 roots that
previously passed the strict matched-rollout hard-change rule. This proves the
counterfactual corpus is still losslessly applicable; N6R's earlier failure was
not caused by feature staleness or live-policy drift.

The analyzed N12 v3 learner trained a 1,153-parameter post-attention context
adapter against those 48 direct max-margin targets while preserving 591 N6R
roots and 6,753 exact N9 actions. All 7,392 cached decisions had exact logit and
top-action parity before optimization. The locked 400-step run completed with
tiny preservation KL, but moved 0/48 corrections at every recorded boundary,
so no checkpoint existed and no match panel was authorized.

This clean negative result diagnoses intervention scale, not strategic value:
the un-clipped gradient stayed around 4.4, was capped to 0.25, and the `5e-7`
learning rate produced only `1.11e-7` relative parameter displacement. A next
experiment can transparently calibrate a stronger single path from this
offline result, with behavior-indexed checkpoints and the same preservation
and independent match gates. It cannot treat N12 as strength evidence. The
immutable result is in `N12_COUNTERFACTUAL_MARGIN_RESULT.md`.

## N13 result: global context reweighting cannot localize safely

N13 ran one preregistered 100x learning-rate calibration of the same exact
1,153-parameter norm2/head path. It completed 400 steps and moved the strict
teacher fit from 0/48 to 11/48, distributed across both opponents and turns.
The weakest fit12 threshold was never reached, so no checkpoint or match panel
exists.

This near miss is not a reason to extend the run. At step 400 the sampled N6R
and N9 preservation KL values were already about 0.172 and 0.106, far above
their final gate ceilings of 0.05 and 0.02. Correction loss improved while
broad drift grew, showing that one global reweighting of the frozen contextual
channels cannot isolate the desired states cleanly. A credible successor needs
an explicitly residual, state/option-conditioned adapter initialized at zero,
with unchanged preservation and match gates. The immutable negative result is
in `N13_CALIBRATED_NORM2_RESULT.md`.

## What "strong" means here

Local improvement against two frozen opponents is not rank-one evidence. A strong result must preserve the exact deck, execute only public/own-visible information, remain legal and load-safe, improve the second-seat Marnie weakness without destroying first-seat strength, and then sustain the rating in the live arena. Rank one is the target, not a promised outcome. Any campaign that fails its gates remains a research artifact and is not submitted on hope.

## N14 result: sufficient correction capacity, unsafe first-hit policies

N14 replaced N13's global context reweighting with a zero-initialized,
6,928-parameter nonlinear residual on each frozen contextual option embedding.
Construction had exact live-policy parity on all 7,392 cached support examples,
and every trained checkpoint matched its full runtime exactly.

Unlike N12 and N13, N14 fit all 48/48 strict counterfactual corrections. The
first-hit `fit48` checkpoint also passed both KL ceilings, but retained only
93.739% of non-target N6R top actions and 95.735% of N9 top actions. The gates
required 98% and 99%, respectively, plus 98.5% within each N9 opponent group.
Every earlier first-hit checkpoint was less preserving. No N14 checkpoint was
admitted to match play, and the live bot remains unchanged.

The fixed run continued to step 2,000 with 48/48 fit while sampled preservation
KL declined. Because the protocol saved only first-hit checkpoints, there is no
eligible terminal model and no full-corpus terminal retention result. This does
not permit post-hoc selection, but it motivates a separately preregistered
consolidation experiment: begin from immutable `fit48`, enforce the corrections
as margin constraints, and project toward the live policy on the complete
preservation supports at fixed checkpoints. The immutable N14 result is in
`N14_LOCALIZED_CONTEXTUAL_RESIDUAL_RESULT.md`.

## N15 result: offline-safe terminal model, promising but failed match gate

N15 prospectively replayed the exact N14 path and proved tensor equality at all
four archived prefixes before saving the sole step-2,000 terminal state. The
terminal adapter fit 48/48 corrections while retaining 99.831% of N6R anchors
and 99.126% of N9 actions. Every offline, serialization, and cached/full-runtime
gate passed. Packaged `NNAgent` parity then matched 1,049/1,049 official-engine
decisions.

The locked 960-game panel produced +6.25 percentage points against V110,
-0.42 points against RecoveryV4, and +2.92 points equally stratified. Its 95%
interval was [-3.33, +9.17] points. The candidate missed both the strictly
positive Recovery gate and the lower-bound-above -0.5 gate, so it cannot advance
or replace the live bot.

This is still useful diagnosis: a highly localized residual can improve one
frozen opponent materially while appearing nearly neutral against the other.
The next admissible test is an outcome-independent minimum-scale projection of
the fixed residual direction: preserve all 48 corrections at a locked positive
margin while minimizing incidental action changes. It must use new seeds and
the same offline and match gates. The immutable N15 result is in
`N15_TERMINAL_CONSOLIDATION_RESULT.md`.

## N16 result: safer direction, effect still below the gate

N16 analytically scaled the N15 residual to the smallest global magnitude that
kept all 48 corrections at margin 0.02. The unique scale was 0.859655. Offline
N9 retention improved to 99.230%, every gate passed, and 964 packaged-path
official decisions matched exactly.

In a fresh 960-game panel, N16 gained +2.50 points against V110 and +0.83
against RecoveryV4, for +1.67 points equally stratified. Its 95% interval was
[-4.58, +7.92]. It therefore failed the locked +2-point aggregate and
lower-bound-above -0.5 gates and cannot advance.

The projection improved the worst-opponent direction, but it also confirms that
the current 48-correction residual family is too small for the strength target.
The next experiment should broaden high-confidence counterfactual coverage from
the existing 639-root official-descendant corpus, not merely resample or tune
the selected N15/N16 panels. The immutable result is in
`N16_MINIMUM_SCALE_RESIDUAL_RESULT.md`.
