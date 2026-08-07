# Team Rocket deck pause and revisit note

Status: **paused by user; no active deck work**  
Originally recorded: 2026-08-01  
Latest pause checkpoint: 2026-08-06 (after the BC2 Recovery V8 internal-arena
submission and GitHub publication update)

## 2026-08-06 superseding checkpoint

`HelbrosRocketSpidopsBC2RecoveryV8` is the final submitted identity. It
restores the byte-exact ScaleBCV2/V5 deck and model, retains the early-Spidops
and exact-lethal Bangle guards, adds the public-information Articuno bench-slot
reservation, and constrains only selected Wobbuffet Headbutt Bounce actions.
The accepted archive and full claims boundary are recorded in
`BC2_RECOVERY_V8_SUBMISSION_AND_PAUSE.md`.

The earlier zero-Wobbuffet restart requirement below is now historical rather
than mandatory. A preregistered third-Murkrow removal arm went 14-10 against
both V5 and V7 but failed 0-4 against Marnie and 0-4 against Starmie, so it was
not submitted. Future work may still pursue a zero-Wobbuffet package, but it
must freeze and train a coherent exact-deck policy and pass those matchup
gates; do not treat a blind removal as an already validated improvement.

No local monitor, evaluation, training, packaging, submission, or GitHub job
should be considered active after the V8 publication handoff. Do not submit to
Kaggle unless separately requested.

The first read-only live V8 snapshot was 8-3 at rating 778.4. Treat this only
as proof of promotion and execution; it is not a mature strength estimate.

The subsequent Target-80 effort produced the immutable Wobbuffet-free V6 and,
in the final authorized pass, `HelbrosRocketSpidopsArticunoV7`. V7 was accepted
into the internal arena inbox on 2026-08-05. It is the final submitted identity
for this pause. The user has now paused the deck again. Do not silently restart
collection, training, evaluation, packaging, arena submission, GitHub
publication, or Kaggle work.

## Historical 2026-08-05 restart brief (superseded above)

If this deck project is revisited, begin with a **deck-level rebuild that
removes Wobbuffet**. The user's explicit direction is to find a higher-value
use of that slot and incorporate the other queued improvements below. Do not
resume by adding another narrow heuristic to the current Wobbuffet list.

Treat removal as a design constraint, but do not treat any one card as a
validated replacement. Freeze complete 60-card candidates whose engines make
sense without Wobbuffet, then compare them against an untouched control in the
official engine. A package-level redesign is preferred to a blind one-for-one
swap.

The 80% internal-arena goal is **not complete**. V6 failed that gate, and V7's
mature arena estimate was not available at the pause. No matching local
monitor, evaluation, training, or submission process was running after V7 was
accepted into the inbox.

The user's latest direction is explicit: pause here. If work resumes, keep
Wobbuffet out of every new deck candidate, search for genuinely higher-value
additions, and carry forward the improvement queue in this document. Do not
interpret unfinished candidate files as permission to resume automatically.

The next session must not silently resume collection, training, evaluation, or
submission. First reread this note, confirm that the user wants to resume, and
then preregister the next experiment. V5, V6, and V7 remain immutable. Do not
submit anything to the internal arena or Kaggle merely because an unfinished
candidate directory exists.

The final pre-pause work admitted the exact-deck elite behavior-cloning model
into V7, but did not admit the failed Crustle/Neutralization Zone,
Miraidon/Crispin, Lightning-package, or raw-action sampled-teacher experiments.
V7 adds only the already retained early-Spidops and exact-lethal guards plus a
narrow public-information Articuno fifth-bench-slot reservation.

## Immediate resume marker

The next deck candidate must contain **zero Wobbuffet**. No tested direct swap
or attacker package has yet earned promotion. In particular, the coherent
Joltik/Galvantula ex and Team Rocket's Zapdos/Maximum Belt hypotheses were
implemented, smoke-tested in the official engine, and rejected in locked
side-swapped panels. Galvantula went 2-10 and reached a ready attacker in only
3/12 games. Zapdos executed its intended attack package much more reliably but
also went 2-10, versus 4-8 for its deck-identical control. Do not rerun either
exact package or describe it as an untested lead.

On a future restart, first refresh the legal card pool and search for a
genuinely higher-value opening Basic / second-attacker package with repeat
attacker continuity. The raw-action paired milestone teacher was executed in
V1R and failed its preregistered integrity, stability, and second-seat gates;
do not rerun that exact configuration. A successor learning experiment must
repair the Alakazam collector path and establish stronger game-disjoint label
stability before it can train the native policy.

## Exact pause checkpoint

The latest submitted identity is `HelbrosRocketSpidopsArticunoV7`. It is
immutable, contains zero Wobbuffet, and was queued in the internal arena inbox
for promotion on the next job. Do not pool its future results with V5 or V6.
Its bundle is:

- `work/HelbrosRocketSpidopsArticunoV7.tar.gz`
- SHA-256:
  `8858324f993656691caa59700b968b88390e3edce94068df8dfb5cf28e455289`
- deck SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`
- model SHA-256:
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`
- main SHA-256:
  `d690d9f316180ebd87f5648257deacbaaf8a9bb7bd50a5231ed8b128f6be2484`

V6 did not approach the 80% target, and V7 had no mature rated sample at the
pause. The contemporaneous mature arena leaders were near 57%, so a future 80%
claim requires a genuinely structural advance, not another narrow overlay. The
arena may continue scheduling V7 while local work is paused, but do not start a
monitor or pull loop unless the user explicitly resumes evaluation.

The prior V5 identity remains the latest deeply archived diagnostic baseline.
It is immutable and contains one Wobbuffet. Its latest exact archived pull was:

- 1,025 rated games
- 563 wins, 462 losses, 0 draws
- 54.93% raw win rate
- exact pull: `work/arena_v5_pull_20260805_0144/`
- archived tree digest:
  `e3897738ea62f6e231dc4d5e2256290408ad0c01ddbe96002d597252e1ec1fcf`

The latest read-only arena snapshot at this pause showed V5 at 1,324-1,051
(55.75%) after 2,375 rated games. That snapshot was not pulled and archived
locally, so the 1,025-game artifact above remains the latest byte-exact evidence
package.

V5 also did not approach the 80% target. Do not combine its games with V4,
ScaleBCV2, V6, or any future identity.

Across the earlier exact 519-game diagnostic slice, V5's main loss signals
were structural:

- 144 of 231 losses did not establish a second ready attacker;
- 127 of 231 losses never attacked with Mewtwo;
- losses first attacked on turn 6.00 on average versus 4.86 for wins;
- Marnie was 141-140, Rocket mirrors 52-42, Alakazam 42-19, and
  Starmie 1-3;
- opening Tarountula was 103-59, Wobbuffet 32-25, Mewtwo 68-58,
  Murkrow 64-58, and Articuno 21-31.

The last row does not establish that Wobbuffet is harmful or uniquely useful.
The rebuild must deliberately replace its basic-Pokemon/opening role even
though Wobbuffet removal is now required.

The later 1,025-game pull confirms that this was not merely an early-sample
problem. Its largest opponent rows were Marnie Full at 163-198, Training
Marnie V90 at 90-91, the V4 Rocket training pin at 109-73, and the Alakazam
training pin at 65-22. The aggregate is useful diagnosis, not evidence that
any one replacement card is sufficient.

## Latest exploratory work at this pause

These experiments happened after the earlier 112-game checkpoint. V6 was the
first zero-Wobbuffet successor; the exact-deck elite checkpoint and Articuno
guard were later admitted together as V7. All other listed variants remained
research-only.

- **Wobbuffet-free V6:** the public-state Marnie-first overlay had strong
  development cells, but its immutable arena identity failed the first-200
  Target-80 gate. The latest read-only snapshot was 162-186-3. Keep the package
  as evidence, not as the restart substrate.
- **Paired Starmie causal panel:** 32 public roots and 768 official-engine
  branches found a positive oracle alternative at 17/32 roots, but the signal
  was fragmented. No fixed public rule met the locked support, gain, precision,
  and harm gates. The PPO route was also closed. Do not repeat either route
  unchanged.
- **Hancang structural recovery list:** the exact zero-Wobbuffet deck scored
  63-48-1 (56.25%) versus the transferred-policy control's 58-53-1 (51.79%) in
  a 224-game, seven-opponent, side-swapped panel. Its recovery spine was the
  strongest later structural signal, but it failed the absolute, comparison,
  start-order, and Starmie gates. Close the frozen transferred-policy package;
  retain the deck structure only as a candidate for exact-deck learning.
- **Maximum Belt Mewtwo/Starmie package:** replacing Hero's Cape with Maximum
  Belt plus a narrow lethal guard scored 21-59 versus the control's 13-67 in
  the locked 160-game Starmie panel. The `+8` wins were directionally useful,
  but 26.25% absolute strength, 5/40 starting second, and only two exact lethal
  attachments failed the gates. Close that exact intervention. Belt remains a
  possible component of a broader rebuild, not a validated one-card answer.
- **Crustle / Neutralization Zone package:** the structural wall route scored
  3-12-1 across its two locked Starmie cells while its deck-identical control
  scored 7-9. Dwebble appeared often, but Crustle was established in only four
  games, became Active in two, and Neutralization Zone was never established.
  Close this exact package; Mega Starmie's attack also bypasses effect-based
  damage prevention, so this is not the right primary counter architecture.
- **Exact-Hancang elite behavior cloning:** a fresh, zero-Wobbuffet,
  exact-deck-trained run completed 1,428 official-engine games with clean
  numerical and integrity checks. Its fresh baseline was 129-80-1 and its best
  selected iteration was 136-73-1 (64.76%). That was a real improvement, but
  it missed the locked 147/210 absolute gate, improved by only seven wins
  rather than the required fourteen, and failed both Starmie floors. Preserve
  `iteration_03.pt` as research evidence only; do not package or deploy it.
- **Exact-Hancang Starmie specialist:** four fresh specialist iterations used
  only the two locked Starmie opponents and completed 1,624 official-engine
  games. The frozen baseline scored 34-83-3. The best iteration improved to
  41-78-1, but missed the 90/120 overall floor, both 42/60 opponent floors, and
  the required +24-win improvement. Close this exact specialist route; its
  modest +7 result does not justify a router, package, or submission.
- **Miraidon / Crispin Starmie conversion:** the materially different
  Lightning-weakness route was mechanically legal, and Miraidon appeared in
  21/24 routed games, but the policy never produced a ready Miraidon or a
  Hadron Spark attack. The routed arm scored 2-21-1 against the two locked
  Starmie opponents versus 4-20 for its deck-identical control. Stage B was
  correctly not run. Close the exact list, route, and model combination rather
  than adding Sparkling Crystal as a post-hoc rescue.
- **Native turn-3 Rocket Rush V1R:** this formerly queued route is now closed.
  It executed Rocket Rush in 22/24 candidate games, including 15 turn-3 uses
  and 14 attacks into Staryu, but scored only 4-18-2 versus the deck-identical
  control's 8-16. Only 11/24 games reached two Spidops, two reached three, and
  two reached two simultaneously ready Spidops. Early Staryu pressure did not
  solve attacker continuity; do not train on or rerun this exact wrapper.
- **Tapu Koko ex / Crispin / Maximum Belt exploration:** the official prompt
  mapping was corrected and the mechanism was then tested in a locked,
  deck-matched 32-game panel. The routed candidate scored 5-11 versus its
  control's 4-11-1. Tapu Koko reached play in every routed game, became ready
  and used Linked Lightning in 11/16, and made six belted attacks, but the
  package failed every decisive advancement gate. Close the exact
  Koko/Crispin/Petrel package; do not resume from its earlier debug bundles.
- **Pikachu ex / Crispin / Sparkling Crystal exploration:** the official engine
  verified the intended two-Energy Topaz Bolt route. In the corrected open
  smoke, Pikachu reached play in four of five games; it became ready in two,
  and three Topaz Bolt attacks hit full-health Mega Starmie. The route went
  only 1-4, with Crystal acquisition and repeat-attacker/recharge continuity
  remaining unreliable. This was a health screen, not a sealed strength
  panel. No training, package, or submission was authorized. Treat the exact
  V1/V1R list and wrapper as closed unless a future deck-level redesign solves
  those structural costs.
- **Joltik / Galvantula ex package:** the corrected route established
  Galvantula in all 12 locked games, but produced a ready Galvantula in only
  three and an attack in four; the panel finished 2-10. `Charged Web` did reach
  at least 200 damage when used, so the failure was setup and prize-race
  continuity rather than attack text. Close this exact package.
- **Team Rocket's Zapdos / Maximum Belt package:** the corrected route proved
  the intended native-Rocket mechanism and attacked in 8/12 games, with ten
  attacks reaching at least 200 damage, but finished 2-10 versus 4-8 for the
  deck-identical control. Close this exact package; reliable execution of one
  attacker did not translate into a sustainable prize race.

## Top queued route when the user explicitly resumes

1. Keep **zero Wobbuffet** as a hard deck constraint.
2. Do **not** resume the rejected native Rocket Rush V1R, Tapu Koko/Crispin/
   Maximum Belt, or Pikachu ex/Crispin/Sparkling Crystal routes unchanged.
   Their intended attacks executed, but attacker continuity and win conversion
   remained poor.
3. Perform a fresh deck-level card screen for a stronger second attacker and a
   higher-value opening Basic. Reconsider whether Articuno, Murkrow, Mimikyu,
   or other support slots return enough value in a zero-Wobbuffet package.
   Score complete packages on setup cost, prize liability, energy recovery,
   repeat attacks, and broad-matchup value rather than on a single headline
   knockout.
4. Prefer the next materially different route: freeze the best coherent
   Wobbuffet-free list, then learn and calibrate a policy on that exact deck.
   The preserved Hancang recovery spine and exact-deck elite checkpoint are
   useful evidence, but neither is a deployable answer. If a new attacker is
   proposed, require an official-engine mechanism smoke before spending a
   sealed panel on it.
5. Start from the useful parts of the Hancang recovery spine—Night Stretcher,
   energy density, Poke Pad/Transceiver access, and damage-tool flexibility—
   but freeze a new coherent 60-card list before training.
6. Do not repeat the rejected Joltik/Galvantula ex or Team Rocket's
   Zapdos/Maximum Belt packages unchanged. Their locked screens confirmed that
   merely exploiting Lightning weakness or producing one high-damage attacker
   is insufficient. A successor must improve setup survival, repeated attacks,
   second-attacker readiness, and the full prize race.
7. Search higher-value additions by role, not novelty: a reliable opening
   Basic, a real second attacker, energy preservation/recovery, and exact prize
   conversion. Refresh the legal card pool and current competitive evidence at
   resume time; do not assume today's card pool or metagame is unchanged.
8. Train and calibrate the policy for that exact deck. Do not transfer the V5,
   V6, or ScaleBC model and interpret the result as the deck's ceiling.
9. Make starting second and both Starmie variants hard development gates. Also
   retain Marnie, Rocket mirror, Archaludon/Cinderace, Alakazam, Crustle, and a
   broad generalist in the regression panel.
10. Preserve official-engine legality, telemetry, replay capture, public-belief
   boundaries, safe-baseline behavior, and packaging. Use paired causal action
   values or a safe information-set resolver for learning; do not make a
   heuristic overlay or sampled rollout search the final strategic approach.
11. Require a fresh side-swapped broad panel and then an untouched 200-game
   confirmation before any successor arena submission. Do not submit to Kaggle
   unless separately asked.

- **Marnie-specific V5 diagnosis:** building multiple Spidops was the strongest
  observable correlate in the 281-game Marnie cell. Peak one Spidops was 9-54,
  peak two was 64-52, peak three was 60-27, and peak four was 8-2. This is
  observational, not causal proof. It motivates deliberate three-line setup,
  evolution access, and bench reservation in a rebuilt package.
- **Pinsir / Crispin acceleration:** the official engine verified that Crispin
  can put one basic Energy in hand and attach a different type, and that Pinsir
  can execute Slow Crunch. The radical build scored 1-11 against Marnie; the
  compact bridge scored 4-8 while the untouched V5 control scored 8-4. Close
  those exact packages.
- **Setup/evolution guards:** forcing a Tarountula start plus early evolution
  scored 6-6 against Marnie; the evolution-only version scored 7-5 versus the
  same 8-4 V5 reference. They did not justify advancement.
- **Mega Heracross ex counter package:** official traces verified the intended
  damage-response interaction and a four-prize line, but energy access and
  backup-attacker continuity failed. The coherent third build scored 3-9
  versus Marnie. Close the three exact builds.
- **Marnie Spidops swarm exploration:** V1 lost its official smoke because the
  bench filled with non-line Pokemon and produced only one Spidops. V2 contains
  an unfinished bench-reservation helper that is not wired into `agent()` and
  has never passed a smoke test. It is an abandoned scratch candidate, not a
  resumable or packageable bot. Preserve only the research idea: reserve bench
  capacity and actively build three Spidops lines without sacrificing attacker
  readiness or broad-matchup strength.
- **Team Rocket's Grimer / Corrosive:** the frozen Wobbuffet-free candidate
  scored 2-10, the same-deck control scored 2-10, and V5 scored 7-5. In five
  activation games Grimer was knocked out and discarded Energy by its effect;
  all five were losses. Close this exact package.
- **Team Rocket's Kangaskhan ex backup attacker:** scored 3-9 versus V5's 9-3.
  Kangaskhan reached play in three games, attacked in none, and all three were
  losses. Close this exact package.
- **Third Murkrow:** the direct Wobbuffet-to-third-Murkrow build scored 6-6
  versus a contemporaneous 3-9 V5 cell, but missed its preregistered 9-3
  absolute gate. Murkrow reached play in 11 games; Deceit was used on 26 game
  turns, but Torment was never used. It is not a validated replacement.
- **Learned public-value planner V2:** offline training on the exact 1,025-game
  V5 corpus cleared its held-out gates (validation AUC 0.787, test AUC 0.836;
  early-turn validation/test AUC 0.727/0.760), and the official resolver was
  healthy. In the sealed development panel it scored 8-11-1, exactly tying the
  same-deck control in wins and missing the required 15/20. Its 58 searches
  made 11 blueprint changes with zero errors, but those changes created no
  aggregate gain. Close this exact resolver. A future value layer must learn
  paired counterfactual action advantages rather than infer action quality
  from trajectory-outcome AUC.
- **Full NAIC Watchtower proxy:** the coherent available-card proxy (two
  Watchtower substituted for the unavailable Prism Tower) scored 11-17 versus
  V5's recorded 13-15 across Marnie, Starmie, and Alakazam. It collapsed 1-7
  against Starmie and won only 4/14 from seat 1. Close this exact proxy under
  the frozen V5 policy; it is neither the real Prism Tower list nor a suitable
  substrate for counterfactual training.
- **Mimikyu / Maximum Belt preservation package:** replacing Wobbuffet with
  Mimikyu and Cape with Maximum Belt produced 36-44 in an 80-game Marnie panel,
  versus 33-47 for the same frozen-deck control. The +3-win result remained
  only 45% absolute and did not clear advancement gates. Close the exact
  hand-authored preservation policy. Its clean 60-card deck can be retained
  only as a low-risk, Wobbuffet-free data-collection substrate for a genuinely
  learned policy.
- **Randomized-support intervention corpus:** the Wobbuffet-free
  Mimikyu/Maximum-Belt substrate produced 1,200 official-engine games across
  Marnie, Starmie, and Alakazam with both seats represented. The locked corpus
  contains 74,694 audited Rocket decisions, 57,741 exploration-eligible
  decisions, and 8,620 supported randomized choices, with exact behavior
  propensities and no hidden-hand leakage, illegal actions, fallbacks,
  duplicate games, or truncated games. This successfully establishes a clean
  intervention dataset; its corpus SHA-256 is
  `3dde57e3420070bf43f31ffd9fa38ac6a9015c6fa6cc3878a63dff776eb69602`.
- **Randomized Advantage V1:** three joint state/action models learned signal,
  but all failed the preregistered effective-sample-size gate and none improved
  on the state-only value baseline. The minimum opponent/seat ESS was 0.210.
  No checkpoint was opened on the locked V1 test split, packaged, or submitted.
- **Residual Advantage V2:** a five-fold cross-fitted state baseline plus
  behavior-centered residual action heads repaired the pre-outcome ESS problem
  (minimum group ESS 0.295), but all three regularization variants regressed
  against the state-only baseline on held-out development data. The best
  explored-choice Q-minus-state AUC delta was -0.0044; all variants also lost
  Brier and residual-error accuracy. No fresh confirmation, runtime package,
  or submission was authorized. The terminal win/loss label is too weak for
  reliable per-decision credit in this setting.

## Wobbuffet evidence to retain

- In the 20-game Kaggle review, Wobbuffet made only 2 of 103 attacks. Both were
  `Headbutt Bounce` choices in one episode where retreating to a ready Spidops
  appeared strategically preferable.
- A narrow Headbutt-to-ready-Spidops exit guard changed exactly those two
  recorded decisions but activated zero times across 4,044 decisions in 60
  fresh official-engine games. It was not submitted.
- In a later 16-game arena downswing slice, Wobbuffet used `Rocket Mirror`
  once and never used Headbutt in a loss. Larger problems were matchup
  pressure, backup-attacker development, and prize conversion.
- Competitive evidence supported Wobbuffet as a legitimate minority one-of,
  not a mandatory core card. The reason to remove it on restart is the user's
  deck direction and the opportunity to redesign the attacker package, not a
  false claim that the existing data proved the card harmful.

## Highest-value deck directions

Evaluate these as separate, frozen 60-card hypotheses.

1. **Honchkrow / Rocket Feathers rebuild**

   This is the most ambitious queued alternative and is not a drop-in
   replacement. It needs Supporter-discard planning, a resource generator, and
   likely a Porygon2-style engine. A historical replay exposed this unverified
   `TrainingPin - AshRocketFeathersV4` composition: 4 Murkrow, 3 Honchkrow,
   2 Porygon, 1 Porygon2, 2 Articuno; four each of Ariana, Archer, Giovanni,
   Petrel, and Proton; 4 Poke Pad, 4 Transceiver, 4 Roto-Stick, 3 Night
   Stretcher, 1 Ultra Ball, 1 Miracle Headset, 3 Factory, 4 Rocket Energy, and
   4 Ignition Energy. That exact deterministic package has now been tested and
   is closed: it scored 11-29 against a frozen five-opponent panel versus the
   V5 control's 24-16. A one-Spidops hybrid scored 7-33. Directly transferring
   V5's frozen neural policy was worse (0-20 pure; 1-19 hybrid). Do not rerun
   any of those four exact packages.

   The strongest preserved Wobbuffet-free baseline is the historical
   `HelbrosRocketFeathersMKV10` package. Its exact archived identity contains
   no Wobbuffet, uses Cody Graham's Air Balloon Honchkrow/Porygon2 list, and
   recorded 240-156 (60.6%) across 396 historical arena games. It is preserved
   at `artifacts/historical_bundles/HelbrosRocketFeathersMKV10.tar.gz`. This is
   a research baseline, not an 80% solution: its heuristic spine plus sampled
   rollout search must not be mistaken for the intended final game-theoretic
   planner.

2. **Wobbuffet-free Mewtwo / Spidops package rebuild**

   Preserve the core identity but use the released slot plus any necessary
   supporting slots to improve second-attacker readiness, Mewtwo activation,
   and start-second recovery. Reconsider opening-basic density, energy access,
   and whether Articuno or Murkrow is consuming more space than it returns.

3. **Mimikyu utility package**

   Mimikyu is the strongest direct competitive role comparator. The simple
   one-for-one Wobbuffet-to-Mimikyu candidate scored 26-14 in its frozen panel,
   below the untouched control's 31-9, so that exact swap is closed. Revisit
   only as part of a materially redesigned package with different opening and
   matchup assumptions.

4. **Lillie's Clefairy ex / closer package**

   The simple Clefairy replacement also scored 26-14 versus the 31-9 control.
   A larger NAIC-style attacker replacement scored 29-11 and was also
   rejected. A future version must justify Clefairy as a prize-map closer and
   support it coherently rather than repeat those exact swaps.

5. **Fresh high-value legal one-of search**

   Refresh the legal card pool and current competitive lists when work resumes.
   Shortlist cards by the specific deficits above: reliable opening basic,
   second attacker, energy acceleration/recovery, prize conversion, or
   disruption resilience. Verify card IDs, legality, 60-card count, and the
   packaged deck hash before policy testing.

6. **Joltik / Galvantula ex acceleration package (untested priority lead)**

   Build this only as a coherent package: enough Joltik and Galvantula ex to
   survive the vulnerable setup turn, sufficient Lightning Energy without
   starving Spidops/Mewtwo, and an explicit backup-attacker plan. Compare it
   with a Team Rocket's Zapdos / Maximum Belt package, which is more native to
   the Rocket core but appears slower. First run deterministic mechanism
   smokes; only then freeze at most one exact list for exact-deck learning.

## Queued strategic improvements

- **Second-attacker readiness:** make this an explicit development metric, not
  just an end-of-game autopsy. Track a ready backup before the first attacker
  falls, by seat and matchup.
- **Mewtwo utilization:** measure games in which Mewtwo attacks, the turn it
  first becomes ready, and whether resources committed to it delay Spidops.
- **Starting-second recovery:** V5 retained a severe seat asymmetry. Candidate
  selection must use side-swapped cells and a hard no-regression gate for the
  second-player row.
- **Early Spidops consistency:** the EarlyPad behavior is real and should be
  revalidated on each rebuilt deck, not assumed to transfer. V5's live replay
  pull recorded 29 activations.
- **Lethal prize conversion:** V5's lethal-only Brave Bangle rule reproduced
  exactly and its first five live activation games went 4-1. Preserve the idea
  only if the new deck still contains the same legal card/action context.
- **Marnie and Starmie:** keep both exact Starmie variants and the strongest
  available Marnie policy in the hard-matchup panel. Do not optimize solely on
  pooled win rate.
- **Three-line Spidops setup and bench discipline:** test whether Proton,
  Transceiver, Poke Pad, Bug Catching Set, evolution selection, and reserved
  bench slots can produce three useful Spidops without clogging the board.
  Measure the setup rate directly and require side-swapped, broad-matchup
  confirmation; do not infer causation from the archived Marnie correlation.
- **Higher-value Wobbuffet slot/package:** score candidates on opening quality,
  backup-attacker readiness, energy access, prize liability, and matchup value.
  Prefer a coherent multi-slot package over a cosmetic one-card substitution.
- **Lightning-package continuity:** for Joltik/Galvantula, Zapdos, or any later
  attacker, record setup-turn survival, time to first attack, Energy retained
  after attacking, availability of a second ready attacker, and performance
  when starting second. A one-time Mega Starmie knockout is not enough.
- **Archaludon/Cinderace resource walls:** include this family because prior
  Kaggle losses exposed backup-energy and Erasure Ball problems.
- **Learned strategic planning:** the public-belief SafePlan V1 resolver was
  runtime-clean but scored only 14-22 versus its V5 control's 12-24, failing
  the frozen gate because the aggregate gain was too small and one Starmie row
  regressed by two wins. Do not hand-retune that fixed leaf model. A future
  planning layer should learn matchup-conditioned public values on held-out
  data while preserving official-engine legality and fail-closed behavior.
- **Opponent-range upgrade:** use the preserved MKV10 evidence to replace its
  point-estimate deck identifier and prize-difference-only rollout leaf with a
  calibrated public-belief range and learned matchup-conditioned public value.
  Evaluate candidate actions on paired hidden-state particles and require a
  conservative improvement over the safe baseline before exploiting a
  predicted opponent tendency.
- **Target weak cells, not search depth:** historical MKV10 analysis identifies
  Trevenant, flat Starmie, Garchomp, and other weak matchup cells as the useful
  frontier. Deeper flat Monte Carlo was already expensive and did not establish
  a stronger general policy. Preserve official-engine legality, replay capture,
  telemetry, packaging, and benchmark gates while moving toward the documented
  neural/game-theoretic architecture.
- **Relearn after the deck is frozen:** do not transfer the current public-value
  network or V5 policy blindly to a new 60-card list. Freeze the Wobbuffet-free
  deck first, then rebuild training examples, matchup priors, action support,
  and held-out calibration for that exact identity.
- **Paired official-engine critical-root supervision:** the randomized-support
  corpus and terminal-outcome learners above are complete and must not be
  repeated unchanged. The next learning experiment should branch a small,
  preregistered set of strategically critical or genuinely ambiguous roots in
  the official engine, evaluate supported actions under identical
  public-consistent hidden-state particles and common random numbers, and
  supervise on paired action deltas or short-horizon strategic milestones.
  Focus on second-attacker readiness, Mewtwo activation, evolution/setup,
  prize conversion, and start-second recovery. Begin with a small stability
  pilot before authorizing another large corpus. Preserve the public-belief,
  safe-baseline, official-legality, telemetry, and fail-closed boundaries; do
  not make sampled rollout search the final strategic architecture.

## Closed exact interventions

Do not rerun these unchanged merely because the project has restarted:

- Wobbuffet to one Mimikyu: 26-14 versus 31-9 control.
- Wobbuffet to one Lillie's Clefairy ex: 26-14 versus 31-9 control.
- Wobbuffet plus two Murkrow to two Mimikyu plus Clefairy: 29-11 versus
  31-9 control.
- Unconditional EarlyPad extension through turn six: 11-13 in development.
- Turn-six extension capped at two Spidops: tied control, then tied an
  independent Marnie confirmation.
- Broad backup-Spidops Bangle rule: historical regression.
- Manual Starmie attachment guard: 9-31 versus 17-21-2 control.
- SafePlan V1 sampled public-belief resolver: failed the preregistered strength
  gate.
- Exact deterministic Honchkrow/Feathers package: 11-29 versus 24-16 control.
- Exact one-Spidops Honchkrow hybrid: 7-33 versus 24-16 control.
- Direct V5 neural transfer to pure Honchkrow: 0-20.
- Direct V5 neural transfer to Honchkrow/Spidops hybrid: 1-19.
- Pinsir/Crispin radical acceleration package: 1-11 against Marnie.
- Pinsir/Crispin compact bridge: 4-8 versus 8-4 V5 reference.
- Tarountula-start plus evolution guard: 6-6 versus 8-4 V5 reference.
- Evolution-only guard: 7-5 versus 8-4 V5 reference.
- Mega Heracross/Crispin/Energy Search package V3: 3-9 versus 8-4 V5
  reference; do not repeat V1-V3 unchanged.
- Team Rocket's Grimer / Corrosive package: 2-10 versus V5's 7-5.
- Team Rocket's Kangaskhan ex backup package: 3-9 versus V5's 9-3.
- Direct Wobbuffet-to-third-Murkrow package: 6-6; failed its 9-3 absolute gate.
- Mimikyu / Maximum Belt hand-authored preservation V2: 36-44 versus 33-47
  same-deck control; failed its absolute and improvement gates.
- Learned public-value plan V2 on the third-Murkrow substrate: 8-11-1, tied the
  same-deck control's eight wins despite 11 resolver changes.
- Full NAIC Watchtower proxy under the frozen V5 policy: 11-17 versus V5's
  recorded 13-15; failed absolute, matchup, comparison, and seat gates.
- Randomized Advantage V1 on the 1,200-game intervention corpus: all three
  variants failed the locked ESS gate and failed to beat the state-only value
  baseline; no test opening or packaging.
- Residual Advantage V2 with cross-fitted state values: all three variants
  regressed on held-out action-value metrics; no confirmation or packaging.
- Paired critical-root V1: valid causal branches, but only 11/16 roots met the
  particle-stability gate and just one stable plan change was found; closed.
- Persistent two-turn plan milestone V2: four positive stable changes across
  both seats, but only 50% particle and 41.7% coin stability; closed.
- End-of-own-turn plan milestone V3: passed all locked gates (12/16 stable by
  particles and coins, five stable changes, both seats, +0.310 mean paired
  gain), authorizing a larger selector corpus only.
- State-conditioned selector V4: 160 fresh training and 48 held-out roots were
  integrity-complete, but the best conservative selector changed only 3/48
  roots, reached 66.7% precision and +0.203 changed-root gain, and failed its
  recall and all-root value gates; no test, integration, or packaging.
- Compact engine-resilience rule V5: on 96 fresh roots the rule changed 14,
  averaged +0.253 paired gain per change and stayed positive in both start
  orders, but reached only 50% stable-label precision, had 21.4% negative
  changes, and narrowly missed the +0.04 all-root gate at +0.0368; closed with
  no full-match integration.
- Narrow Spidops-finish action guard V1: 51 interventions in a clean 200-game
  identical-deck direct match produced 106-94 (53%), with 61% starting first
  and 45% starting second; failed the 55% overall and both-start-order gates,
  so no broad panel or packaging.
- Exact Hancang structural recovery list under the transferred ScaleBC policy:
  63-48-1 versus 58-53-1 control; directionally positive but failed absolute,
  comparison, matchup, and both-seat gates.
- Hancang Cape-to-Maximum-Belt plus narrow Mewtwo/Starmie lethal guard: 21-59
  versus 13-67 control; failed absolute, family, both-seat, and mechanism gates.
- Starmie paired causal public rules and the locked PPO route: no fixed rule or
  trained policy cleared its preregistered safety/strength gates.
- Earlier narrow Marnie fine-tunes and residual routes documented in the N-series
  reports: no validated gain.
- Native Rocket Rush V1R: 4-18-2 versus its deck-identical control's 8-16;
  opening-window attacks were common, but the multi-Spidops attacker chain and
  win conversion failed.
- Exact-Hancang Starmie-specialist BC V1: best iteration 41-78-1 versus its
  34-83-3 baseline; failed every preregistered admission threshold, so no
  router, packaging, or submission was authorized.
- Joltik/Galvantula ex Lightning package V1R: 2-10; only 3/12 games produced a
  ready Galvantula and only four produced a Galvantula attack.
- Team Rocket's Zapdos/Maximum Belt package V1R: 2-10 versus its deck-identical
  control's 4-8 despite healthy attack execution; failed absolute and
  comparison gates.

## Required restart sequence

1. Confirm that the user explicitly wants to resume, then pull and archive the
   latest exact V6 and V5 arena runs if their counts have advanced; report each
   immutable record separately without pooling identities.
2. Refresh legal-card and current competitive-list evidence. Explicitly search
   for higher-value packages that replace Wobbuffet's opening-basic role while
   improving backup attack, energy access, prize conversion, or a weak matchup.
   Treat the completed Joltik/Galvantula and Rocket Zapdos/Belt packages as
   closed evidence, and search for a materially different higher-value package.
3. Freeze at most two materially different Wobbuffet-free hypotheses: a
   rebuilt Mewtwo/Spidops list and, only if still useful, an upgraded
   public-belief/value version of the preserved MKV10 baseline. Do not repeat
   the rejected one-card swaps, Grimer, Kangaskhan, third-Murkrow, or four
   Honchkrow packages above. Do not resume the unfinished swarm V2 or learned
   public-value V2 as if either were validated.
4. Prefer the clean Mimikyu/Maximum-Belt deck only as a provisional research
   substrate if no stronger coherent package clears the deck screen. Do not
   repeat the completed 1,200-game randomized-support collection or its two
   failed terminal-outcome learners unchanged.
5. Verify counts, legality, model/deck identity, serialization, and package
   safety; run a terminal official-engine smoke.
6. If pursuing the learned route, preregister a small paired critical-root
   pilot first. Use exact official-engine branches, public-consistent opponent
   ranges, common random numbers, and paired action/milestone deltas. Advance
   to a new game-disjoint corpus only if root rankings are stable enough to
   justify it; do not reuse N24 or the two failed randomized terminal-outcome
   models as causal supervision.
7. Run preregistered, seat-balanced development cells against Marnie, both
   Starmies, Rocket mirror, Archaludon/Cinderace, and broad control opponents.
8. Report both outcomes and intermediate strategic metrics: second attacker,
   first Mewtwo attack, early Spidops, prize conversion, and start-second rate.
9. Select at most one candidate, then run a fresh untouched confirmation panel.
10. Submit one new internal-arena identity only if confirmation and broad
   regression gates pass. Make no Kaggle submission unless separately asked.

## Durable artifacts

- V5 submission and live-pull report:
  `TARGET80_EARLYPAD_LETHAL_V5_SUBMISSION.md`
- V6 immutable Wobbuffet-free submission and failed gate:
  `TARGET80_WOBBFREE_V6_SUBMISSION.md`
- Hancang structural result:
  `TARGET80_HANCANG_STRUCTURAL_V1_RESULT.md`
- Maximum Belt/Starmie locked protocol and result:
  `TARGET80_MAXBELT_MEWTWO_STARMIE_V1_PROTOCOL.md` and
  `TARGET80_MAXBELT_MEWTWO_STARMIE_V1_RESULT.md`
- Starmie causal and PPO closures:
  `TARGET80_STARMIE_CAUSAL_V1_RESULT.md` and
  `TARGET80_STARMIE_PPO_V1_RESULT.md`
- rejected public-belief resolver: `SAFE_PLAN_RESOLVER_V1_RESULT.md`
- Kaggle/Wobbuffet review:
  `KAGGLE_SCALEBCV2_WOBBUFFET_AND_GAME_REVIEW.md`
- rejected exit guard: `WOBBUFFET_EXIT_GUARD_V1_RESULT.md`
- prior ladder diagnosis: `CURRENT_LADDER_DIAGNOSTIC_20260801.md`
- rejected Starmie attachment rule:
  `N26_STARMIE_SPIDOPS_ATTACHMENT_GUARD_RESULT.md`
- V5 archive:
  `artifacts/submissions/HelbrosRocketSpidopsEarlyPadLethalV5.tar.gz`
- V5 first replay pull: `work/arena_v5_pull_first50/`
- V5 exact 519-game pull: `work/arena_v5_pull_518/`
- V5 exact 1,025-game pull: `work/arena_v5_pull_20260805_0144/`
- V5 519-game autopsy: `work/arena_v5_519_autopsy/`
- stopped local monitor log: `work/arena_v5_to200_monitor.log`
- SafePlan candidate and traces: `work/target80_candidates/earlypad_bangle_safe_plan_v1/`
  and `work/target80_safe_plan_dev_v1/`
- rejected Wobbuffet-free package screen:
  `work/target80_honchkrow_screen/`
- rejected direct-neural transfer screen:
  `work/target80_honchkrow_neural_screen/`
- rejected Pinsir/Crispin candidates:
  `work/target80_candidates/pinsir_crispin_exchange_v2/` and
  `work/target80_candidates/pinsir_crispin_bridge_v3/`
- rejected Mega Heracross candidate:
  `work/target80_candidates/mega_heracross_counter_v3/`
- abandoned, unvalidated scratch swarm candidate:
  `work/target80_candidates/marnie_spidops_swarm_v2/`
- interrupted, unvalidated learned public-value candidate:
  `work/target80_candidates/murkrow_public_value_plan_v2/`
- learned public-value frozen protocol and serialization amendment:
  `TARGET80_LEARNED_PUBLIC_VALUE_PLAN_V2_PROTOCOL.md` and
  `TARGET80_PUBLIC_VALUE_V2_SERIALIZATION_AMENDMENT.md`
- learned public-value locked result:
  `TARGET80_LEARNED_PUBLIC_VALUE_PLAN_V2_RESULT.md`
- NAIC Watchtower proxy locked result:
  `TARGET80_NAIC_WATCHTOWER_PROXY_NEURAL_V1_RESULT.md`
- preserved historical Wobbuffet-free MKV10 package:
  `artifacts/historical_bundles/HelbrosRocketFeathersMKV10.tar.gz`
- provisional Wobbuffet-free Mimikyu/Maximum-Belt neural-control substrate:
  `work/target80_candidates/mimikyu_maxbelt_neural_control_v1/`
- N24 diagnosis motivating fresh randomized action support:
  `N24_CONSERVATIVE_NATIVE_IQL_RESULT.md`
- randomized-support protocol, audited corpus, and locked learning results:
  `TARGET80_RANDOMIZED_SUPPORT_ADVANTAGE_V1_PROTOCOL.md`,
  `work/target80_randomized_support_v1_corpus/CORPUS_AUDIT.json`,
  `TARGET80_RANDOMIZED_ADVANTAGE_V1_RESULT.md`, and
  `TARGET80_RESIDUAL_ADVANTAGE_V2_RESULT.md`
- paired-plan protocols and results:
  `TARGET80_PAIRED_CRITICAL_ROOT_V1_RESULT.md`,
  `TARGET80_PAIRED_MILESTONE_ROOT_V2_RESULT.md`,
  `TARGET80_OWN_TURN_PLAN_MILESTONE_V3_RESULT.md`, and
  `TARGET80_PLAN_SELECTOR_V4_RESULT.md`, and
  `TARGET80_COMPACT_ENGINE_RULE_V5_TEST_RESULT.md`, and
  `TARGET80_SPIDOPS_FINISH_GUARD_V1_DEVELOPMENT_RESULT.md`
- final pre-pause structural and learning screens:
  `TARGET80_CRUSTLE_ZONE_STRUCTURAL_V1_PROTOCOL.md`,
  `TARGET80_EXACT_DECK_ELITE_BC_V1R_PROTOCOL.md`, and
  `TARGET80_MIRAIDON_CRISPIN_CONVERSION_V1_PROTOCOL.md`
- rejected native Rocket Rush protocol/result:
  `TARGET80_NATIVE_ROCKET_RUSH_V1R_PROTOCOL.md`
- rejected Tapu Koko/Crispin/Maximum Belt locked panel:
  `TARGET80_TAPU_KOKO_CRISPIN_MAXBELT_V4_PROTOCOL.md`
- Tapu Koko and Pikachu/Crispin debug implementation and smoke artifacts:
  `scripts/target80_tapu_koko_maxbelt_main_v1.py`,
  `scripts/build_target80_tapu_koko_maxbelt_v1.py`,
  `work/target80_candidates/tapu_koko_crispin_maxbelt_route_v2r/`,
  `work/target80_tapu_koko_v4_locked/`, and
  `work/target80_candidates/pikachu_crispin_crystal_route_v1r/`
- Pikachu/Crispin/Sparkling Crystal health-screen closure:
  `TARGET80_PIKACHU_CRYSTAL_V1R_SMOKE_RESULT.md`
- exact-deck elite learning report and research-only selected checkpoint:
  `work/target80_exact_deck_elite_bc_v1r_seed806051/training_report.json` and
  `work/target80_exact_deck_elite_bc_v1r_seed806051/iteration_03.pt`
- rejected exact-deck Starmie specialist:
  `TARGET80_HANCANG_STARMIE_SPECIALIST_BC_V1_PROTOCOL.md`
- rejected Lightning package protocol, result, and machine-readable audit:
  `TARGET80_LIGHTNING_PACKAGE_SCREEN_V1_PROTOCOL.md`,
  `TARGET80_LIGHTNING_PACKAGE_SCREEN_V1_RESULT.md`, and
  `work/target80_lightning_screen_v1/analysis.json`
- rejected raw-action milestone teacher:
  `TARGET80_RAW_ACTION_MILESTONE_POLICY_V1R_PROTOCOL.md`,
  `TARGET80_RAW_ACTION_MILESTONE_POLICY_V1R_RESULT.md`, and
  `work/target80_raw_action_milestone_v1r_pilot/analysis.json`
- final V7 source, package, and provenance:
  `work/target80_candidates/hancang_wobbfree_elite_articuno_v7/`,
  `work/HelbrosRocketSpidopsArticunoV7.tar.gz`, and
  `work/target80_candidates/hancang_wobbfree_elite_articuno_v7/FINAL_ITERATION_PROVENANCE.json`
- publication-safe snapshot and private GitHub draft PR:
  `publication/team-rocket-mewtwo-spidops-research/`, branch
  `agent/publish-mewtwo-spidops-research`, commit
  `cb4c0e3aa459c2cc6c906faef39740010940c521`, and
  `https://github.com/Barrel-Aged/barrelaged-ptcg-bot/pull/1`

No local monitor, evaluation, training, packaging, submission, or publishing
process should be considered active after this checkpoint. The Target-80
objective is paused at the user's direction and remains unmet; pausing is not
evidence of completion or permission for another submission.
