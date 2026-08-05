# Target-80 Crustle + Neutralization Zone Structural V1 Protocol

Status: **completed; rejected**

Preregistered: 2026-08-05 (America/Denver)

## Purpose

Test one new, Wobbuffet-free structural answer to the live arena's dominant
Mega Starmie ex matchups. Crustle (card 345) prevents damage from opposing
Pokemon ex while Neutralization Zone (card 1247) gives the same protection to
all non-Rule-Box Pokemon. The test asks whether adding this wall package to the
frozen Hancang neural policy creates enough first-seat value to justify a
separate, explicitly routed successor.

This is an official-engine development experiment, not an arena result or a
performance claim. The authoritative arena currently schedules the submitted
bot as the first player in every observed game, so this first stage locks that
live seat. Any advancing successor must later pass a mixed-opponent, both-seat
gate before packaging.

## Frozen arms

Control: `work/target80_candidates/hancang_wobbfree_scale_control_v1`

- tree SHA-256: `e3775e90ff127022e6db1b66f62e721c20a26817afba117b448c708a7f094808`
- deck SHA-256: `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`
- main SHA-256: `3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341`
- model SHA-256: `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`

Candidate: `work/target80_candidates/hancang_crustle_zone_deckonly_v1`

- tree SHA-256: `5e9bc9fcc0d6e4854cf0e56b37a8e99a5efc86df8a980dc755639fd2856329d0`
- deck SHA-256: `21c66edb485fba33b8af0331a6294c1390c0e1e3b76c6a81177a5dcdfac89201`
- main SHA-256: `99165c1cc455dcae545603aabdecb8bbbfaad6770cfda59ccaff438deb3f5f24`
- model SHA-256: `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`

The candidate changes only the deck and provenance/docstring. Relative to the
control it removes one Team Rocket's Articuno, one Team Rocket's Mimikyu, one
Poke Pad, one Team Rocket's Factory, and Hero's Cape; it adds two Dwebble, two
Crustle, and one Neutralization Zone. Both decks contain exactly 60 cards and
zero Wobbuffet. The candidate has exactly one ACE SPEC.

## Frozen opponents and execution

- `SparkStarmieMistFlatV20_20260801_extracted`
  - tree SHA-256: `19bc05c8db7caa12c939a418c541951780bedfd4a43f9259b64699b5978d6964`
  - seed: `806201`
- `SparkStarmieFroslassCFV13_extracted`
  - tree SHA-256: `c693adec6062422241f2b52caf6beba6518913c2c1ed66581b369eba15b92f76`
  - seed: `806202`

Collector:
`scripts/collect_target80_deck_matched_bc_v1_panel.py`

- SHA-256: `37bb80f515afa7fbaa62cddf750bcfd080fe0abdf9dbd5ac5243ac3417d90c11`
- official evaluator: the arena lab's `tools/evaluate.py`
- seat: first
- schedule: 8 candidate and 8 control games per opponent, randomly interleaved
- total: 32 official-engine games
- strict packaged-worker legality and zero hidden-state access
- output files must not exist before collection

Neither opponent panel may be opened, summarized, or otherwise inspected until
both collectors have terminated and both manifests exist.

## Integrity gates

All are required:

1. Exactly 32 terminal rows and four 8-game arm/opponent cells.
2. Zero illegal actions, worker exceptions, timeouts, or engine failures.
3. Every manifest reproduces the frozen tree, deck, main, model, collector, and
   opponent hashes above.
4. Every focal game reports `started_first=true`.

An integrity failure invalidates the panel and permits only a corrected rerun
from game one under a new documented protocol revision.

## Admission gates

The deck-only candidate advances only if all are true:

1. At least 12 wins in its 16 games (75%).
2. At least four more wins than control across the locked panel.
3. At least five wins in each 8-game opponent family.
4. Dwebble, Crustle, or Neutralization Zone is publicly observed in at least 12
   of 16 candidate games, and Crustle is active in at least six candidate games.

Passing does not authorize packaging or arena submission. It authorizes one
new immutable routed candidate that prioritizes establishing and using the wall
against publicly identified Starmie lines. That routed candidate must undergo
fresh preregistered development, broad both-seat testing, and independent
confirmation.

If the candidate gains at least three wins over control and wins at least 10 of
16 but misses only the Crustle-active mechanism threshold, one routed candidate
may still be built to test the specific preregistered diagnosis of policy
under-execution. No other near miss advances.

## Claim boundary

This panel can reject or provisionally advance a structural hypothesis. It
cannot establish an 80% arena win rate. The project goal remains one immutable
successor with at least 160 wins in 200 official internal-arena games; no bot is
to be submitted merely because this small screen passes.

## Locked result (added after both manifests completed)

Both collectors completed with all integrity gates satisfied. The candidate
finished 3-12-1 (3 wins in 16); control finished 7-9. By opponent, candidate was
2-6 against MistFlat and 1-6-1 against Froslass, versus control's 3-5 and 4-4.
The candidate showed Dwebble in 10 games, Crustle in four, made Crustle active
in two, and never established Neutralization Zone. It therefore missed every
admission path and is permanently rejected. No routed derivative is authorized
from this experiment.

Artifacts:

- `work/target80_crustle_zone_structural_v1/mist_first.jsonl`
  (`81153c5d3e6d5735e98aa16f8be50e655288729220f4734c4c2d8feac2eb9eb5`)
- `work/target80_crustle_zone_structural_v1/froslass_first.jsonl`
  (`b85bb5ff4ee3f64e0b035be8f6f57b4fb9e0dd083cbef8ce0114d9a572d0dc6d`)
