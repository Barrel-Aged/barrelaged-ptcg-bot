# Target-80 Wobbuffet-free V6 submission checkpoint

Status: **submitted to the internal arena; outcome gate in progress**  
Submitted: 2026-08-05  
Arena identity: `HelbrosRocketSpidopsWobbFreeV6`

## Immutable artifact

- Bundle: `work/target80_submission/HelbrosRocketSpidopsWobbFreeV6.tar.gz`
- Bundle SHA-256: `65beecf25b92cc510d963057c80052deaa0eb327fdf20694ec1c068cf26c9108`
- Bundle size: 33,861,682 bytes
- Tested package source:
  `work/target80_candidates/flg_wobbfree_safe_marnie_first_overlay_v1r_package`
- Research source:
  `work/target80_candidates/flg_wobbfree_safe_marnie_first_overlay_v1r`
- `main.py` SHA-256:
  `0fb90ea7adf0ab9518992b8749424fc152e78da536cd881e8fdb9c291902d5fc`
- Base model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`
- Marnie-first overlay model SHA-256:
  `a60a0b1b1d2115b579eb60fe37349b238a03d3e7d61cb005f5b17ff7e795ece9`
- Ordered deck SHA-256:
  `e2f02e3d29a37c3c126daf92c920693a6d25ec00943e2ee725422dbf27604b73`
- Wobbuffet count: **0**

The packaging-only copy adds the packager-required top-level `bot/` directory
as a byte copy of `soldier/`. The tested `main.py`, deck, and both model files
are unchanged from the research candidate.

## Candidate design

The bot uses the exact Wobbuffet-free public flg Rocket deck and the frozen
deck-matched neural control by default. It changes policy only when both of the
following are true:

1. definitive Marnie-family Pokemon have become publicly visible; and
2. Rocket is the first player.

At that point it lazily activates a model fine-tuned on exact-deck public wins
with extra Marnie and second-seat training weight. Loading the dormant model
preserves the Torch RNG state. All other routes remain on the frozen baseline.
The existing early Spidops Poke Pad and Brave Bangle lethal guards are retained.

## Pre-submission evidence

### Dormant-route exact parity

- Marnie, Rocket second: 20 real public replay streams, 2,902 decisions,
  0 action mismatches, 0 illegal actions.
- Non-Marnie, both seats: 20 real public replay streams, 2,913 decisions,
  0 action mismatches, 0 illegal actions.

Artifacts:

- `work/target80_flg_safe_overlay_v1r_development/second_marnie_parity.json`
- `work/target80_flg_safe_overlay_v1r_development/other_allseat_parity.json`

### Active-route official-engine panels

Against `AshV178MarnieFullBest_20260801_extracted`, with Rocket starting first:

- independent panel 1: overlay 32/50, control 24/50;
- independent panel 2: overlay 35/50, control 23/50;
- combined: overlay 67/100, control 47/100.

These are development panels, not arena-strength claims.

### Arena-tool smoke

The exact packaging copy completed two terminal, side-swapped games through
the arena evaluator: 1-1, zero crashes or legality failures.

Artifact:
`work/target80_flg_safe_overlay_v1r_development/arena_tool_packagecopy_smoke_2.jsonl`

## Live gate

The only admission rule is the user's exact target:

- at least 200 rated games for this immutable arena identity;
- at least 160 wins;
- draws count as non-wins;
- no pooling with any earlier Rocket identity.

The first accepted arena snapshot was 6-5-0 after 11 games. This is only a
progress checkpoint and is far below the admission gate. The contemporary
mature arena leaders are near 57% lifetime win rate, so no 80% performance
claim is justified in advance.

### Gate failure recorded after submission

At the authoritative 2026-08-05 refresh, V6 was **37-42-0 after 79 rated
games (46.84%)**. It had already accumulated 42 losses, so reaching 160 wins
in its first 200 games became mathematically impossible: even 121 consecutive
wins would finish 158-42. V6 therefore failed the exact first-200 Target-80
gate. Its identity and bundle remain immutable, and its games may not be pooled
with a successor.

The first 19-game pull was 10-9. Its public-observation autopsy showed Marnie
2-0, Starmie 0-4, Soldier 1-2, and six of nine losses without a second ready
attacker. The early Marnie result is consistent with the overlay hypothesis,
but it does not compensate for the broader weakness.

`HelbrosRocketSpidopsEarlyPadLethalV5` was not modified. No Kaggle submission
was made.
