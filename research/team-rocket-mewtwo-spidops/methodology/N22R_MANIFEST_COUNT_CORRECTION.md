# N22R Manifest-Count Correction

Status: **frozen after preflight failure and before retry**

The locked N22R command failed before loading a model or reconstructing any
example. The script expected 160 N19 training games, but the frozen N19 audit
declares:

- teacher: 48 train games per seat x 2 seats = **96**;
- retention: 24 train games per seat/outcome x 4 strata = **96**;
- total: **192**.

The only authorized source change is replacing the incorrect total-only check
with exact checks for 96 teacher and 96 retention records. The tie rule,
threshold, sources, metrics, gates, output path, and zero-validation/test
boundary remain unchanged. Because the failure occurred before example loading,
this retry has observed no training or held-out decision outcome.

Failed script SHA-256:
`94dd22a0b9d367e037dbb845ff012776b6b7667364b3f79c6b8b1afa336ce408`.

Failed execution-lock SHA-256:
`e23389c6386574038ca6d3a04549fb3df935a34a9bb9276dd683fcb27a6fa80f`.
