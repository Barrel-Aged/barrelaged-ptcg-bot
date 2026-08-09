# Spidops-finish guard V1 analyzer mechanical amendment

Amended: 2026-08-05, after the 200-game evaluator completed and after its raw
106-94 summary was visible, but before any analysis artifact was written.

## Problem

The frozen candidate and control tree constants were computed with the
project's established `tree_sha256` contract: relative path plus file bytes,
excluding interpreter caches. The initial analyzer independently reimplemented
that contract but accidentally inserted each file's byte length into the
digest. It therefore rejected both unchanged bundles at the identity precheck.

Read-only comparison with the canonical helper proved that the live candidate
and control trees still exactly match their frozen hashes:

- candidate:
  `952fc72d0cb2eb39c122b0f29e3aee3dac98c3cc17dbc97238c22a7ca3eae70c`;
- control:
  `30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f`.

No `analysis.json` existed when the amendment was made.

## Authorized mechanical change

The only source change removes
`digest.update(path.stat().st_size.to_bytes(8, "big"))` from the analyzer's
local tree-hash helper, making it byte-equivalent to the canonical project
helper. Winner accounting, trace parsing, telemetry parsing, execution checks,
all seven development gates, authority, candidate/control hashes, evaluator
hash, and every input path remain unchanged.

- original analyzer SHA-256:
  `cf002e05c885b75979a45e88f5c9ccc7699738ef6ebecf8d142ab3430b0cac99`;
- amended analyzer SHA-256:
  `c5fab30abe08e14202ea11ff5336d523d61222b1f3bba5b73d88852cd6bae5b6`.

This amendment repairs provenance checking only. It cannot rescue the already
visible 106/200 raw score from the locked 110-win gate.

## Trace-schema correction

The identity-corrected analyzer then reached trace parsing and stopped before
writing output because the first trace row in every game is a pregame choice
with `current.firstPlayer == -1`. The start order is populated on later rows.
The analyzer was mechanically corrected to retain the first row per game whose
`firstPlayer` is 0 or 1, rather than unconditionally retaining step zero. The
definition of candidate-started-first, the 50% per-start-order gate, and every
other gate remain identical. No analysis artifact existed before this fix.

The final amended analyzer SHA-256 is
`ec99128a2b7ab310224e124acf41195e136db0c8e16ed06c9129eaed2a336623`.
