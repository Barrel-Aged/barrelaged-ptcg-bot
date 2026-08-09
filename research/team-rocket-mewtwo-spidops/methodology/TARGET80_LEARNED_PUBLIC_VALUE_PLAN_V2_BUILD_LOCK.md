# Target-80 learned public-value plan V2 build lock

Frozen: 2026-08-05, after offline admission and before the first official-engine smoke

Status: **immutable candidate for the frozen V2 match protocol only**

## Candidate identity

Directory: `work/target80_candidates/murkrow_public_value_plan_v2/`

The deck contains exactly 60 cards, zero card ID `432` (Wobbuffet), and three
card ID `463` (Team Rocket's Murkrow).

| Artifact | SHA-256 |
|---|---|
| `main.py` | `692cf67c166c6d28c019007fbdb9efbaddf1a47ce8a7edda59f63ec2430086f8` |
| `deck.csv` | `3778b1c3bcd8c74230af90ecddd849123490a945679ef67b75e2b9a5dac57193` |
| `model.pt` | `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b` |
| `public_value.pt` | `dfb3cac65e99f2756408d3577e88417a1fe5e6f48a4ca0bf719e4a77b1e56483` |
| `deck_library.json` | `ed56d55cec6e2a87415db422b72f16baa8ab552e87c625de75cd0d8cb7e8d4e0` |
| `soldier/public_value.py` | `247334546fc1dc2d20380f3788a3abc4592ecaa7ed13577e51e414de6ce74166` |
| `soldier/public_belief.py` | `ce65be013140905ee2a9ed4a4f9e3350eb04dba4264248686e1bb39a40a08f2f` |
| `soldier/safe_plan_search.py` | `c6d7292f3f56d2aa8f33692d1fb0e8c751c7ef9d5b719992f34878e2f35967c7` |
| `tests/test_target80_public_value_v2.py` | `be31348eeb4806c0ae420dfcd21b3680d16bf23628009e71918858d5ead03a7e` |

Changing any candidate artifact voids this lock and requires a new prospective
identity and smoke. Test-only changes require recording a replacement test
hash but do not authorize changes to the candidate.

## Offline evidence

The exact 1,025-game source digest reproduced as
`e3897738ea62f6e231dc4d5e2256290408ad0c01ddbe96002d597252e1ec1fcf`.
The epoch-40 artifact passed every frozen offline gate:

- validation overall AUC `0.7869604609864821` and early-turn AUC
  `0.726712084010559`;
- test overall AUC `0.8361893907043474` and early-turn AUC
  `0.760069351827154`;
- validation Brier `0.19990536778713527`, below constant `0.2352567037460493`;
- test Brier `0.17949721692766163`, below constant `0.24698839976204642`;
- every eligible major-family AUC gate passed; and
- byte-exact normalizer/state serialization and single-row reload inference
  passed under the frozen serialization amendment.

Authoritative report: `work/target80_public_value_v2_training_report.json`.

## Pre-smoke checks

The three deterministic test functions in
`tests/test_target80_public_value_v2.py` passed with Python 3.12 on 2026-08-05:

1. deck count, Wobbuffet absence, Murkrow count, and model shape;
2. invariance to opponent hidden hand/prize identities while retaining own
   hand sensitivity; and
3. finite probability output in `[0, 1]`.

`main.py`, `soldier/public_value.py`, and `soldier/safe_plan_search.py` also
passed Python bytecode compilation. The base interpreter has no `pytest`
installation, so the functions were executed directly and retained their
ordinary assertion semantics; no dependency was installed.

This lock authorizes only the smoke and sealed development work in
`TARGET80_LEARNED_PUBLIC_VALUE_PLAN_V2_PROTOCOL.md`. It does not authorize
packaging, submission, or any strength claim.
