# Target-80 raw-action milestone policy V1R result

Completed and first analyzed: 2026-08-05 (America/Denver)

Decision: **reject and close this sampled teacher configuration**. No learned
or rollout-derived policy change was admitted to the live bot.

The complete locked panel was collected before any outcome inspection. It
contained all 14 manifests and 28 unique, balanced roots. The preregistered
analyzer then reported `all_gates_pass: false`.

## Failed gates

- Integrity and complete pairs failed because the Alakazam first/second cells
  contained collector errors and incomplete paired samples.
- Stable-label coverage was 15/28, below the required 21/28.
- First-start mean stable-label advantage was `+0.08994`, but second-start was
  `+0.02855`, below the locked `+0.03` per-start gate.

## Gates that passed

- All 28 balanced roots and 14 cell manifests were present.
- Nine stable action changes were found, spanning both starts and remaining in
  the locked 4-16 range.
- Stable changed-root mean advantage was `+0.09533`, median `+0.02250`, with a
  `0.15972` negative underlying-sample fraction.
- Seven opponent families were nonnegative and both Starmie safety bounds
  passed.

These partial signals are not sufficient to override any failed gate. The
final iteration therefore retains the frozen elite behavioral-cloning policy
and only the separately justified deterministic public-information guards.

Machine-readable result:
`work/target80_raw_action_milestone_v1r_pilot/analysis.json`
