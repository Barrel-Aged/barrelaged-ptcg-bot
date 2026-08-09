# Majkel Lucario Public-BC V3 Safe-Hybrid Result

Date: 2026-08-08

## Outcome

V3 passed every frozen offline and official-engine admission gate. It is
approved for one internal-arena publication as
`HelbrosMegaLucarioV123PublicBCSafeHybrid`. It is not approved for direct
Kaggle submission; promotion remains a user action after sufficient arena
evidence.

## Frozen implementation

- Exact V122 neural checkpoint, SHA-256
  `e80b691f50b82fbaecd61d3a9354cd727737a4e302d9a243fa5f91fcfffb8eea`.
- Exact V121 `LucarioPromotionClonePolicyV17` as the safety provider.
- Exact current Majkel1337 public 60-card Mega Lucario deck.
- Neural attack-safe `END` guard retained.
- Both policies advance on every observation and reset between games.
- Safety route if neural confidence is unavailable, its action is illegal, or
  top-action probability is below `0.35228917002677923`.

## Calibration

The threshold was selected using only the 17 frozen V1 validation games. The
18 frozen test games were opened once after selection.

| Metric | Neural | Hybrid | Delta |
|---|---:|---:|---:|
| Validation complete action (766 rows) | 60.84% | 61.88% | +1.04 pp |
| Test complete action (937 rows) | 57.52% | 58.27% | +0.75 pp |
| Test first action (937 rows) | 60.51% | 61.47% | +0.96 pp |

The hybrid routed 84 validation rows (10.97%) and 88 test rows (9.39%) to the
safety policy. Re-running calibration produced the same threshold, actions,
metrics, and byte-identical JSON (SHA-256
`b7ac571a748dfafcd4299a24de75190c07d8933cb5d4138d4da531cf38db0c74`).

## Official-engine panels

All games used the official evaluator, side swapping, fixed evaluator seeds,
and terminal-result checks.

| Opponent | V3 record | Frozen gate | Result |
|---|---:|---:|---|
| V122 parent | 7-5 | at least 7-5 | pass |
| V121 safety baseline | 11-1 | at least 7-5 | pass |
| Team Rocket ScaleBCv2 | 2-10 | no material regression from V122 1-11 | pass |
| Starmie CFV13 | 7-5 | at least 6-6 | pass |

The Rocket result confirms that Mega Lucario retains a structural counter; V3
does not claim to solve that matchup. Local panels are admission evidence, not
an Elo estimate and not proof of public-leader equivalence.

## Durable artifacts

- Protocol: `MAJKEL_LUCARIO_PUBLIC_BC_V3_SAFE_HYBRID_PROTOCOL.md`
- Candidate: `work/kaggle_candidates/helbros_mega_lucario_public_bc_v3_safe_hybrid`
- Calibration: candidate `HYBRID_CALIBRATION.json`
- Official panels: lab `runs/v123_safe_hybrid_*.jsonl`
- Packaged archive: lab
  `submissions/helbros_mega_lucario_v123_public_bc_safe_hybrid.tar.gz`
- Archive bytes: `18,392,630`
- Archive SHA-256:
  `1e9f32a1aa8c4725c424cfea42a4cc73915489a3555c9028797221d07bf2a975`
- Extracted-archive smoke: 2-0 versus V122, two terminal games.
- Internal arena: accepted into the inbox as
  `HelbrosMegaLucarioV123PublicBCSafeHybrid`; the worker acknowledged that it
  would promote the bot on the next job.
- Kaggle: no submission performed.

## Post-publication arena result

The first read-only arena snapshot after pickup was 3-25 at rating 206.4 over
28 games. This is a severe adverse signal and fails the promotion decision,
despite the favorable frozen local gates. V123 must not be promoted to Kaggle.
The sample is still too small to estimate a stable true rating, but it is more
than sufficient to reject the requested promotion under the project's
conservative evidence standard.

The discrepancy demonstrates that public-action imitation and small matched
official-engine panels did not generalize to the arena opponent distribution.
No post-hoc threshold change, rescue candidate, additional submission, or
Kaggle action was made. Development was paused immediately after recording the
result.
