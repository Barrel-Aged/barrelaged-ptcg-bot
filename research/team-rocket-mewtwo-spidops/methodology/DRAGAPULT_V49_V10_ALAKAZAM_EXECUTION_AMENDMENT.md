# V10 Alakazam execution amendment

Frozen: 2026-08-07, before inspecting any incomplete Alakazam outcomes.

The V10 protocol assigned 24 games per arm to
`VioletAlakazamMKV93_extracted`. After roughly eight minutes, the three
independent evaluator files contained only 5, 6, and 7 completed rows. The
opponent's search cost makes the locked 72-game cell incompatible with the
10-minute command bound and with timely completion of the larger confirmation.

All three evaluator parents and their three active opponent workers were
stopped. No completed-game winner, arm score, trace action, or partial cell
rate was read. The incomplete files under
`work/dragapult_v49_v10_development/alakazam_*.jsonl` are quarantined as
execution evidence only and must not be analyzed, completed, or pooled.

The Alakazam development cell is replaced prospectively with the frozen local
`agents/violet_alakazam_threat_v1_proxy` package from the arena lab. It runs in
the same official engine, with the same three arms, side swapping, 24 games,
2,000-step cap, 30-second action cap, and the original 10/24 candidate floor
and three-win control-relative harm ceiling. All other opponent cells,
selection rules, and confirmation thresholds remain unchanged.

This substitution changes the local regression opponent, not the training
data or model. The proxy result remains a local health gate and is not a claim
about the live Lotus Alakazam training pin.
