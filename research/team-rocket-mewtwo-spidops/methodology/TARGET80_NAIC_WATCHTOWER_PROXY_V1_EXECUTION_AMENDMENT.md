# Target-80 NAIC Watchtower Proxy V1 execution amendment

Frozen: 2026-08-05, after the orchestration timeout and before any game outcome was inspected

The six frozen arms were launched together. The outer PowerShell orchestration
process reached its 600-second command limit while the proxy-versus-Alakazam
arm was still running. Read-only completion checks established only these row
counts:

- proxy/Marnie: 12/12;
- control/Marnie: 12/12;
- proxy/Starmie: 8/8;
- control/Starmie: 8/8;
- control/Alakazam: 8/8; and
- proxy/Alakazam: 6/8.

No winner, result, reason, trace action, score, or partial arm comparison was
read. The interrupted six-row proxy/Alakazam prefix and its trace are retained
unchanged but are operationally excluded from all selection, training,
strength analysis, and claims.

The sole authorized recovery is to rerun only proxy/Alakazam from game 1 with
the exact frozen candidate, opponent, evaluator arguments, default side
swapping, and eight-game count, writing a new `_rerun` output. All other arms
remain sealed and may not be rerun. Analysis is forbidden until that complete
eight-row rerun exists and all six eligible arms pass count checks.

The candidate hashes and every original strength gate remain unchanged. This
amendment fixes orchestration only; it cannot select a favorable prefix, seed,
seat, opponent, deck, policy, or threshold.
