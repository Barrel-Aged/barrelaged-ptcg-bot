# Arena 300-game finalization gate

Status: **implemented and fail-closed; waiting for the live 300-game condition**  
Frozen auditable identities: `HelbrosRocketSpidopsScaleBCV2` and
`HelbrosRocketSpidopsScaleStopV3`

`scripts/finalize_arena_rank_goal.py` is the only completion sealer for the
active Team Rocket rank-one-at-300 target. It follows whichever of the two
frozen identities the monitor admits, does not treat an early rank-one snapshot
as completion, and never combines games from multiple bot names.

The finalizer requires all of the following in one run:

1. the v2 read-only monitor has admitted the exact immutable identity at rank
   1 with at least 300 of its own rated games;
2. authenticated `arena.py ratings` calls immediately before and after the
   official-engine smoke independently show that same exact bot at current
   rank 1 with at least 300 games;
3. the monitored source deck/model still match their frozen hashes;
4. the admitted identity's original arena-hosted archive matches its frozen
   SHA-256 (`27dff66b...2c6f9` for ScaleBCV2 or
   `366a3b4b...fcdae` for ScaleStopV3), contains exactly its frozen set of 26
   or 27 safe regular-file members, resolves to the exact 60-card deck, and
   contains the exact frozen model;
5. the archive's own entry point returns that deck and the exact extracted
   package completes two seat-swapped games through its bundled official `cg`
   engine without a deck, agent, timeout, legality, or engine error.

Only after every gate passes does the script atomically create
`artifacts/team_rocket_rank1_300/final_audit/`. That seal contains the parsed
and raw live ratings evidence, all frozen hashes, official-engine smoke records,
worker logs, a machine-readable audit, and a concise report. A SHA-256 manifest
fingerprints every evidence file, and an existing seal is never overwritten.

`scripts/watch_and_finalize_arena_rank_goal.py` is a one-shot companion. It
waits read-only for monitor admission, invokes the independent finalizer, and
exits only after the final evidence seal exists. A stale admitted snapshot is
not enough: exit code 3 means the fresh ratings gate disagreed, so the watcher
continues until the live conditions pass simultaneously.

`python scripts/finalize_arena_rank_goal.py --preflight` exercises the static
archive and packaged-runtime gates without checking rank and explicitly emits
`completion_claim: false`. Running the script without `--preflight` before the
live threshold returns exit code 3 and writes no final audit.

Evidence boundary: the independent ratings call is genuine current network
arena evidence. The package smoke is genuine official-engine execution using
the exact archive's `cg` runtime, but it is local and unrated. The finalizer
does not submit or replace a bot and performs no Kaggle action.

## Preflight result

The static preflight passed on 2026-08-01. It verified all 26 archive members,
22,803,396 uncompressed bytes, the exact deck/model/archive hashes, and the
60-card entry-point deck query. The exact extracted package then completed two
seat-swapped official-engine self-play games in 144 and 116 decisions with
terminal results and no evaluator stderr or runtime error record. This is
runtime/legality evidence only; it is not the final rank certificate and adds
zero rated arena games.

The independently downloaded ScaleStopV3 arena archive also passed on
2026-08-01: SHA-256
`366a3b4bf2e646cfb39754fc518183114f8e6b2dd6c0f2f7942946be4a8fcdae`,
27 safe members, exact 60-card deck, exact model SHA-256
`c6b808e3e50b5fffcfd44b59ad2a541c6661e8d22915b1cbb5ade5a57464d8f2`,
and two terminal seat-swapped official-engine games. Thus either monitor target
can now satisfy the same exact-archive completion contract independently.
