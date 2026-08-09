"""Replay final V8 decisions and count each disclosed public guard."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "work" / "target80_candidates" / "bc2_recovery_exact_guard_v8"
BOT_NAME = "BC2RecoveryExactV8"


def _load_main():
    spec = importlib.util.spec_from_file_location("bc2_recovery_exact_v8_audit", CANDIDATE / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load final V8 main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate = _load_main()
    totals = {
        "decisions": 0,
        "exact_matches": 0,
        "set_only_matches": 0,
        "mismatches": 0,
        "early_spidops": 0,
        "lethal_bangle": 0,
        "articuno_forced": 0,
        "articuno_masked": 0,
        "wobbuffet_exit": 0,
    }
    mismatch_examples: list[dict] = []

    for trace_path in args.traces:
        current_game = None
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("bot") != BOT_NAME:
                continue
            game = row.get("game")
            if game != current_game:
                candidate._AGENT = candidate._fresh_agent()
                current_game = game

            observation = row["obs"]
            expected = list(row["action"])
            totals["decisions"] += 1

            early = candidate._early_spidops_search_override(observation)
            if early is not None:
                actual = list(early)
                totals["early_spidops"] += 1
            else:
                bangle = candidate._spidops_bangle_lethal_override(observation)
                if bangle is not None:
                    actual = list(bangle)
                    totals["lethal_bangle"] += 1
                else:
                    articuno, forbidden = candidate._articuno_bench_plan(observation)
                    if articuno is not None:
                        actual = list(articuno)
                        totals["articuno_forced"] += 1
                    else:
                        if forbidden:
                            baseline = list(candidate._masked_greedy_action(observation, forbidden))
                            totals["articuno_masked"] += 1
                        else:
                            baseline = list(candidate._AGENT(observation))
                        actual = candidate._wobbuffet_exit_guard_action(observation, baseline)
                        if actual != baseline:
                            totals["wobbuffet_exit"] += 1

            if actual == expected:
                totals["exact_matches"] += 1
            elif sorted(actual) == sorted(expected):
                totals["set_only_matches"] += 1
            else:
                totals["mismatches"] += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(
                        {
                            "trace": str(trace_path),
                            "game": game,
                            "step": row.get("step"),
                            "expected": expected,
                            "actual": actual,
                        }
                    )

    report = {
        "schema": "bc2-recovery-exact-v8-trace-audit/v1",
        "candidate": str(CANDIDATE),
        "traces": [str(path) for path in args.traces],
        "totals": totals,
        "mismatch_examples": mismatch_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if totals["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
