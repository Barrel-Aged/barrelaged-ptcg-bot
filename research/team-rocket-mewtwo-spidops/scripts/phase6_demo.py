"""Run Phase 6 against the official engine and persist the evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_demo import run_phase6_benchmark  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--runtime-directory",
        type=Path,
        default=PROJECT_ROOT / "work" / "official_kaggle" / "sample_submission",
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_evidence",
    )
    parser.add_argument("--full-output", action="store_true")
    args = parser.parse_args()
    artifacts = run_phase6_benchmark(
        runtime_directory=args.runtime_directory,
        repetitions=args.repetitions,
    )
    target = args.artifact_directory.resolve()
    _write(target / "phase6_evaluation.json", artifacts.report.to_json() + "\n")
    _write(target / "phase6_evaluation.md", artifacts.report.to_markdown())
    summary = artifacts.summary()
    _write(
        target / "phase6_demo_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    replays = dict(artifacts.replay_data)
    for review in artifacts.reviews:
        safe_id = review.match_id.replace("/", "_").replace("\\", "_")
        _write(
            target / "replay_reviews" / f"{safe_id}.json",
            json.dumps(review.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        _write(
            target / "replay_reviews" / f"{safe_id}.md",
            review.to_markdown(),
        )
        _write(target / "replays" / f"{safe_id}.json", replays[review.match_id] + "\n")
    output = summary if args.full_output else {
        "status": summary["status"],
        "software_status": summary["software_status"],
        "external_status": summary["external_status"],
        "benchmark_id": artifacts.report.benchmark_id,
        "matches": len(artifacts.report.matches),
        "evidence_fingerprint": artifacts.report.fingerprint,
        "artifact_directory": str(target),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
