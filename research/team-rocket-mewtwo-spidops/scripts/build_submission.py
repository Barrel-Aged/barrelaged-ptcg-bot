"""Build the deterministic Phase 6 neural game-theoretic arena archive.

The exact deck cannot currently be built because Prism Tower is absent from the
competition pool.  Creating the Watchtower compatibility archive therefore
requires the explicit ``--allow-watchtower-compatibility`` flag.
"""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.competition import (  # noqa: E402
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    resolve_official_deck,
)


REQUIRED_CG_FILES = (
    "__init__.py",
    "api.py",
    "game.py",
    "sim.py",
    "utils.py",
    "libcg.so",
)
OPTIONAL_CG_FILES = ("cg.dll", "libcg-arm64.so", "libcg.dylib")


DEFAULT_TEACHER_ROOT = Path(
    r"C:\Users\jhos4\PokemonDocs\_local_archive_2026-07-26\evidence"
    r"\rocket_spidops_mewtwo_20260719\package_pro_v3_extracted"
)


MAIN_SOURCE = '''"""Generated Phase 6 belief-aware plan-controller entry point."""

from pathlib import Path

from rocketbot_competition import (
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    resolve_official_deck,
)
from rocketbot_integrated_planner import IntegratedBeliefPlannerAgent
from bot.rocket_spidops import RocketSpidopsPolicy


DECK = resolve_official_deck(substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION)
TACTICAL_PROVIDER = RocketSpidopsPolicy(DECK.card_ids)
BOT = IntegratedBeliefPlannerAgent(
    DECK,
    Path(__file__).resolve().parent / "integrated_planner.json",
    telemetry_limit=256,
    tactical_teacher=TACTICAL_PROVIDER,
)


def agent(obs_dict: dict) -> list[int]:
    return BOT.choose(obs_dict)
'''


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    """Write generated content; source edits themselves still use apply_patch."""

    path.write_text(text, encoding="utf-8", newline="\n")


def build_submission(
    *,
    runtime_directory: Path,
    output_path: Path,
    allow_watchtower_compatibility: bool,
    teacher_root: Path = DEFAULT_TEACHER_ROOT,
) -> dict[str, object]:
    if not allow_watchtower_compatibility:
        exact = resolve_official_deck()
        missing = ", ".join(exact.unresolved)
        raise RuntimeError(
            "exact-deck build is blocked by the official card pool "
            f"({missing}); pass --allow-watchtower-compatibility only after "
            "accepting the documented substitution"
        )
    resolution = resolve_official_deck(
        substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
        fail_on_unresolved=True,
    )
    runtime_directory = runtime_directory.resolve()
    cg_source = runtime_directory / "cg"
    missing_files = [name for name in REQUIRED_CG_FILES if not (cg_source / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"official runtime is missing: {missing_files}")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rocketbot_submission_") as temporary:
        stage = Path(temporary)
        cg_target = stage / "cg"
        cg_target.mkdir()
        for name in REQUIRED_CG_FILES + OPTIONAL_CG_FILES:
            source = cg_source / name
            if source.is_file():
                shutil.copy2(source, cg_target / name)
        shutil.copy2(
            PROJECT_ROOT / "src" / "rocketbot" / "competition.py",
            stage / "rocketbot_competition.py",
        )
        shutil.copy2(
            PROJECT_ROOT / "src" / "rocketbot" / "integrated_planner.py",
            stage / "rocketbot_integrated_planner.py",
        )
        model_source = (
            PROJECT_ROOT
            / "artifacts"
            / "integrated_controller"
            / "integrated_planner.json"
        )
        shutil.copy2(model_source, stage / "integrated_planner.json")
        teacher_root = teacher_root.resolve()
        teacher_target = stage / "bot"
        teacher_target.mkdir()
        for name in ("__init__.py", "rocket_feathers.py", "rocket_spidops.py"):
            source = teacher_root / "bot" / name
            if not source.is_file():
                raise FileNotFoundError(f"tactical provider source is missing: {source}")
            shutil.copy2(source, teacher_target / name)
        _write_text(stage / "main.py", MAIN_SOURCE)
        _write_text(stage / "deck.csv", "\n".join(map(str, resolution.card_ids)) + "\n")

        files = sorted(path for path in stage.rglob("*") if path.is_file())
        manifest = {
            "schema_version": 2,
            "agent_version": "0.7.0",
            "profile": "watchtower_compatibility_neural_public_belief_plan_controller",
            "strategic_status": "integrated_frozen_neural_blueprint_is_final_strategy_provider",
            "game_theoretic_status": {
                "neural_blueprint": "trained_on_repeated_official_continuations_and_packaged",
                "public_belief": "six_archetype_six_intent_and_multi_resource_hypotheses",
                "safe_resolving": "selectively_gated_with_admitted_counterfactual_advantage_bounds",
                "opponent_exploitation": "evaluated_rejected_and_disabled",
                "official_information_set_bridge": "structured_public_private_history_features_with_native_legality",
            },
            "tactical_policy": {
                "autoregressive_neural_controller": True,
                "low_confidence_teacher_provider": True,
                "teacher_is_strategy_provider": False,
                "deference_threshold": json.loads(model_source.read_text(encoding="utf-8"))["calibration"]["tactical_teacher_deference_threshold"],
            },
            "model": {
                "fingerprint": json.loads(model_source.read_text(encoding="utf-8"))["fingerprint"],
                "sha256": _sha256(model_source),
            },
            "deck_resolution": resolution.to_dict(),
            "compatibility_disclosure": (
                "This arena archive replaces the exact list's two Prism Tower "
                "cards with two Team Rocket's Watchtower cards because Prism "
                "Tower is absent from the current official engine card pool."
            ),
            "execution_boundary": {
                "genuine_official": [
                    "cg engine transitions",
                    "native legal options",
                    "deck validation",
                    "selection execution",
                ],
                "research_abstractions": [
                    "public-belief posterior",
                    "five strategic plans",
                    "six value heads",
                    "counterfactual interval model",
                    "selective resolver",
                ],
            },
            "files": {
                path.relative_to(stage).as_posix(): _sha256(path) for path in files
            },
            "license_boundary": (
                "cg runtime is competition-use-only; do not redistribute and delete "
                "when the competition ends"
            ),
        }
        _write_text(stage / "submission_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        with output_path.open("wb") as compressed_handle:
            with gzip.GzipFile(
                filename="", fileobj=compressed_handle, mode="wb", mtime=0
            ) as gzip_handle:
                with tarfile.open(
                    fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    for path in sorted(stage.rglob("*")):
                        if path.is_file():
                            info = archive.gettarinfo(
                                str(path), path.relative_to(stage).as_posix()
                            )
                            info.mtime = 0
                            info.uid = 0
                            info.gid = 0
                            info.uname = ""
                            info.gname = ""
                            with path.open("rb") as handle:
                                archive.addfile(info, handle)
        manifest["archive"] = {
            "path": str(output_path),
            "size": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        }
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-directory",
        type=Path,
        default=PROJECT_ROOT / "work" / "official_kaggle" / "sample_submission",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "submissions"
            / "HelbrosMewtwoSpidopsGTWTV1.tar.gz"
        ),
    )
    parser.add_argument("--allow-watchtower-compatibility", action="store_true")
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    args = parser.parse_args()
    result = build_submission(
        runtime_directory=args.runtime_directory,
        output_path=args.output,
        allow_watchtower_compatibility=args.allow_watchtower_compatibility,
        teacher_root=args.teacher_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
