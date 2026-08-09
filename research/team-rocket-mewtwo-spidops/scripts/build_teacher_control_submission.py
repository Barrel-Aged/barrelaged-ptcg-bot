"""Build a deterministic teacher-only control on the compatibility deck.

This is an experimental control, not the final recovery architecture.  It
isolates deck/policy strength from the failed neural planner weights.
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


MAIN_SOURCE = '''"""Generated ProV3 compatibility-deck control entry point."""

from rocketbot_competition import (
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    resolve_official_deck,
)
from bot.rocket_spidops import RocketSpidopsPolicy


DECK = resolve_official_deck(substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION)
BOT = RocketSpidopsPolicy(DECK.card_ids)


def agent(obs_dict: dict) -> list[int]:
    if obs_dict.get("select") is None:
        return list(DECK.card_ids)
    return list(BOT.act(obs_dict))
'''


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def build(
    *,
    runtime_directory: Path,
    teacher_root: Path,
    output_path: Path,
) -> dict[str, object]:
    resolution = resolve_official_deck(
        substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
        fail_on_unresolved=True,
    )
    runtime_directory = runtime_directory.resolve()
    teacher_root = teacher_root.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rocket_teacher_control_") as temporary:
        stage = Path(temporary)
        cg_target = stage / "cg"
        cg_target.mkdir()
        for name in REQUIRED_CG_FILES + OPTIONAL_CG_FILES:
            source = runtime_directory / "cg" / name
            if name in REQUIRED_CG_FILES and not source.is_file():
                raise FileNotFoundError(source)
            if source.is_file():
                shutil.copy2(source, cg_target / name)
        shutil.copy2(
            PROJECT_ROOT / "src" / "rocketbot" / "competition.py",
            stage / "rocketbot_competition.py",
        )
        teacher_target = stage / "bot"
        teacher_target.mkdir()
        for name in ("__init__.py", "rocket_feathers.py", "rocket_spidops.py"):
            source = teacher_root / "bot" / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, teacher_target / name)
        _write_text(stage / "main.py", MAIN_SOURCE)
        _write_text(stage / "deck.csv", "\n".join(map(str, resolution.card_ids)) + "\n")
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        manifest = {
            "schema_version": 1,
            "agent_version": "teacher-control-1",
            "profile": "prov3_teacher_watchtower_compatibility_control",
            "purpose": (
                "Recovery experiment isolating compatibility-deck and published "
                "teacher strength; not the final neural strategy"
            ),
            "deck_resolution": resolution.to_dict(),
            "compatibility_disclosure": (
                "Two Prism Tower cards are replaced by two Team Rocket's "
                "Watchtower cards because Prism Tower is absent from the engine."
            ),
            "files": {
                path.relative_to(stage).as_posix(): _sha256(path) for path in files
            },
        }
        _write_text(
            stage / "submission_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        with output_path.open("wb") as compressed:
            with gzip.GzipFile(filename="", fileobj=compressed, mode="wb", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for path in sorted(stage.rglob("*")):
                        if not path.is_file():
                            continue
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
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "submissions"
            / "HelbrosMewtwoSpidopsTeacherGTWT.tar.gz"
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                runtime_directory=args.runtime_directory,
                teacher_root=args.teacher_root,
                output_path=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
