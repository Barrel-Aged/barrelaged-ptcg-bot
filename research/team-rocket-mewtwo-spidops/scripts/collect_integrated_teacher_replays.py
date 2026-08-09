"""Collect exact-compatibility tactical-teacher games in the licensed engine.

The archived ProV3 policy is used only to produce tactical imitation labels.
It is not copied into the final archive and is not the deployed strategy
provider. The target deck is the documented Prism Tower -> Watchtower
compatibility resolution, not the archived policy's original deck.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.competition import (  # noqa: E402
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    SelectionContract,
    _safe_int,
    _state,
    resolve_official_deck,
)
from rocketbot.official_evaluation import (  # noqa: E402
    OFFICIAL_PANEL_DECKS,
    PriorityBaseline,
)


DEFAULT_RUNTIME = PROJECT_ROOT / "work" / "official_kaggle" / "sample_submission"
DEFAULT_TEACHER = Path(
    r"C:\Users\jhos4\PokemonDocs\_local_archive_2026-07-26\evidence"
    r"\rocket_spidops_mewtwo_20260719\package_pro_v3_extracted"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "work" / "integrated_teacher_replays"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load_teacher(root: Path, deck: tuple[int, ...]) -> Any:
    root = root.resolve()
    if not (root / "bot" / "rocket_spidops.py").is_file():
        raise FileNotFoundError(f"archived tactical teacher is missing: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from bot.rocket_spidops import RocketSpidopsPolicy  # type: ignore[import-not-found]

    return RocketSpidopsPolicy(deck)


def collect(
    *,
    runtime: Path,
    teacher_root: Path,
    output: Path,
    replicas: int,
    maximum_selections: int,
) -> dict[str, Any]:
    runtime_text = str(runtime.resolve())
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore[import-not-found]

    resolution = resolve_official_deck(
        substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
        fail_on_unresolved=True,
    )
    target_deck = tuple(resolution.card_ids)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    styles = ("tempo", "attack_rush", "control")
    for replica in range(replicas):
        for panel in OFFICIAL_PANEL_DECKS:
            for style in styles:
                for target_seat in (0, 1):
                    match_id = (
                        f"phase6an-{panel.archetype}-{style}-s{target_seat}-r{replica}"
                    )
                    teacher = _load_teacher(teacher_root, target_deck)
                    opponent = PriorityBaseline(panel.card_ids, style=style, reverse=bool(replica % 2))
                    decks = [target_deck, tuple(panel.card_ids)]
                    if target_seat == 1:
                        decks.reverse()
                    observation, start = battle_start(list(decks[0]), list(decks[1]))
                    if observation is None:
                        raise RuntimeError(
                            f"official engine rejected {match_id}: "
                            f"player={start.errorPlayer} error={start.errorType}"
                        )
                    rows: list[dict[str, Any]] = [
                        {
                            "meta": {
                                "match_id": match_id,
                                "players": [
                                    "phase6an_tactical_teacher" if index == target_seat else f"panel_{panel.archetype}_{style}"
                                    for index in range(2)
                                ],
                                "decks": [list(deck) for deck in decks],
                                "target_seat": target_seat,
                                "target_profile": "watchtower_compatibility",
                                "teacher_source": "HelbrosRocketSpidopsProV3 frozen policy",
                                "opponent_archetype": panel.archetype,
                                "opponent_style": style,
                                "replica": replica,
                            }
                        }
                    ]
                    selections = 0
                    illegal = 0
                    engine_error: str | None = None
                    try:
                        while (
                            observation is not None
                            and _safe_int(_state(observation).get("result"), -1) == -1
                            and selections < maximum_selections
                        ):
                            actor = _safe_int(_state(observation).get("yourIndex"), 0)
                            contract = SelectionContract.from_observation(observation)
                            try:
                                raw = (
                                    teacher.act(observation)
                                    if actor == target_seat
                                    else opponent.choose(observation)
                                )
                                action = list(contract.validate(raw))
                            except Exception:
                                illegal += int(actor == target_seat)
                                action = list(contract.fallback())
                            if actor == target_seat:
                                rows.append(
                                    {
                                        "p": actor,
                                        "obs": deepcopy(observation),
                                        "action": action,
                                    }
                                )
                            observation = battle_select(action)
                            selections += 1
                        if selections >= maximum_selections:
                            engine_error = f"selection cap reached ({maximum_selections})"
                    except Exception as exc:  # Preserve failures as evidence.
                        engine_error = f"{type(exc).__name__}: {exc}"
                    finally:
                        final_state = _state(observation or {})
                        rows.append(
                            {
                                "result": {
                                    "winner": _safe_int(final_state.get("result"), -1),
                                    "turns": _safe_int(final_state.get("turn"), 0),
                                    "selections": selections,
                                    "teacher_illegal_fallbacks": illegal,
                                    "engine_error": engine_error,
                                }
                            }
                        )
                        battle_finish()
                    path = output / f"{match_id}.jsonl"
                    path.write_text(
                        "".join(_canonical(row) + "\n" for row in rows),
                        encoding="utf-8",
                        newline="\n",
                    )
                    record = {
                        "match_id": match_id,
                        "path": path.name,
                        "sha256": sha256(path.read_bytes()).hexdigest(),
                        "winner": rows[-1]["result"]["winner"],
                        "turns": rows[-1]["result"]["turns"],
                        "selections": selections,
                        "target_decisions": len(rows) - 2,
                        "teacher_illegal_fallbacks": illegal,
                        "engine_error": engine_error,
                    }
                    records.append(record)
                    print(_canonical(record), flush=True)
    manifest = {
        "schema_version": 1,
        "protocol": "Phase 6AN preregistered integrated-controller tactical labels",
        "official_engine_execution": True,
        "teacher_is_deployed_strategy": False,
        "target_deck_resolution": resolution.to_dict(),
        "records": records,
        "accounting": {
            "games": len(records),
            "completed_games": sum(record["engine_error"] is None for record in records),
            "target_decisions": sum(record["target_decisions"] for record in records),
            "teacher_illegal_fallbacks": sum(record["teacher_illegal_fallbacks"] for record in records),
        },
    }
    manifest["fingerprint"] = _fingerprint(manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--maximum-selections", type=int, default=2_000)
    args = parser.parse_args()
    result = collect(
        runtime=args.runtime,
        teacher_root=args.teacher_root,
        output=args.output,
        replicas=args.replicas,
        maximum_selections=args.maximum_selections,
    )
    print(json.dumps(result["accounting"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
