#!/usr/bin/env python3
"""Independently audit a Target-80 randomized-support replay shard."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REPLAY_SCHEMA = "target80-randomized-support-replay/v1"
MANIFEST_SCHEMA = "target80-randomized-support-shard/v1"
PROTOCOL = "target80-randomized-support-advantage/v1"
BEHAVIOR_SCHEMA = "target80-supported-behavior/v1"
EPSILON = 0.15
MIN_SUPPORT = 0.03
SPLIT_COUNTS = {"smoke": 2, "train": 120, "validation": 40, "test": 40}
OPPONENT_INDEX = {"marnie": 1, "starmie": 2, "alakazam": 3}
SPLIT_PREFIX = {"smoke": 850, "train": 851, "validation": 852, "test": 853}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--require-smoke-gates",
        action="store_true",
        help="Also require at least one eligible and one explored decision.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(paths: Sequence[Path]) -> str:
    result = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        name = path.name.encode("utf-8")
        result.update(len(name).to_bytes(4, "big"))
        result.update(name)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                result.update(chunk)
    return result.hexdigest()


def legal(action: Sequence[int], selection: Mapping[str, Any]) -> bool:
    options = selection.get("option") or []
    minimum = min(max(0, int(selection.get("minCount") or 0)), len(options))
    maximum = min(max(0, int(selection.get("maxCount") or 0)), len(options))
    return (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(isinstance(value, int) and 0 <= value < len(options) for value in action)
    )


def hidden_safe(observation: Mapping[str, Any], actor: int) -> bool:
    current = observation.get("current") or {}
    players = current.get("players") or []
    return (
        actor in (0, 1)
        and len(players) == 2
        and isinstance(players[1 - actor], Mapping)
        and players[1 - actor].get("hand") is None
    )


def close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return abs(left - right) <= tolerance


def audit_behavior(
    behavior: Mapping[str, Any], action: Sequence[int], selection: Mapping[str, Any]
) -> tuple[bool, bool, bool]:
    if behavior.get("schema") != BEHAVIOR_SCHEMA:
        raise ValueError("behavior schema mismatch")
    if list(behavior.get("chosen") or []) != list(action):
        raise ValueError("logged behavior choice differs from applied action")
    if not legal(action, selection):
        raise ValueError("illegal replay action")
    options = selection.get("option") or []
    count = len(options)
    if int(behavior.get("option_count") or 0) != count:
        raise ValueError("behavior option count mismatch")
    propensity = float(behavior.get("chosen_propensity") or 0.0)
    if not math.isfinite(propensity) or not 0.0 < propensity <= 1.0:
        raise ValueError("invalid chosen propensity")
    probabilities = behavior.get("behavior_probabilities")
    if probabilities is not None:
        if len(probabilities) != count:
            raise ValueError("behavior probability length mismatch")
        values = [float(value) for value in probabilities]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("invalid behavior probability")
        if not close(sum(values), 1.0, 1e-6):
            raise ValueError("behavior probabilities do not sum to one")
        if len(action) == 1 and not close(values[action[0]], propensity):
            raise ValueError("chosen propensity mismatch")
    reference = behavior.get("reference_probabilities")
    source = str(behavior.get("reference_source") or "")
    eligible = bool(behavior.get("eligible"))
    explored = bool(behavior.get("explored"))
    guarded = source.endswith("_guard")
    if guarded:
        if eligible or explored or not close(propensity, 1.0):
            raise ValueError("packaged guard was not deterministic")
    if reference is not None:
        values = [float(value) for value in reference]
        if len(values) != count or any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("invalid reference probabilities")
        if not close(sum(values), 1.0, 1e-6):
            raise ValueError("reference probabilities do not sum to one")
        baseline = int(behavior["reference_argmax"])
        if baseline != max(range(count), key=lambda index: values[index]):
            raise ValueError("reference argmax mismatch")
        expected_supported = [
            index
            for index, value in enumerate(values)
            if index != baseline and value + 1e-12 >= MIN_SUPPORT
        ]
        supported = [int(value) for value in behavior.get("supported_alternatives") or []]
        if supported != expected_supported:
            raise ValueError("supported-alternative set mismatch")
        expected_eligible = int(selection.get("type") or 0) != 9 and bool(supported)
        if eligible != expected_eligible:
            raise ValueError("eligibility mismatch")
        if eligible:
            assert probabilities is not None
            actual = [float(value) for value in probabilities]
            if not close(actual[baseline], 1.0 - EPSILON):
                raise ValueError("baseline behavior mass mismatch")
            if not close(sum(actual[index] for index in supported), EPSILON):
                raise ValueError("exploratory behavior mass mismatch")
            if explored != (len(action) == 1 and action[0] != baseline):
                raise ValueError("exploration flag mismatch")
        elif explored:
            raise ValueError("ineligible action marked explored")
    if behavior.get("attack_guard_applied"):
        if not any(option.get("type") == 13 for option in options):
            raise ValueError("attack guard logged without an attack")
        if probabilities is not None:
            for index, option in enumerate(options):
                if option.get("type") == 14 and float(probabilities[index]) != 0.0:
                    raise ValueError("END has behavior mass while ATTACK is legal")
    return eligible, explored, guarded


def audit(manifest_path: Path, require_smoke_gates: bool) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("protocol") != PROTOCOL:
        raise ValueError("manifest identity mismatch")
    if not manifest.get("official_engine_execution") or not manifest.get("strict_no_fallback_execution"):
        raise ValueError("manifest does not assert strict official-engine execution")
    if manifest.get("hidden_information_used_by_actor") is not False:
        raise ValueError("manifest hidden-information flag mismatch")
    split = str(manifest.get("split"))
    opponent = str(manifest.get("opponent_key"))
    own_seat = int(manifest.get("own_seat"))
    if split not in SPLIT_COUNTS or opponent not in OPPONENT_INDEX or own_seat not in (0, 1):
        raise ValueError("unknown locked shard identity")
    if split == "smoke" and opponent != "marnie":
        raise ValueError("smoke opponent is not Marnie")
    expected_seed = SPLIT_PREFIX[split] * 1000 + OPPONENT_INDEX[opponent] * 100 + own_seat + 1
    if int(manifest.get("shard_behavior_seed")) != expected_seed:
        raise ValueError("locked behavior seed mismatch")
    if int(manifest.get("games")) != SPLIT_COUNTS[split]:
        raise ValueError("locked shard game count mismatch")
    if not close(float(manifest.get("epsilon")), EPSILON):
        raise ValueError("epsilon mismatch")
    if not close(float(manifest.get("minimum_reference_support")), MIN_SUPPORT):
        raise ValueError("support threshold mismatch")

    replay_root = Path(str(manifest_path)[: -len(".manifest.json")])
    replay_rows = manifest.get("replays") or []
    if len(replay_rows) != SPLIT_COUNTS[split]:
        raise ValueError("manifest replay-row count mismatch")
    replay_paths: list[Path] = []
    seen_names: set[str] = set()
    seen_hashes: set[str] = set()
    seen_exploration_seeds: set[int] = set()
    totals: Counter[str] = Counter()
    selection_types: Counter[str] = Counter()
    for game_index, replay_row in enumerate(replay_rows, start=1):
        name = str(replay_row.get("file"))
        path = replay_root / name
        if name in seen_names or not path.is_file():
            raise ValueError(f"missing or duplicate replay: {name}")
        seen_names.add(name)
        fingerprint = sha256(path)
        if fingerprint != replay_row.get("sha256") or fingerprint in seen_hashes:
            raise ValueError(f"replay digest mismatch or duplicate: {name}")
        seen_hashes.add(fingerprint)
        replay_paths.append(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) < 3 or "meta" not in rows[0] or "result" not in rows[-1]:
            raise ValueError(f"truncated replay: {name}")
        meta = rows[0]["meta"]
        if meta.get("schema") != REPLAY_SCHEMA or meta.get("protocol") != PROTOCOL:
            raise ValueError(f"replay identity mismatch: {name}")
        if meta.get("split") != split or meta.get("opponent_key") != opponent:
            raise ValueError(f"replay cell mismatch: {name}")
        if int(meta.get("own_seat")) != own_seat or int(meta.get("game_index")) != game_index:
            raise ValueError(f"replay seat/index mismatch: {name}")
        if meta.get("engine_seeded") is not False or meta.get("engine_seed") is not None:
            raise ValueError(f"native-engine seed claim mismatch: {name}")
        exploration_seed = int(meta.get("exploration_seed"))
        if exploration_seed != expected_seed * 1_000_000 + game_index:
            raise ValueError(f"exploration seed mismatch: {name}")
        if exploration_seed in seen_exploration_seeds:
            raise ValueError(f"duplicate exploration seed: {name}")
        seen_exploration_seeds.add(exploration_seed)
        game_counts: Counter[str] = Counter()
        for row in rows[1:-1]:
            actor = int(row.get("p"))
            observation = row.get("obs") or {}
            current = observation.get("current") or {}
            if int(current.get("yourIndex")) != actor:
                raise ValueError(f"actor/observation mismatch: {name}")
            selection = observation.get("select") or {}
            action = row.get("action") or []
            if not legal(action, selection):
                raise ValueError(f"illegal action: {name}")
            game_counts["decisions"] += 1
            if actor == own_seat:
                if not hidden_safe(observation, actor):
                    raise ValueError(f"hidden-hand boundary violation: {name}")
                behavior = row.get("behavior")
                if not isinstance(behavior, Mapping):
                    raise ValueError(f"missing Rocket behavior record: {name}")
                is_eligible, is_explored, is_guarded = audit_behavior(behavior, action, selection)
                game_counts["own_decisions"] += 1
                game_counts["eligible"] += int(is_eligible)
                game_counts["explored"] += int(is_explored)
                game_counts["guarded"] += int(is_guarded)
                selection_types[str(behavior.get("selection_type"))] += 1
            elif "behavior" in row:
                raise ValueError(f"opponent row contains Rocket behavior metadata: {name}")
        result = rows[-1]["result"]
        if int(result.get("raw_result")) not in (0, 1, 2):
            raise ValueError(f"nonterminal result: {name}")
        for key in ("decisions", "own_decisions", "eligible", "explored", "guarded"):
            result_key = "steps" if key == "decisions" else key
            if int(result.get(result_key)) != game_counts[key] or int(replay_row.get(key)) != game_counts[key]:
                raise ValueError(f"{key} count mismatch: {name}")
            totals[key] += game_counts[key]

    if tree_digest(replay_paths) != manifest.get("replay_tree_sha256"):
        raise ValueError("replay tree digest mismatch")
    manifest_map = {
        "decisions": "decisions",
        "own_decisions": "own_decisions",
        "eligible": "eligible_decisions",
        "explored": "explored_decisions",
        "guarded": "guarded_decisions",
    }
    for local_key, manifest_key in manifest_map.items():
        if totals[local_key] != int(manifest.get(manifest_key)):
            raise ValueError(f"manifest {manifest_key} mismatch")
    if dict(sorted(selection_types.items())) != manifest.get("selection_type_counts"):
        raise ValueError("selection-type counts mismatch")
    if int(manifest.get("hidden_boundary_audits")) != totals["own_decisions"]:
        raise ValueError("hidden-boundary audit count mismatch")
    if require_smoke_gates and (totals["eligible"] < 1 or totals["explored"] < 1):
        raise ValueError("smoke lacks an eligible or realized exploratory decision")
    return {
        "manifest": str(manifest_path),
        "split": split,
        "opponent": opponent,
        "own_seat": own_seat,
        "games": len(replay_paths),
        "decisions": totals["decisions"],
        "own_decisions": totals["own_decisions"],
        "eligible": totals["eligible"],
        "explored": totals["explored"],
        "guarded": totals["guarded"],
        "replay_tree_sha256": manifest.get("replay_tree_sha256"),
        "audit_passed": True,
    }


def main() -> int:
    args = parse_args()
    result = audit(args.manifest, args.require_smoke_gates)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
