#!/usr/bin/env python3
"""V5-retained policy adaptation from exact Soldier V49 trajectories.

This is deliberately conservative offline policy improvement.  Complete games
are split before feature extraction.  Logged actions from wins receive larger
advantage-weighted regression weights than actions from losses within the same
seat/opponent-family stratum, and a frozen-reference KL penalty limits drift.
The result is only a candidate: fresh official-engine matchup gates remain the
deployment authority.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as Fnn

try:
    # Direct script execution places this directory on sys.path.
    from train_public_leader_v110_bc import Example, collate, configure_trainable
except ModuleNotFoundError:  # pragma: no cover - exercised by module-style tests
    from scripts.train_public_leader_v110_bc import (
        Example,
        collate,
        configure_trainable,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    PROJECT_ROOT / "work" / "target80_candidates" / "earlypad_bangle_lethal_v1"
)
DEFAULT_REPLAYS = PROJECT_ROOT / "work" / "arena_v8_pull_20260807"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "work" / "dragapult_v49_learning" / "training"
)
DEFAULT_EXACT_PARITY_REPORT = (
    PROJECT_ROOT / "work" / "dragapult_v49_learning" / "v5_vs_v8_parity.json"
)
LIVE_MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
LIVE_DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
EXACT_PARITY_REPORT_SHA256 = (
    "3f57b075f57474104ce27588d8494b4bcff4c28e2134692bddf1f7c7367e9e41"
)
EXPECTED_EXCLUDED_GAMES = 77
EXPECTED_ELIGIBLE_GAMES = 183
EXPECTED_REJECTED_DRAWS = 0
EXPECTED_AUDITED_DECISIONS = 14267
EXPECTED_MISMATCHED_DECISIONS = 99


@dataclass
class WeightedExample(Example):
    won: bool
    group: str
    advantage: float = 0.0
    weight: float = 1.0
    deployment_end_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class GameInfo:
    path: Path
    seat: int
    started_first: bool
    opponent: str
    family: str
    won: bool
    draw: bool

    @property
    def group(self) -> str:
        return f"first={self.started_first},family={self.family}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--recorded-bot", default="HelbrosRocketSpidopsBC2RecoveryV8")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("head", "last", "full"), default="last")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--family-filter", default="soldier")
    parser.add_argument("--started", choices=("any", "first", "second"), default="any")
    parser.add_argument(
        "--split-mode",
        choices=("stratified", "chronological"),
        default="stratified",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--exact-parity-report",
        type=Path,
        default=DEFAULT_EXACT_PARITY_REPORT,
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="extract and verify exact reference-action parity without writing or training",
    )
    return parser.parse_args()


def family(name: str) -> str:
    lowered = name.lower()
    for token in (
        "marnie", "alakazam", "soldier", "starmie", "rocket", "dragapult",
        "lucario", "archaludon", "trevenant", "garchomp", "crustle",
    ):
        if token in lowered:
            return token
    return "other"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_parity_exclusions(
    path: Path,
    *,
    candidate: Path,
    replays: Path,
    recorded_bot: str,
    family_filter: str | None,
    started: str,
) -> tuple[set[str], dict[str, Any]]:
    """Validate the immutable exact-agent audit and return off-policy games."""

    report_path = path.resolve()
    report_hash = sha256(report_path)
    if report_hash != EXACT_PARITY_REPORT_SHA256:
        raise ValueError(
            "exact-agent parity report hash changed: "
            f"expected {EXACT_PARITY_REPORT_SHA256}, found {report_hash}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "exact-live-agent-replay-parity/v2":
        raise ValueError("unsupported exact-agent parity report schema")
    expected_fields = {
        "candidate": str(candidate),
        "replays": str(replays),
        "reference_model_sha256": LIVE_MODEL_SHA256,
        "deck_sha256": LIVE_DECK_SHA256,
    }
    for key, expected in expected_fields.items():
        if report.get(key) != expected:
            raise ValueError(f"exact-agent parity report {key} changed")
    expected_filters = {
        "recorded_bot": recorded_bot,
        "family": family_filter,
        "started": started,
        "include_draws": False,
    }
    if report.get("filters") != expected_filters:
        raise ValueError("exact-agent parity report filters changed")
    totals = report.get("totals") or {}
    if int(report.get("eligible_games", -1)) != EXPECTED_ELIGIBLE_GAMES:
        raise ValueError("exact-agent parity eligible-game count changed")
    if int(report.get("rejected_draws", -1)) != EXPECTED_REJECTED_DRAWS:
        raise ValueError("exact-agent parity draw count changed")
    if int(totals.get("decisions", -1)) != EXPECTED_AUDITED_DECISIONS:
        raise ValueError("exact-agent parity decision count changed")
    if int(totals.get("mismatched", -1)) != EXPECTED_MISMATCHED_DECISIONS:
        raise ValueError("exact-agent parity mismatch count changed")
    if int(totals.get("games_with_mismatch", -1)) != EXPECTED_EXCLUDED_GAMES:
        raise ValueError("exact-agent parity exclusion count changed")
    rows = list(report.get("per_game") or [])
    exclusions = {
        str(row["file"])
        for row in rows
        if int(row.get("mismatched", 0)) > 0
    }
    if len(exclusions) != EXPECTED_EXCLUDED_GAMES or len(rows) != len(exclusions):
        raise ValueError("exact-agent parity exclusion manifest is inconsistent")
    return exclusions, {
        "path": str(report_path),
        "sha256": report_hash,
        "eligible_games": int(report["eligible_games"]),
        "decisions": int(totals["decisions"]),
        "matched_or_equivalent": int(totals["matched"]),
        "equivalent_set_order": int(totals.get("equivalent_set_order", 0)),
        "excluded_games": sorted(exclusions),
    }


def read_game(path: Path, bot: str) -> tuple[GameInfo, list[dict[str, Any]]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    players = list((records[0].get("meta") or {}).get("players") or [])
    if bot not in players:
        raise ValueError(f"{path.name}: bot absent")
    seat = players.index(bot)
    terminal = records[-1].get("result") or {}
    winner_value = terminal.get("winner")
    winner = int(winner_value) if winner_value is not None else None
    won = winner == seat
    draw = winner is None
    decisions = [
        record for record in records[1:-1]
        if "obs" in record and int(record.get("p", -1)) == seat
    ]
    first_player = next(
        (
            (record.get("obs") or {}).get("current", {}).get("firstPlayer")
            for record in decisions
            if (record.get("obs") or {}).get("current", {}).get("firstPlayer") in (0, 1)
        ),
        -1,
    )
    opponent = players[1 - seat]
    return (
        GameInfo(
            path,
            seat,
            first_player == seat,
            opponent,
            family(opponent),
            won,
            draw,
        ),
        decisions,
    )


def split_games(
    games: list[GameInfo],
    seed: int,
    *,
    mode: str = "stratified",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[GameInfo]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation fraction leaves no chronological test split")
    if mode == "chronological":
        ordered = sorted(games, key=lambda game: game.path.name)
        train_end = int(len(ordered) * train_fraction)
        validation_end = train_end + int(len(ordered) * validation_fraction)
        if train_end <= 0 or validation_end <= train_end or validation_end >= len(ordered):
            raise ValueError("chronological AWR split is empty")
        return {
            "train": ordered[:train_end],
            "validation": ordered[train_end:validation_end],
            "test": ordered[validation_end:],
        }
    if mode != "stratified":
        raise ValueError(f"unknown split mode: {mode}")
    # Round-robin each outcome/seat/family stratum after deterministic shuffle,
    # keeping rare matchups represented without leaking a game across splits.
    strata: dict[tuple[str, bool, bool], list[GameInfo]] = defaultdict(list)
    for game in games:
        strata[(game.family, game.started_first, game.won)].append(game)
    splits: dict[str, list[GameInfo]] = {"train": [], "validation": [], "test": []}
    for key, values in sorted(strata.items()):
        random.Random(f"{seed}:{key}").shuffle(values)
        for index, game in enumerate(values):
            bucket = index % 10
            split = "validation" if bucket == 8 else "test" if bucket == 9 else "train"
            splits[split].append(game)
    return splits


def group_rates(games: list[GameInfo]) -> dict[str, float]:
    global_rate = sum(game.won for game in games) / max(1, len(games))
    grouped: dict[str, list[GameInfo]] = defaultdict(list)
    for game in games:
        grouped[game.group].append(game)
    rates = {}
    for key, values in grouped.items():
        # Five-game empirical-Bayes shrinkage avoids extreme weights in rare groups.
        rates[key] = (
            sum(game.won for game in values) + 5.0 * global_rate
        ) / (len(values) + 5.0)
    rates["__global__"] = global_rate
    return rates


def deployment_end_indices(obs: dict[str, Any]) -> tuple[int, ...]:
    """Return END options masked by the deployed post-logit attack guard."""

    selection = obs.get("select") or {}
    options = selection.get("option") or []
    upper = min(int(selection.get("maxCount") or 0), len(options))
    # NNAgent only invokes attack_safe_logits in its single-selection branch.
    if upper != 1:
        return ()
    if not any(option.get("type") == 13 for option in options):
        return ()
    return tuple(
        index for index, option in enumerate(options) if option.get("type") == 14
    )


def extract_examples(
    games: list[GameInfo],
    bot: str,
    deck: list[int],
    deck_library: Any,
    belief_class: Any,
    features: Any,
) -> tuple[list[WeightedExample], Counter[str]]:
    examples: list[WeightedExample] = []
    counts: Counter[str] = Counter()
    for game in games:
        _, decisions = read_game(game.path, bot)
        belief = belief_class(deck, opponent_deck_priors=deck_library)
        game_examples: list[WeightedExample] = []
        for record in decisions:
            obs = record["obs"]
            action = list(record.get("action") or [])
            selection = obs.get("select") or {}
            options = selection.get("option") or []
            try:
                # Extract every acted observation, including trivial prompts,
                # so the sequential belief tracker matches deployed runtime.
                base = features.extract(obs, belief=belief, slot_link=True)
            except Exception as error:
                counts[f"feature_error:{type(error).__name__}"] += 1
                continue
            lower = int(selection.get("minCount") or 0)
            upper = min(int(selection.get("maxCount") or 0), len(options))
            legal = (
                len(action) == len(set(action))
                and lower <= len(action) <= upper
                and all(isinstance(index, int) and 0 <= index < len(options) for index in action)
            )
            if not legal:
                counts["invalid"] += 1
                continue
            if len(options) <= 1:
                counts["trivial"] += 1
                continue
            if not action or len(action) != upper:
                counts["unsupported_optional_or_early_stop"] += 1
                continue
            if upper == len(options):
                # The engine consumes a selection set, not the serialized list
                # order.  When every option is selected there is no tactical
                # choice; archived [1, 0] versus runtime [0, 1] is equivalent.
                counts["forced_full_set"] += 1
                continue
            try:
                selected: list[int] = []
                for target in action:
                    conditioned = features.mark_selection(base, selected, upper)
                    if conditioned["opt_avail"][target] < 0.5:
                        counts["guard_mismatch"] += 1
                        break
                    game_examples.append(
                        WeightedExample(
                            conditioned,
                            int(target),
                            game.path.name,
                            game.won,
                            game.group,
                            deployment_end_indices=deployment_end_indices(obs),
                        )
                    )
                    selected.append(int(target))
                counts["supported_decisions"] += 1
            except Exception as error:
                counts[f"unroll_error:{type(error).__name__}"] += 1
        # Equal total base mass per game; otherwise long losses dominate updates.
        per_choice = 1.0 / max(1, len(game_examples))
        for example in game_examples:
            example.weight = per_choice
        examples.extend(game_examples)
    counts["examples"] = len(examples)
    return examples, counts


def apply_outcome_weights(
    examples: list[WeightedExample], rates: dict[str, float], temperature: float
) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    for example in examples:
        baseline = rates.get(example.group, rates["__global__"])
        example.advantage = float(example.won) - baseline
        example.weight *= math.exp(example.advantage / temperature)
    scale = len(examples) / max(1.0e-12, sum(example.weight for example in examples))
    for example in examples:
        example.weight *= scale


def deployment_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    examples: list[WeightedExample],
) -> torch.Tensor:
    """Apply ordinary support plus the live agent's post-logit attack guard."""

    guarded = logits.masked_fill(mask < 0.5, -1.0e9)
    for row, example in enumerate(examples):
        for index in example.deployment_end_indices:
            if 0 <= index < guarded.shape[1]:
                guarded[row, index] = -1.0e9
    return guarded


def collate_like_live_agent(
    examples: list[WeightedExample],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate without changing the live transformer's option-token context.

    The shared historical collator stores autoregressive availability in
    ``opt_mask``.  ``RelationalNet.forward_batch`` also uses that field as its
    transformer padding mask, which removes already-selected *real* options
    from attention.  The deployed single-state ``forward`` keeps every real
    option token in attention and masks unavailable logits only afterwards.
    Preserve the availability mask separately and give ``forward_batch`` a
    token-presence mask so batched training is behaviorally faithful.
    """

    batch = collate(examples, device)
    availability = batch["opt_mask"].clone()
    token_presence = torch.zeros_like(availability)
    for row, example in enumerate(examples):
        option_count = int(example.feat["opt_feats"].shape[0])
        token_presence[row, :option_count] = 1.0
    batch["opt_mask"] = token_presence
    return batch, availability


@torch.no_grad()
def reference_parity(
    model: torch.nn.Module,
    examples: list[WeightedExample],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    matched = mismatched = 0
    game_examples: Counter[str] = Counter()
    game_mismatches: Counter[str] = Counter()
    for start in range(0, len(examples), batch_size):
        selected = examples[start:start + batch_size]
        batch, availability = collate_like_live_agent(selected, device)
        labels = torch.tensor([e.action for e in selected], dtype=torch.long, device=device)
        logits, _ = model.forward_batch(batch)
        logits = deployment_logits(logits, availability, selected)
        predictions = logits.argmax(dim=1)
        matched += int((predictions == labels).sum())
        mismatched += int((predictions != labels).sum())
        for example, prediction in zip(selected, predictions.tolist()):
            game_examples[example.game] += 1
            game_mismatches[example.game] += int(prediction != example.action)
    mismatch_names = sorted(name for name, count in game_mismatches.items() if count)
    return {
        "examples": len(examples),
        "matched": matched,
        "mismatched": mismatched,
        "games": len(game_examples),
        "games_with_mismatch": len(mismatch_names),
        "first_mismatch_game": mismatch_names[0] if mismatch_names else None,
        "last_mismatch_game": mismatch_names[-1] if mismatch_names else None,
        "worst_mismatch_games": [
            {"file": name, "mismatches": count, "examples": game_examples[name]}
            for name, count in game_mismatches.most_common(10)
            if count
        ],
    }


@torch.no_grad()
def metrics(
    model: torch.nn.Module,
    reference: torch.nn.Module,
    examples: list[WeightedExample],
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    totals = Counter()
    nll = weighted_nll = kl = 0.0
    win_nll = loss_nll = 0.0
    wins = losses = 0
    pg_score = 0.0
    for start in range(0, len(examples), batch_size):
        selected = examples[start:start + batch_size]
        batch, availability = collate_like_live_agent(selected, device)
        labels = torch.tensor([e.action for e in selected], dtype=torch.long, device=device)
        logits, _ = model.forward_batch(batch)
        ref_logits, _ = reference.forward_batch(batch)
        logits = deployment_logits(logits, availability, selected)
        ref_logits = deployment_logits(ref_logits, availability, selected)
        logp = Fnn.log_softmax(logits, dim=1)
        chosen = -logp.gather(1, labels[:, None]).squeeze(1)
        weights = torch.tensor([e.weight for e in selected], device=device)
        advantages = torch.tensor([e.advantage for e in selected], device=device)
        win_mask = torch.tensor([e.won for e in selected], dtype=torch.bool, device=device)
        nll += float(chosen.sum())
        weighted_nll += float((chosen * weights).sum())
        pg_score += float(((-chosen) * advantages).sum())
        win_nll += float(chosen[win_mask].sum())
        loss_nll += float(chosen[~win_mask].sum())
        wins += int(win_mask.sum())
        losses += int((~win_mask).sum())
        ref_prob = Fnn.softmax(ref_logits, dim=1)
        ref_logp = Fnn.log_softmax(ref_logits, dim=1)
        kl += float((ref_prob * (ref_logp - logp)).sum())
        totals["correct"] += int((logits.argmax(1) == labels).sum())
        totals["count"] += len(selected)
    count = max(1, totals["count"])
    return {
        "examples": totals["count"],
        "top1": totals["correct"] / count,
        "nll": nll / count,
        "weighted_nll": weighted_nll / count,
        "reference_kl": kl / count,
        "advantage_logp": pg_score / count,
        "win_nll": win_nll / max(1, wins),
        "loss_nll": loss_nll / max(1, losses),
    }


def main() -> int:
    args = parse_args()
    if args.threads != 1:
        raise ValueError("exact replay parity requires --threads 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")
    candidate = args.candidate.resolve()
    replays = args.replays.resolve()
    sys.path.insert(0, str(candidate))
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy, save_policy  # type: ignore

    deck = [int(value) for value in (candidate / "deck.csv").read_text().split()]
    if len(deck) != 60:
        raise ValueError("candidate deck is not 60 cards")
    if sha256(candidate / "model.pt") != LIVE_MODEL_SHA256:
        raise ValueError("frozen live model hash changed")
    if sha256(candidate / "deck.csv") != LIVE_DECK_SHA256:
        raise ValueError("frozen Team Rocket deck hash changed")
    excluded_games, exact_parity_report = exact_parity_exclusions(
        args.exact_parity_report,
        candidate=candidate,
        replays=replays,
        recorded_bot=args.recorded_bot,
        family_filter=args.family_filter,
        started=args.started,
    )
    deck_library_path = candidate / "deck_library.json"
    deck_library = json.loads(deck_library_path.read_text()) if deck_library_path.exists() else None
    replay_paths = sorted(replays.glob("*.jsonl"))
    all_games = []
    rejected_games: Counter[str] = Counter()
    for path in replay_paths:
        try:
            game = read_game(path, args.recorded_bot)[0]
        except (ValueError, KeyError, json.JSONDecodeError):
            rejected_games["parse_or_identity"] += 1
            continue
        if game.draw:
            rejected_games["draw"] += 1
            continue
        if args.family_filter and game.family != args.family_filter:
            rejected_games["family_filter"] += 1
            continue
        if args.started == "first" and not game.started_first:
            rejected_games["seat_filter"] += 1
            continue
        if args.started == "second" and game.started_first:
            rejected_games["seat_filter"] += 1
            continue
        all_games.append(game)
    games = all_games
    if len(games) < 30:
        raise ValueError(f"need at least 30 complete games, found {len(games)}")
    pre_exclusion_splits = split_games(
        games,
        args.seed,
        mode=args.split_mode,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    splits = {
        name: [game for game in values if game.path.name not in excluded_games]
        for name, values in pre_exclusion_splits.items()
    }
    excluded_by_split = {
        name: sorted(
            game.path.name for game in values if game.path.name in excluded_games
        )
        for name, values in pre_exclusion_splits.items()
    }
    observed_exclusions = {
        name for values in excluded_by_split.values() for name in values
    }
    if observed_exclusions != excluded_games:
        raise ValueError(
            "exact-agent exclusions do not match the filtered replay population"
        )
    rates = group_rates(splits["train"])
    reference = load_policy(candidate / "model.pt", map_location=device).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    data: dict[str, list[WeightedExample]] = {}
    extraction = {}
    parity = {}
    parity_failed = False
    for name, split in splits.items():
        data[name], counts = extract_examples(
            split, args.recorded_bot, deck, deck_library, Belief, features
        )
        apply_outcome_weights(data[name], rates, args.temperature)
        extraction[name] = dict(counts)
        parity[name] = reference_parity(
            reference, data[name], args.batch_size, device
        )
        parity_failed = parity_failed or bool(parity[name]["mismatched"])
    if args.audit_only:
        audit = {
            "audit_only": True,
            "candidate": str(candidate),
            "reference_model_sha256": hashlib.sha256(
                (candidate / "model.pt").read_bytes()
            ).hexdigest(),
            "replays": str(replays),
            "replay_files": len(replay_paths),
            "eligible_games": len(games),
            "rejected_games": dict(rejected_games),
            "exact_agent_parity": exact_parity_report,
            "excluded_by_split": excluded_by_split,
            "filters": {
                "recorded_bot": args.recorded_bot,
                "family": args.family_filter,
                "started": args.started,
            },
            "split": {
                "mode": args.split_mode,
                "train_fraction": args.train_fraction,
                "validation_fraction": args.validation_fraction,
                "outcomes": {
                    name: {
                        "games": len(values),
                        "wins": sum(game.won for game in values),
                        "first_file": values[0].path.name if values else None,
                        "last_file": values[-1].path.name if values else None,
                    }
                    for name, values in splits.items()
                },
            },
            "extraction": extraction,
            "reference_action_parity": parity,
        }
        print(json.dumps(audit, indent=2))
        return 2 if parity_failed else 0
    if parity_failed:
        raise ValueError(f"replay/reference action mismatch: {parity}")
    model = copy.deepcopy(reference)
    trainable = configure_trainable(model, args.mode)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite AWR output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    base_metrics = {
        name: metrics(reference, reference, values, args.batch_size, device)
        for name, values in data.items()
    }
    history = []
    order = list(range(len(data["train"])))
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(order)
        train_loss = train_awr = train_kl = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            selected = [data["train"][index] for index in order[start:start + args.batch_size]]
            batch, availability = collate_like_live_agent(selected, device)
            labels = torch.tensor([e.action for e in selected], dtype=torch.long, device=device)
            weights = torch.tensor([e.weight for e in selected], device=device)
            logits, _ = model.forward_batch(batch)
            with torch.no_grad():
                ref_logits, _ = reference.forward_batch(batch)
            logits = deployment_logits(logits, availability, selected)
            ref_logits = deployment_logits(ref_logits, availability, selected)
            logp = Fnn.log_softmax(logits, dim=1)
            chosen = -logp.gather(1, labels[:, None]).squeeze(1)
            awr = (chosen * weights).mean()
            ref_prob = Fnn.softmax(ref_logits, dim=1)
            ref_logp = Fnn.log_softmax(ref_logits, dim=1)
            kl = (ref_prob * (ref_logp - logp)).sum(1).mean()
            loss = awr + args.kl_coef * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            count = len(selected)
            train_loss += float(loss.detach()) * count
            train_awr += float(awr.detach()) * count
            train_kl += float(kl.detach()) * count
            seen += count
        epoch_path = output / f"epoch_{epoch:02d}.pt"
        save_policy(model, epoch_path)
        validation = metrics(model, reference, data["validation"], args.batch_size, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, seen),
            "train_awr": train_awr / max(1, seen),
            "train_kl": train_kl / max(1, seen),
            "validation": validation,
            "elapsed_seconds": time.monotonic() - started,
            "model": str(epoch_path),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    # Offline metrics are diagnostic only.  Keep every epoch as an immutable
    # official-engine arm and make model.pt an explicit final-epoch alias.
    save_policy(model, output / "model.pt")
    final_model = load_policy(output / "model.pt", map_location=device).to(device)
    tuned_metrics = {
        name: metrics(final_model, reference, values, args.batch_size, device)
        for name, values in data.items()
    }
    summary = {
        "schema_version": 1,
        "method": "game-balanced advantage-weighted regression with reference KL",
        "deployment_authority": "none; fresh official-engine gates required",
        "source": {
            "candidate": str(candidate),
            "model_sha256": hashlib.sha256((candidate / "model.pt").read_bytes()).hexdigest(),
            "replays": str(replays),
            "replay_files": len(replay_paths),
            "games": len(games),
            "recorded_bot": args.recorded_bot,
            "rejected_games": dict(rejected_games),
            "exact_agent_parity": exact_parity_report,
            "excluded_by_split": excluded_by_split,
        },
        "split_games": {name: [game.path.name for game in values] for name, values in splits.items()},
        "split_outcomes": {
            name: {"games": len(values), "wins": sum(game.won for game in values)}
            for name, values in splits.items()
        },
        "group_train_win_rates": rates,
        "extraction": extraction,
        "reference_action_parity": parity,
        "training": {
            "mode": args.mode,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "kl_coef": args.kl_coef,
            "threads": args.threads,
            "family_filter": args.family_filter,
            "started": args.started,
            "split_mode": args.split_mode,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "checkpoint_selection": "none from offline metrics; epoch_01..epoch_04 are separate engine arms",
            "final_epoch_alias": args.epochs,
            "history": history,
        },
        "base_metrics": base_metrics,
        "tuned_metrics": tuned_metrics,
        "tuned_model": str(output / "model.pt"),
        "tuned_model_sha256": hashlib.sha256((output / "model.pt").read_bytes()).hexdigest(),
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "final_epoch_alias": args.epochs,
        "base_validation": base_metrics["validation"],
        "tuned_validation": tuned_metrics["validation"],
        "tuned_test": tuned_metrics["test"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
