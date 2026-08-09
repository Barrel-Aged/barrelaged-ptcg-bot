#!/usr/bin/env python3
"""Audit exact live-versus-zero-residual parity on all N14 supports."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
ZERO = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_base"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_parity.json"
LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
ZERO_MODEL_SHA = "0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_n12_counterfactual_margin_adapter as support  # noqa: E402
from train_live_outcome_awr import (  # noqa: E402
    collate_like_live_agent,
    deployment_logits,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def purge_soldier() -> None:
    for name in list(sys.modules):
        if name == "soldier" or name.startswith("soldier."):
            del sys.modules[name]


def compare(
    live: torch.nn.Module,
    zero: torch.nn.Module,
    examples: Sequence[support.N12Example],
    *,
    batch_size: int = 128,
) -> dict[str, float | int]:
    counts: Counter[str] = Counter()
    maximum = 0.0
    live.eval()
    zero.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            selected = list(examples[start:start + batch_size])
            batch, availability = collate_like_live_agent(selected, torch.device("cpu"))
            live_logits, _ = live.forward_batch(batch)
            zero_logits, _ = zero.forward_batch(batch)
            live_logits = deployment_logits(live_logits, availability, selected)
            zero_logits = deployment_logits(zero_logits, availability, selected)
            live_top = live_logits.argmax(dim=1)
            zero_top = zero_logits.argmax(dim=1)
            labels = torch.tensor([example.action for example in selected], dtype=torch.long)
            counts["examples"] += len(selected)
            counts["live_zero_top_mismatch"] += int((live_top != zero_top).sum())
            counts["live_label_mismatch"] += int((live_top != labels).sum())
            counts["zero_label_mismatch"] += int((zero_top != labels).sum())
            for row, example in enumerate(selected):
                width = int(example.feat["opt_feats"].shape[0])
                anchor = int(live_top[row])
                left = live_logits[row, :width] - live_logits[row, anchor]
                right = zero_logits[row, :width] - zero_logits[row, anchor]
                finite = (left > -1.0e8) & (right > -1.0e8)
                if bool(finite.any()):
                    maximum = max(maximum, float((left[finite] - right[finite]).abs().max()))
    return {**dict(counts), "max_finite_relative_logit_error": maximum}


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N14 zero parity accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N14 parity output: {OUTPUT}")
    if digest(LIVE / "model.pt") != LIVE_MODEL_SHA:
        raise ValueError("N14 live model changed")
    if digest(ZERO / "model.pt") != ZERO_MODEL_SHA:
        raise ValueError("N14 zero model changed")
    if digest(LIVE / "deck.csv") != DECK_SHA or digest(ZERO / "deck.csv") != DECK_SHA:
        raise ValueError("N14 deck changed")
    live_state = torch.load(LIVE / "model.pt", map_location="cpu", weights_only=True)
    zero_state = torch.load(ZERO / "model.pt", map_location="cpu", weights_only=True)
    original_keys = set(live_state["state_dict"])
    if any(not torch.equal(value, zero_state["state_dict"][name]) for name, value in live_state["state_dict"].items()):
        raise ValueError("N14 zero base changed an original tensor")
    extra = sorted(set(zero_state["state_dict"]) - original_keys)
    if extra != [
        "logit_adapter.0.bias",
        "logit_adapter.0.weight",
        "logit_adapter.1.weight",
        "logit_adapter.3.weight",
    ]:
        raise ValueError(f"N14 adapter state keys changed: {extra}")
    if int(torch.count_nonzero(zero_state["state_dict"]["logit_adapter.3.weight"])) != 0:
        raise ValueError("N14 residual projection is not zero")

    sys.path.insert(0, str(LIVE))
    from soldier.net import load_policy as load_live  # type: ignore
    live = load_live(LIVE / "model.pt", map_location="cpu").eval()
    purge_soldier()
    sys.path = [entry for entry in sys.path if Path(entry).resolve() != LIVE.resolve()]
    sys.path.insert(0, str(ZERO))
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy as load_zero  # type: ignore
    zero = load_zero(ZERO / "model.pt", map_location="cpu").eval()

    torch.set_num_threads(4)
    deck = [int(value) for value in (ZERO / "deck.csv").read_text().replace(",", "\n").split()]
    library = json.loads((ZERO / "deck_library.json").read_text(encoding="utf-8"))
    n6, n6_provenance = support.load_n6_examples(support.CORPUS, support.AUDIT)
    n9, n9_provenance = support.load_n9_examples(
        LIVE.resolve(), support.REPLAYS, support.PARITY, deck, library, Belief, features
    )
    result = {
        "schema": "rocket-n14-zero-residual-parity/v1",
        "live_model_sha256": LIVE_MODEL_SHA,
        "zero_model_sha256": ZERO_MODEL_SHA,
        "deck_sha256": DECK_SHA,
        "original_tensors_exact": True,
        "adapter_extra_keys": extra,
        "adapter_final_projection_exact_zero": True,
        "n6": compare(live, zero, n6),
        "n9": compare(live, zero, n9),
        "n6_provenance": {"rows": n6_provenance["rows"], "corrections": n6_provenance["corrections"]},
        "n9_provenance": {
            "examples": n9_provenance["examples"],
            "examples_by_group": n9_provenance["examples_by_group"],
        },
        "claims": {"training_authority": False, "strength_improvement": False},
    }
    for name in ("n6", "n9"):
        row = result[name]
        if row["live_zero_top_mismatch"] or row["live_label_mismatch"] or row["zero_label_mismatch"]:
            raise ValueError(f"N14 zero top-action parity failed: {name}={row}")
        if float(row["max_finite_relative_logit_error"]) > 2.0e-5:
            raise ValueError(f"N14 zero logit parity failed: {name}={row}")
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "n6": result["n6"], "n9": result["n9"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
