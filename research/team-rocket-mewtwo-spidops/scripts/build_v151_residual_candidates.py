#!/usr/bin/env python3
"""Build conservative V151 residual-seed candidates from a frozen archive."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import torch


EXPECTED_SOURCE_SHA256 = (
    "744fc3511ac965903ae9635f896c3c8ce6ad9af6954dc8d0c1c6f85ece5409a4"
)
EXPECTED_SEED2_SHA256 = (
    "9947415b659ad6b43c42cbd1b7142e5249b2955b388c80fd0c01f2942cb0fe4a"
)
EXPECTED_V110_MODEL_SHA256 = (
    "552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b"
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_payload(raw: bytes) -> dict:
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def save_payload(payload: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def interpolate_payload(first: dict, second: dict, alpha: float) -> dict:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    for key in ("schema", "global_width", "option_width", "memory", "base"):
        if first[key] != second[key]:
            raise ValueError(f"incompatible residual payload field: {key}")
    if first["state_dict"].keys() != second["state_dict"].keys():
        raise ValueError("residual state dictionaries have different keys")

    result = copy.deepcopy(first)
    for key in first["state_dict"]:
        a = first["state_dict"][key]
        b = second["state_dict"][key]
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {key}")
        result["state_dict"][key] = a.mul(1.0 - alpha).add(b, alpha=alpha)
    if not torch.equal(first["projection"], second["projection"]):
        result["projection"] = first["projection"].mul(1.0 - alpha).add(
            second["projection"], alpha=alpha
        )
    result["recovery_provenance"] = {
        "kind": "linear_weight_soup",
        "alpha_seed2": alpha,
        "seed1_sha256": None,
        "seed2_sha256": EXPECTED_SEED2_SHA256,
    }
    return result


def replace_archive_member(source_raw: bytes, member_name: str, replacement: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as sink:
            with tarfile.open(fileobj=io.BytesIO(source_raw), mode="r:gz") as source:
                found = False
                for original in source.getmembers():
                    info = copy.copy(original)
                    if original.isfile():
                        handle = source.extractfile(original)
                        if handle is None:
                            raise RuntimeError(f"could not read {original.name}")
                        payload = handle.read()
                        if original.name == member_name:
                            payload = replacement
                            info.size = len(payload)
                            found = True
                        sink.addfile(info, io.BytesIO(payload))
                    else:
                        sink.addfile(info)
                if not found:
                    raise ValueError(f"source archive has no {member_name}")
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed2", type=Path, required=True)
    parser.add_argument("--v110-source", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    source_raw = args.source.read_bytes()
    if sha256(source_raw) != EXPECTED_SOURCE_SHA256:
        raise ValueError("source archive is not the frozen published V151 bundle")
    seed2_raw = args.seed2.read_bytes()
    if sha256(seed2_raw) != EXPECTED_SEED2_SHA256:
        raise ValueError("seed-2 residual does not match the published branch artifact")

    with tarfile.open(fileobj=io.BytesIO(source_raw), mode="r:gz") as archive:
        seed1_member = archive.getmember("history_residual.pt")
        handle = archive.extractfile(seed1_member)
        if handle is None:
            raise RuntimeError("could not read source history_residual.pt")
        seed1_raw = handle.read()

    seed1 = load_payload(seed1_raw)
    seed2 = load_payload(seed2_raw)
    soup = interpolate_payload(seed1, seed2, args.alpha)
    soup["recovery_provenance"]["seed1_sha256"] = sha256(seed1_raw)
    soup_raw = save_payload(soup)
    # Ensure the generated checkpoint is immediately loadable before packaging.
    load_payload(soup_raw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        "seed2": replace_archive_member(source_raw, "history_residual.pt", seed2_raw),
        "soup": replace_archive_member(source_raw, "history_residual.pt", soup_raw),
    }
    if args.v110_source is not None:
        v110_raw = args.v110_source.read_bytes()
        with tarfile.open(fileobj=io.BytesIO(v110_raw), mode="r:gz") as archive:
            handle = archive.extractfile(archive.getmember("model.pt"))
            if handle is None:
                raise RuntimeError("could not read V110 model.pt")
            v110_model_raw = handle.read()
        if sha256(v110_model_raw) != EXPECTED_V110_MODEL_SHA256:
            raise ValueError("V110 source contains an unexpected model.pt")
        v110_seed1 = replace_archive_member(source_raw, "model.pt", v110_model_raw)
        candidates["v110_seed1"] = v110_seed1
        candidates["v110_seed2"] = replace_archive_member(
            v110_seed1, "history_residual.pt", seed2_raw
        )
    outputs = {}
    for label, raw in candidates.items():
        path = args.out_dir / f"HelbrosMarnieGeneralRecovery_{label}.tar.gz"
        path.write_bytes(raw)
        outputs[label] = {
            "path": str(path.resolve()),
            "bytes": len(raw),
            "sha256": sha256(raw),
            "residual_sha256": sha256(
                seed2_raw
                if label in {"seed2", "v110_seed2"}
                else soup_raw
                if label == "soup"
                else seed1_raw
            ),
        }

    manifest = {
        "schema": "v151-residual-recovery-candidates/v1",
        "source_archive_sha256": sha256(source_raw),
        "seed1_residual_sha256": sha256(seed1_raw),
        "seed2_residual_sha256": sha256(seed2_raw),
        "soup_alpha_seed2": args.alpha,
        "candidates": outputs,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
