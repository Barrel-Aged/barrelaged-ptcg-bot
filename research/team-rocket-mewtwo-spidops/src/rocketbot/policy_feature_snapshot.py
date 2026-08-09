"""Typed, JSON-safe snapshots of actor-visible neural policy features.

Snapshots contain only tensors already produced from the acting agent's normal
observation and persistent public/own-visible belief state. They are training
artifacts, never official hidden-state completions or deployment inputs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


SNAPSHOT_SCHEMA = "rocket-policy-feature-snapshot/v1"
_ALLOWED_DTYPES = {
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "int64": np.dtype("int64"),
    "int32": np.dtype("int32"),
    "bool": np.dtype("bool"),
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def serialize_policy_feature(feature: Mapping[str, np.ndarray]) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for key, raw in sorted(feature.items()):
        array = np.asarray(raw)
        dtype = str(array.dtype)
        if dtype not in _ALLOWED_DTYPES:
            raise ValueError(f"unsupported policy-feature dtype for {key}: {dtype}")
        if array.ndim > 3:
            raise ValueError(f"policy-feature rank exceeds three for {key}")
        tensors[str(key)] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "values": array.tolist(),
        }
    payload: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "tensors": tensors,
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def deserialize_policy_feature(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("unexpected policy-feature snapshot schema")
    expected = str(payload.get("sha256") or "")
    unsigned = {"schema": payload.get("schema"), "tensors": payload.get("tensors")}
    actual = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    if expected != actual:
        raise ValueError("policy-feature snapshot hash mismatch")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("policy-feature snapshot has no tensors")
    feature: dict[str, np.ndarray] = {}
    for key, item in tensors.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid policy-feature tensor: {key}")
        dtype_name = str(item.get("dtype") or "")
        dtype = _ALLOWED_DTYPES.get(dtype_name)
        if dtype is None:
            raise ValueError(f"unsupported policy-feature dtype for {key}: {dtype_name}")
        shape = tuple(int(value) for value in (item.get("shape") or ()))
        array = np.asarray(item.get("values"), dtype=dtype)
        if array.shape != shape:
            raise ValueError(
                f"policy-feature shape mismatch for {key}: {array.shape} != {shape}"
            )
        feature[str(key)] = array
    return feature

