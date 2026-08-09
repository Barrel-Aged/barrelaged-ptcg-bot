"""Safely extract the submission payload embedded in a public Kaggle notebook.

The Rocket replay-clone notebooks store their three submission source files as a
literal ``PAYLOADS`` mapping of base64 strings.  This utility parses the cell's
AST and evaluates only that literal assignment; it never executes notebook
code.  Extraction is deliberately restricted to the expected filenames.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
from pathlib import Path


ALLOWED_FILENAMES = {"bc_agent.py", "deck.csv", "main.py"}


def extract_payloads(notebook_path: Path) -> dict[str, bytes]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assignments: list[ast.Assign] = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "PAYLOADS" not in source:
            continue
        tree = ast.parse(source, filename=str(notebook_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "PAYLOADS" for target in node.targets):
                assignments.append(node)

    if len(assignments) != 1:
        raise ValueError(f"expected one PAYLOADS assignment, found {len(assignments)}")

    encoded = ast.literal_eval(assignments[0].value)
    if not isinstance(encoded, dict) or set(encoded) != ALLOWED_FILENAMES:
        raise ValueError(f"unexpected payload filenames: {sorted(encoded) if isinstance(encoded, dict) else type(encoded)}")

    decoded: dict[str, bytes] = {}
    for name, value in encoded.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("PAYLOADS must map strings to base64 strings")
        decoded[name] = base64.b64decode(value, validate=True)
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    payloads = extract_payloads(args.notebook)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        destination = args.output_dir / name
        destination.write_bytes(content)
        print(f"extracted {name}: {len(content)} bytes")


if __name__ == "__main__":
    main()
