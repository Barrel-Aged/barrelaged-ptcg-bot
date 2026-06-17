#!/usr/bin/env bash
# Build the Kaggle submission bundle: main.py + deck.csv + the bot/ package at the top level.
# (cg is provided by the Environment — not bundled.)
set -euo pipefail
cd "$(dirname "$0")"

for required in main.py deck.csv bot/agent.py; do
  [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
[[ $(grep -cve '^[[:space:]]*$' deck.csv) -eq 60 ]] || { echo "ERROR: deck.csv must have 60 card IDs" >&2; exit 1; }

rm -f submission.tar.gz
tar -czf submission.tar.gz \
  --exclude='__pycache__' --exclude='*.pyc' \
  main.py deck.csv bot
echo "Built submission.tar.gz:"
tar -tzf submission.tar.gz | sed 's/^/  /'
