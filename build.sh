#!/usr/bin/env bash
# Build the Kaggle submission bundle: main.py + deck.csv + bot/ + cg/ at the top level.
#
# We BUNDLE the cg engine, matching the official sample submission (which does
# `tar.add(.../cg-lib/cg, arcname="cg")`). Shipping without cg means `import cg` fails at eval and the
# Agent silently falls back to the weak DefaultSafePolicy — so we hard-require cg here.
set -euo pipefail
cd "$(dirname "$0")"

for required in main.py deck.csv bot/agent.py bot/knowledge/heuristics_beginner.json; do
  [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
[[ $(grep -cve '^[[:space:]]*$' deck.csv) -eq 60 ]] || { echo "ERROR: deck.csv must have 60 card IDs" >&2; exit 1; }

# Locate the cg engine to bundle (override with PTCG_CG_DIR; default = the extracted sample SDK).
CG_DIR="${PTCG_CG_DIR:-$HOME/ptcg-abc/notebooks/_sdk/sample_submission/cg}"
if [[ ! -f "$CG_DIR/api.py" ]] || { [[ ! -f "$CG_DIR/libcg.so" ]] && [[ ! -f "$CG_DIR/cg.dll" ]]; }; then
  echo "ERROR: cg engine not found at '$CG_DIR' (need api.py + libcg.so/cg.dll)." >&2
  echo "       Set PTCG_CG_DIR to your extracted competition cg/ directory and re-run." >&2
  exit 1
fi

rm -f submission.tar.gz
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp main.py deck.csv "$STAGE"/
cp -r bot "$STAGE"/bot
cp -r "$CG_DIR" "$STAGE"/cg
# drop runtime-irrelevant weight: the local eval harness + caches (cg is provided at eval anyway).
rm -rf "$STAGE"/bot/eval
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGE" -name '*.pyc' -delete

tar -czf submission.tar.gz -C "$STAGE" main.py deck.csv bot cg
echo "Built submission.tar.gz (cg from $CG_DIR):"
tar -tzf submission.tar.gz | grep -E '^(main.py|deck.csv|cg/|bot/[^/]*/?$|bot/knowledge)' | sed 's/^/  /'
echo "  ... ($(tar -tzf submission.tar.gz | wc -l) entries, $(du -h submission.tar.gz | cut -f1))"
