#!/usr/bin/env bash
# Run hybrid privacy tradeoff + maxauc_vs_degree for a results folder.
# Usage: ./scripts/plot_results_folder.sh <input_folder> <output_folder>
#
# - Hybrid plots go to <output_folder>/hybrid/
# - Max-AUC plots: one subfolder per CSV under <output_folder>/max/<basename>/ 

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR="$1"
OUT_DIR="$2"

if [[ -z "$INPUT_DIR" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 <input_folder> <output_folder>"
  echo "  input_folder   e.g. results/a/er_p_0.08 (relative to project root)"
  echo "  output_folder e.g. plots/a (relative to project root)"
  exit 1
fi

cd "$REPO_ROOT"

# Resolve paths: if not absolute, treat as relative to project root
[[ "$INPUT_DIR" != /* ]] && INPUT_DIR="$REPO_ROOT/$INPUT_DIR"
[[ "$OUT_DIR" != /* ]] && OUT_DIR="$REPO_ROOT/$OUT_DIR"
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# --- Hybrid privacy tradeoff → output_folder/hybrid/
HYBRID_OUT="${OUT_DIR}/hybrid"
mkdir -p "$HYBRID_OUT"
echo "Running hybrid_privacy_tradeoff.py -> $HYBRID_OUT"
python3 src/plotting/hybrid_privacy_tradeoff.py \
  --results-dir "$INPUT_DIR" \
  --out-dir "$HYBRID_OUT"

# --- Max-AUC vs degree: one run per CSV → output_folder/max/<basename>/
MAX_BASE="${OUT_DIR}/max"
mkdir -p "$MAX_BASE"
shopt -s nullglob
for csv in "$INPUT_DIR"/*.csv; do
  base="$(basename "$csv" .csv)"
  run_out="${MAX_BASE}/${base}"
  mkdir -p "$run_out"
  echo "Running maxauc_vs_degree.py for $base -> $run_out"
  python3 src/plotting/maxauc_vs_degree.py \
    --results-glob "$csv" \
    --round 100 \
    --out-dir "$run_out" \
    --auc-col max_auc \
    --plot-global-test-acc \
    --ylim 0.5 1 \
    --y2lim 0.2 0.7
done
shopt -u nullglob

echo "Done. Hybrid plots: $HYBRID_OUT"
echo "Max-AUC per-file: $MAX_BASE/<file_stem>/"
