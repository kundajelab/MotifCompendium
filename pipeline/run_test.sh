#!/bin/bash

# Set failure
set -euo pipefail

# Load modules
# module load cuda/11.2

# Load conda environment
eval "$(conda shell.bash hook)"
conda activate motifcompendium-gpu

# Variables
NAME="fetal-pipeline"
OUTPUT_DIR="/oak/stanford/groups/akundaje/cmyun/motifcompendium/fetal/pipeline"
MODISCO_PATHS="${OUTPUT_DIR}/raw/modisco_paths.tsv"
mapfile -t INPUT_NAMES < <(cut -f 1 "$MODISCO_PATHS")
mapfile -t INPUT_PATHS < <(cut -f 2 "$MODISCO_PATHS")
MC_PATH="${OUTPUT_DIR}/motifcompendium.mc"
REFERENCE_PATH="/oak/stanford/groups/akundaje/cmyun/software/motifcompendium/pipeline/data/JASPAR2024-HOCOMOCOv13.meme.txt"

SIM_THRESHOLD=0.9
SIM_SCAN=(0.8 0.85 0.9 0.95 0.98)
MIN_SEQLETS=100

MAX_CHUNK=1200
MAX_CPUS=32

SCRIPT_PATH="/oak/stanford/groups/akundaje/cmyun/software/motifcompendium/pipeline/pipeline.py"
LOG_PATH="${OUTPUT_DIR}/logs/log_pipeline_v2_$(date +'%Y%m%d_%H%M%S').o"
ERROR_PATH="${OUTPUT_DIR}/logs/log_pipeline_v2_$(date +'%Y%m%d_%H%M%S').e"

# Run script
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$(dirname "$ERROR_PATH")"

python3 -u "$SCRIPT_PATH" \
    -im "$MC_PATH" \
    -o "$OUTPUT_DIR" \
    -r "$REFERENCE_PATH" \
    --sim-threshold "$SIM_THRESHOLD" \
    --sim-scan "${SIM_SCAN[@]}" \
    --min-seqlets "$MIN_SEQLETS" \
    --quality \
    --html-collection \
    --html-table \
    --html-removed \
    -ch "$MAX_CHUNK" \
    -cp "$MAX_CPUS" \
    --use-gpu \
    --fast-plot \
    --time \
    --verbose > >(tee -a "$LOG_PATH") 2> >(tee -a "$ERROR_PATH" >&2)
    # -ih "${INPUT_PATHS[@]}" \
    # -nh "${INPUT_NAMES[@]}" \

echo "Completed run."