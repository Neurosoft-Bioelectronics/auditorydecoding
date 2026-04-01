#!/bin/bash
#SBATCH --job-name=freq-sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sweep_%j.out

set -e

CONFIG=${1:?Usage: sbatch scripts/submit_sweep.sh <config.yaml> [output_dir]}
OUTPUT_DIR=${2:-results}

echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "Config:   $CONFIG"
echo "Output:   $OUTPUT_DIR"
echo "CPUs:     $SLURM_CPUS_PER_TASK"

CONFIG_NAME=$(basename "$CONFIG" .yaml)

mkdir -p "$OUTPUT_DIR" logs

srun uv run --offline python scripts/frequency_decoding_sweep.py \
    --config "$CONFIG" \
    --output-csv "$OUTPUT_DIR/${CONFIG_NAME}_${SLURM_JOB_ID}.csv" \
    --output-jsonl "$OUTPUT_DIR/${CONFIG_NAME}_${SLURM_JOB_ID}.jsonl" \
    --num-cpus "$SLURM_CPUS_PER_TASK" \
    --log-level INFO
