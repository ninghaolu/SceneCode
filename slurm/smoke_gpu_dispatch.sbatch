#!/bin/bash
#SBATCH --job-name=scenecode-render-smoke
#SBATCH --output=/scratch/nl2752/scenecode_logs/%j.out
#SBATCH --error=/scratch/nl2752/scenecode_logs/%j.err
#SBATCH --chdir=/scratch/nl2752/SceneCode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_56_tandon_advanced
set -eo pipefail
source /scratch/nl2752/scenecode_setup/activate.sh
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=8
export OPENAI_AGENTS_DISABLE_TRACING=1
export SCENECODE_GPU_STAGE_DIR="$PWD/outputs/gpu_split_smoke_${SLURM_JOB_ID}/gpu_tasks"
export TMPDIR="$PWD/outputs/gpu_split_smoke_${SLURM_JOB_ID}/tmp"
mkdir -p "$TMPDIR"
trap 'python scripts/cancel_pending_gpu_tasks.py "$SCENECODE_GPU_STAGE_DIR" || true' EXIT
python scripts/smoke_gpu_dispatch.py "$PWD/outputs/gpu_split_smoke_${SLURM_JOB_ID}"
