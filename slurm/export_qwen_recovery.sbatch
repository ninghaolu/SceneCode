#!/bin/bash
#SBATCH --job-name=scenecode-export
#SBATCH --output=/scratch/nl2752/scenecode_logs/%j.out
#SBATCH --error=/scratch/nl2752/scenecode_logs/%j.err
#SBATCH --chdir=/scratch/nl2752/SceneCode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --account=torch_pr_56_tandon_advanced
set -eo pipefail
source /scratch/nl2752/scenecode_setup/activate.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENAI_AGENTS_DISABLE_TRACING=1
RUN_DIR="$PWD/outputs/qwen_recovery_export_${SLURM_JOB_ID}"
SOURCE_DIR="${1:?Pass the CPU recovery output directory}"
python scripts/qwen_run_record.py "$RUN_DIR" --snapshot --stage export
finish() {
  status=$?
  trap - EXIT
  python scripts/qwen_run_record.py "$RUN_DIR" --stage export --exit-code "$status" || true
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
python scripts/export_qwen_recovery.py --source "$SOURCE_DIR" --output "$RUN_DIR"
if [[ -f "$SOURCE_DIR/integration.json" ]]; then
  blender -b "$RUN_DIR/room/scene_000/combined_house_after_furniture/house.blend" -t "$SLURM_CPUS_PER_TASK" --python-exit-code 1 \
    --python scripts/render_recovered_room.py -- --output "$RUN_DIR/room_review"
fi
