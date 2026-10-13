#!/bin/bash
#SBATCH --job-name=sofa-room-physics
#SBATCH --output=/scratch/nl2752/scenecode_logs/%j.out
#SBATCH --error=/scratch/nl2752/scenecode_logs/%j.err
#SBATCH --chdir=/scratch/nl2752/SceneCode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_56_tandon_advanced
set -eo pipefail
source /scratch/nl2752/scenecode_setup/activate.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENAI_AGENTS_DISABLE_TRACING=1
RUN_DIR="$PWD/outputs/sofa_room_${SLURM_JOB_ID}"
python scripts/qwen_run_record.py "$RUN_DIR" --snapshot --stage sofa_integration
finish() {
  status=$?
  trap - EXIT
  python scripts/qwen_run_record.py "$RUN_DIR" --stage sofa_integration --exit-code "$status" || true
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
python scripts/integrate_sofa_repair.py --source "$PWD/outputs/qwen_recovery_export_19084523" --repair "${1:?Pass reviewed repair output directory}" --output "$RUN_DIR"
