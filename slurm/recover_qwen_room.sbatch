#!/bin/bash
#SBATCH --job-name=scenecode-recover
#SBATCH --output=/scratch/nl2752/scenecode_logs/%j.out
#SBATCH --error=/scratch/nl2752/scenecode_logs/%j.err
#SBATCH --chdir=/scratch/nl2752/SceneCode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --account=torch_pr_56_tandon_advanced
set -eo pipefail
source /scratch/nl2752/scenecode_setup/activate.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENAI_AGENTS_DISABLE_TRACING=1
RUN_DIR="$PWD/outputs/qwen_recovery_${SLURM_JOB_ID}"
python scripts/qwen_run_record.py "$RUN_DIR" --snapshot --stage recovery
finish() {
  status=$?
  trap - EXIT
  python scripts/qwen_run_record.py "$RUN_DIR" --stage recovery --exit-code "$status" || true
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
python scripts/recover_qwen_room.py --source "${SCENECODE_RECOVERY_SOURCE:-$PWD/outputs/qwen_19051280}" --output "$RUN_DIR" --defer-blend
