#!/bin/bash
#SBATCH --job-name=qwen-sofa-plan
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
source "${SCENECODE_QWEN_ENV:-/scratch/nl2752/.config/scenecode/qwen.env}"
export OPENAI_API_KEY="$QWEN_API_KEY"
export OPENAI_API_BASE="${QWEN_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export SCENECODE_QWEN_MODEL="${SCENECODE_QWEN_MODEL:-qwen3.7-plus}"
export SCENECODE_REQUIRED_MODEL="$SCENECODE_QWEN_MODEL"
export SCENECODE_QWEN_REASONING=low
export SCENECODE_THINKING_BUDGET=2048
export SCENECODE_MAX_TOKENS=16384
export OPENAI_AGENTS_DISABLE_TRACING=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
RUN_DIR="$PWD/outputs/sofa_plan_${SLURM_JOB_ID}"
export SCENECODE_AUDIT_DIR="$RUN_DIR/api"
python scripts/qwen_run_record.py "$RUN_DIR" --snapshot --stage supplied_plan
finish() {
  status=$?
  trap - EXIT
  python scripts/qwen_run_record.py "$RUN_DIR" --stage supplied_plan --exit-code "$status" || true
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
python scripts/qwen_sofa_supplied_plan.py --output "$RUN_DIR" --plan "${1:?Pass reviewed plan directory}" --attempts 2
