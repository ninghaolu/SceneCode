#!/bin/bash
#SBATCH --job-name=scenecode-room
#SBATCH --output=/scratch/nl2752/scenecode_logs/%j.out
#SBATCH --error=/scratch/nl2752/scenecode_logs/%j.err
#SBATCH --chdir=/scratch/nl2752/SceneCode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_56_tandon_advanced
set -eo pipefail
source /scratch/nl2752/scenecode_setup/activate.sh
source /scratch/nl2752/.config/scenecode/qwen.env
export OPENAI_API_KEY="${QWEN_API_KEY:?Missing Qwen credentials}"
export OPENAI_API_BASE="${QWEN_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export SCENECODE_QWEN_MODEL="${SCENECODE_QWEN_MODEL:?Select the Qwen model}"
export SCENECODE_REQUIRED_MODEL="$SCENECODE_QWEN_MODEL"
export SCENECODE_QWEN_REASONING=low
export SCENECODE_THINKING_BUDGET=2048
export SCENECODE_MAX_TOKENS=16384
export OPENAI_AGENTS_DISABLE_TRACING=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
RUN_DIR="${1:?Prepared output required}"
SMOKE_REPORT="${2:?GPU smoke report required}"
export SCENECODE_RENDER_BACKEND=slurm
unset SCENECODE_GPU_WORKER
export SCENECODE_GPU_STAGE_DIR="$RUN_DIR/gpu_tasks"
export SCENECODE_GPU_WAIT_SECONDS=7200
export SCENECODE_AUDIT_DIR="$RUN_DIR/api"
export SCENECODE_BLENDER_GLOBAL_LOCK="$RUN_DIR/blender.lock"
export TMPDIR="$RUN_DIR/tmp"
mkdir -p "$TMPDIR"
stage=setup
finish() {
    status=$?
    trap - EXIT
    python scripts/cancel_pending_gpu_tasks.py "$SCENECODE_GPU_STAGE_DIR" || true
    python scripts/summarize_qwen_run.py "$RUN_DIR" --exit-code "$status" || true
    python scripts/qwen_run_record.py "$RUN_DIR" --stage "$stage" --exit-code "$status" || true
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python scripts/check_gpu_split_ready.py "$SMOKE_REPORT"
python scripts/qwen_run_record.py "$RUN_DIR" --snapshot --stage "$stage"
stage=preflight
python scripts/qwen_run_record.py "$RUN_DIR" --stage "$stage"
REFERENCE="$(python -c 'import json; print(json.load(open("slurm/qwen_saved_images.json"))[0]["path"])')"
python scripts/qwen_preflight.py --image "$REFERENCE" --output "$RUN_DIR/preflight.json"
stage=room_generation
python scripts/qwen_run_record.py "$RUN_DIR" --stage "$stage"
python scripts/run_prepared_room.py "$RUN_DIR/room_config.yaml"
stage=final_validation
python scripts/qwen_run_record.py "$RUN_DIR" --stage "$stage"
python scripts/validate_qwen_room.py "$RUN_DIR/room"
stage=complete_pending_visual_review
