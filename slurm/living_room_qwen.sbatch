#!/bin/bash
#SBATCH --job-name=scenecode-qwen
#SBATCH --output=/scratch/nl2752/scenecode_logs/%j.out
#SBATCH --error=/scratch/nl2752/scenecode_logs/%j.err
#SBATCH --chdir=/scratch/nl2752/SceneCode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --export=ALL
#SBATCH --account=torch_pr_56_tandon_advanced

set -eo pipefail
cd "${SCENECODE_SOURCE_ROOT:-/scratch/nl2752/SceneCode}"
source /scratch/nl2752/scenecode_setup/activate.sh
if [[ -z "${QWEN_API_KEY:-}" ]]; then
  source "${SCENECODE_QWEN_ENV:-/scratch/nl2752/.config/scenecode/qwen.env}"
fi
export OPENAI_API_KEY="${QWEN_API_KEY:?Set QWEN_API_KEY}"
export OPENAI_API_BASE="${QWEN_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export SCENECODE_QWEN_MODEL="${SCENECODE_QWEN_MODEL:-qwen3.7-plus}"
export SCENECODE_REQUIRED_MODEL="$SCENECODE_QWEN_MODEL"
export SCENECODE_QWEN_REASONING="${SCENECODE_QWEN_REASONING:-low}"
export SCENECODE_THINKING_BUDGET="${SCENECODE_THINKING_BUDGET:-2048}"
export SCENECODE_MAX_TOKENS="${SCENECODE_MAX_TOKENS:-16384}"
export OPENAI_AGENTS_DISABLE_TRACING=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR="${SCENECODE_RUN_DIR:-$PWD/outputs/qwen_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
mkdir -p "$RUN_DIR"
RUN_DIR="$(realpath "$RUN_DIR")"
# Keep API orchestration on CPU; use the verified short GPU dispatch path.
export SCENECODE_RENDER_BACKEND=slurm
unset SCENECODE_GPU_WORKER
export SCENECODE_GPU_STAGE_DIR="$RUN_DIR/gpu_tasks"
export SCENECODE_GPU_WAIT_SECONDS=7200
export SCENECODE_BLENDER_GLOBAL_LOCK="$RUN_DIR/blender.lock"
export TMPDIR="$RUN_DIR/tmp"
mkdir -p "$TMPDIR"
REFERENCE="/scratch/nl2752/SceneCode/outputs/2026-09-27/08-25-27/scene_000/room_living_room/generated_assets/furniture/images/coffee_table_1790514726863.png"
echo "Qwen pilot: model=$SCENECODE_QWEN_MODEL reasoning=$SCENECODE_QWEN_REASONING output=$RUN_DIR"
python scripts/prepare_qwen_pilot.py --output "$RUN_DIR/configs"
REFERENCE_PROMPT="$(python -c 'import json; entries=json.load(open("slurm/qwen_saved_images.json")); print(" ".join(e["description"] for e in entries))')"
ROOM_PROMPT="A living room with exactly four furniture items: one TV console, one sofa, one bookshelf, and one coffee table. Use only the four saved reference images and these matching specifications: $REFERENCE_PROMPT Do not add other furniture, lighting, rugs, wall art, or decorations."
room_args=( \
  experiment=slurm_living_room \
  "experiment.prompts=[\"$ROOM_PROMPT\"]" \
  +name="qwen_living_room_${SLURM_JOB_ID:-local}" \
  hydra.run.dir="$RUN_DIR/room" \
  openai.service_tier=null \
  floor_plan_agent.openai.model="$SCENECODE_QWEN_MODEL" \
  furniture_agent.openai.model="$SCENECODE_QWEN_MODEL" \
  floor_plan_agent.session_memory.summarization_model=agent \
  furniture_agent.session_memory.summarization_model=agent \
  furniture_agent.agents.designer_agent.max_turns=100 \
  furniture_agent.agents.planner_agent.max_turns=60 \
  furniture_agent.agents.critic_agent.max_turns=8 \
  +furniture_agent.agents.critic_agent.finish_on_turn_limit=true \
  floor_plan_agent.materials.generator.enabled=false \
  floor_plan_agent.image_generation.backend=saved \
  +floor_plan_agent.image_generation.saved.manifest="$PWD/slurm/qwen_saved_images.json" \
  floor_plan_agent.image_generation.fallback_backend=null \
  furniture_agent.asset_manager.image_generation.backend=saved \
  +furniture_agent.asset_manager.image_generation.saved.manifest="$PWD/slurm/qwen_saved_images.json" \
  furniture_agent.asset_manager.image_generation.fallback_backend=null \
  furniture_agent.asset_manager.router.strategies.thin_covering.enabled=false \
  furniture_agent.asset_manager.code_object.config_path="$RUN_DIR/configs/config.yaml" \
  furniture_agent.asset_manager.code_object.simple_object_config_path="$RUN_DIR/configs/config_simplemanip.yaml" \
  furniture_agent.asset_manager.code_object.manipuland_config_path="$RUN_DIR/configs/config_structmanip.yaml" \
  +furniture_agent.asset_manager.code_object.wall_art_config_path="$RUN_DIR/configs/config_WallArt.yaml" \
  furniture_agent.asset_manager.code_object.max_attempts=2 \
  furniture_agent.placement_noise.mode=perfect \
  furniture_agent.placement_noise.perfect_profile.position_xy_std_meters=0 \
  furniture_agent.placement_noise.perfect_profile.rotation_yaw_std_degrees=0 \
  furniture_agent.placement_noise.natural_profile.position_xy_std_meters=0 \
  furniture_agent.placement_noise.natural_profile.rotation_yaw_std_degrees=0 \
  "$@" )
if [[ "${SCENECODE_CONFIG_ONLY:-0}" == 1 ]]; then
  python main.py "${room_args[@]}" --cfg job
  exit 0
fi
echo "CPU orchestration with short GPU rendering/export jobs"
export SCENECODE_AUDIT_DIR="$RUN_DIR/api"
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
python scripts/check_gpu_split_ready.py "${SCENECODE_GPU_SMOKE_REPORT:-$PWD/slurm/gpu_split_verified.json}"
python scripts/qwen_run_record.py "$RUN_DIR" --snapshot --stage "$stage"
stage=preflight
python scripts/qwen_run_record.py "$RUN_DIR" --stage "$stage"
python scripts/qwen_preflight.py --image "$REFERENCE" --output "$RUN_DIR/preflight.json"
stage=asset_pilot
python scripts/qwen_run_record.py "$RUN_DIR" --stage "$stage"
if [[ -n "${SCENECODE_ASSET_VALIDATION:-}" ]]; then
  python -c 'import json,os; from pathlib import Path; p=Path(os.environ["SCENECODE_ASSET_VALIDATION"]); r=json.loads(p.read_text()); preflight=json.loads((p.parent.parent / "preflight.json").read_text()); assert r["passed"] and preflight["model"] == os.environ["SCENECODE_REQUIRED_MODEL"] and all(c["passed"] for c in preflight["checks"].values()), "Asset pilot did not pass for this model"'
  cp "$SCENECODE_ASSET_VALIDATION" "$RUN_DIR/asset_validation.json"
else
  python scripts/qwen_asset_pilot.py --image "$REFERENCE" --config "$RUN_DIR/configs/config.yaml" \
    --model "$SCENECODE_QWEN_MODEL" --output "$RUN_DIR/asset_pilot"
fi
if [[ "${SCENECODE_PILOT_ONLY:-0}" == 1 ]]; then
  stage=pilot_complete
  exit 0
fi
stage=room_generation
python scripts/qwen_run_record.py "$RUN_DIR" --stage "$stage"
python main.py "${room_args[@]}"
stage=final_validation
python scripts/qwen_run_record.py "$RUN_DIR" --stage "$stage"
python scripts/validate_qwen_room.py "$RUN_DIR/room"
stage=complete_pending_visual_review
