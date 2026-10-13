#!/bin/bash
#SBATCH --job-name=scenecode-living
#SBATCH --output=/scratch/nl2752/scenecode_logs/%j.out
#SBATCH --error=/scratch/nl2752/scenecode_logs/%j.err
#SBATCH --chdir=/scratch/nl2752/SceneCode
#SBATCH --nodes=1
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --account=torch_pr_56_tandon_advanced

cd /scratch/nl2752/SceneCode
source /scratch/nl2752/scenecode_setup/activate.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Export API credentials before sbatch. Default image backend: openai (gpt-image-2).
# For local Flux: export SCENECODE_IMAGE_BACKEND=flux-klein,
# FLUX_PYTHON=/path/to/flux/python, and FLUX_MODEL_PATH=/path/to/checkpoint.
python main.py \
  experiment=slurm_living_room \
  +name=living_room_${SLURM_JOB_ID:-local} \
  floor_plan_agent.openai.model="${SCENECODE_ROOM_MODEL:-gpt-5.4}" \
  furniture_agent.openai.model="${SCENECODE_ROOM_MODEL:-gpt-5.4}" \
  furniture_agent.asset_manager.code_object.config_path="${SCENECODE_OBJECT_CONFIG:-agent_framework/config.yaml}" \
  floor_plan_agent.image_generation.backend="${SCENECODE_IMAGE_BACKEND:-openai}" \
  floor_plan_agent.materials.generator.backend="${SCENECODE_IMAGE_BACKEND:-openai}" \
  furniture_agent.asset_manager.image_generation.backend="${SCENECODE_IMAGE_BACKEND:-openai}" \
  furniture_agent.asset_manager.router.strategies.thin_covering.generator.backend="${SCENECODE_IMAGE_BACKEND:-openai}" \
  floor_plan_agent.image_generation.fallback_backend=null \
  furniture_agent.asset_manager.image_generation.fallback_backend=null \
  "$@"
