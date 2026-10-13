#!/bin/bash
#SBATCH --job-name=scenecode-render-test
#SBATCH --output=/scratch/nl2752/scenecode_logs/%j.out
#SBATCH --error=/scratch/nl2752/scenecode_logs/%j.err
#SBATCH --chdir=/scratch/nl2752/SceneCode
#SBATCH --account=torch_pr_56_tandon_advanced
#SBATCH --nodes=1
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00

source /scratch/nl2752/scenecode_setup/activate.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
nvidia-smi
python -m pytest tests/integration/test_blender.py::TestBlenderIntegration::test_blender_integration_pipeline -q -s
