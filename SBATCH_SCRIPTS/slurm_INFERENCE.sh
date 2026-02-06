#!/bin/bash
#SBATCH --job-name=INFER
#SBATCH --output=/idia/projects/roadtoska/projectF/LOGS/slurm_INFERENCE.out
#SBATCH --error=/idia/projects/roadtoska/projectF/LOGS/slurm_INFERENCE.err
#SBATCH --time=03:00:00
#SBATCH --partition=GPU
#SBATCH --constraint=A100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --reservation=roadtoska-gpu

# SIREN Inferences - GPU Required

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================================
# CONFIGURATION
# ============================================================================
REQUIREMENTS="${REQUIREMENTS:-/idia/projects/roadtoska/projectF/DEPENDENCIES/requirements.txt}"
CONTAINER="${CONTAINER:-/idia/projects/roadtoska/projectF/pytorch_projectF.sif}"
SCRIPTS_DIR="${SCRIPTS_DIR:-/project_workspace/scripts}"
CONFIG_FILE="${CONFIG_FILE:-/project_workspace/scripts/config.yaml}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load modules
module purge
module load apptainer
#module load cuda/11.7

# ============================================================================
# GPU CHECK
# ============================================================================
echo ""
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
GPU_CHECK=$?
if [ $GPU_CHECK -ne 0 ]; then
    echo "ERROR: GPU not available or nvidia-smi failed"
    exit 1
fi
echo "✓ GPU available"
echo "=========================================="
echo ""

# ============================================================================
# INFERENCES TESTING
# ============================================================================

export CONTAINER=/idia/projects/roadtoska/projectF/pytorch_projectF.sif

apptainer exec --nv \
  --bind $PWD  \
  --bind /idia/projects/roadtoska/projectF:/project_workspace \
  "$CONTAINER" \
  python $SCRIPTS_DIR/inference.py --config $CONFIG_FILE


TUNE_EXIT_CODE=$?

if [ $INFERENCE_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: failed with exit code $INFERENCE_EXIT_CODE"
    exit $INFERENCE_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Completed successfully"
echo "End time: $(date)"
echo "=========================================="
