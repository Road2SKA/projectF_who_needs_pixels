#!/bin/bash
#SBATCH --job-name=TRAIN
#SBATCH --output=/idia/projects/roadtoska/projectF/LOGS/slurm_TRAIN.out
#SBATCH --error=/idia/projects/roadtoska/projectF/LOGS/slurm_TRAIN.err
#SBATCH --time=01:00:00
#SBATCH --partition=GPU
#SBATCH --constraint=A100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --reservation=roadtoska-gpu

# SIREN Model Training on the Milky Way galactic center mosiac - GPU Required
# This script trains the SIREN model on prepared data

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================================
# CONFIGURATION - Edit these paths as needed
# ============================================================================
SCRIPTS_DIR="${SCRIPTS_DIR:-/project_workspace/scripts}"
CONFIG_FILE="${CONFIG_FILE:-/project_workspace/scripts/config.yaml}"
REQUIREMENTS="${REQUIREMENTS:-/idia/projects/roadtoska/projectF/DEPENDENCIES/requirements.txt}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load modules
module purge
module load apptainer

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
# RUN TRAINING
# ============================================================================

echo "Starting model training..."
echo ""

export CONTAINER=/idia/projects/roadtoska/projectF/pytorch_projectF.sif

apptainer exec --nv \
  --bind $PWD  \
  --bind /idia/projects/roadtoska/projectF:/project_workspace \
  "$CONTAINER" \
  python $SCRIPTS_DIR/train.py --config $CONFIG_FILE


TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Training completed successfully"
echo "End time: $(date)"
echo "=========================================="
