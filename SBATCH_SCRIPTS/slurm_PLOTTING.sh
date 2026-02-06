#!/bin/bash
#SBATCH --job-name=PLOT
#SBATCH --output=/idia/projects/roadtoska/projectF/LOGS/slurm_PLOT.out
#SBATCH --error=/idia/projects/roadtoska/projectF/LOGS/slurm_PLOT.err
#SBATCH --time=00:05:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --reservation=roadtoska-gpu

# SIREN Data Preparation - CPU
# This script prepares the FITS data for training

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
REQUIREMENTS="${REQUIREMENTS:-/idia/projects/roadtoska/projectF/DEPENDENCIES/requirements.txt}"
#VENV_PATH="${VENV_PATH:-siren_env}"
SCRIPTS_DIR="${SCRIPTS_DIR:-/project_workspace/scripts}"
CONFIG_FILE="${CONFIG_FILE:-/project_workspace/scripts/config.yaml}"


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load modules
module purge
module load apptainer


# Install requirements
# Uncomment if we need to reinstall dependencies
# echo ""
# echo "Installing requirements from: $REQUIREMENTS"
# if [ ! -f "$REQUIREMENTS" ]; then
#     echo "ERROR: Requirements file not found at $REQUIREMENTS"
#     exit 1
# fi
# pip install -r $REQUIREMENTS --quiet

# ============================================================================
# VALIDATION CHECKS
# ============================================================================

echo ""
echo "=========================================="
echo "Validation Checks"
echo "=========================================="

# Check Python installation
python --version || { echo "ERROR: Python not available"; exit 1; }
echo "✓ Python available"

echo "=========================================="
echo ""

# ============================================================================
# RUN DATA PREPARATION
# ============================================================================

echo "Starting Image plotting sequence beep boop boop"
echo ""
export CONTAINER=/idia/projects/roadtoska/projectF/pytorch_projectF.sif
apptainer exec "$CONTAINER" pip install --user python-dateutil

apptainer exec \
  --bind $PWD  \
  --bind /idia/projects/roadtoska/projectF:/project_workspace \
  "$CONTAINER" \
  python $SCRIPTS_DIR/plot_training_frames.py.py --config $CONFIG_FILE

PREPARE_EXIT_CODE=$?

if [ $PLOT_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Data preparation failed with exit code $PLOT_EXIT_CODE"
    exit $PLOT_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Image plotting complete completed successfully"
echo "End time: $(date)"
echo "=========================================="
