#!/bin/bash
#SBATCH --job-name=PREP
#SBATCH --output=LOGS/slurm_PREPARE.out
#SBATCH --error=LOGS/slurm_PREPARE.err
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
CONFIG_FILE="${CONFIG_FILE:-/idia/projects/roadtoska/projectF/DEPENDENCIES/config.yaml}"
REQUIREMENTS="${REQUIREMENTS:-/idia/projects/roadtoska/projectF/DEPENDENCIES/requirements.txt}"
#VENV_PATH="${VENV_PATH:-siren_env}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load modules
module purge
module load apptainer
#module load python/3.12

# Activate virtual environment
#echo ""
#echo "Activating virtual environment: $VENV_PATH"
#if [ ! -d "$VENV_PATH" ]; then
#    echo "ERROR: Virtual environment not found at $VENV_PATH"
#    exit 1
#fi
#source $VENV_PATH/bin/activate

# Install requirements
echo ""
echo "Installing requirements from: $REQUIREMENTS"
if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: Requirements file not found at $REQUIREMENTS"
    exit 1
fi
pip install -r $REQUIREMENTS --quiet

# ============================================================================
# VALIDATION CHECKS
# ============================================================================

echo ""
echo "=========================================="
echo "Validation Checks"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi
echo "✓ Config file found: $CONFIG_FILE"

# Check if prepare.py exists
if [ ! -f "prepare.py" ]; then
    echo "ERROR: prepare.py not found in current directory"
    exit 1
fi
echo "✓ prepare.py found"

# Check Python installation
python --version || { echo "ERROR: Python not available"; exit 1; }
echo "✓ Python available"

echo "=========================================="
echo ""

# ============================================================================
# RUN DATA PREPARATION
# ============================================================================

echo "Starting data preparation..."
echo ""
export CONTAINER=/idia/projects/roadtoska/projectF/pytorch_projectF.sif
apptainer exec "$CONTAINER" pip install --user python-dateutil

apptainer exec \
  --bind $PWD \
  "$CONTAINER" \
  python prepare.py --config config.yaml

PREPARE_EXIT_CODE=$?

if [ $PREPARE_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Data preparation failed with exit code $PREPARE_EXIT_CODE"
    exit $PREPARE_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Data preparation completed successfully"
echo "End time: $(date)"
echo "=========================================="

# Deactivate virtual environment
# Deactivate virtual environment if active
#if command -v deactivate >/dev/null 2>&1; then#
#    deactivate || true
#fi

#exit 0

