#!/bin/bash
#SBATCH --job-name=TRAIN
#SBATCH --output=LOGS/slurm_TRAIN.out
#SBATCH --error=LOGS/slurm_TRAIN.err
#SBATCH --time=01:00:00
#SBATCH --partition=GPU
#SBATCH --constraint=A100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --reservation=roadtoska-gpu

# SIREN Model Training - GPU Required
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
CONFIG_FILE="${CONFIG_FILE:-/idia/projects/roadtoska/projectF/DEPENDENCIES/config.yaml}"
REQUIREMENTS="${REQUIREMENTS:-/idia/projects/roadtoska/projectF/DEPENDENCIES/requirements.txt}"
CONTAINER="${CONTAINER:-/idia/projects/roadtoska/projectF/pytorch_projectF.sif}" 
#VENV_PATH="${VENV_PATH:-siren_env}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load modules
module purge
module load apptainer
#module load cuda/11.7

# Activate virtual environment
#echo ""
#echo "Activating virtual environment: $VENV_PATH"#
#if [ ! -d "$VENV_PATH" ]; then
#    echo "ERROR: Virtual environment not found at $VENV_PATH"
#    exit 1
#fi
#source $VENV_PATH/bin/activate

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
# VALIDATION CHECKS
# ============================================================================

echo "=========================================="
echo "Validation Checks"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi
echo "✓ Config file found: $CONFIG_FILE"

# Check if train.py exists
if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found in current directory"
    exit 1
fi
echo "✓ train.py found"

# Check if prepared data exists (assuming it's at paths.data from config)
# This is a basic check - train.py will do more thorough validation
DATA_FILE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['paths']['data'])" 2>/dev/null)
if [ -n "$DATA_FILE" ] && [ ! -f "$DATA_FILE" ]; then
    echo "WARNING: Prepared data file not found at $DATA_FILE"
    echo "         Training may fail if data preparation didn't complete"
fi

# Check Python and PyTorch
apptainer exec "$CONTAINER" python --version || { echo "ERROR: Python not available"; exit 1; }
apptainer exec --nv "$CONTAINER" python -c "import torch; print(f'PyTorch {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}')" || { echo "ERROR: PyTorch not properly installed"; exit 1; }

echo "=========================================="
echo ""

# ============================================================================
# RUN TRAINING
# ============================================================================

echo "Starting model training..."
echo ""

apptainer exec --nv "$CONTAINER" python train.py --config "$CONFIG_FILE"

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

# Deactivate virtual environment
# Deactivate virtual environment if active
#if command -v deactivate >/dev/null 2>&1; then
#    deactivate || true
#fi

#exit 0

