#!/bin/bash
#SBATCH --job-name=TUNE
#SBATCH --output=LOGS/slurm_TUNE.out
#SBATCH --error=LOGS/slurm_TUNE.err
#SBATCH --time=03:00:00
#SBATCH --partition=GPU
#SBATCH --constraint=A100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --reservation=roadtoska-gpu

# SIREN Hyperparameter Tuning - GPU Required
# This script performs hyperparameter optimization using Optuna

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

# Check if tune.py exists
if [ ! -f "tune.py" ]; then
    echo "ERROR: tune.py not found in current directory"
    exit 1
fi
echo "✓ tune.py found"

# Check if prepared data exists
DATA_FILE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['paths']['data'])" 2>/dev/null)
if [ -n "$DATA_FILE" ] && [ ! -f "$DATA_FILE" ]; then
    echo "WARNING: Prepared data file not found at $DATA_FILE"
    echo "         Tuning may fail if data preparation didn't complete"
fi

# Check if trained model exists (optional - tuning might train from scratch)
MODEL_FILE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['paths']['model'])" 2>/dev/null)
if [ -n "$MODEL_FILE" ] && [ ! -f "$MODEL_FILE" ]; then
    echo "INFO: Trained model not found at $MODEL_FILE"
    echo "      Tuning will proceed (may train models from scratch)"
fi

# Check Python and PyTorch
python --version || { echo "ERROR: Python not available"; exit 1; }
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || { echo "ERROR: PyTorch not properly installed"; exit 1; }
echo "✓ Python and PyTorch available"

# Check Optuna
python -c "import optuna; print(f'Optuna {optuna.__version__}')" || { echo "WARNING: Optuna not available - tuning may fail"; }

echo "=========================================="
echo ""

# ============================================================================
# RUN HYPERPARAMETER TUNING
# ============================================================================

echo "Starting hyperparameter tuning..."
echo "This may take a while (config specifies ${n_trials:-20} trials)"
echo ""

apptainer exec \
  --bind $PWD \
  "$CONTAINER" \
  python tune.py --config config.yaml

TUNE_EXIT_CODE=$?

if [ $TUNE_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Hyperparameter tuning failed with exit code $TUNE_EXIT_CODE"
    exit $TUNE_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Hyperparameter tuning completed successfully"
echo "End time: $(date)"
echo "=========================================="

# Deactivate virtual environment
# Deactivate virtual environment if active
#if command -v deactivate >/dev/null 2>&1; then
#    deactivate || true
#fi

#exit 0

