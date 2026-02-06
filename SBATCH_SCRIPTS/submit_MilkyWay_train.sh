#!/bin/bash

# ============================================================================
# Master Submission Script for SIREN MilkyWay Pipeline
# ============================================================================
# This script submits a complete pipeline using Singularity containers:
#   1. PREPARE (CPU) - Data preparation + install dependencies
#   2. TRAIN (GPU)   - Model training (depends on PREPARE)
#
# Directory structure (run from projectF/ directory):
#	projectF/
#	├── submit_MilkyWay_train.sh --> this script
#       ├── LOGS/
#	├── DEPENDENCIES/
#	│   ├── config.yaml
#	│   └── requirements.txt
#	└── SBATCH_SCRIPTS/
#	    ├── slurm_PREPARE.sh
#	    ├── slurm_TRAIN.sh
#	    ├── slurm_TUNE.sh
#	    ├── prepare.py
#	    ├── train.py
#	    ├── tune.py
#	    └── configs.py
#	    ├── data/
#
# Usage:
#   cd /idia/projects/road2ska/projectF
#   ./submit_MilkyWay_train.sh with sbatch submit_MilkyWay_train.sh
# ============================================================================

echo "========================================================================"
echo "SIREN MilkyWay Pipeline Submission"
echo "========================================================================"
echo "Submission time: $(date)"
echo ""

# ============================================================================
# CONFIGURATION - Edit these paths as needed
# ============================================================================

# Path to config file (YAML) - relative to projectF/
CONFIG_FILE="${CONFIG_FILE:-/idia/projects/roadtoska/projectF/DEPENDENCIES/config.yaml}"

# Path to requirements file - relative to projectF/
export REQUIREMENTS="${REQUIREMENTS:-/idia/projects/roadtoska/projectF/DEPENDENCIES/requirements.txt}"

# Path to Singularity container
export CONTAINER="${CONTAINER:-/idia/projects/roadtoska/projectF/pytorch_projectF.sif}"

# Directory containing SBATCH scripts
SBATCH_DIR="${SBATCH_DIR:-/idia/projects/roadtoska/projectF/SBATCH_SCRIPTS}"

# Individual submission scripts
PREPARE_SCRIPT="slurm_PREPARE.sh"
TRAIN_SCRIPT="slurm_TRAIN.sh"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo "Configuration:"
echo "  Container:      $CONTAINER"
echo "  Config file:    $CONFIG_FILE"
echo "  Requirements:   $REQUIREMENTS"
echo "  SBATCH dir:     $SBATCH_DIR"
echo ""

# Create necessary directories
#echo "Setting up directories..."

if [ ! -d "LOGS" ]; then
    echo "  Creating LOGS/ directory..."
    mkdir -p LOGS
fi

#echo "✓ Directories ready"
#echo ""

# Check if SBATCH_SCRIPTS directory exists
#if [ ! -d "$SBATCH_DIR" ]; then
#    echo "ERROR: SBATCH_SCRIPTS directory not found: $SBATCH_DIR"
#    echo "       Please run this script from the projectF/ directory"
#    exit 1
#fi

# Check if all submission scripts exist
#echo "Checking for submission scripts in $SBATCH_DIR/..."
#MISSING_SCRIPTS=()
#for script in "$PREPARE_SCRIPT" "$TRAIN_SCRIPT"; do
#    if [ ! -f "$SBATCH_DIR/$script" ]; then
#        MISSING_SCRIPTS+=("$script")
#    fi
#done

#if [ ${#MISSING_SCRIPTS[@]} -ne 0 ]; then
#    echo "ERROR: The following submission scripts are missing from $SBATCH_DIR/:"
#    for script in "${MISSING_SCRIPTS[@]}"; do
#        echo "  - $script"
#    done
#    exit 1
#fi
#echo "✓ All submission scripts found"

# Check if Python scripts exist
#echo "Checking for Python scripts in $SBATCH_DIR/..."
#MISSING_PY_SCRIPTS=()
#for script in "prepare.py" "train.py" "tune.py"; do
#    if [ ! -f "$SBATCH_DIR/$script" ]; then
#        MISSING_PY_SCRIPTS+=("$script")
#    fi
#done

#if [ ${#MISSING_PY_SCRIPTS[@]} -ne 0 ]; then
#    echo "ERROR: The following Python scripts are missing from $SBATCH_DIR/:"
#    for script in "${MISSING_PY_SCRIPTS[@]}"; do
#        echo "  - $script"
#    done
#    echo ""
#    echo "NOTE: Make sure configs.py is copied to SBATCH_SCRIPTS/ directory"
#    exit 1
#fi
#echo "✓ All Python scripts found"

# Check if container exists
#if [ ! -f "$CONTAINER" ]; then
#    echo "ERROR: Container not found: $CONTAINER"
#    exit 1
#fi
#echo "✓ Container found"

# Check if config file exists
#if [ ! -f "$CONFIG_FILE" ]; then
#    echo "ERROR: Config file not found: $CONFIG_FILE"
#    exit 1
#fi
#echo "✓ Config file found"

# Check if requirements file exists
#if [ ! -f "$REQUIREMENTS" ]; then
#    echo "ERROR: Requirements file not found: $REQUIREMENTS"
#    exit 1
#fi
#echo "✓ Requirements file found"

#echo ""
#echo "✓ All pre-flight checks passed"
#echo ""

# ============================================================================
# JOB SUBMISSION WITH DEPENDENCIES
# ============================================================================

echo "------------------------------------------------------------------------"
echo "Submitting jobs with dependency chain..."
echo "------------------------------------------------------------------------"
echo ""

# Change to SBATCH_SCRIPTS directory for job submission
cd "$SBATCH_DIR" || { echo "ERROR: Failed to change to $SBATCH_DIR"; exit 1; }

# Submit PREPARE job
echo "[1/2] Submitting PREPARE job..."
PREPARE_JOB=$(sbatch --parsable "$PREPARE_SCRIPT")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit PREPARE job"
    cd ..
    exit 1
fi

echo "      ✓ PREPARE job submitted: Job ID = $PREPARE_JOB"
echo ""

# Submit TRAIN job (depends on PREPARE)
echo "[2/2] Submitting TRAIN job (depends on PREPARE=$PREPARE_JOB)..."
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$PREPARE_JOB "$TRAIN_SCRIPT")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit TRAIN job"
    echo "      Cancelling PREPARE job $PREPARE_JOB..."
    scancel $PREPARE_JOB
    cd ..
    exit 1
fi

echo "      ✓ TRAIN job submitted: Job ID = $TRAIN_JOB"
echo ""

# Return to original directory
cd ..

# ============================================================================
# SUMMARY
# ============================================================================

echo "========================================================================"
echo "Pipeline submitted successfully!"
echo "========================================================================"
echo ""
echo "Job IDs:"
echo "  PREPARE: $PREPARE_JOB (CPU, installs dependencies)"
echo "  TRAIN:   $TRAIN_JOB (GPU, waits for PREPARE)"
echo ""
echo "Dependency chain:"
echo "  $PREPARE_JOB → $TRAIN_JOB → $TUNE_JOB"
echo ""
echo "Container:"
echo "  $CONTAINER"
echo ""
echo "Output directories:"
echo "  LOGS/    - Job logs (.out and .err files)"
echo "  RESULTS/ - Model outputs, plots, etc."
echo ""
echo "Log files:"
echo "  LOGS/prepare_${PREPARE_JOB}.out"
echo "  LOGS/train_${TRAIN_JOB}.out"
echo ""
echo "Monitor job status with:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific jobs with:"
echo "  squeue -j $PREPARE_JOB,$TRAIN_JOB"
echo ""
echo "View job output in real-time:"
echo "  tail -f LOGS/prepare_${PREPARE_JOB}.out"
echo "  tail -f LOGS/train_${TRAIN_JOB}.out"
echo ""
echo "Cancel entire pipeline:"
echo "  scancel $PREPARE_JOB $TRAIN_JOB"
echo ""
echo "========================================================================"
echo "Submission completed at: $(date)"
echo "========================================================================"
