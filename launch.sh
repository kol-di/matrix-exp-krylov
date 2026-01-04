#!/bin/bash

# Simple Matrix Exponential Launcher Script
# Usage: ./launch.sh [gpus] [time] [size|matrix_file]
#   or:  ./launch.sh --size SIZE [--gpus GPUS] [--time TIME]
#   or:  ./launch.sh --matrix FILE [--gpus GPUS] [--time TIME]
#
# Examples:
#   ./launch.sh                    # defaults: 2 GPU, 5 min, size 1000
#   ./launch.sh 4 00:10:00 2000    # 4 GPU, 10 min, size 2000 (positional)
#   ./launch.sh matrices/A.mtx     # load matrix file (defaults for gpus/time)
#   ./launch.sh --matrix matrices/A.mtx --gpus 4 --time 00:15:00
#   ./launch.sh --size 500         # size 500, other params default
#   ./launch.sh --size 2000 --gpus 4 --time 00:15:00

# Default values
GPUS=2
TIME="00:05:00"
SIZE=1000
MATRIX_FILE=""

# Parse arguments
# If first argument starts with '--', use named arguments
if [[ $# -gt 0 && "$1" == --* ]]; then
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpus)
                GPUS="$2"
                shift 2
                ;;
            --time)
                TIME="$2"
                shift 2
                ;;
            --size)
                SIZE="$2"
                shift 2
                ;;
            --matrix)
                MATRIX_FILE="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: ./launch.sh [gpus] [time] [size|matrix_file]"
                echo "   or: ./launch.sh --size SIZE [--gpus GPUS] [--time TIME]"
                echo "   or: ./launch.sh --matrix FILE [--gpus GPUS] [--time TIME]"
                exit 1
                ;;
        esac
    done
else
    # Positional arguments (old style)
    GPUS=${1:-2}
    TIME=${2:-"00:05:00"}
    THIRD=${3:-""}
    if [[ -n "$THIRD" ]]; then
        if [[ "$THIRD" == *.* || "$THIRD" == */* || "$THIRD" == *.mtx ]]; then
            MATRIX_FILE="$THIRD"
        else
            SIZE="$THIRD"
        fi
    fi
fi

# Decide what to pass to the executable
RUN_ARG="$SIZE"
JOB_TAG="size${SIZE}"
if [[ -n "$MATRIX_FILE" ]]; then
    RUN_ARG="$MATRIX_FILE"
    JOB_TAG=$(basename "$MATRIX_FILE")
    # sanitize job tag (remove slashes/spaces)
    JOB_TAG=${JOB_TAG//[^A-Za-z0-9._-]/_}
fi

echo "=== Simple Matrix Exponential Job Launcher ==="
echo "Configuration:"
echo "  GPUs: $GPUS"
echo "  Time limit: $TIME"
if [[ -n "$MATRIX_FILE" ]]; then
    echo "  Matrix file: $MATRIX_FILE"
else
    echo "  Matrix size: $SIZE"
fi
echo ""

# Create temporary SLURM script
TEMP_SLURM=$(mktemp)
cat > "$TEMP_SLURM" << EOF
#!/bin/bash
#SBATCH --job-name=matrix_exp_${JOB_TAG}_${GPUS}gpu
#SBATCH --account=proj_1720
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:$GPUS
#SBATCH --time=$TIME
#SBATCH --output=matrix_exp_%j.out
#SBATCH --error=matrix_exp_%j.err

echo "=== Matrix Exponential SLURM Job ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "GPUs allocated: \$CUDA_VISIBLE_DEVICES"
echo "Start time: \$(date)"
echo ""

# Load required modules
echo "Loading modules..."
module purge
module load EasyBuild/modules
module load GCC/12.3.0 CMake/3.26.3-GCCcore-12.3.0 CUDA/12.4 Eigen/3.4.0-GCCcore-12.3.0

# Set NCCL environment variables
export NCCL_ROOT=/opt/software/nvidia/hpc_sdk/v24.5/Linux_x86_64/24.5/comm_libs/12.4/nccl
export NCCL_REDIST=/opt/software/nvidia/hpc_sdk/v24.5/Linux_x86_64/24.5/REDIST/comm_libs/12.4/nccl
export LD_LIBRARY_PATH="\${NCCL_REDIST}/lib:\${NCCL_ROOT}/lib:\${LD_LIBRARY_PATH}"

echo "Environment setup:"
echo "  NCCL_ROOT: \$NCCL_ROOT"
echo "  NCCL_REDIST: \$NCCL_REDIST"
echo ""

# Check GPU availability
echo "GPU information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Change to project directory
cd /home/dyukolobaev/matrix_exp

# Check if executable exists
if [ ! -f "./build/matrix_exp" ]; then
    echo "ERROR: Executable not found. Building project..."
    bash build.bash
    if [ \$? -ne 0 ]; then
        echo "ERROR: Build failed!"
        exit 1
    fi
fi

echo "Running matrix exponential computation with Nsight Systems profiling..."
if [[ -n "$MATRIX_FILE" ]]; then
echo "Matrix file: $MATRIX_FILE"
else
echo "Matrix size: $SIZE"
fi
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys not found. Make sure CUDA module is loaded."
    exit 1
fi

echo "Nsight Systems version:"
nsys --version
echo ""

# Prepare matrix file if provided
ACTUAL_ARG="$RUN_ARG"
TEMP_MATRIX=""
if [[ -n "$MATRIX_FILE" ]]; then
    if [[ ! -f "$MATRIX_FILE" ]]; then
        echo "ERROR: Matrix file not found: $MATRIX_FILE"
        exit 1
    fi
    if [[ "$MATRIX_FILE" == *.gz ]]; then
        TEMP_MATRIX="/tmp/matrix_\${SLURM_JOB_ID}.mtx"
        echo "Decompressing matrix to \$TEMP_MATRIX ..."
        gzip -dc "$MATRIX_FILE" > "\$TEMP_MATRIX"
        if [[ \$? -ne 0 ]]; then
            echo "ERROR: Failed to decompress $MATRIX_FILE"
            exit 1
        fi
        ACTUAL_ARG="\$TEMP_MATRIX"
    fi
fi

# Run the program with nsys profiling
cd build
PROFILE_OUTPUT="matrix_exp_profile_\${SLURM_JOB_ID}.nsys-rep"
echo "Profiling output will be saved to: \$PROFILE_OUTPUT"
echo ""

nsys profile \\
    --output="\$PROFILE_OUTPUT" \\
    --force-overwrite=true \\
    --trace=cuda,nvtx,osrt \\
    --stats=true \\
    --cuda-memory-usage=true \\
    ./matrix_exp "\$ACTUAL_ARG"

PROFILE_EXIT_CODE=\$?

# Check exit status
if [ \$PROFILE_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Job completed successfully ==="
    if [ -f "\$PROFILE_OUTPUT" ]; then
        echo "Profile report saved: \$PROFILE_OUTPUT"
        echo "To view the report, use: nsys-ui \$PROFILE_OUTPUT"
        echo "Or generate a report: nsys stats \$PROFILE_OUTPUT"
    fi
else
    echo ""
    echo "=== Job failed with exit code \$PROFILE_EXIT_CODE ==="
fi

echo "End time: \$(date)"
EOF

# Submit the job
echo "Submitting job to SLURM..."
JOB_ID=$(sbatch "$TEMP_SLURM" | awk '{print $4}')
echo "Job submitted with ID: $JOB_ID"
echo ""

# Clean up temporary file
rm "$TEMP_SLURM"

# Show job status
echo "Job status:"
squeue -j "$JOB_ID"

echo ""
echo "To monitor the job:"
echo "  squeue -u \$USER"
echo "  tail -f matrix_exp_${JOB_ID}.out"
echo "  tail -f matrix_exp_${JOB_ID}.err"
