#!/bin/bash

# Simple Matrix Exponential Launcher Script
# Usage: ./launch.sh [gpus] [time] [size]
#   or:  ./launch.sh --size SIZE [--gpus GPUS] [--time TIME]
#
# Examples:
#   ./launch.sh                    # defaults: 2 GPU, 5 min, size 1000
#   ./launch.sh 4 00:10:00 2000    # 4 GPU, 10 min, size 2000
#   ./launch.sh --size 500         # size 500, other params default
#   ./launch.sh --size 2000 --gpus 4 --time 00:15:00

# Default values
GPUS=2
TIME="00:05:00"
SIZE=1000

# Parse arguments
# If first argument starts with '--', use named arguments
if [[ "$1" == --* ]]; then
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
            *)
                echo "Unknown option: $1"
                echo "Usage: ./launch.sh [gpus] [time] [size]"
                echo "   or: ./launch.sh --size SIZE [--gpus GPUS] [--time TIME]"
                exit 1
                ;;
        esac
    done
else
    # Positional arguments (old style)
    GPUS=${1:-2}
    TIME=${2:-"00:05:00"}
    SIZE=${3:-1000}
fi

echo "=== Simple Matrix Exponential Job Launcher ==="
echo "Configuration:"
echo "  GPUs: $GPUS"
echo "  Time limit: $TIME"
echo "  Matrix size: $SIZE"
echo ""

# Create temporary SLURM script
TEMP_SLURM=$(mktemp)
cat > "$TEMP_SLURM" << EOF
#!/bin/bash
#SBATCH --job-name=matrix_exp_${SIZE}_${GPUS}gpu
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
echo "Matrix size: $SIZE"
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys not found. Make sure CUDA module is loaded."
    exit 1
fi

echo "Nsight Systems version:"
nsys --version
echo ""

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
    ./matrix_exp $SIZE

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
