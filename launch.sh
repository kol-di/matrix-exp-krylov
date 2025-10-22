#!/bin/bash

# Simple Matrix Exponential Launcher Script
# Usage: ./launch_simple.sh [gpus] [time] [size]

# Default values
GPUS=${1:-2}
TIME=${2:-"00:05:00"}
SIZE=${3:-1000}

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

echo "Running matrix exponential computation..."
echo "Matrix size: $SIZE"
echo ""

# Run the program
cd build
./matrix_exp

# Check exit status
if [ \$? -eq 0 ]; then
    echo ""
    echo "=== Job completed successfully ==="
else
    echo ""
    echo "=== Job failed with exit code \$? ==="
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
