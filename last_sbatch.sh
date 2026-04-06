#!/bin/bash
#SBATCH --job-name=matrix_exp_synthetic_n20_d2.0e-01_s1.00e_00_diag1.00.mtx.gz_1gpu
#SBATCH --account=proj_1720
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/home/dyukolobaev/matrix_exp/logs/runtime/matrix_exp_%j.out
#SBATCH --error=/home/dyukolobaev/matrix_exp/logs/runtime/matrix_exp_%j.err
#SBATCH --export=ALL

# Propagate OUTPUT_Y_FILE explicitly (if set at submission)
export OUTPUT_Y_FILE="/home/dyukolobaev/matrix_exp/logs/y_mtx.txt"

echo "=== Matrix Exponential SLURM Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Load required modules
echo "Loading modules..."
module purge
module load EasyBuild/modules
module load GCC/12.3.0 CMake/3.26.3-GCCcore-12.3.0 CUDA/12.4 Eigen/3.4.0-GCCcore-12.3.0

# Set NCCL environment variables
export NCCL_ROOT=/opt/software/nvidia/hpc_sdk/v24.5/Linux_x86_64/24.5/comm_libs/12.4/nccl
export NCCL_REDIST=/opt/software/nvidia/hpc_sdk/v24.5/Linux_x86_64/24.5/REDIST/comm_libs/12.4/nccl
export LD_LIBRARY_PATH="${NCCL_REDIST}/lib:${NCCL_ROOT}/lib:${LD_LIBRARY_PATH}"

echo "Environment setup:"
echo "  NCCL_ROOT: $NCCL_ROOT"
echo "  NCCL_REDIST: $NCCL_REDIST"
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
    if [ $? -ne 0 ]; then
        echo "ERROR: Build failed!"
        exit 1
    fi
fi

echo "Running matrix exponential computation with profiler: none"
if [[ -n "/home/dyukolobaev/matrix_exp/matrices/generated/synthetic_n20_d2.0e-01_s1.00e+00_diag1.00.mtx.gz" ]]; then
echo "Matrix file: /home/dyukolobaev/matrix_exp/matrices/generated/synthetic_n20_d2.0e-01_s1.00e+00_diag1.00.mtx.gz"
else
echo "Matrix size: 1000"
fi
echo ""

# Prepare matrix file if provided.
# For MatrixMarket .gz, decompress to a temporary .mtx.
# For Binary CSR (.bincsr), pass through as-is.
ACTUAL_ARG="/home/dyukolobaev/matrix_exp/matrices/generated/synthetic_n20_d2.0e-01_s1.00e+00_diag1.00.mtx.gz"
TEMP_MATRIX=""
if [[ -n "/home/dyukolobaev/matrix_exp/matrices/generated/synthetic_n20_d2.0e-01_s1.00e+00_diag1.00.mtx.gz" ]]; then
    if [[ ! -f "/home/dyukolobaev/matrix_exp/matrices/generated/synthetic_n20_d2.0e-01_s1.00e+00_diag1.00.mtx.gz" ]]; then
        echo "ERROR: Matrix file not found: /home/dyukolobaev/matrix_exp/matrices/generated/synthetic_n20_d2.0e-01_s1.00e+00_diag1.00.mtx.gz"
        exit 1
    fi
    if [[ "/home/dyukolobaev/matrix_exp/matrices/generated/synthetic_n20_d2.0e-01_s1.00e+00_diag1.00.mtx.gz" == *.gz ]]; then
        TEMP_MATRIX="/tmp/matrix_${SLURM_JOB_ID}.mtx"
        echo "Decompressing matrix to $TEMP_MATRIX ..."
        gzip -dc "/home/dyukolobaev/matrix_exp/matrices/generated/synthetic_n20_d2.0e-01_s1.00e+00_diag1.00.mtx.gz" > "$TEMP_MATRIX"
        if [[ $? -ne 0 ]]; then
            echo "ERROR: Failed to decompress /home/dyukolobaev/matrix_exp/matrices/generated/synthetic_n20_d2.0e-01_s1.00e+00_diag1.00.mtx.gz"
            exit 1
        fi
        ACTUAL_ARG="$TEMP_MATRIX"
    fi
fi

# Run the program with nsys profiling
cd build
PROFILE_EXIT_CODE=0

# Build extra args for Arnoldi params
EXTRA_ARGS=""
if [[ -n "8" ]]; then
    EXTRA_ARGS+=" --m 8"
fi
if [[ -n "0.1" ]]; then
    EXTRA_ARGS+=" --t 0.1"
fi
if [[ -n "1" ]]; then
    EXTRA_ARGS+=" --max-restarts 1"
fi

if [[ "none" == "nsys" ]]; then
    # Check nsys
    if ! command -v nsys &> /dev/null; then
        echo "ERROR: nsys not found. Make sure CUDA module is loaded."
        exit 1
    fi
    echo "Nsight Systems version:"
    nsys --version
    echo ""
    PROFILE_OUTPUT="/home/dyukolobaev/matrix_exp/logs/nsys_profiles/matrix_exp_profile_${SLURM_JOB_ID}"
    echo "Profiling output will be saved to: ${PROFILE_OUTPUT}.nsys-rep"
    nsys profile \
        --output="$PROFILE_OUTPUT" \
        --force-overwrite=true \
        --trace=cuda,nvtx,cublas,osrt \
        --stats=true \
        --cuda-memory-usage=true \
        ./matrix_exp "$ACTUAL_ARG" $EXTRA_ARGS
    PROFILE_EXIT_CODE=$?
elif [[ "none" == "ncu" ]]; then
    # Check ncu
    if ! command -v ncu &> /dev/null; then
        echo "ERROR: ncu not found. Make sure CUDA module is loaded."
        exit 1
    fi
    echo "Nsight Compute version:"
    ncu --version
    echo ""
    PROFILE_OUTPUT="/home/dyukolobaev/matrix_exp/logs/ncu_profiles/matrix_exp_profile_${SLURM_JOB_ID}"
    echo "Profiling output will be saved to: ${PROFILE_OUTPUT}.ncu-rep"
    # Focus on kernels inside NVTX range total_compute_expmv and collect flop-related metrics
    ncu \
        --target-processes all \
        --nvtx \
        --metrics \"sm__sass_thread_inst_executed_ops_fadd_pred_on.sum,sm__sass_thread_inst_executed_ops_ffma_pred_on.sum,sm__sass_thread_inst_executed_ops_fmul_pred_on.sum,sm__sass_thread_inst_executed_ops_dadd_pred_on.sum,sm__sass_thread_inst_executed_ops_dfma_pred_on.sum,sm__sass_thread_inst_executed_ops_dmul_pred_on.sum\" \
        --export \"$PROFILE_OUTPUT\" \
        ./matrix_exp "$ACTUAL_ARG" $EXTRA_ARGS
    PROFILE_EXIT_CODE=$?
else
    echo "Profiler disabled; running binary directly."
    ./matrix_exp "$ACTUAL_ARG" $EXTRA_ARGS
    PROFILE_EXIT_CODE=$?
fi

# Check exit status
if [ $PROFILE_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Job completed successfully ==="
    if [ -f "$PROFILE_OUTPUT" ]; then
        echo "Profile report saved: $PROFILE_OUTPUT"
        echo "To view the report, use: nsys-ui $PROFILE_OUTPUT"
        echo "Or generate a report: nsys stats $PROFILE_OUTPUT"
    fi
else
    echo ""
    echo "=== Job failed with exit code $PROFILE_EXIT_CODE ==="
fi

echo "End time: $(date)"
