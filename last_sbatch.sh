#!/bin/bash
#SBATCH --job-name=matrix_exp_synthetic_n500000_d5.0e-04_s1.00e_00_diag5.00.bincsr_4gpu
#SBATCH --account=proj_1720
#SBATCH --partition=gpu-ef-quick
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=type_e
#SBATCH --time=00:03:00
#SBATCH --output=/home/dyukolobaev/matrix_exp/experiments/v8_a100/_job_logs/gpu4/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00/r05/runtime/matrix_exp_%j.out
#SBATCH --error=/home/dyukolobaev/matrix_exp/experiments/v8_a100/_job_logs/gpu4/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00/r05/runtime/matrix_exp_%j.err
#SBATCH --export=ALL

# Propagate OUTPUT_Y_FILE only when non-empty
if [[ -n "" ]]; then
  export OUTPUT_Y_FILE
else
  unset OUTPUT_Y_FILE
fi

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
module load GCC/12.3.0 CMake/3.26.3-GCCcore-12.3.0 CUDA/12.4
module --ignore-cache load Eigen/3.4.0-GCCcore-12.3.0 2>/dev/null || true

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
if [[ -n "/home/dyukolobaev/matrix_exp/artifacts/scalability_20260224_220745/matrices/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00.bincsr" ]]; then
echo "Matrix file: /home/dyukolobaev/matrix_exp/artifacts/scalability_20260224_220745/matrices/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00.bincsr"
else
echo "Matrix size: 1000"
fi
echo ""

# Prepare matrix file if provided.
# For MatrixMarket .gz, decompress to a temporary .mtx.
# For Binary CSR (.bincsr), pass through as-is.
ACTUAL_ARG="/home/dyukolobaev/matrix_exp/artifacts/scalability_20260224_220745/matrices/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00.bincsr"
TEMP_MATRIX=""
if [[ -n "/home/dyukolobaev/matrix_exp/artifacts/scalability_20260224_220745/matrices/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00.bincsr" ]]; then
    if [[ ! -f "/home/dyukolobaev/matrix_exp/artifacts/scalability_20260224_220745/matrices/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00.bincsr" ]]; then
        echo "ERROR: Matrix file not found: /home/dyukolobaev/matrix_exp/artifacts/scalability_20260224_220745/matrices/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00.bincsr"
        exit 1
    fi
    if [[ "/home/dyukolobaev/matrix_exp/artifacts/scalability_20260224_220745/matrices/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00.bincsr" == *.gz ]]; then
        TEMP_MATRIX="/tmp/matrix_${SLURM_JOB_ID}.mtx"
        echo "Decompressing matrix to $TEMP_MATRIX ..."
        gzip -dc "/home/dyukolobaev/matrix_exp/artifacts/scalability_20260224_220745/matrices/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00.bincsr" > "$TEMP_MATRIX"
        if [[ $? -ne 0 ]]; then
            echo "ERROR: Failed to decompress /home/dyukolobaev/matrix_exp/artifacts/scalability_20260224_220745/matrices/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00.bincsr"
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
if [[ -n "" ]]; then
    EXTRA_ARGS+=" --m "
fi
if [[ -n "" ]]; then
    EXTRA_ARGS+=" --t "
fi
if [[ -n "" ]]; then
    EXTRA_ARGS+=" --max-restarts "
fi

if [[ "none" == "nsys" ]]; then
    # Prefer newer Nsight Systems from NVIDIA HPC SDK, fallback to module-provided nsys.
    NSYS_BIN=""
    for cand in         /opt/software/nvidia/hpc_sdk/v24.11/Linux_x86_64/24.11/profilers/Nsight_Systems/bin/nsys         /opt/software/nvidia/hpc_sdk/v24.11/Linux_x86_64/2024/profilers/Nsight_Systems/bin/nsys         /opt/software/nvidia/hpc_sdk/v24.5/Linux_x86_64/24.5/profilers/Nsight_Systems/bin/nsys         /opt/software/nvidia/hpc_sdk/v24.5/Linux_x86_64/2024/profilers/Nsight_Systems/bin/nsys; do
        if [[ -x "$cand" ]]; then
            NSYS_BIN="$cand"
            break
        fi
    done
    if [[ -z "$NSYS_BIN" ]]; then
        NSYS_BIN="$(command -v nsys || true)"
    fi
    if [[ -z "$NSYS_BIN" ]]; then
        echo "ERROR: nsys not found. Make sure CUDA module is loaded."
        exit 1
    fi
    echo "Nsight Systems version:"
    "$NSYS_BIN" --version
    echo ""
    PROFILE_OUTPUT="/home/dyukolobaev/matrix_exp/experiments/v8_a100/_job_logs/gpu4/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00/r05/nsys_profiles/matrix_exp_profile_${SLURM_JOB_ID}"
    echo "Profiling output will be saved to: ${PROFILE_OUTPUT}.nsys-rep"
    NSYS_GPU_METRICS_ARGS=""
    if [[ "0" == "1" ]]; then
        NSYS_GPU_METRICS_ARGS="--gpu-metrics-devices=all --gpu-metrics-frequency=10000"
        if [[ -n "" ]]; then
            NSYS_GPU_METRICS_ARGS+=" --gpu-metrics-set="
        fi
        echo "Nsight GPU metrics collection is enabled for NVLink analysis."
        echo "Tip: if needed, inspect available sets with: $NSYS_BIN profile --gpu-metrics-set=help"
    fi
    # Keep trace minimal to reduce importer instability on some stacks.
    "$NSYS_BIN" profile \
        --output="$PROFILE_OUTPUT" \
        --force-overwrite=true \
        --trace=cuda,nvtx,cublas,osrt \
        --stats=true \
        --cuda-memory-usage=true \
        $NSYS_GPU_METRICS_ARGS \
        ./matrix_exp "$ACTUAL_ARG" $EXTRA_ARGS
    PROFILE_EXIT_CODE=$?
    # Treat missing report as a profiler failure even when nsys returns 0.
    if [[ $PROFILE_EXIT_CODE -eq 0 && ! -f "${PROFILE_OUTPUT}.nsys-rep" ]]; then
        echo "ERROR: nsys finished but did not produce ${PROFILE_OUTPUT}.nsys-rep"
        PROFILE_EXIT_CODE=2
    fi
elif [[ "none" == "ncu" ]]; then
    # Check ncu
    if ! command -v ncu &> /dev/null; then
        echo "ERROR: ncu not found. Make sure CUDA module is loaded."
        exit 1
    fi
    echo "Nsight Compute version:"
    ncu --version
    echo ""
    PROFILE_OUTPUT="/home/dyukolobaev/matrix_exp/experiments/v8_a100/_job_logs/gpu4/diag_shift/synthetic_n500000_d5.0e-04_s1.00e+00_diag5.00/r05/ncu_profiles/matrix_exp_profile_${SLURM_JOB_ID}"
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
    if [ -f "${PROFILE_OUTPUT}.nsys-rep" ]; then
        echo "Profile report saved: ${PROFILE_OUTPUT}.nsys-rep"
        echo "To view the report, use: nsys-ui ${PROFILE_OUTPUT}.nsys-rep"
        echo "Or generate a report: nsys stats ${PROFILE_OUTPUT}.nsys-rep"
    fi
else
    echo ""
    echo "=== Job failed with exit code $PROFILE_EXIT_CODE ==="
fi

echo "End time: $(date)"
exit $PROFILE_EXIT_CODE
