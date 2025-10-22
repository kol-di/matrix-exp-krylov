#!/usr/bin/bash

set -euo pipefail

# Clean module environment and load a compatible toolchain
module purge
module load EasyBuild/modules
module load GCC/12.3.0 CMake/3.26.3-GCCcore-12.3.0 CUDA/12.2 Eigen/3.4.0-GCCcore-12.3.0

# NCCL from NVIDIA HPC SDK 24.5 (CUDA 12.4 comm_libs)
export NCCL_ROOT=/opt/software/nvidia/hpc_sdk/v24.5/Linux_x86_64/24.5/comm_libs/12.4/nccl
export NCCL_REDIST=/opt/software/nvidia/hpc_sdk/v24.5/Linux_x86_64/24.5/REDIST/comm_libs/12.4/nccl

# Runtime search path for NCCL if needed (CMake also sets RPATH in target)
export LD_LIBRARY_PATH="${NCCL_REDIST}/lib:${NCCL_ROOT}/lib:${LD_LIBRARY_PATH-}"

# Build directory
rm -rf build
mkdir -p build
cd build

# Allow overriding CUDA arch via env var CUDA_ARCH; default to 80 (A100)
CUDA_ARCH_VAL=${CUDA_ARCH:-"70;80;90"}

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_VAL} \
  -DNCCL_ROOT=$NCCL_ROOT \
  -DNCCL_REDIST=$NCCL_REDIST

cmake --build . -j
