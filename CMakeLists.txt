
cmake_minimum_required(VERSION 3.18)
project(matrix_exp LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Hint on where to find NCCL
set(NCCL_HINT_INC
  $ENV{NCCL_ROOT}/include
  $ENV{NCCL_REDIST}/include
  /opt/software/nvidia/hpc_sdk/v24.5/Linux_x86_64/24.5/comm_libs/12.4/nccl/include
)
set(NCCL_HINT_LIB
  $ENV{NCCL_REDIST}/lib
  $ENV{NCCL_ROOT}/lib
  /opt/software/nvidia/hpc_sdk/v24.5/Linux_x86_64/24.5/comm_libs/12.4/nccl/lib
)

# Find NCCL (assuming it's installed)
find_path(NCCL_INCLUDE_DIR nccl.h PATHS ${NCCL_HINT_INC})
find_library(NCCL_LIBRARY nccl PATHS ${NCCL_HINT_LIB})


if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL not found. Set NCCL_ROOT and NCCL_REDIST or adjust hint paths.")
endif()

add_library(nccl SHARED IMPORTED)
set_target_properties(nccl PROPERTIES
  IMPORTED_LOCATION "${NCCL_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
)

add_executable(matrix_exp main.cu)

target_link_libraries(matrix_exp
  CUDA::cudart
  CUDA::cublas
  CUDA::cusparse
  CUDA::nvToolsExt
  nccl
  Eigen3::Eigen
)

# Set CUDA properties
# V100=70, A100=80, H100=90
# For faster compilation during development, compile only for V100 (available on cn nodes)
set_property(TARGET matrix_exp PROPERTY CUDA_ARCHITECTURES "70")
set_property(TARGET matrix_exp PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Flags: -O3 for performance (explicit, in addition to Release defaults)
target_compile_options(matrix_exp PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3 -Xcompiler=-Wall -diag-suppress=20014>
)

# Set C++ standard
set_property(TARGET matrix_exp PROPERTY CUDA_STANDARD 17)
set_property(TARGET matrix_exp PROPERTY CUDA_STANDARD_REQUIRED ON)

# RPATH: чтобы не настраивать LD_LIBRARY_PATH при запуске
set_target_properties(matrix_exp PROPERTIES
  BUILD_RPATH "${NCCL_HINT_LIB}"
  INSTALL_RPATH "${NCCL_HINT_LIB}"
)

# CPU-only reference implementation (no CUDA, no NCCL)
add_executable(matrix_exp_cpu main_cpu.cpp)
target_link_libraries(matrix_exp_cpu Eigen3::Eigen)
target_compile_options(matrix_exp_cpu PRIVATE -O3)
set_property(TARGET matrix_exp_cpu PROPERTY CXX_STANDARD 17)
set_property(TARGET matrix_exp_cpu PROPERTY CXX_STANDARD_REQUIRED ON)