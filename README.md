# Matrix Exponential Action on Multiple GPUs

This project implements the matrix exponential action `y = exp(t·A)·v` using the Arnoldi method on multiple CUDA GPUs, leveraging cuSPARSE, cuBLAS, and NCCL for distributed computation.

## Features

- **Distributed Arnoldi Algorithm**: Implements the Arnoldi method for computing matrix exponential actions
- **Multi-GPU Support**: Uses NCCL for inter-GPU communication and peer-to-peer memory transfers
- **CSR Matrix Format**: Efficient sparse matrix storage and operations
- **Ghost Element Management**: Handles off-diagonal matrix elements through ghost exchanges
- **Overlapped Communication**: Overlaps computation and communication using CUDA streams
- **Restart Capability**: Supports Arnoldi restarts for improved convergence

## Requirements

- CUDA 12.4 or later
- CMake 3.18 or later
- Eigen3 library
- NCCL library (for multi-GPU communication)
- Multiple CUDA-capable GPUs
- Python 3 (for matrix generation tools)

## Building

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

The program creates a test diagonal matrix and computes `exp(tA)v` where:
- `A` is a 1000×1000 diagonal matrix
- `t = 1.0` (time parameter)
- `v` is a vector of all ones

```bash
./matrix_exp
```

### Matrix Input Formats

`matrix_exp` supports:

- MatrixMarket text: `.mtx`
- Binary CSR v1: `.bincsr` (fast path for generation/loading)

Binary CSR v1 format details are documented in `docs/binary_csr_v1.md`.

### Fast Matrix Generation (Binary CSR)

Generate matrices (default output is Binary CSR):

```bash
python tools/generate_matrices.py \
  --out-dir matrices/generated \
  --sizes 1000 2000 \
  --densities 1e-3 5e-3 \
  --scales 1.0 \
  --diag-shifts 1.0 \
  --format bincsr \
  --index-dtype u32 \
  --value-dtype f64
```

Compatibility output options:

- `--format mmgz` for `.mtx.gz` only
- `--format both` for both `.bincsr` and `.mtx.gz`

## Architecture

### Key Classes

1. **CSRHost**: Host-side CSR matrix handling, partitioning, and ghost map creation
2. **DeviceContext**: GPU resource management (streams, handles, memory, descriptors)
3. **NcclContext**: NCCL communication wrapper for AllReduce operations
4. **ArnoldiRunner**: High-level orchestrator for the Arnoldi algorithm

### Algorithm Flow

1. **Initialization**: Partition matrix across GPUs, build ghost maps, initialize CUDA resources
2. **Arnoldi Iterations**: For each iteration `j`:
   - Exchange ghost elements for current vector `q_j`
   - Perform SpMV: `w = A·q_j` (on-diag + off-diag)
   - Orthogonalize using Modified Gram-Schmidt
   - Normalize to get `q_{j+1}`
   - Store Hessenberg matrix column
3. **Small Matrix Exponentiation**: Compute `exp(t·H_m)·e1` on CPU using Eigen
4. **Result Lifting**: Project result back to original space: `y = ||v||·V_m·wH`
5. **Convergence Check**: Estimate residual and restart if needed

## Implementation Details

- **Memory Management**: All computation stays on device; minimal CPU↔GPU transfers
- **Communication**: Peer-to-peer transfers for ghost elements, NCCL AllReduce for norms/dot products
- **Numerical Stability**: Double precision throughout, optional reorthogonalization
- **Error Handling**: Comprehensive CUDA/cuBLAS/cuSPARSE/NCCL error checking

## Testing

The current implementation includes a simple test with a diagonal matrix. For more comprehensive testing:

1. Load real sparse matrices from Matrix Market format
2. Compare results with CPU reference implementations
3. Verify convergence properties with different tolerance settings
4. Test scaling behavior with varying numbers of GPUs

## Future Enhancements

- Support for symmetric matrices (Lanczos method)
- Communication-avoiding techniques (s-step methods)
- Multi-node support with MPI
- More sophisticated convergence criteria
- Performance profiling and optimization

