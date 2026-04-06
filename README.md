# Matrix Exponential Action on Multiple GPUs

This project implements the matrix exponential action `y = exp(t·A)·v` using the Arnoldi method on one or more CUDA GPUs, leveraging **cuSPARSE** (`cusparseSpMV`), cuBLAS, and NCCL for distributed computation.

## Features

- **Distributed Arnoldi Algorithm**: Implements the Arnoldi method for computing matrix exponential actions
- **Multi-GPU Support**: Uses NCCL for inter-GPU communication and peer-to-peer memory transfers (works on a single GPU as well)
- **CSR Matrix Format**: Efficient sparse matrix storage and operations
- **SpMV via cuSPARSE**: On- and off-diagonal parts use `cusparseSpMV` (CSR) on each device
- **Ghost Element Management**: Handles off-diagonal matrix elements through ghost exchanges
- **Overlapped Communication**: CUDA streams and events to overlap SpMV/communication where applicable
- **Restart Capability**: Supports Arnoldi restarts for improved convergence

## Requirements

- CUDA 12.4 or later
- CMake 3.18 or later
- Eigen3 library (small dense `exp(t·H)` on CPU)
- NCCL library (linked even for single-GPU runs)
- One or more CUDA-capable GPUs
- Python 3 (for matrix generation tools)

## Building

Default build targets **sm_70, sm_80, sm_90** (V100, A100, H100) in one binary. For faster iteration on a single architecture:

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80   # e.g. A100 only
make -j$(nproc)
```

Full default:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Artifacts: `build/matrix_exp` (GPU), `build/matrix_exp_cpu` (CPU reference, Eigen only).

## Command-line usage (`matrix_exp`)

Run from `build/` (or pass the path to the binary).

**Positional**

- `./matrix_exp` — generated diagonal test matrix, default size **1000**, `t=1`, Arnoldi subspace size and restarts from defaults
- `./matrix_exp SIZE` — generated `SIZE×SIZE` diagonal test matrix
- `./matrix_exp MATRIX_PATH` — load `.bincsr`, `.mtx`, or `.mtx.gz`
- `./matrix_exp ... M MAX_RESTARTS` — optional 2nd/3rd positional: Arnoldi dimension `m`, max restarts (`≤0` = until convergence)

**Flags**

| Flag | Meaning |
|------|--------|
| `--m N` | Arnoldi subspace dimension (if omitted, derived from matrix size) |
| `--t T` | Time parameter `t` in `exp(t·A)` (default `1.0`; may be clamped if `\|A\|` is large) |
| `--max-restarts K` | Max Arnoldi restarts (`≤0` = until convergence) |

**Environment**

- `OUTPUT_Y_FILE` — if set, writes the full result vector `y` to that path (see `launch.sh` for cluster use)

**Default no-file run**

- `A`: generated diagonal test matrix (size 1000 unless overridden)
- `t = 1.0`
- `v`: vector of 0.1's
- Arnoldi tolerance in code: `1e-6` (see `ArnoldiParams` in `main.cu`)

### Matrix input formats

- MatrixMarket: `.mtx`, `.mtx.gz`
- Binary CSR v1: `.bincsr` (see `docs/binary_csr_v1.md`)

### Fast matrix generation (Binary CSR)

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

- `--format mmgz` — `.mtx.gz` only  
- `--format both` — `.bincsr` and `.mtx.gz`

## Cluster runs (`launch.sh`)

`launch.sh` builds a temporary **SLURM** job script and submits it with `sbatch`. It sets modules (GCC, CMake, CUDA), Eigen (optional load), **NCCL** paths, runs from the project root, and executes `./build/matrix_exp` under `build/`.

**Typical workflow**

1. Load modules on the login node (or rely on the script’s `module load` inside the batch job).
2. Ensure `build/matrix_exp` exists (the job script can invoke `build.bash` if missing).
3. Set **`LOG_ROOT`** if you want logs outside the default `logs/` (runtime logs go under `$LOG_ROOT/runtime/`).

**Examples**

```bash
# Default: 2 GPUs, 5 min wall time, generated size 1000
./launch.sh

# Named: matrix file, 4 GPUs, partition and GPU constraint (site-specific)
./launch.sh --matrix /path/to/matrix.bincsr --gpus 4 --time 00:15:00 \
  --partition gpu-ef-quick --constraint type_e --profiler none

# Arnoldi parameters forwarded to the binary
./launch.sh --matrix matrices/A.bincsr --gpus 2 --time 00:10:00 \
  --m 30 --t 1.0 --max-restarts 10 --profiler none
```

**Useful options**

| Option | Role |
|--------|------|
| `--gpus N` | SLURM `--gres=gpu:N` |
| `--time HH:MM:SS` | Wall time |
| `--partition NAME` | SLURM partition |
| `--constraint EXPR` | e.g. GPU type |
| `--profiler nsys\|ncu\|none` | Profiling (default in script is `nsys`; use `none` for timing-only runs) |
| `--m`, `--t`, `--max-restarts` | Passed through to `matrix_exp` |

**Account / paths**: the generated script includes `#SBATCH` directives (e.g. `--account=proj_1720`) and `cd` to the project directory — adjust `launch.sh` for your site if needed.

After submission, the script prints the job id and keeps a copy of the batch script as `last_sbatch.sh`.

## Architecture

### Key classes

1. **CSRHost**: Host-side CSR matrix handling, partitioning, and ghost map creation  
2. **DeviceContext**: GPU resources (streams, handles, memory, cuSPARSE descriptors)  
3. **NcclContext**: NCCL wrapper for collectives (norms, dots)  
4. **ArnoldiRunner**: High-level Arnoldi / time-stepping driver  

### Algorithm flow

1. **Initialization**: Partition matrix across GPUs, build ghost maps, initialize CUDA resources  
2. **Arnoldi iterations** (per step `j`): ghost exchange → **SpMV** (`cusparseSpMV` for on/off-diagonal parts) → modified Gram–Schmidt → normalize → Hessenberg column  
3. **Small matrix exponentiation**: `exp(t·H_m)·e1` on CPU (Eigen)  
4. **Lifting**: `y = ‖v‖ · V_m · w_H`  
5. **Convergence / restarts**: residual estimate and restart if needed  

## Implementation details

- **Memory**: Core vectors stay on device; minimal host transfers  
- **Communication**: P2P for ghosts; NCCL for global reductions  
- **Numerical stability**: FP64; optional safeguards on `t` vs `‖A‖`  
- **Errors**: CUDA/cuBLAS/cuSPARSE/NCCL checks in debug-oriented paths  

## Testing

- **Smoke test**: `./matrix_exp` with the built-in diagonal matrix  
- **Files**: run with `.mtx` / `.bincsr` from `tools/generate_matrices.py` or external suites  
- **CPU check**: build `matrix_exp_cpu` and compare against GPU output where applicable  
- **Verification**: `python verify_result.py` (see script for inputs)  
- **Scaling**: vary GPU count and matrix manifests under `experiments/` as needed  

## Future enhancements

- Symmetric structure (Lanczos)  
- Communication-reducing Krylov variants  
- Multi-node MPI  
- Richer convergence controls  
