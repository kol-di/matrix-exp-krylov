#!/usr/bin/env python3
"""
Generate synthetic sparse matrices in MatrixMarket format (.mtx.gz).

- Cartesian product over provided lists: sizes × densities × scales (nothing extra).
- Optional symmetry flag; otherwise matrices are general.
- Adds diagonal shift for stability.
- Uses NumPy for faster sampling/coalescing.
"""

import argparse
import gzip
import os
import struct
import time
from itertools import product
from typing import Tuple

import numpy as np


def coalesce(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((cols, rows))
    rows_sorted = rows[order]
    cols_sorted = cols[order]
    vals_sorted = vals[order]

    diffs = np.ones(len(rows_sorted), dtype=bool)
    diffs[1:] = (rows_sorted[1:] != rows_sorted[:-1]) | (cols_sorted[1:] != cols_sorted[:-1])
    idx = np.nonzero(diffs)[0]

    cumsum = np.cumsum(vals_sorted)
    sums = cumsum[idx]
    prev = np.concatenate(([0], cumsum[idx[:-1]]))
    sums -= prev
    return rows_sorted[idx], cols_sorted[idx], sums


def generate_matrix(
    n: int,
    density: float,
    scale: float,
    symmetric: bool,
    diag_shift: float,
    seed: int,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nnz_target = max(1, int(density * n * n))

    rows = rng.integers(0, n, size=nnz_target, dtype=np.int64)
    cols = rng.integers(0, n, size=nnz_target, dtype=np.int64)
    vals = rng.uniform(-scale, scale, size=nnz_target)

    if symmetric:
        mask_lower = rows > cols
        tmp = rows[mask_lower].copy()
        rows[mask_lower] = cols[mask_lower]
        cols[mask_lower] = tmp

    rows, cols, vals = coalesce(rows, cols, vals)

    # Row sums for diagonal dominance
    if symmetric:
        offmask = rows != cols
        off_rows = rows[offmask]
        off_cols = cols[offmask]
        off_vals = np.abs(vals[offmask])
        # Contributions go to both incident rows for symmetric storage
        row_sum = np.bincount(off_rows, weights=off_vals, minlength=n) + np.bincount(off_cols, weights=off_vals, minlength=n)
    else:
        row_sum = np.bincount(rows, weights=np.abs(vals), minlength=n)

    diag_shift_vals = diag_shift * (row_sum + 1.0)

    # add diagonal shifts (vectorized), then coalesce once more
    diag_indices = np.nonzero(diag_shift_vals != 0.0)[0]
    if diag_indices.size > 0:
        rows = np.concatenate([rows, diag_indices])
        cols = np.concatenate([cols, diag_indices])
        vals = np.concatenate([vals, diag_shift_vals[diag_indices]])
        rows, cols, vals = coalesce(rows, cols, vals)

    return n, n, rows, cols, vals


def csr_from_coo(n: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # rows/cols are already coalesced and row-major sorted by construction.
    row_counts = np.bincount(rows, minlength=n)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    return indptr, cols.astype(np.int64, copy=False), vals.astype(np.float64, copy=False)


def _check_u32_fit(name: str, value: int):
    if value < 0 or value > np.iinfo(np.uint32).max:
        raise ValueError(f"{name}={value} does not fit into u32")


def write_binary_csr(
    path: str,
    n: int,
    m: int,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    symmetric_upper: bool,
    index_dtype: str,
    value_dtype: str,
):
    indptr64, indices64, data64 = csr_from_coo(n, rows, cols, vals)
    nnz = int(indices64.size)

    if index_dtype == "u32":
        idx_np_dtype = np.uint32
        idx_code = 1
        _check_u32_fit("nrows", n)
        _check_u32_fit("ncols", m)
        _check_u32_fit("nnz", nnz)
        if nnz > 0:
            _check_u32_fit("max_col_index", int(indices64.max()))
        _check_u32_fit("max_indptr", int(indptr64[-1]))
    elif index_dtype == "u64":
        idx_np_dtype = np.uint64
        idx_code = 2
    else:
        raise ValueError(f"Unsupported index dtype: {index_dtype}")

    if value_dtype == "f32":
        val_np_dtype = np.float32
        val_code = 1
    elif value_dtype == "f64":
        val_np_dtype = np.float64
        val_code = 2
    else:
        raise ValueError(f"Unsupported value dtype: {value_dtype}")

    indptr = indptr64.astype(idx_np_dtype, copy=False)
    indices = indices64.astype(idx_np_dtype, copy=False)
    data = data64.astype(val_np_dtype, copy=False)

    flags = 0
    flags |= 1 << 0  # indices_sorted
    flags |= 1 << 1  # no_duplicates
    if symmetric_upper:
        flags |= 1 << 2

    magic = b"CSR\x00\x00\x00\x00\x01"
    header = struct.pack(
        "<8sIIQQQIIQQ",
        magic,
        1,              # version
        flags,
        int(n),
        int(m),
        int(nnz),
        idx_code,
        val_code,
        0,              # reserved0
        0,              # reserved1
    )

    with open(path, "wb") as f:
        f.write(header)
        indptr.tofile(f)
        indices.tofile(f)
        data.tofile(f)


def write_mm_gz(path: str, n: int, m: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, symmetric: bool):
    header_sym = "symmetric" if symmetric else "general"
    # Faster bulk write via numpy; use low compression for speed
    with gzip.open(path, "wb", compresslevel=1) as gz:
        from io import TextIOWrapper
        f = TextIOWrapper(gz, encoding="utf-8", newline="\n")
        f.write(f"%%MatrixMarket matrix coordinate real {header_sym}\n")
        f.write(f"{n} {m} {len(rows)}\n")
        data = np.column_stack((rows + 1, cols + 1, vals))
        np.savetxt(f, data, fmt=["%d", "%d", "%.16e"], delimiter=" ")
        f.flush()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sparse matrices (.mtx.gz).")
    parser.add_argument("--out-dir", default="matrices/generated", help="Output directory")
    parser.add_argument("--sizes", nargs="+", type=int, required=True, help="Matrix sizes (list)")
    parser.add_argument("--densities", nargs="+", type=float, required=True, help="Densities (fraction of nnz)")
    parser.add_argument("--scales", nargs="+", type=float, required=True, help="Value scales (uniform in [-scale, scale])")
    parser.add_argument("--symmetric", action="store_true", help="Store as symmetric MatrixMarket (upper triangle)")
    parser.add_argument("--diag-shifts", nargs="+", type=float, default=[1.0], help="Diagonal dominance factors (>=0)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--format",
        choices=["bincsr", "mmgz", "both"],
        default="bincsr",
        help="Output format: binary CSR (fast), MatrixMarket gzip, or both",
    )
    parser.add_argument(
        "--index-dtype",
        choices=["u32", "u64"],
        default="u32",
        help="Index dtype for Binary CSR output",
    )
    parser.add_argument(
        "--value-dtype",
        choices=["f32", "f64"],
        default="f64",
        help="Value dtype for Binary CSR output",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    combos = list(product(args.sizes, args.densities, args.scales, args.diag_shifts))
    t_all_start = time.perf_counter()
    for idx, (n, dens, scl, dshift) in enumerate(combos):
        name = f"synthetic_n{n}_d{dens:.1e}_s{scl:.2e}_diag{dshift:.2f}"
        print(f"Generating {name} ...")
        t_case_start = time.perf_counter()
        nrows, ncols, rows, cols, vals = generate_matrix(
            n=n,
            density=dens,
            scale=scl,
            symmetric=args.symmetric,
            diag_shift=dshift,
            seed=args.seed + idx,
        )
        t_after_generate = time.perf_counter()
        if args.format in ("bincsr", "both"):
            bin_path = os.path.join(args.out_dir, f"{name}.bincsr")
            write_binary_csr(
                path=bin_path,
                n=nrows,
                m=ncols,
                rows=rows,
                cols=cols,
                vals=vals,
                symmetric_upper=args.symmetric,
                index_dtype=args.index_dtype,
                value_dtype=args.value_dtype,
            )
            print(f"  wrote {bin_path}")
        if args.format in ("mmgz", "both"):
            mm_path = os.path.join(args.out_dir, f"{name}.mtx.gz")
            write_mm_gz(mm_path, nrows, ncols, rows, cols, vals, args.symmetric)
            print(f"  wrote {mm_path}")
        print(f"  nnz = {len(rows)}")
        t_case_end = time.perf_counter()
        print(f"  timing: generate={t_after_generate - t_case_start:.4f}s total_case={t_case_end - t_case_start:.4f}s")
    t_all_end = time.perf_counter()
    print(f"Done. Generated {len(combos)} matrix set(s) in {t_all_end - t_all_start:.4f}s")


if __name__ == "__main__":
    main()


