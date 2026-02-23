# Binary CSR v1 Specification

This document defines the Binary CSR v1 file format used in this repository for fast matrix generation and loading.

## Scope

- Endianness: little-endian for all integer and floating-point fields.
- Storage layout: one fixed-size header followed by raw payload arrays.
- CSR semantics: standard compressed sparse row representation.

## Fixed-width Types

- `u32`: unsigned 32-bit integer
- `u64`: unsigned 64-bit integer
- `f32`: IEEE-754 float32
- `f64`: IEEE-754 float64

## File Layout

The file is:

1. Header (64 bytes, fixed size)
2. Payload:
   - `indptr` array, length `nrows + 1`, dtype = `index_dtype`
   - `indices` array, length `nnz`, dtype = `index_dtype`
   - `data` array, length `nnz`, dtype = `value_dtype`

No padding is required between sections.

## Header (64 bytes)

| Offset | Size | Type    | Field         | Meaning |
|--------|------|---------|---------------|---------|
| 0      | 8    | char[8] | magic         | ASCII bytes `CSR\0\0\0\0\1` |
| 8      | 4    | u32     | version       | Format version, must be `1` |
| 12     | 4    | u32     | flags         | Bitfield (see below) |
| 16     | 8    | u64     | nrows         | Number of rows |
| 24     | 8    | u64     | ncols         | Number of columns |
| 32     | 8    | u64     | nnz           | Number of nonzeros |
| 40     | 4    | u32     | index_dtype   | `1 = u32`, `2 = u64` |
| 44     | 4    | u32     | value_dtype   | `1 = f32`, `2 = f64` |
| 48     | 8    | u64     | reserved0     | Must be `0` |
| 56     | 8    | u64     | reserved1     | Must be `0` |

### Magic

The magic byte sequence is exactly:

- Hex: `43 53 52 00 00 00 00 01`
- Text form: `CSR\0\0\0\0\1`

## Flags

- bit 0 (`1 << 0`): `indices_sorted` (column indices sorted ascending within each row)
- bit 1 (`1 << 1`): `no_duplicates` (no repeated column index in the same row)
- bit 2 (`1 << 2`): `symmetric_upper` (stored entries represent upper triangle including diagonal)

Unset bits are reserved and must be ignored by readers.

## CSR Invariants

Writer must enforce:

- `len(indptr) == nrows + 1`
- `indptr[0] == 0`
- `indptr` is non-decreasing
- `indptr[nrows] == nnz`
- `len(indices) == nnz`
- `len(data) == nnz`
- for all `i`: `0 <= indices[i] < ncols`

Strongly recommended and used in this repo:

- sorted indices within each row (`indices_sorted` flag set)
- no duplicates within each row (`no_duplicates` flag set)

## Symmetry Semantics

If `symmetric_upper` flag is set, payload stores upper triangle including diagonal.
Consumer behavior in this repo:

- the C++ loader expands triangular storage to full matrix during load.

If flag is not set, matrix is interpreted as already fully stored.

## Compatibility Policy

- Current version: `1`
- Reader should reject unknown major format (`version != 1`).
- Future versions may extend header semantics while preserving little-endian raw payload model.

## Minimal Validation Checklist

Reader validation should include:

- magic match
- `version == 1`
- known dtype codes
- `reserved0 == 0` and `reserved1 == 0`
- payload size matches expected bytes exactly
- CSR invariants listed above

