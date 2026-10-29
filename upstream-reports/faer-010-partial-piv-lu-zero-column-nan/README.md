# faer-010: partial pivoted LU returns NaN factors when a pivot column is zero

**What happens:** `Mat::partial_piv_lu` returns `L` and `U` with NaN entries when a pivot column is exactly zero, and reports no error.
**Which input:** a singular matrix whose pivot column is zero at some step. The program below uses `A = [[0, 1], [0, 2]]`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program factors `A = [[0, 1], [0, 2]]` and rebuilds it with `reconstruct`.
`P = I`, `L = I`, `U = A` is a valid factorization, so the reconstruction must equal `A`.

| | `L[1,0]` | reconstruction |
|---|---|---|
| expected | 0 | `[[0, 1], [0, 2]]` |
| `faer` 0.24.4 | **NaN** | **`[[0, NaN], [NaN, NaN]]`** |

## Cause

The unblocked factorization picks the largest entry of the column as the pivot.
It then forms `inv = matrix[(row, row)].recip()` (`src/linalg/lu/partial_pivoting/factor.rs:50`).
It multiplies the entries below the pivot by `inv` (lines 51 to 53).
When the whole column is zero, the pivot is 0, so `inv` is infinity.
The entries below it become `0 * inf`, which is NaN.
The rank one update of the trailing block (lines 54 to 63) then copies the NaN into `U` and into later columns.
The matrix is singular, but its LU factors exist and are finite.

## How the fork fixed it

The fork skips the scaling when the pivot is exactly zero, as OpenBLAS's `getf2_k` does.
The zero column of `L` then stays zero, and the singularity is left in `U`.
The fork also divides elementwise when the pivot is subnormal (below the smallest normal number), because the reciprocal of such a pivot overflows; reference `dgetf2` does the same.
Run against the fork, the program reconstructs `A` exactly.

Fork regression test: `linalg::lu::partial_pivoting::factor::tests::zero_pivot_column_reconstructs` (`f32`, `c32`, `f64`, `c64`).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#46 (commit `2c5d45a`).
* Where it was found: tensor4all/faer#38.
