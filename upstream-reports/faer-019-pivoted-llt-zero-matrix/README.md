# faer-019: pivoted Cholesky of the zero matrix does not report rank 0

**What happens:** pivoted Cholesky of a zero matrix reports rank 1 for size 1, and fails with `NonPositivePivot` for larger sizes. The zero matrix is positive semidefinite with rank 0.
**Which input:** any matrix whose diagonal is all zero, for example the zero matrix.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program calls `linalg::cholesky::llt_pivoting::factor::cholesky_in_place` on the `n` x `n` zero matrix for `n` = 1 and 3.
LAPACK `dpstrf` returns rank 0 for this input.

| `n` | expected | `faer` 0.24.4 |
|---|---|---|
| 1 | `Ok`, rank 0 | **`Ok`, rank 1** |
| 3 | `Ok`, rank 0 | **`Err(NonPositivePivot { index: 1 })`** |

## Cause

In `src/linalg/cholesky/llt_pivoting/factor.rs`, the rank tolerance is `tol = eps * n * ajj`, where `ajj` is the largest diagonal entry (line 86).
For a zero diagonal, `ajj = 0`, so `tol = 0`.
The check `ajj < tol` that stops at the numerical rank runs only for `j > 0` (lines 105 and 120).
The first pivot is never compared with the tolerance.
So the first step takes `ajj = sqrt(0) = 0` (line 155) and scales the column below it by `1 / 0 = inf` (lines 167 to 169).
For `n = 1` there is no column below, and the loop ends with rank 1.
For `n > 1` the scaled column holds `0 * inf = NaN`, and the next step reports `NonPositivePivot` because it finds a `NaN` diagonal (lines 110 to 113).

## How the fork fixed it

Before the pivot loop the fork checks the largest diagonal entry.
If it is zero, the fork sets rank 0, fills the factor with zeros, and returns, as LAPACK `dpstf2` and `dpstrf` do.
Positive and negative diagonals are unchanged.
With that change this program prints `Ok, rank 0` for both sizes.

Fork regression test: `linalg::cholesky::llt_pivoting::reconstruct::tests::test_zero_matrix_rank_zero` (`n` = 1, 2 and 5; rank and reconstruction).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#65 (commit `cdb56da`).
* Where it was found: tensor4all/faer#31, the LAPACK validation of the Cholesky factorizations.
