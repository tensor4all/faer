# faer-026: the SVD returns `NoConvergence` on a 12 x 10 complex matrix

**What happens:** `Mat::thin_svd` of a 12 x 10 complex matrix returns `Err(NoConvergence)`. LAPACK's `zgesdd` and `zgesvd` decompose the same matrix.
**Which input:** a matrix whose bidiagonal QR iteration ends on a 2 x 2 block with two diagonal entries one ulp apart. The program below reads such a 12 x 10 matrix from `src/matrix.txt`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The matrix has nonzero entries of size 1e-8 to 3, plus twelve entries of size 1e-19 to 1e-16.
The decimals in `src/matrix.txt` round trip to the exact bits.
The singular values from LAPACK `zgesdd` are 3.94, 3.94, 0.4815, 0.4815, 0.1133, 0.1133, 0.0119, 0.0119, 2.9e-17 and 7.8e-18.

| | result |
|---|---|
| expected | `Ok`, singular values as above |
| `faer` 0.24.4 | **`Err(NoConvergence)`** |

The program gives the same result in debug and release builds on x86_64 Linux.
The failure depends on the exact rounding path.
A different processor or compiler may make the call succeed; it did succeed on one aarch64 macOS build.
Setting the twelve tiny entries to zero makes the call succeed.
The smaller dimension, 10, is below the default `recursion_threshold` of 128, so the SVD uses the QR algorithm directly.

## Cause

The bidiagonal QR iteration is in `qr_algorithm` (`src/linalg/svd/bidiag_svd.rs`).
At each iteration it sets an off diagonal entry to zero when `subdiag[i0].abs2() <= eps^2 |diag[i0] diag[i1]| + sml` (lines 308 to 314).
A trace of this routine on this matrix was reported with the original finding (on code that is the same as 0.24.4 for this input).
The iteration reaches a 2 x 2 block with diagonal `(0.02874731110238686, 0.028747311102386864)`, two numbers one ulp apart, and off diagonal `±2^-57`.
`|e|^2` is 1.18 times the right side of the test, so the block is never split.
A QR step on this block only flips the sign of the off diagonal entry, so the state repeats exactly.
The loop then runs until `max_iters` (line 286) and returns `SvdError::NoConvergence` (lines 352 to 359 and 425 to 432).
The code has no direct solve for a 2 x 2 block.
LAPACK's `dbdsqr` handles a remaining 2 x 2 block directly with `dlasv2`.

## How the fork fixed it

The fork diagonalizes each remaining 2 x 2 block directly instead of iterating on it.
A Givens rotation makes the block symmetric, and a Jacobi rotation diagonalizes it.
Both rotations are accumulated into `U` and `V`.
The smaller signed value is recovered from the determinant, as `(f / larger) * h`, to avoid cancellation.
The deflation test and the iteration limit are unchanged.
Run against the fork, the program returns `Ok` with singular values 3.9414676652919907, 3.9414676652919862, 0.4815088318581345, ..., 0.01191983011413392, 3.6e-17, 1.7e-17.

Fork regression tests: `linalg::svd::bidiag_svd::tests::test_issue81_qr_near_repeated_pair` (the 2 x 2 block on its own, `f32` and `f64`) and `test_issue81_complex_qr_stall` in `faer/src/linalg/svd/tests/qr_stall.rs` (this matrix, `c32` and `c64`, several scales and output choices).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#84 (commit `f83ffb0`).
* Where it was found: tensor4all/faer#81.
