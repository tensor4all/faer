# faer-024: the tridiagonal QR eigenvalues are less accurate than LAPACK's test bound

**What happens:** the eigenvalues from the values only self adjoint solver are off by 166 `eps` in `f32` and 62 `eps` in `f64`. The divide and conquer solver on the same matrix is off by about 5 `eps`.
**Which input:** a symmetric matrix of moderate size with a spread out spectrum. The program uses a 127 x 127 matrix with evenly spaced eigenvalues in [-1, 1].
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program builds `A = Q diag(s) Q^T` with known eigenvalues `s_i = 1 - 2 i / 126`.
`Q` is a product of adjacent plane rotations with cosine 0.6 and sine 0.8.
It computes the eigenvalues of the same `A` in two ways.

1. `self_adjoint_eigenvalues`, which runs the tridiagonal QR algorithm.
2. `self_adjoint_evd` with eigenvectors and recursion threshold 8, which runs divide and conquer.

It prints `max_i |lambda_i - s_i| / eps`.
LAPACK's CHKST test accepts a ratio below 60.

| | QR (values only) | divide and conquer (control) |
|---|---|---|
| expected | below 60 | below 60 |
| `faer` 0.24.4, `f32` | **166.0** | 5.0 |
| `faer` 0.24.4, `f64` | **62.0** | 4.5 |

The `f64` ratio is only just above the bound.
The gap to divide and conquer on the same input, more than ten times in both types, is the clearer sign.

## Cause

The QR sweep in `qr_algorithm` (`src/linalg/evd/tridiag_evd.rs:147-167`) applies each Givens rotation as an explicit 2 x 2 similarity.
It recomputes the diagonal entry from scratch: `diag[k] = c (c d_k - s e_k) - s (c e_k - s d_(k+1))` (lines 151 to 152).
Each such update rounds a quantity of the size of `|d|`.
A full solve updates each diagonal entry `O(n)` times, so the rounding errors add up.
LAPACK's `dsteqr` instead carries the correction from the previous rotation (its `p` and `g` recurrence).
It updates a diagonal entry as `g + p`, and only rounds a small correction at each step.

## How the fork fixed it

The fork runs the QR inner loop of `dsteqr`, with its `p` and `g` recurrence and its shift formula.
Block splitting, the deflation test, the ordering and the eigenvector updates are unchanged.
With that change this program prints 9.0 for `f32` and 14.5 for `f64` on the QR path.

Fork regression test: `lapack_self_adjoint_evd_qr_spectrum_accuracy`, in `faer/src/linalg/evd/tests/lapack_self_adjoint_evd.rs` (thresholds 8 and 128, with and without eigenvectors, all four scalar types, bound 60).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#87 (merge commit `ba77be4`, fix commit `2d00c64`).
* Where it was found: tensor4all/faer#76.
