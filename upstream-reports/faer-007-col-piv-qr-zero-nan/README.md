# faer-007: column pivoted QR of a zero matrix returns NaN

**What happens:** `Mat::col_piv_qr` of a zero matrix returns `R` and `Q R` full of NaN, and reports no error.
**Which input:** any nonempty matrix whose entries are all zero. The program below uses a 3 x 2 zero matrix; `[[0]]` behaves the same.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program factors a 3 x 2 zero matrix with column pivoted QR and rebuilds it with `reconstruct`.
A zero matrix has the exact factorization `Q = I`, `R = 0`, so every entry must come back as 0.

| | `R[0,0]` | `(Q R)[0,0]` |
|---|---|---|
| expected | 0 | 0 |
| `faer` 0.24.4 | **NaN** | **NaN** |

## Cause

Before the factorization, the matrix is scaled so that its largest column norm is one.
The scale is `best`, the largest column norm (`src/linalg/qr/col_pivoting/factor.rs:144-150`).
The inverse scale is `scale_bwd = best.recip()` (line 152), and every entry is multiplied by it (line 153).
For a zero matrix `best` is 0, so `scale_bwd` is infinity.
Each entry becomes `0 * inf`, which is NaN.
The NaN then spreads into `R`, the Householder factors and the reconstruction.

## How the fork fixed it

The fork uses the scale only when it is positive: `scale_fwd = if best > 0 { best } else { 1 }`, and `scale_bwd = 1 / scale_fwd`.
A zero matrix is then left unscaled, and each zero column gets the identity reflector, as LAPACK's `dlarfg` does.
Run against the fork, the program prints 0 for both entries.

Fork regression test: `linalg::qr::col_pivoting::factor::tests::zero_matrix_reconstructs` (shapes 1 x 1 to 64 x 64, `f32`, `c32`, `f64`, `c64`, sequential and parallel).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#46 (commit `2c5d45a`).
* Where it was found: tensor4all/faer#35.
