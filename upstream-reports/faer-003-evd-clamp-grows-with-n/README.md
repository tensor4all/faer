# faer-003: the eigenvector residual of the nonsymmetric EVD grows with n

**What happens:** `Mat::eigen` returns eigenvectors whose residual `|A U - U S|` grows in proportion to `n`, instead of staying a few `eps` of `|A| |U|`.
**Which input:** a matrix with a defective or tightly clustered spectrum, for example a Jordan block (ones on the diagonal and on the subdiagonal).
**Which version:** `faer` 0.24.4.

## Reproduce

```sh
cargo run
```

The program builds the lower Jordan block of size `n`, calls `Mat::eigen`, and prints the residual ratio of LAPACK's GET22: `|A U - U S|_1 / (|A|_1 |U|_1 eps)`.
LAPACK's CHKHS test accepts a ratio below 20.

| n | expected | `faer` 0.24.4 |
|---|---|---|
| 16 | below 20 (LAPACK `dgeev`, OpenBLAS 0.3.34: 0.50) | 15.53 |
| 63 | below 20 (LAPACK `dgeev`: 0.50) | **62.61** |
| 65 | below 20 (LAPACK `dgeev`: 0.50) | **64.38** |

The ratio is about `(2n - 1) / 2`, so it crosses 20 at `n` of about 21.

## Cause

The eigenvectors are computed from the Schur form `T` by back substitution with the shifts `t_ii - lambda`.
When a shift is smaller than a floor, it is replaced by the floor.
The floor is `eps * norm`, where `norm` is the sum of `|t_ij|` over the whole Schur form.
It is computed in `src/linalg/evd/mod.rs:567-572` (real Schur form) and `:632-637` (complex), and applied at `:701`, `:794`, `:828` and `:917`.
A sum over all entries grows with `n`: on the Jordan block it is `2n - 1`.
Each clamped row leaves a residual of the size of the floor in the eigenvector, so the residual ratio grows like `n`.
LAPACK's `?TREVC` uses a floor relative to the eigenvalue instead, `max(ulp * (|wr| + |wi|), safmin * n / ulp)`, on a matrix that `?GEEV` has first scaled into range.

## How the fork fixed it

The fork's floor is `eps * max(|lambda|, max |t_ij|)`, relative to the eigenvalue and the largest entry instead of the sum.
`|re| + |im|` is scaled by `eps` before the sum, so it does not overflow, and the floor is at least the smallest positive subnormal.
A floor this small is range safe only on entries near one.
So a Schur form whose largest entry is outside `[2^-20, 2^20]` is first scaled by a power of two, as `?GEEV` scales its input.
With that change this program prints 0.50 at every `n`.

Fork regression tests, in `faer/src/linalg/evd/`:
* `clustered_eigenvector_residuals_do_not_grow_with_n`, in `tests/lapack_nonsymmetric_evd.rs`;
* `jordan_chains_at_both_ends_of_the_range`, in `mod.rs`, for the scaled Schur forms.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.

## References

* Fork fix: tensor4all/faer#121 (commit `0c806f1`).
* Where it was found: tensor4all/faer#112, the bounded LAPACK validation's nonsymmetric EVD family.
