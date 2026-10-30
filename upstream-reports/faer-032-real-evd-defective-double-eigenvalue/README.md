# faer-032: the real nonsymmetric EVD returns a wrong eigenvector for a defective double eigenvalue

**What happens:** `Mat::eigen` returns an eigenvector that is not an eigenvector. Its residual `|A v - lambda v|` is `sqrt(2)`, and its norm is `2.25e15` instead of one.
**Which input:** `A = [[2, 1], [-1, 0]]`, a real matrix whose eigenvalue 1 is double and defective (a 2 x 2 Jordan block).
**Which version:** `faer` 0.24.4, the latest release on crates.io.
**Added to this list:** 2026-10-30.

## Reproduce

```sh
cargo run --release
```

The program computes the eigendecomposition of `A` and prints the residual and the norm of each returned eigenvector.
It exits with status 1 when a residual exceeds `1e-14`.

| eigenvalue | expected residual | `faer` 0.24.4 |
|---|---|---|
| 1 (first) | rounding level | 3.1e-16 |
| 1 (second) | rounding level | **1.41**, with `|v| = 2.25e15` |

LAPACK `dgeev` returns two nearly parallel eigenvectors with residuals at rounding level.

## Cause

The Schur iteration standardizes each 2 x 2 diagonal block with `lahqr_schur22`, faer's version of LAPACK `dlanv2` (`src/linalg/evd/schur/real_schur.rs`).
In its last branch, for nearly equal or complex eigenvalues, `dlanv2` ends with two cases.
If `B` and `C` have the same sign, the eigenvalues are real and the block is made upper triangular.
If `B = 0` and `C != 0`, the rows and columns are swapped (`B = -C`, `C = 0`).
faer has the first case but not the second.
So a rotation that leaves `[[1, 0], [c, 1]]` returns that lower triangular block as it is.
That block is not in standard form: a nonzero subdiagonal entry marks a conjugate pair.
For `A` above the block is `[[1, 0], [2, 1]]`.
The eigenvector back substitution then treats the block as a complex pair with imaginary part `sqrt(|b| |c|) = 0` and returns a wrong vector.

## How the fork fixed it

The fork adds `dlanv2`'s `B = 0` case: `b = -c`, `c = 0`, and the rotation `(cs, sn)` becomes `(-sn, cs)`.
With that change the fork passes GET22 (threshold 20) on this matrix and three others with a defective double eigenvalue.

Fork regression tests:

* `lahqr_schur22_standardizes_a_lower_triangular_result` in `faer/src/linalg/evd/schur/real_schur.rs`: every input in a grid of 13^4 2 x 2 blocks returns a standardized block, and the rotation reproduces the input.
* `geev_scaling_tests::defective_double_eigenvalue_has_small_residuals` in `faer/src/linalg/evd/mod.rs`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.

## References

* Where it was found: LAPACK's `dget34` adjacent-swap records, added to the fork's `schur_reorder` validation family for tensor4all/faer#31. Swapping two 2 x 2 blocks with nearly equal eigenvalues produced such a block.
