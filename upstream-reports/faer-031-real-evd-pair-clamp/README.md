# faer-031: the real nonsymmetric EVD returns a wrong eigenvector for a complex pair

**What happens:** `Mat::eigen` of a real matrix returns an eigenvector of a complex eigenvalue pair whose residual `|A u - lambda u|` is about 9e4 `eps` of `|A| |u|`, where LAPACK's `dgeev` gives 1e-13.
**Which input:** a real matrix with two complex pairs that are small and close compared with the rest of the matrix. The program below uses two 2 x 2 rotation blocks (eigenvalues `+-1e-5 i` and `+-2e-5 i`) coupled by entries of size `1e8`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program builds the 4 x 4 matrix shown at the top of `src/main.rs`.
It calls `Mat::eigen` and prints `|A u - lambda u|_1 / (|A|_1 |u|_1 eps)` for each eigenvector.

| eigenvalue | expected (LAPACK `dgeev`) | `faer` 0.24.4 |
|---|---|---|
| `+-1e-5 i` | about 1e-13 | 0 |
| `+-2e-5 i` | about 1e-13 | **8.97e4** |

## Cause

The eigenvectors of a real matrix come from a back substitution on its real Schur form `T`.
For a complex eigenvalue `lambda = p + i q`, each 2 x 2 diagonal block `B` of `T` above it is solved as `(B - lambda I) x = r` (a 2 x 2 complex system).
`solve_cplx_shifted_upper_quasi_triangular_system` in `src/linalg/evd/mod.rs` solves it with the inverse of the shifted block, through its determinant `z = det(B - lambda I)` (line 825).
It guards against a nearly singular block with `if z < eps * norm` (line 828).
Here `norm` is the sum of the magnitudes of all entries of `T` (lines 567 to 570), so `eps * norm` has the size of an entry.
A determinant is a product of two entries, so it is compared with a threshold of the wrong size.
In this matrix the shifted blocks have entries near `1e-5` and determinants near `1e-10`, below `eps * norm` (about `9e-8`, from the entries `1e8`).
The determinant is then replaced by `eps * norm` while the numerators of the inverse stay those of the original block, so the two entries of that row are solved from a system that does not exist, and they are wrong.
The 1 x 1 case in the same function (line 794) compares `|T[i, i] - lambda|` with `eps * norm`, which has the right size.

LAPACK's `dtrevc3` solves the same blocks with `dlaln2`.
That routine compares its threshold with the largest entry of the shifted block.
It then solves the block by Gaussian elimination with complete pivoting, and raises the second pivot to the threshold when it is smaller, so it always solves a slightly perturbed system exactly.

## How the fork fixed it

The fork solves each shifted 2 x 2 block the way `dlaln2` does.
A block whose entries are all below the threshold is taken as the threshold times the identity.
Any other block is solved by elimination with complete pivoting, with the second pivot raised to the threshold when it is smaller.
The 2 x 2 blocks above a real eigenvalue use the same solve with a zero imaginary part.
With that solve in place of lines 825 to 843 of the 0.24.4 source, this program prints residual ratios of 7e-14 for all four eigenvectors.

Fork regression tests: `linalg::evd::lapack::{f32,f64}::lapack_nonsymmetric_evd_graded_pairs_keep_their_residual` (this matrix, and strongly graded matrices `D A D^-1` with `D = diag(10^linspace(-6, 6, n))`, left and right eigenvectors) and `lapack_nonsymmetric_evd_pair_solve_with_a_small_determinant`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: the pull request that closes tensor4all/faer#145.
* Where it was found: tensor4all/faer#145, strongly graded matrices in the fork's comparison with LAPACK. In the fork, a Schur form is scaled to entries near one before this solve, which made every 2 x 2 block of such a matrix hit the guard.
