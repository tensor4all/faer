# Upstream reports

This list is updated over time. Last updated: 2026-10-29.

This directory lists defects that we found while working on a port of `faer` and that also occur in the published upstream crates.
Each entry was reproduced by running its program against the upstream release named in the table.
faer-028 was first reported upstream (sarah-quinones/faer#339, simplicial backend only). It is listed because the supernodal backend has the same defect.
Defects that exist only in this fork are not listed here.

Each subdirectory is a standalone Cargo project.
It depends only on the upstream crate from crates.io, pinned to the version in the table.
Run it with `cargo run` inside the subdirectory.
The program prints the expected and the actual value, and exits with status 1 when the defect shows.

| # | Title | Upstream crate, version confirmed | Kind of symptom | Fix in this fork | Fixed upstream | Added |
|---|---|---|---|---|---|---|
| [faer-001](faer-001-full-piv-lu-lane-tie/) | Full pivoting LU picks a different pivot depending on the SIMD width | `faer` 0.24.4 | result depends on the CPU | tensor4all/faer#105 (`efaa5de`) | not yet | 2026-10-29 |
| [faer-002](faer-002-partial-svd-residual/) | Partial SVD reports a pair as converged when it is not | `faer` 0.24.4 | wrong value | tensor4all/faer#108 (`1ac38c6`) | not yet | 2026-10-29 |
| [faer-003](faer-003-evd-clamp-grows-with-n/) | The eigenvector residual of the nonsymmetric EVD grows with n | `faer` 0.24.4 | inaccurate value | tensor4all/faer#121 (`0c806f1`) | not yet | 2026-10-29 |
| [faer-004](faer-004-svd-dc-secular-stagnation/) | The divide and conquer SVD returns wrong factors for a 128 x 128 bidiagonal matrix | `faer` 0.24.4 | wrong value | tensor4all/faer#29 (`65dd42f`) | not yet | 2026-10-29 |
| [faer-005](faer-005-svd-values-only-zero-subproblem/) | The values only SVD loses a nonzero singular value | `faer` 0.24.4 | wrong value | tensor4all/faer#39 (`ccfaa56`) | not yet | 2026-10-29 |
| [faer-006](faer-006-lblt-reconstruct-diag/) | `Lblt::reconstruct` ignores the block diagonal | `faer` 0.24.4 | wrong value | tensor4all/faer#46 (`4e2c241`) | not yet | 2026-10-29 |
| [faer-007](faer-007-col-piv-qr-zero-nan/) | Column pivoted QR of a zero matrix returns NaN | `faer` 0.24.4 | NaN | tensor4all/faer#46 (`2c5d45a`) | not yet | 2026-10-29 |
| [faer-008](faer-008-norm-l2-strided-range/) | `norm_l2` of a strided vector overflows or underflows | `faer` 0.24.4 | overflow or underflow | tensor4all/faer#46 (`4e2c241`) | not yet | 2026-10-29 |
| [faer-009](faer-009-gevd-reused-output-buffer/) | Generalized eigenvectors depend on the old contents of the output buffer | `faer` 0.24.4 | wrong value | tensor4all/faer#46 (`4e2c241`) | not yet | 2026-10-29 |
| [faer-010](faer-010-partial-piv-lu-zero-column-nan/) | Partial pivoted LU returns NaN factors when a pivot column is zero | `faer` 0.24.4 | NaN | tensor4all/faer#46 (`2c5d45a`) | not yet | 2026-10-29 |
| [faer-011](faer-011-svd-lstsq-zero-singular-value-nan/) | The SVD least squares solve returns NaN for a rank deficient matrix | `faer` 0.24.4 | NaN | tensor4all/faer#47 (`ee70868`) | not yet | 2026-10-29 |
| [faer-012](faer-012-svd-full-u-stale-buffer/) | The full U of a tall SVD keeps old entries of the output buffer | `faer` 0.24.4 | wrong value | tensor4all/faer#47 (`ee70868`) | not yet | 2026-10-29 |
| [faer-013](faer-013-evd-2x2-overflow/) | 2 x 2 self adjoint eigenvalues overflow for a finite spectrum | `faer` 0.24.4 | overflow | tensor4all/faer#47 (`946ac9f`) | not yet | 2026-10-29 |
| [faer-014](faer-014-trsv-subnormal-pivot/) | Triangular solve returns inf or NaN for a subnormal pivot | `faer` 0.24.4 | inf or NaN | tensor4all/faer#47 (`946ac9f`), tensor4all/faer#83 (`017d504`) | not yet | 2026-10-29 |
| [faer-015](faer-015-evd-empty-scratch/) | Self adjoint EVD of a 0 x 0 matrix panics in a debug build | `faer` 0.24.4 | panic | tensor4all/faer#47 (`946ac9f`) | not yet | 2026-10-29 |
| [faer-016](faer-016-pivoted-llt-rank-exit/) | Pivoted Cholesky leaves input entries in the factor after a rank stop | `faer` 0.24.4 | wrong value | tensor4all/faer#47 (`946ac9f`) | not yet | 2026-10-29 |
| [faer-017](faer-017-tridiag-evd-range/) | Tridiagonal EVD fails for finite spectra near the ends of the `f64` range | `faer` 0.24.4 | overflow or no convergence | tensor4all/faer#62 (`f59395a`) | not yet | 2026-10-29 |
| [faer-018](faer-018-lblt-stale-subdiag/) | LBL^T factorization leaves old values in the caller's `subdiag` buffer | `faer` 0.24.4 | wrong value | tensor4all/faer#67 (`ff1df37`) | not yet | 2026-10-29 |
| [faer-019](faer-019-pivoted-llt-zero-matrix/) | Pivoted Cholesky of the zero matrix does not report rank 0 | `faer` 0.24.4 | wrong rank or error | tensor4all/faer#65 (`cdb56da`) | not yet | 2026-10-29 |
| [faer-020](faer-020-complex-recip-subnormal/) | Complex reciprocal of a subnormal value is `NaN - inf i` | `faer` 0.24.4 | NaN | tensor4all/faer#68 (`57a06de`) | not yet | 2026-10-29 |
| [faer-021](faer-021-tridiag-evd-empty-complex/) | Complex tridiagonal EVD of a 0 x 0 matrix panics | `faer` 0.24.4 | panic | tensor4all/faer#69 (`8a199b0`) | not yet | 2026-10-29 |
| [faer-022](faer-022-svd-scratch-too-small/) | `svd` panics inside the workspace that `svd_scratch` sized | `faer` 0.24.4 | panic | tensor4all/faer#80 (`36b787c`) | not yet | 2026-10-29 |
| [faer-023](faer-023-svd-merge-rotation-nonorthogonal-v/) | The divide and conquer SVD returns a non orthogonal V | `faer` 0.24.4 | inaccurate value | tensor4all/faer#87 (`b1ceefe`) | not yet | 2026-10-29 |
| [faer-024](faer-024-tridiag-qr-spectrum-accuracy/) | The tridiagonal QR eigenvalues are less accurate than LAPACK's test bound | `faer` 0.24.4 | inaccurate value | tensor4all/faer#87 (`2d00c64`) | not yet | 2026-10-29 |
| [faer-025](faer-025-evd-norm-sum-overflow/) | Nonsymmetric EVD returns a wrong eigenvector when the Schur norm sum overflows | `faer` 0.24.4 | wrong value | tensor4all/faer#121 (`0c806f1`) | not yet | 2026-10-29 |
| [faer-026](faer-026-svd-qr-two-by-two-no-convergence/) | The SVD returns `NoConvergence` on a 12 x 10 complex matrix | `faer` 0.24.4 | no convergence | tensor4all/faer#84 (`f83ffb0`) | not yet | 2026-10-29 |
| [faer-027](faer-027-qr-near-dependent-column-skip/) | Unpivoted QR skips a nearly dependent column of a tall matrix | `faer` 0.24.4 | inaccurate value | tensor4all/faer#1 (`2e6ea52`, `d16929b`) | not yet | 2026-10-29 |
| [faer-028](faer-028-sparse-lu-numeric-singular/) | Sparse LU factors a numerically singular matrix without an error | `faer` 0.24.4 | NaN | tensor4all/faer#132 | not yet | 2026-10-29 |
| [faer-029](faer-029-real-evd-pair-clamp/) | The real nonsymmetric EVD returns a wrong eigenvector for a complex pair | `faer` 0.24.4 | wrong value | fix for tensor4all/faer#145 | not yet | 2026-10-29 |
| [gemm-x86-001](gemm-x86-001-diag-negative-stride/) | A product with a diagonal gives a wrong value when the left matrix is stored backwards | `private-gemm-x86` 0.1.22 | wrong value | tensor4all/t4a-private-gemm-x86#5 (`84ca6db`) | not yet | 2026-10-29 |

"Fixed upstream" was checked against upstream `main`: `faer` at `0ecd846` (2026-10-22) and `private-gemm-x86` at `71872b9` (its 0.1.22 release).

Each README states, in its first three lines, what goes wrong, for which input and in which version.
It then gives the command, a table of expected and actual output, the cause with file and line in the upstream source, and how this fork fixed it.

Upstream sources:
[faer](https://codeberg.org/sarah-quinones/faer),
[private-gemm-x86](https://github.com/sarah-quinones/gemm-x64-v2).
