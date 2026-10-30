# Defects found by the LAPACK validation

This page lists every defect that the bounded LAPACK validation found. That validation is
[docs/lapack-validation.md](lapack-validation.md), run by `scripts/run-lapack-validation.py`; tensor4all/faer#31 tracks the plan.
The page also lists the defects in the checker itself, the defects found elsewhere that the suite now carries as regression evidence, what remains, and what the suite does not cover.

Inventory date: 2026-10-30, on `main` `56e5d95` (t4a-faer 0.27.0).
Sources:

* the history of `scripts/lapack-known-failures.json`;
* the linked issues and pull requests;
* `PATCHSET.md`;
* `upstream-reports/`.

Nothing was measured for this page.

## State on `main`

The known-failures ledger is empty.
Its history has three commits:

* #116 (5c917a1) added the eight `nonsymmetric_evd` groups under #112.
* #127 (32f808b) left only `c32` `lapack_nonsymmetric_evd_cases`.
* #154 (b903fa1) removed that last entry.

A manual run of the scheduled workflow on `56e5d95` plus the #162 test fix passed every family: run [40999522695](https://github.com/tensor4all/faer/actions/runs/40999522695), ubuntu-latest, x86_64.
The family counts were common 92, svd 88, hessenberg 8, schur 12, nonsymmetric_evd 40, schur_reorder 8, balance 8, tridiagonal_evd 12, self_adjoint_evd 20, lu 16, cholesky 12, indefinite 12, qr 12, triangular 12, lstsq 12 and svd_large 1, all with exit status 0.

The "Final combined execution" section of [lapack-validation.md](lapack-validation.md) records the first full run, on 2026-10-26 on macOS arm64.
Every failure listed there is fixed; rows 1-8 below give the mapping.

## Library defects found by the validation

"First release" is the first t4a-faer version on crates.io that contains the fix.
0.24.10 was bumped in the repository but never published.
"Upstream" refers to the upstream-reports entry.
Each entry was reproduced against the upstream crate named in its table row, and none is fixed upstream (`Fixed upstream: not yet` in every row).
Nothing was reported upstream: under this repository's rules, upstream feedback is deferred.

| # | Symptom | Found by (family, LAPACK source) | Issue | Fix | First release | Regression test | Upstream |
|---|---|---|---|---|---|---|---|
| 1 | Pivoted LLT never compared the first pivot with its rank tolerance. A 1 x 1 zero matrix returned rank 1, and a larger zero matrix returned `NonPositivePivot` instead of rank 0, which is `DPSTRF`'s answer. | `cholesky`, `lapack_cholesky_cases` (CHKPS), 26 diagnostics per type | none; found during #31 | #65 | 0.25.1 | `test_zero_matrix_rank_zero` | [faer-019](../upstream-reports/faer-019-pivoted-llt-zero-matrix/) |
| 2 | LBL^T left the old value in the caller's `subdiag` buffer after a 1 x 1 pivot with no update, so a prefilled buffer produced invalid metadata. | `indefinite`, `lapack_indefinite_*` (S/DCHKSY, C/ZCHKHE; SYT01/HET01), 178 diagnostics per type | none; found during #31 | #67 | 0.25.1 | `test_factor_clears_stale_subdiag` | [faer-018](../upstream-reports/faer-018-lblt-stale-subdiag/) |
| 3 | The complex tridiagonal EVD of a 0 x 0 matrix with eigenvectors panicked on `scale[0]`. | `tridiagonal_evd` (CHKST), 36 panicking calls per complex type | none; found during #31 | #69 | 0.25.1 | `empty_tridiagonal_with_vectors` | [faer-021](../upstream-reports/faer-021-tridiag-evd-empty-complex/) |
| 4 | `svd_scratch` was too small for an 8 x 8 `f32`/`c32` rank-one SVD with V only, and `svd` panicked inside the workspace it had sized. | `svd` (8 x 8 rank-one fixture, S/C types) | #73 | #80 | 0.25.2 | `test_v_only_rank_one_exact_scratch`, `test_divide_and_conquer_exact_scratch` | [faer-022](../upstream-reports/faer-022-svd-scratch-too-small/) |
| 5 | A triangular solve with a subnormal pivot returned inf or NaN where the exact solution is finite. The minimal input is a 1 x 1 pivot `i * MIN_POSITIVE / 2` with the same right-hand side, whose solution is 1. Real `n >= 2` was affected too. | `triangular` (CHKTR, TRT02), 48 diagnostics per complex type | #74 | #83 | 0.25.2 | `subnormal_complex_diagonal_solve`, `subnormal_real_diagonal_solve`, `subnormal_coupled_solve` | [faer-014](../upstream-reports/faer-014-trsv-subnormal-pivot/) (with #43), [faer-020](../upstream-reports/faer-020-complex-recip-subnormal/) |
| 6 | The `f32` SVD returned a non-orthogonal V, with Gram error about 2e-3, on the 127 x 127 logarithmic lower-bidiagonal fixture. The cause was the merge-boundary rotation of the divide and conquer. | `svd`, `lapack_svd_f32::lapack::bidiagonal_family` (DCHKBD type 16) | #75 | #87 | 0.25.2 | `bidiagonal_merge_boundary_rotation_is_orthogonal` | [faer-023](../upstream-reports/faer-023-svd-merge-rotation-nonorthogonal-v/) |
| 7 | The values-only self-adjoint EVD, through the tridiagonal QR, missed CHKST's spectrum and mode-agreement threshold of 60 for `f64`/`f32`/`c32`. For example, `f64` n = 127, kind 2 gave 71.5. The cause was the diagonal update of the QR sweep. | `self_adjoint_evd`, `lapack_self_adjoint_evd_cases` (CHKST, SYT21/HET21) | #76 | #87 | 0.25.2 | `lapack_self_adjoint_evd_qr_spectrum_accuracy` | [faer-024](../upstream-reports/faer-024-tridiag-qr-spectrum-accuracy/) |
| 8 | Schur eigenvector extraction reported success with wrong vectors when its internal norm sum overflowed. | the nonsymmetric eigenvector validation (CHKHS, GET22), extended to the range limit | #77 | #87, superseded by #121 | 0.25.2 | `range_limit_vector_tests::*` | [faer-025](../upstream-reports/faer-025-evd-norm-sum-overflow/) |
| 9 | DCHKBD type 16 (entries `eps^U(-2, 2)`) returned `NoConvergence` at n = 127-129 in all four dtypes. dqds's running pivot underflowed and every retry was rejected. | `svd`, `dchkbd_15_to_19` and `bidiagonal_family` | #110 | #114 | 0.26.0 | `dchkbd_type_16_values_only_finishes` | not upstream: the dqds route is the fork's own (#52) |
| 10 | The nonsymmetric EVD's eigenvector residual grew with n: GET22 ratio `(2n - 1) / 2` on Jordan and tightly clustered fixtures, against LAPACK's 0.5. The back-substitution clamp was `eps * sum |t_ij|`. | `nonsymmetric_evd` (CHKHS, GET22) | #112 | #121 | 0.26.0 | `lapack_nonsymmetric_evd_clustered_residuals_do_not_grow_with_n` | [faer-003](../upstream-reports/faer-003-evd-clamp-grows-with-n/) |
| 11 | The nonsymmetric EVD did not scale its input into range as `?GEEV` does. For `c32` kinds 6/7 at n = 63-65 the ratio was 41.7-59.4. At scales `2^-100` (32-bit) or `2^-960` (64-bit) and beyond it was `1e4` to `2e15`, where LAPACK gives 2-70. | `nonsymmetric_evd` `c32`, then the #112 comparison sweep | #112 | #154 | 0.26.0 | `geev_scaling_tests::evd_scales_its_input_into_range` | [faer-029](../upstream-reports/faer-029-evd-input-not-scaled/) |
| 12 | The real Schur QR never returned when the deflation-window sort could not swap a pair (a hang). | #112 comparison sweep (4 matrix families, n = 16-257, scales to the range limits) | #144 | #154 | 0.26.0 | `aed_sort_terminates_when_a_swap_fails` | [faer-030](../upstream-reports/faer-030-real-schur-aed-sort-hang/) |
| 13 | Real eigenvectors of strongly graded matrices `D A D^-1` had GET22 residuals up to 2.6e9, where `dgeev` is below 5e-4. The cause was the 2 x 2 block solves, which now follow `?LALN2`. | #112 comparison sweep | #145 | #154 | 0.26.0 | `lapack_nonsymmetric_evd_graded_pairs_keep_their_residual`, `lapack_nonsymmetric_evd_pair_solve_with_a_small_determinant`, `lapack_nonsymmetric_evd_real_shift_block_below_the_square_root_of_min_positive` | [faer-031](../upstream-reports/faer-031-real-evd-pair-clamp/) |
| 14 | The unpivoted LDL at n >= 65, in the two layouts that reverse rows and columns, had reconstruction, solve and inverse ratios of 1e9 to 1e15, positive definite input included. The x86 kernel reversed a negative depth stride without reversing the diagonal `D`. | `indefinite`, `lapack_indefinite_*`, layouts 2 and 3 | #113 | #117 (workaround), then #122 (`t4a-private-gemm-x86` 0.1.27, tensor4all/t4a-private-gemm-x86#5) | 0.26.0 | `a_diagonal_product_reads_the_depth_in_any_direction` | [gemm-x86-001](../upstream-reports/gemm-x86-001-diag-negative-stride/) (`private-gemm-x86` 0.1.22; upstream faer 0.24.4's unpivoted LDL is affected) |
| 15 | A rejected real Schur swap changed T and Q: the 2 x 2 blocks were standardized after the kernel had rejected. LAPACK `DLAEXC` returns with T and Q untouched. | `schur_reorder`, LAPACK `dget34` records (2 x 2 / 2 x 2 section): 148 rejected swaps per real type | none; this inventory's follow-up for #31 | this inventory's follow-up PR | unreleased | `lapack_schur_reorder_checkers` (one rejected `dget34` record bit for bit), `lapack_schur_reorder_cases` (all 42,258 `dget34` records per real type) | not reported: the upstream code does the same, but it is internal and reports every swap as a success |
| 16 | `lahqr_schur22` lacked `dlanv2`'s `B = 0` case, so a lower triangular 2 x 2 block (`[[x, 0], [c, x]]`) stayed in the Schur form. The real EVD of `[[2, 1], [-1, 0]]` returned an eigenvector with residual `sqrt(2)`. | `schur_reorder`, `dget34`'s standardization check on accepted 2 x 2 swaps | none; this inventory's follow-up for #31 | this inventory's follow-up PR | unreleased | `lahqr_schur22_standardizes_a_lower_triangular_result`, `geev_scaling_tests::defective_double_eigenvalue_has_small_residuals` | [faer-032](../upstream-reports/faer-032-real-evd-defective-double-eigenvalue/) |
| 17 | The generic `spicy_matmul` path (every target but x86_64 with AVX2/FMA, and non-native types) wrote nothing for a `Rectangular` block with row or column indices, and wrote the strictly upper columns of a lower block with more columns than rows. | the three strict-triangle matmul tests, failing on the weekly aarch64 lane (both hosts) as on the first run's macOS arm64 | none; this inventory's follow-up for #31 | this inventory's follow-up PR | unreleased | the strict-triangle tests in `linalg::matmul::internal`, now on both routes on every host | not reported: same code upstream, but no public caller reaches either case |

Rows 12 and 13 were not found by a family test.
They came from the comparison sweep against LAPACK that #112 asked for.
They are listed because that sweep is part of the validation work, and their regression tests sit in the `nonsymmetric_evd` family's module.

Every defect that also occurs upstream and shows through a public upstream entry point has an upstream-reports entry.
Row 9 is the only fork-only defect. Row 15 is in upstream code too, but no public upstream entry point shows it.

## Defects in the checker or the runner

These made the validation report a failure that was not a library defect, or kept it from running.

| Defect | Where | Fix |
|---|---|---|
| The mode-agreement check between values-only and vector modes used the single-mode budget of 20 epsilons. Two spectra, each within 20 epsilons of an exact-input double reference, differed by 21 (`c32` n = 129, clustered). | `nonsymmetric_evd` | #78, fixed in #87: composed budget 2 x 20 |
| The first local relative oracle for GET34/36 swaps rejected valid swaps. | `schur_reorder` | normalization corrected before the suite landed; threshold 20 unchanged |
| The count check counted four SVD calibration tests as numerical groups. | runner | fixed before the suite landed; an offline check rejects that misclassification |
| The classifier rejected the `linalg::evd::lapack::*` tests added by #87. | runner | #100, fixed in #101 |
| The runner did not reach the end of the `svd` family, and its parallel tests timed out on four cores. | runner | #101 (noted in #110) |
| The offline self-test required a non-empty known-failures ledger, so the empty ledger failed the scheduled run before validation. | runner self-test | #161, fixed in #162 |

One calibration was changed on purpose, and it is not a defect fix.
The GET22 residual threshold is now `20 * max(1, sqrt(n / 16))` (#127; maintainer decision in #112).
CHKHS calibrates its 20 at n <= 16, and LAPACK's own `dgeev` exceeds 20 at n = 63 and 129 on the same fixtures.

## Defects found elsewhere that a family now covers

These were not found by the suite.
Each family carries the fix's test as regression evidence.

| Issue | Defect | Fix, first release | Family | Upstream |
|---|---|---|---|---|
| #28 | The divide and conquer SVD returned wrong factors for a 128 x 128 bidiagonal matrix (secular-root stagnation). | #29, 0.24.8 | `svd` | [faer-004](../upstream-reports/faer-004-svd-dc-secular-stagnation/) |
| #33 | The values-only SVD lost a nonzero singular value. | #39, 0.24.9 | `svd` | [faer-005](../upstream-reports/faer-005-svd-values-only-zero-subproblem/) |
| #34 | `Lblt::reconstruct` ignored the block diagonal. | #46, 0.24.9 | `indefinite` | [faer-006](../upstream-reports/faer-006-lblt-reconstruct-diag/) |
| #35 | Column-pivoted QR of a zero matrix returned NaN. | #46, 0.24.9 | `qr` | [faer-007](../upstream-reports/faer-007-col-piv-qr-zero-nan/) |
| #36 | `norm_l2` of a strided vector overflowed or underflowed. | #46, 0.24.9 | outside the decompositions; the checkers use their own scaled norms | [faer-008](../upstream-reports/faer-008-norm-l2-strided-range/) |
| #37 | Generalized eigenvectors depended on the old contents of the output buffer. | #46, 0.24.9 | none: generalized EVD is not validated | [faer-009](../upstream-reports/faer-009-gevd-reused-output-buffer/) |
| #38 | Partial-pivoting LU returned NaN for a zero pivot column. | #46, 0.24.9 | `lu` | [faer-010](../upstream-reports/faer-010-partial-piv-lu-zero-column-nan/) |
| #40 | SVD least squares returned NaN for a rank-deficient matrix. | #47, 0.24.9 | `lstsq` | [faer-011](../upstream-reports/faer-011-svd-lstsq-zero-singular-value-nan/) |
| #41 | The full U of a tall SVD kept old entries of the output buffer. | #47, 0.24.9 | `svd` | [faer-012](../upstream-reports/faer-012-svd-full-u-stale-buffer/) |
| #42 | 2 x 2 self-adjoint eigenvalues overflowed for a finite spectrum. | #47, 0.24.9 | `self_adjoint_evd` | [faer-013](../upstream-reports/faer-013-evd-2x2-overflow/) |
| #43 | A real 1 x 1 triangular solve overflowed for a subnormal pivot. | #47, 0.24.9 | `triangular` | [faer-014](../upstream-reports/faer-014-trsv-subnormal-pivot/) |
| #44 | The self-adjoint EVD of a 0 x 0 matrix panicked in the scratch query. | #47, 0.24.9 | `self_adjoint_evd` | [faer-015](../upstream-reports/faer-015-evd-empty-scratch/) |
| #45 | Pivoted LLT's early rank exit left input entries in the factor. | #47, 0.24.9 | `cholesky` | [faer-016](../upstream-reports/faer-016-pivoted-llt-rank-exit/) |
| #57 | The tridiagonal EVD returned inf or `NoConvergence` near the `f64` range limits (found by krylov-rs). | #62, 0.25.0 | `tridiagonal_evd` | [faer-017](../upstream-reports/faer-017-tridiag-evd-range/) |
| #81 | The SVD returned `NoConvergence` on a 12 x 10 complex matrix (a 2 x 2 bidiagonal QR block stalled). | #84, 0.25.2 | `svd` | [faer-026](../upstream-reports/faer-026-svd-qr-two-by-two-no-convergence/) |
| #82 | Singular values came out of order with a subnormal at a divide and conquer split (0.24.9). | #87, 0.25.2 | `svd` | not checked |

## What remains

* **Ledger:** no known failure; the ledger is empty.
* **Open plan:** tensor4all/faer#31 (the validation plan) is still open.
* **GET34 rejection contract:** it now runs. LAPACK `dget34`'s own records give 148 rejected swaps per real type. It found rows 15 and 16.
* **Host:** the nightly runs on x86_64 (ubuntu-latest). A weekly aarch64 lane (`lapack-validation-aarch64.yml`, `ubuntu-24.04-arm` and `macos-latest`) runs the same families. The lane also runs the three strict-triangle tests below and the #58 allocation test with `--ignored`; that test is the known per-call allocation of the `gemm` crate fallback ([gemm-001](../upstream-reports/gemm-001-per-call-allocation/)), to be replaced by #56. It is reported as a warning and does not fail the lane. Its results are in the pull request that added the lane.
* **Strict-triangle tests:** they failed on aarch64, in the weekly lane's first run on `ubuntu-24.04-arm` and `macos-latest`. They passed on x86_64 only because the x86 kernel never takes the generic path. They found row 17, which is now fixed, and the tests run both routes on every host.
* **#82 upstream:** checked on 2026-10-30 against `faer` 0.24.4, the latest release. The issue's matrix returns sorted singular values. A diagonal 128 x 128 matrix with `1`, `0.5` and a subnormal at the split positions `65`, `100` and `127` (the fork's direct regression input) also returns sorted values through `Mat::svd`. Upstream's `deflate` has the same `sml` comparisons that the fork replaced (`src/linalg/svd/bidiag_svd.rs:845`, `:871`), but no public reproduction was found, so there is no upstream-reports entry. In the fork, the trigger was 0.24.9's `norm_l2` change.

## What the suite does not validate

### Whole areas

| Area | Why |
|---|---|
| Generalized EVD (QZ), generalized Schur | not in #31's candidate list |
| Sparse factorizations | not a LAPACK test-suite area |
| Matrix-free and partial SVD/EVD | no LAPACK counterpart |
| Norms (`norm_l2` and others) | outside the decompositions; checkers use independent scaled norms |
| Condition estimation, iterative refinement, equilibration (`?CON`, `?RFS`, `?EQU`), expert drivers | no matching faer interface |
| Packed and banded storage | no matching faer interface |
| QL, RQ and LQ factorizations | not provided by faer |
| Bisection, inverse iteration, MRRR, selected eigenvalue ranges, Sturm counts | no matching faer interface |
| LAPACK's random matrix generators (LATMS/LATMR) | replaced by deterministic fixtures: seed `0x4c415041434b` and fixed unitary rotations |

### Omissions within each family

| Family | Omitted |
|---|---|
| `svd` | see [lapack-svd-pilot.md](lapack-svd-pilot.md); the parallel cases use only thin or omitted factors |
| `hessenberg`, `schur` | condition-weighted spectrum comparison: there is no faer estimator, so Jordan and non-normal cases are checked by backward residual |
| `nonsymmetric_evd` | GET22's max-entry normalization is replaced by faer's unit 2-norm; the Schur-vector helper is right-only |
| `schur_reorder` | other EC input records, condition estimation, alternative reordering drivers; `dget34`'s scaled-residual summary (RMAX, LMAX) is replaced by per-record DHST01 and exact checks |
| `balance` | other BAL/BAK records, separate job flags, left-vector back-transformation; the combined API is serial |
| `tridiagonal_evd` | MRRR, bisection, inverse iteration, range selection, Sturm certification, vendor-algorithm comparison |
| `self_adjoint_evd` | packed and banded interfaces, alternate UPLO, other LAPACK drivers |
| `lu` | RFS, CON, equilibration, expert drivers, singular solves |
| `cholesky` | packed and banded storage, regularization, RFS, CON |
| `indefinite` | complex-symmetric (non-Hermitian) LDL, alternate pivot strategies, singular solves, packed IPIV comparison, error estimates |
| `qr` | fixed JPVT columns, QL/RQ/LQ, QRT12's cross-decomposition singular-value check |
| `triangular` | overflow-scaled LATRS, condition and error estimators |
| `lstsq` | wide and epsilon-cutoff pseudoinverse contracts; QRT14 is checked by literal minimum-norm and null-space solutions instead of an augmented factorization |
