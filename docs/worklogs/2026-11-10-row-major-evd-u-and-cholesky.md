# Row-major `u` in the self-adjoint EVD; row-major `cholesky_in_place` measured, not changed (tensor4all/faer#314)

## Decisions

- **`self_adjoint_evd` with a row-major `u`** (`|row_stride| != 1`, `|col_stride| == 1`)
  computes the eigenvectors in an `n x n` column-major stand-in taken from the scratch and
  copies them into `u` once at the end (`linalg::copy_across_layouts`, tiled and split over
  the team). Before, the real eigenvector solver (QR iteration below
  `recursion_threshold`, divide and conquer from it) and the Householder
  back-transformation wrote the caller's `u` in place a column at a time with a stride of
  `n`. The issue's "O(n^2) once is the right answer" holds: the copy is one pass against
  an O(n^3) decomposition.
- The arithmetic is the column-major call's, so `u` and the eigenvalues are the
  column-major ones **bit for bit**. Before, the layout of `u` changed the result: on
  `main`, 92-97% of the entries of `u` differed in their last bits between the two
  layouts (f64 and c64, n = 64-256), and at f64 n = 256 fourteen eigenvalues did too,
  because divide and conquer computes with `u`.
- A failed real solve (`NoConvergence`, e.g. a NaN input) leaves its partial output in a
  column-major `u`, and the entries it did not reach as the caller left them. The
  stand-in therefore starts from the caller's `u` for real `T` (one more O(n^2) pass,
  inside the measured times below), and on failure it is copied back, so a failed call
  hands back the same `u` in both layouts. A complex solve writes the real temporary, and
  `u` is untouched on failure in both layouts, as before. The result and the eigenvalues
  of a failed call are unchanged.
- `self_adjoint_evd_scratch` reserves the stand-in for every caller with eigenvectors,
  because the query cannot see the layout of `u`; a column-major caller never touches it.
  No heap allocation (the row-major arm of `evd_svd_vectors_no_alloc`).
- The perf warning now fires only for a `u` with generic strides; those keep the in-place
  path, which this change does not touch.
- **SVD left alone, warning included.** `svd()` with a row-major `U`/`V` measured
  0.96-1.05x on the issue's Zen 5 (HX 470, n <= 256), so this slice neither stages it nor
  drops its warning: dropping a warning without a change behind it would need the same
  grid measured on more hosts and sizes, and the warning costs nothing.
- **`cholesky_in_place` on a row-major input is not changed.** There is no upper Cholesky
  kernel in the fork. For a row-major `A`, the transpose dual (upper Cholesky of
  `B = A^T`, `U = L^T`) is the same sequence of memory operations as the existing lower
  recursion run on the row-major views: its triangular solve `U01 = U00^-H B01` is the
  call `solve_lower(A00.conj(), A10^T)`, and so on. The only layout-dependent piece of
  the Cholesky code is the leaf, `simd_cholesky`, which runs SIMD only on a column-major
  view and otherwise falls back to a scalar loop. Four ways of running the existing
  column-major lower kernel on a transposed view were measured (local prototypes, not
  committed; see the evidence below), and none reaches the issue's 1.05x at n = 64-256,
  which is the stop condition the issue's plan set for this case. So no new kernel was
  started.

## Where row-major `cholesky_in_place` loses (Zen 5, f64, 1T, `main`)

- The scalar leaf (`n <= 64`): 8.8 against 19.7 us at n = 64.
- **The leaf after a matrix product is 4x slower again**: the second leaf of n = 128 takes
  82.7 us row-major, 9.0 column-major, and 20.6 us when a `vzeroupper` is executed just
  before it. So a kernel on the triangular-matmul path returns with the upper vector state
  dirty, and the SSE-encoded scalar fallback pays the transition penalty. Which kernel
  does it is **not identified**. It affects any non-VEX code after that product, not only
  this path; the large spread of the `c64` n = 64 row-major cell between sessions (2.7x
  to 11.8x) is consistent with it.
- The triangular solve: in the row-major recursion its right-hand side `A10^T` is
  column-major with a power-of-two stride at n = 128/256/512, and the solve's base case
  vectorizes along rows of the right-hand side: 1.11x / 1.29x / 2.4x the column-major
  recursion's solve. The triangular matrix product is at parity (0.84-1.01x).

## Prototypes measured (row/col of `cholesky_in_place`, n = 64 / 128 / 256 / 512)

Probe, same process, layouts alternated, 41 calls per layout, CPU 8 (1T) or 8-15 (8T):

| leaf on a transposed view | 1T | 8T |
|---|---|---|
| in-place swap transpose of the leaf block | 1.23 / 1.33 / 1.88 / 2.10 | 1.20 / 1.29 / 2.14 / 2.72 |
| leaf's lower triangle through a padded column-major buffer | 1.15 / 1.11 / 1.23 / 1.45 | 1.16 / 1.07 / 1.20 / 1.40 |
| the same, plus the solve's existing `stage` (padded copy of a conflicting right-hand side) for the non-unit solve | 1.16 / 1.14 / 1.19 / 1.01 | 1.12 / 1.07 / 1.08 / 0.97 |
| each block column staged column-major (`O(n * nb)` buffer) | 1.16 / 1.38 / 1.43 / 1.19 | 1.14 / 1.41 / 1.69 / 1.63 |

The best one (third row) through `fbench` in the same session as `main`: f64 1T
1.17 / 1.14 / 1.13 / 0.99, 8T 1.32 / 1.11 / 1.05 / 0.87; c64 1T 1.07 / 1.08 / 1.06 / 1.06,
8T 1.09 / 1.09 / 0.98 / 0.96 (`main`: 1.5-4.0x). The swap loses to L1 set conflicts at a
2-4 KiB row stride; every copy is O(n^2) against O(n^3/3) flops and does not pay at
n <= 256; what is left after the leaf is the solve with a column-major right-hand side.
Closing the remaining 5-17% needs either a leaf kernel native to a row-major lower
triangle (vectorized along the contiguous rows, a different summation order) or a
triangular-solve base case for a column-major right-hand side. Neither is tried here.

## Verification conclusions and constraints

- EVD, `fbench` (`in_place` = `self_adjoint_evd` with prepared `U` in the case layout,
  row-major A and U against column-major A and U), median of 27 samples per layout over
  three sessions, Zen 5 (Ryzen AI MAX+ 395), boost on: f64 1T 1.43 / 3.74 / 8.24 / 5.72
  -> 1.027 / 1.022 / 1.024 / 1.006, 8T 1.44 / 4.00 / 6.57 / 3.68 -> 1.020 / 1.011 /
  1.012 / 1.015; c64 1T 1.03-1.06 -> 1.009-1.019, 8T 1.04-1.14 -> 1.012-1.017
  (n = 64 / 128 / 256 / 512). Every cell is within 1.05x after the change.
- Boost could not be switched off on the host (no sudo); the ratios are within one
  process with the layouts interleaved, which is what the 1.05x criterion compares.
- Only n = 64-512 square, f64 and c64, 1T and 8T were timed; f32/c32 and larger sizes are
  checked for correctness only (the bit-for-bit test covers f32/f64/c32/c64).
- Tests: `faer/tests/self_adjoint_evd_row_major_u.rs` (bit for bit; red on `main` at f64
  n = 5), `faer/tests/evd_perf_warn.rs` (red on `main`: the row-major `u` warned), the
  row-major arm of `faer/tests/evd_svd_vectors_no_alloc.rs` (a guard, green on `main` too).
