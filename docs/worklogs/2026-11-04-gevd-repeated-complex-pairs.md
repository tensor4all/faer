# Generalized eigenvectors for repeated complex pairs

## Decisions

- Protect the second pivot in the real driver's shifted 2x2 solves and return
  after replacing a tiny block by `smin * I`. A repeated eigenvalue makes the
  shifted block singular even for an uncoupled, diagonalizable pencil. Raising
  the pivot to `smin` follows Reference LAPACK v3.12.1's
  [`dlaln2`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dlaln2.f).
  Its caller, `dtgevc`, keeps the perturbed solution; a unit-vector fallback
  would not preserve the eigenvector equation for the coupled pencil.
- Pivot protection alone still overflowed for chains of eight coupled blocks
  in `f32` and 22 in `f64`. Scale the right-hand side before the small solve,
  return that scale, and apply it to every other active entry of the working
  eigenvector. Normalize the working vector between substitution steps, before
  further matrix updates and the back transformation. This covers both left
  and right vectors and preserves the homogeneous eigenvector equations.
- Retain faer's full-pivot LU factorization, but specialize the two scalar
  substitution steps. Form `U01 / U00` before multiplying by `x1`; the product
  `U01 * x1` could overflow even when the final quotient is finite. Complete
  pivoting bounds the multipliers, and the RHS bound uses
  `min(1, min_pivot) / (16 * min_positive)` to leave room for real and complex
  arithmetic. Component maxima avoid overflowing while estimating a complex
  RHS's size. No new heap allocation or scratch is needed.
- The left-vector dot-product guard uses `1 / max(1, xmax)`, so a working
  vector already below unit size cannot be enlarged by an overflow guard.
  For a 2x2 block, include the raw norms of both A and B columns in the bound.
  The second B-column norm previously repeated the A-column norm by mistake;
  its weighted bound alone is insufficient because the B dot product is
  formed before applying the eigenvalue coefficient.

## Verification conclusions and constraints

- The two chain regressions fail with NaN vectors under pivot protection alone.
  With scaling they pass for left-only, right-only, and both-vector requests,
  with nonzero vectors, the known spectrum, and precision-aware normwise
  residuals. The original two-block regressions still pass, including the
  independent vectors of the uncoupled pencil.
- Large-RHS helper tests fail before scaling in both precisions. The final
  tests cover real and complex shifts, ordinary and tiny blocks, row and column
  pivots, and scale propagation to either side of the solved block. They check
  the scaled equation after normalizing the whole working vector.
- LAPACK `ggev` returns finite left and right vectors for both chain inputs;
  its maximum right-vector residuals were approximately `8e-8` (`f32`) and
  `1.5e-16` (`f64`), using Julia 1.13.3's OpenBLAS 0.3.30 / LAPACK 3.12.0.
  The two-block defect also reproduces in upstream 0.24.4 and upstream `main`
  at `0ecd846`; see the [local upstream report](../../upstream-reports/faer-038-gevd-repeated-complex-pair-nan/).
- The inherited complex left-vector convention of `gevd_real` is unchanged:
  it satisfies `y^T A = lambda y^T B`, conjugated relative to LAPACK's and
  `gevd_cplx`'s `y^H A = lambda y^H B`. The tests state and check that convention.
- Focused validation of the guard corrections passes 19 GEVD library tests
  and 21 integration tests across both GEVD suites. Formatting and
  diff-whitespace checks pass for the changed files. Broader validation of the
  preceding overflow-control implementation passed 651 library tests (146
  ignored), the other numerical integration tests, and 70 doctests; those
  broader checks were not repeated for the guard-only corrections.
- The existing tests do not isolate either guard correction as an end-to-end
  regression. An exploratory near-overflow pencil still produced NaNs during
  the complex 2x2 solve with both corrections present, masking the guard's
  effect. The two corrections do not establish full-range overflow safety;
  that remaining case is outside these guard corrections.
- The full crate suite remains red on eight allocation assertions on this
  host. Each also fails against the original source at `bdf71c0`, with matching
  counts in serial execution: `lq_allocates_nothing_128x128`,
  `lq_allocates_nothing_300x60`, `lq_allocates_nothing_33x40`,
  `lq_allocates_nothing_40x33`, `lq_allocates_nothing_60x300`,
  `parallel_regions_allocate_nothing` (Pfaffian at width 4),
  `values_only_svd_allocates_nothing_300x60`, and
  `values_only_svd_allocates_nothing_700x700`. Their causes are outside this
  GEVD change; these failures are not waived or described as a green suite.
- Dependency-inclusive Clippy is blocked by the unchanged
  `faer-traits/src/lib.rs:1280` NaN check (`value != value`, `clippy::eq_op`).
  Checking only the faer crate also encounters existing deny-lints in other
  modules (including constant boolean expressions and SIMD array-index
  diagnostics). There are no Clippy errors in GEVD; the lint gate remains red.
- This is a correctness change. Normalizing between blocks adds bounded
  vector passes to the existing substitution; no timing or performance claim
  is made.
