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

## Conformance follow-up (tensor4all/faer#234)

### Decisions

- The working numerical code is unchanged; the deviations from `dlaln2`
  are documented next to the code instead. The `smin` clamp on the
  factorized second pivot is not a reordering of `dlaln2`: Reference LAPACK
  v3.12.1 also forms the multiplier and the eliminated pivot first and then
  clamps the pivot, so for a 2x2 the two perturb the same entry.
- The 2x2 overflow scale (`shifted_2x2_scale`) bounds the unreduced right
  hand side against the smaller pivot. `dlaln2` bounds the eliminated one and
  rescales again after the solve; here the caller's per-block normalization
  takes that second role. The scale only changes the working vector by a
  positive factor, so the vectors are close to `dtgevc`'s but not
  bit-identical.
- The differential test uses fixtures generated once, offline, from the LAPACK
  that Julia 1.12.5 loads (OpenBLAS 0.3.29, LAPACK 3.12.0). The SciPy wheel
  reachable on the generating host links Apple Accelerate rather than
  Reference LAPACK, so it was not used. The fixtures hold `dggev`'s eigenpairs,
  `dtgsna`-style eigenvalue condition numbers, and per-cluster deflating
  subspaces from `dgges` + `dtgsen` with the exact `Difu`/`Difl`. The test
  compares against first-order bounds with the threshold of LAPACK's
  `TESTING/dgg.in` (20), so a repeated or nearly repeated cluster is compared
  by its subspace, not by its individual vectors.
- Whether a cluster is semisimple comes from how the pencil was built, not
  from the computed vectors: rounding splits a transformed Jordan pair by
  about `sqrt(eps)`, and its reference basis then sits at any `sqrt(eps)`
  rank threshold.

### Verification conclusions and constraints

- All seven fixtures agree with the reference within their conditioning, with
  every ratio at most 1.98 against the threshold of 20. The transformed
  near-defective pencil (eigenvalue `RCONDE` 6.0e-6) and the transformed
  defective one (1.6e-7) are the hard cases. On the defective one, the
  vectors' angle to LAPACK's own vectors is 2.9e-8, about `sqrt(eps)`, as
  expected for a split Jordan pair. On the three fixtures already in Schur
  form, the subspace checks are trivial, since the subspaces are coordinate
  subspaces, and the vectors match LAPACK's directions exactly. A
  `1e-12` perturbation of one entry of the pencil fails all seven.
- The left- and right-vector calls do use different windows of the working
  vector (left: from `je`, offset `j - je`; right: from row 0, offset `j`; the
  issue had the two sides the other way round). The two forms are equivalent
  because the left vector is zero above `je`, and the offset is read only
  when the 2x2 solve returns a scale other than one. Under `dtgevc`'s
  coefficient normalization, that happens only for pencils whose norm is near
  `max_positive`. No GEVD test, no fixture, uniform scalings `1e-320 ..
  1e308` of the faer#220 pencils, and 4000 graded random quasi-triangular
  pencils reached it. Changing the left call to the right call's form,
  changing only its offset, or changing only its window leaves every GEVD
  test green. The new direct left-vector test pins the values that path
  produces. It fails when the sign of the `bcoefi` term in the left right hand
  side is flipped (sine 0.38).
- Out of scope and unchanged: the complex 2x2 solve divides with
  `num_complex`'s unscaled `|d|^2`, not with a `DLADIV` equivalent, and
  `gevd_real` does not prescale the pencil as `dggev` does. In f64 the
  faer#220 pencils scaled by `s` give non-finite vectors for `s` outside
  `[1e-146, 1e154]`, measured at powers of ten. The inputs to the 2x2 solve
  and its LU are finite there, and the quotient by a pivot near `smin` is not.
  `1e-200` already failed before the #220 fix.
- Cost of the per-block rescale and norm pass: about 6% of the eigenvector
  stage. The measurement timed `qz_to_gevd_real` alone, left and right vectors,
  `Par::Seq`, on a pencil already in Schur form, so that stage is all of the
  measured time. It is a local scan on an unpinned laptop, not an acceptance
  run: at n = 1000 the median is 606.6 ms with the pass and 572.0 ms without
  it, and seven interleaved pairs fall between 4.8% and 6.4%. At n = 300 it is
  6.5% (5.2% to 10.9%). Both builds produce the same checksum, since the
  normalization never engages on that pencil. On a dense pencil the QZ stage
  dominates, so the share of the whole `gevd` is smaller; it was not measured.

  Not reproducible from this tree: the timing entry point and the two
  measured binaries were scratch files and were never committed, so the
  percentages above record an ordering measured once, not a benchmark
  that `cargo test` here can re-run.
