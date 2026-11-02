# Self-adjoint definite generalized eigenproblem

Adds the public `self_adjoint_generalized_eigen` and
`self_adjoint_generalized_eigenvalues` solver methods for
$A x = \lambda B x$ with $A$ self-adjoint and $B$ self-adjoint positive
definite, and the `SelfAdjointGeneralizedEigen` result type. Issue
tensor4all/faer#184.

## Decisions

- **Cholesky reduction, not the QZ path.** `B = L L^H`, `C = L^-1 A L^-H`,
  solve `C y = lambda y` with the existing `self_adjoint_eigen`, then
  `x = L^-H y`. Reusing the self-adjoint solver keeps the symmetry that the
  general QZ path (`generalized_eigen`) throws away, and is the reduction
  LAPACK uses in `sygv`/`hegv` for `itype = 1`. The QZ path is left untouched
  and is used as the independent reference in the tests.
- **The right solve is a left solve on the transposed matrix.**
  `C = (L^-1 A) L^-H` needs a right triangular solve, which the crate does not
  expose. Since `(X^H)^H = X`, the right solve can be done as a left solve on
  the transposed matrix: solve `conj(L) Y = D^T` for `Y = conj(L^-1 D^H)` and
  store `Y` through `c.transpose_mut()`, which writes `Y^T = (L^-1 D^H)^H = D L^-H`.
  The `conjugate()` on the factor is exactly what makes the stored transpose
  equal `C` and not `conj(C)` for complex scalars. One comment in
  `self_adjoint_gevd_reduce` marks it, because it is the only non-obvious line
  in the change.
- **Only the lower triangles are read**, matching `self_adjoint_eigen(Side::Lower)`
  and `Llt::new(.., Side::Lower, ..)`. The upper triangle of `A` is mirrored
  before the solves, since an in-place triangular solve needs the full matrix.
- **Errors come from the existing enum.** A failed Cholesky maps to
  `SelfAdjointGevdError::NonPositiveDefinite` (it is the only reason
  `Llt::new` fails), and a failed eigensolver to `NoConvergence`. No new
  variant was added and no failure is swallowed into NaNs.
- **`par` goes to every phase** (Cholesky, both triangular solves, the
  self-adjoint eigensolver, the back substitution). There is no sequential
  fallback.
- **The result type carries `values: Vec<Real<T>>` and `U: Mat<T>`**, as the
  issue proposed, rather than a `Diag` factor like `SelfAdjointEigen`: the
  eigenvalues of a self-adjoint pair are real even for a complex pair, and
  `Vec<Real<T>>` says that without the zero imaginary part a `Diag<Complex>`
  would carry.

## Verification conclusions and constraints

`faer/tests/self_adjoint_generalized_eigen.rs`, 9 tests, all passing locally on
broccoli (2026-11-02):

- `f64` and `c64`, `n = 0, 1, 2, 4, 8, 32, 128`, with the upper triangles of
  both inputs filled with garbage to prove they are not read.
- Eigenvalues agree with the general QZ path (`generalized_eigen`) to
  `1e-9 * (1 + |lambda|)` for `n <= 32` after sorting both.
- Residual `||A X - B X Lambda||_2 / (||A||_2 + max|lambda| ||B||_2)` is
  `<= 6.0e-16` over all sizes and both scalar types; B-orthonormality
  `||X^H B X - I||_2 / sqrt(n)` is `<= 6.1e-15`.
- `B` with a negative pivot returns `NonPositiveDefinite` for both entry
  points, `f64` and `c64`.
- `Par::Seq` and `faer::team::with_team(1|2|4|8, ..)` agree on the eigenvalues
  to `1e-12 * (1 + |lambda|)` for `n = 1, 2, 4, 64`, and the team path keeps
  `X^H B X = I` to `1e-11`. A separate `n = 256`, width 8 test asserts with
  `faer::utils::team::stats` that the team was actually dispatched
  (multi-participant sections > 0), so a silent sequential fallback would fail.
- A conjugated view of a self-adjoint matrix is accepted and gives the right
  residual, covering the non-canonical `Conjugate` path.

Not established, deliberately:

- No performance claim of any kind; nothing was measured, per the issue's
  scope split.
- No `itype = 2`/`itype = 3` (`A B x = lambda x`, `B A x = lambda x`).
- No extreme-magnitude guarantee. `C` is a congruence of `A`, so its entries
  are not bounded by those of `A` and `B`, which is the same limitation
  LAPACK's `sygv` has. A finite, well posed pair can therefore come back
  non-finite: with `A = [[0, 1e200], [1e200, 0]]` and
  `B = diag(1e-300, 1e300)` the exact eigenvalues are `+-1e200`, but
  `L^-1 A` overflows (`1e350`) and both entry points return `Ok([NaN, NaN])`.
  This is recorded, not fixed: a finite check on `C` would only catch this
  half of the problem (the back substitution can overflow on its own), the
  crate has no such convention in the other eigensolvers, and
  `SelfAdjointGevdError` has no variant for it. The choices for the
  maintainer are (a) leave it, as LAPACK does, (b) reject a non-finite `C` or
  non-finite results with `NoConvergence`, or (c) add a scaling variant.
  An early draft test with `1e308` entries hit the same wall and was dropped
  rather than turned into a claim the reduction cannot make.

## Review

An independent review (gpt-6.1-sol, read-only, post-diff) raised three points.
Applied: the team tests now assert the same normwise residual as the sequential
test instead of B-orthonormality alone (a permuted eigenvector block would have
passed), and the `DSYGV` row was corrected to `UPLO = 'L'` with a note that the
entry points only read lower triangles, plus the `itype = 2/3` reduction
described as `L^H A L` rather than "factoring `B` the other way round". The
non-finite result above was recorded instead of patched, for the reasons given
there.
