# From nalgebra

This page translates `nalgebra`'s dense linear algebra to this fork's Rust API.
Both crates store matrices in column-major order, so the storage translation is the easy part; what
changes is ownership, dimensions and the parallel argument.

| nalgebra | faer counterpart |
| --- | --- |
| `DMatrix<f64>`, `Matrix3<f64>` | `Mat<f64>`. Dimensions are runtime values here (`faer/src/lib.rs:861` is the only `Shape` implementation), so there is no `Matrix3` type: check the shape where it matters. |
| `DVector<f64>` | A `Mat<f64>` with one column. A borrowed column is `a.col(0)`, a `ColRef` (`faer/src/col/mod.rs:44`). |
| `m[(i, j)]` | `m[(i, j)]`, zero-based. |
| `m.column(j)`, `m.row(i)` | `m.col(j)`, `m.row(i)`; both are views, not copies. |
| `m.slice((i, j), (r, c))` | `m.submatrix(i, j, r, c)`; `submatrix_mut` for mutation. |
| `m.transpose()` | `m.transpose()`, but zero-copy: it returns a view. `to_owned()` when an owner is needed. |
| `m.adjoint()` | `m.adjoint()`, which conjugates; `transpose()` does not. |
| `&a * &b` | `matmul(dst, Accum::Replace, a, b, 1.0, par)` (`faer/src/linalg/matmul/mod.rs:1944`). |
| `m.lu()` | `m.partial_piv_lu(par)` (`faer/src/linalg/solvers.rs:408`). |
| `lu.solve(&b)` | `lu.solve(&b, par)` (`faer/src/linalg/solvers.rs:231`); `solve_in_place` for a buffer (`:119`). |
| `Cholesky::new(m)` | `m.llt(Side::Lower, par)` (`faer/src/linalg/solvers.rs:451`), which returns `Result<_, LltError>` instead of `Option`. |
| `QR::new(m)` | `m.qr(par)` (`faer/src/linalg/solvers.rs:420`). |
| `QR::q()` / `QR::r()` | `qr.compute_thin_Q(par)` (`faer/src/linalg/solvers.rs:1403`) / `qr.R()` (`:1371`). |
| `SVD::new(m, true, true)` | `m.svd(par)` (`faer/src/linalg/solvers.rs:434`); `thin_svd` (`:442`) for the economy form. |
| `svd.u`, `svd.v_t`, `svd.singular_values` | `svd.U()`, `svd.V()`, `svd.S()` (`faer/src/linalg/solvers.rs:1604`, `:1609`, `:1614`). `V()` is `V`, not `V^T`: `A = U S V^H`. |
| `SymmetricEigen::new(m)` | `m.self_adjoint_eigen(Side::Lower, par)` (`faer/src/linalg/solvers.rs:479`). |
| `Schur::new(m)` | No owning wrapper. The public entry points are Hessenberg and real/complex Schur routines; see `faer/src/linalg/evd/schur/`. |
| `Hessenberg::new(m)` | `faer::linalg::evd::hessenberg` (`faer/src/linalg/evd/hessenberg.rs:567`). |
| `m.norm()` (Frobenius) | `m.norm_l2()` (`faer/src/mat/matref.rs:1669`). |
| `m.determinant()` | `m.determinant(par)` (`faer/src/mat/matref.rs:1718`). |

## Pitfalls

- **Everything takes a `Par`.** nalgebra has no parallel argument; here every factorisation, solve,
  `reconstruct`, `inverse`, `pseudoinverse`, `compute_Q` and `compute_thin_Q` takes one, and it is
  `Par::Seq` unless you build a team (`faer/src/lib.rs:1070`). There is no global thread setting.
- **`transpose()` does not allocate here.** In nalgebra `transpose()` returns an owned matrix; here it
  returns a view that borrows the original, so the borrow checker keeps the original alive. This
  breaks code that transposes into a temporary and lets the original go out of scope.
- **Fixed sizes are gone.** `Matrix3<f64>`, `SMatrix<f64, 3, 3>` and `Const<3>` have no counterpart in
  this revision: dimensions are `usize` values (`faer/src/lib.rs:861`), so shape mismatches that
  nalgebra rejected at compile time are runtime assertions here.
- **`Cholesky::new` returns `Option`; `llt` returns `Result`.** `LltError::NonPositivePivot { index }`
  carries the failing index (`faer/src/linalg/cholesky/llt/factor.rs:22`). `ldlt` returns
  `LdltError::ZeroPivot` (`faer/src/linalg/cholesky/ldlt/factor.rs:691`). This `ldlt` has no pivoting
  (`docs/migration/cpp.md:38`), so a matrix that needs a pivot fails where a pivoted routine
  would succeed; `lblt` is the pivoted route (`faer/src/linalg/solvers.rs:471`).
- **`lu()` in nalgebra is partial pivoting with a solve attached.** Here `partial_piv_lu` is
  infallible and gives you `P A = L U` (`faer/src/linalg/lu/partial_pivoting/mod.rs:2`); a singular
  matrix produces non-finite solves rather than an error. `full_piv_lu` adds column pivoting and a
  `rank` in its in-place info.
- **Traits instead of inherent methods.** `solve` and `solve_lstsq` come from the `Solve` and
  `SolveLstsq` traits (`faer/src/linalg/solvers.rs:115`, `:819`); import them:
  `use faer::linalg::solvers::{Solve, SolveLstsq};`.
- **Genericity is over `ComplexField`, not `RealField`.** A nalgebra generic function bounded by
  `RealField` does not port directly; the faer bound is `faer::traits::ComplexField`, and
  `ComplexField::Real` replaces nalgebra's `RealField::RealField`. Autodiff scalars are not supported.
- **Views and `Copy`.** `MatRef` is `Copy`; an owned `Mat` is not, and cloning one allocates.

## A solve and an eigenproblem side by side

The ownership shape is what changes: factor once, then solve as many right-hand sides as needed,
lending the same thread budget with `par.reborrow()`.

```text
nalgebra                          faer
let lu = a.lu();                  let lu = a.partial_piv_lu(Par::Seq);
let x = lu.solve(&b).unwrap();    let x = lu.solve(&b, Par::Seq);
let e = SymmetricEigen::new(a);   let e = a.self_adjoint_eigen(Side::Lower, Par::Seq)?;
let v = e.eigenvectors;           let v = e.U();
```

`faer/examples/migration_overview.rs` is the compiled version of the same sequence, run by
`.github/workflows/ci.yml`.

## Open gaps

No fixed-size matrix types, no `Matrix3`-style stack allocation, no operator overloading for parallel
products (the operators are sequential conveniences; call `matmul` for a team), and no `Schur`
owning wrapper. Generic code must be written against `ComplexField` rather than `RealField`.
