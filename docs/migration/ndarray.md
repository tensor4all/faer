# From ndarray and ndarray-linalg

This page translates `ndarray`'s `Array2` and the `ndarray-linalg` traits to this fork's Rust API.
The two crates disagree on the default layout, and that is the first difference to plan for:
`ndarray` is row-major, `Mat` is column-major.

| ndarray / ndarray-linalg | faer counterpart |
| --- | --- |
| `Array2<f64>`, `array2![[1., 2.], [3., 4.]]` | `Mat<f64>`, `mat![[1.0, 2.0], [3.0, 4.0]]`. |
| `a[[i, j]]` | `a[(i, j)]`. |
| `a.t()` | `a.transpose()`, a zero-copy view in both crates. |
| `a.slice(s![i..i + r, j..j + c])` | `a.submatrix(i, j, r, c)`. |
| `a.dot(&b)` | `matmul(dst, Accum::Replace, a, b, 1.0, par)` (`faer/src/linalg/matmul/mod.rs:1944`). |
| `a.solve(&b)` (`ndarray_linalg::Solve`) | `a.partial_piv_lu(par).solve(&b, par)` (`faer/src/linalg/solvers.rs:408`, `:231`). |
| `a.solve_into(b)` | `lu.solve_in_place(&mut b, par)` (`faer/src/linalg/solvers.rs:119`). |
| `a.least_squares(&b)` | `a.qr(par).solve_lstsq(&b, par)` (`faer/src/linalg/solvers.rs:420`, `:851`). |
| `a.cholesky(UPLO::Lower)` | `a.llt(Side::Lower, par)` (`faer/src/linalg/solvers.rs:451`). |
| `a.solveh_into(b)` | `a.llt(Side::Lower, par)?.solve_in_place(&mut b, par)`. |
| `a.eigh(UPLO::Lower)` | `a.self_adjoint_eigen(Side::Lower, par)` (`faer/src/linalg/solvers.rs:479`). |
| `a.eig()` | `a.eigen(par)` (`faer/src/linalg/solvers.rs:805`), always complex-valued. |
| `a.svd(true, true)` (`SVD { u, s, vt }`) | `a.svd(par)` (`faer/src/linalg/solvers.rs:434`); `U()`, `S()`, `V()` (`:1604`, `:1614`, `:1609`). |
| `a.qr()` (`QR { q, r }`) | `a.qr(par)` (`faer/src/linalg/solvers.rs:420`); `compute_thin_Q(par)`, `R()`. |
| `a.det()` | `a.determinant(par)` (`faer/src/mat/matref.rs:1718`). |
| `a.norm_l2()` | `a.norm_l2()` (`faer/src/mat/matref.rs:1669`). |
| `LinalgError` | One error enum per algorithm: `LltError`, `LdltError`, `SvdError`, `EvdError`, `SelfAdjointGevdError` (`faer/src/linalg/solvers.rs:7`–`:12`). |

## Layout: the defaults are opposite

`Array2` is row-major (C order). `Mat` is column-major. Two consequences:

- **Reading** a row-major buffer costs nothing: `MatRef::from_row_major_slice(&slice, m, n)`
  (`faer/src/mat/matref.rs:273`) borrows it as a strided view. Building an owned `Mat` from it means
  filling one yourself: `Mat::from_fn(m, n, |i, j| slice[i * n + j])` (`faer/src/mat/matown.rs:381`).
- **Factorising** a row-major view is supported by LU, QR and Cholesky, which copy the input to a
  column-major panel above `64 * 64` elements (`faer/src/linalg/mod.rs:71`,
  `faer/src/linalg/lu/partial_pivoting/factor.rs:649`). Below that size the arithmetic runs directly
  on the strided data.

`ndarray-linalg`'s LAPACK backend needs column-major buffers, so a C-order array is copied before the
call as well. The difference is where the copy happens, not whether there is one.

## Pitfalls

- **`vt` is not `V`.** `ndarray-linalg` returns `V^T`. Here `V()` returns `V`, and the identity is
  `A = U S V^H` (`faer/src/linalg/svd/mod.rs:11`). A reconstruction is `U * S * V.adjoint()`.
- **Element types.** `Array2<T>` accepts any `Clone` element type; a `Mat<T>` requires
  `T: ComplexField` (`f32`, `f64`, `c32`, `c64`). Integer arrays have to be converted first, and an
  `f32` array computes in `f32` throughout.
- **No broadcasting, no `Zip`, no `mapv`.** Shape-broadcast arithmetic is not part of this API. Write
  the loop over elements with `Mat::from_fn`, or the products with `matmul`.
- **`solve` needs a factorisation here.** `a.solve(&b)` in `ndarray-linalg` picks a routine; here the
  choice is explicit and the factor object is reusable across right-hand sides. Reuse it: factoring
  per right-hand side is the usual porting mistake.
- **`eigh` returns values and vectors together; `self_adjoint_eigen` returns an object.** Read
  `U()` and `S()` (`faer/src/linalg/solvers.rs:1697`, `:1702`); `self_adjoint_eigenvalues` (`:545`)
  is the values-only call.
- **Errors are per algorithm.** A function returning `Result<_, LltError>` cannot be `?`-lifted into
  an `LinalgError`; map the variants you care about (`LltError::NonPositivePivot { index }`,
  `faer/src/linalg/cholesky/llt/factor.rs:22`).
- **Slicing is by length, not by range.** `submatrix(i, j, nrows, ncols)` takes extents; there is no
  `s![.., ..-1]` equivalent.
- **Threads are an argument.** Every factorisation and solve takes a `Par`
  (`faer/src/lib.rs:1070`); `Par::Seq` is the sequential path, and there is no process-wide setting
  to configure.

## Open gaps

No broadcasting or `Zip`-style elementwise iteration helpers, no arbitrary element types, no
`Array3`/`IxDyn` generalisation (matrices only), and no unified `LinalgError`. `ndarray`'s views are
more expressive for non-matrix axes; convert at the boundary, keeping the matrix itself as a `Mat` or
a `MatRef`.
