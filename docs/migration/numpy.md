# From NumPy and SciPy

This page translates `numpy.linalg` and `scipy.linalg` calls to this fork's Rust API.
It maps mathematical operations. There is no binding, no `PyObject`, and no dtype dispatch.

| NumPy or SciPy | faer counterpart |
| --- | --- |
| `np.ndarray((m, n), dtype=np.float64)` | `Mat<f64>` owns it (`faer/src/mat/matown.rs:381`); `MatRef` and `MatMut` borrow it. |
| `a.dtype` | The scalar type parameter: `f32`, `f64`, `c32`, `c64`. Nothing is promoted or dispatched at run time. |
| `a[i, j]` | `a[(i, j)]`, zero-based, no slices as indices. A block is `a.submatrix(i, j, rows, cols)`. |
| `a.T` | `a.transpose()` (a zero-copy view, not a copy). `a.adjoint()` for the conjugate transpose. |
| `a @ b` | `matmul(dst, Accum::Replace, lhs, rhs, 1.0, par)` (`faer/src/linalg/matmul/mod.rs:1944`). |
| `np.linalg.solve(a, b)` | `a.partial_piv_lu(par).solve(&b, par)` (`faer/src/linalg/solvers.rs:408`, `:231`). |
| `np.linalg.inv(a)` | `a.partial_piv_lu(par).inverse(par)` (`faer/src/linalg/solvers.rs:58`). |
| `np.linalg.det(a)` | `a.determinant(par)` (`faer/src/mat/matref.rs:1718`). |
| `np.linalg.cholesky(a)` | `a.llt(Side::Lower, par)` (`faer/src/linalg/solvers.rs:451`); `L()` returns `L` with `A = L L^H`. |
| `np.linalg.lstsq(a, b, rcond=None)` | `a.qr(par).solve_lstsq(&b, par)` for a tall full-rank `A` (`faer/src/linalg/solvers.rs:420`, `:851`); `a.svd(par).solve_lstsq(..)` for a rank-deficient or wide problem. |
| `np.linalg.qr(a, mode="reduced")` | `let qr = a.qr(par); qr.compute_thin_Q(par)` and `qr.thin_R()` (`faer/src/linalg/solvers.rs:1403`, `:1376`). |
| `np.linalg.svd(a, full_matrices=False)` | `a.thin_svd(par)` (`faer/src/linalg/solvers.rs:442`); `full_matrices=True` is `a.svd(par)` (`:434`). |
| `np.linalg.svdvals(a)` | `a.singular_values(par)` (`faer/src/linalg/solvers.rs:587`). |
| `np.linalg.eigh(a)` | `a.self_adjoint_eigen(Side::Lower, par)` (`faer/src/linalg/solvers.rs:479`). |
| `np.linalg.eigvalsh(a)` | `a.self_adjoint_eigenvalues(Side::Lower, par)` (`faer/src/linalg/solvers.rs:545`). |
| `np.linalg.eig(a)` | `a.eigen(par)` (`faer/src/linalg/solvers.rs:805`), always complex-valued. |
| `scipy.linalg.eigh(a, b)` | `a.self_adjoint_generalized_eigen(b, par)` (`faer/src/linalg/solvers.rs:498`). |
| `scipy.linalg.lu(a)` | `let lu = a.partial_piv_lu(par)`; see the convention note below. |
| `scipy.linalg.lu_factor(a)` + `lu_solve` | `a.partial_piv_lu(par)`, then `solve` / `solve_in_place`. |
| `scipy.linalg.qr(a, pivoting=True)` | `a.col_piv_qr(par)` (`faer/src/linalg/solvers.rs:426`); `P()`, `R()` (`:1531`, `:1482`). |
| `scipy.linalg.solve_triangular(T, b)` | `faer::linalg::triangular_solve::solve_lower_triangular_in_place(tri, rhs, par)` (`faer/src/linalg/triangular_solve.rs:283`) or `solve_upper_triangular_in_place` (`:385`). |
| `np.linalg.norm(a)` (Frobenius) | `a.norm_l2()` (`faer/src/mat/matref.rs:1669`). |
| `np.linalg.norm(a, np.inf)` | `a.norm_max()` (`faer/src/mat/matref.rs:1658`). |
| `np.linalg.matrix_rank(a)` | No call. Take `a.singular_values(par)` and apply your own tolerance, or read the rank of a truncated LU (see [truncation](#numerical-rank-and-truncation)). |

## Layout: C order versus column-major

A fresh `np.zeros((m, n))` is C order (row-major). An owned `Mat` is column-major.
Data that arrives as a row-major buffer does not need a copy to be *read*: borrow it with
`MatRef::from_row_major_slice(&slice, m, n)` (`faer/src/mat/matref.rs:273`).
To own it, fill a `Mat` yourself with `Mat::from_fn(m, n, |i, j| slice[i * n + j])`
(`faer/src/mat/matown.rs:381`).

The factorisations adapt a row-major input by copying it, above `64 * 64` elements
(`faer/src/linalg/mod.rs:71`, `faer/src/linalg/lu/partial_pivoting/factor.rs:649`).
So a row-major `MatRef` gives the right answer and pays one copy; a `Mat` built by `from_fn` pays
none. The SVD, EVD and GEVD additionally warn, with the `perf-warn` feature, when their eigenvector
or singular-vector *outputs* are row-major or generically strided
(`faer/src/linalg/svd/mod.rs:771`) — allocate those outputs as ordinary `Mat`s.

Generic strides, such as `a[::2, :]`, are slow everywhere. Copy to a contiguous `Mat` before a
factorisation if an upstream `numpy` view would have been non-contiguous too.

## Pitfalls

- **`np.linalg.solve` raises; `partial_piv_lu` does not.** The LU constructors are infallible
  (`faer/src/linalg/solvers.rs:1190`). A singular matrix yields factors and a solve that returns
  non-finite values. `llt` and `ldlt` do return `Result` (`LltError::NonPositivePivot`,
  `LdltError::ZeroPivot`), and the in-place LU reports `rank`.
- **Permutation direction.** SciPy's `lu` returns `A = P L U`. This fork documents `P A = L U`
  (`faer/src/linalg/lu/partial_pivoting/mod.rs:2`), so `A = P^T L U`. `P()` is a `PermRef` whose
  `arrays()` gives the forward and inverse slices (`faer/src/perm/permref.rs:143`); `(P A)` has row
  `i` equal to `A`'s row `perm[i]` (`faer/src/perm/mod.rs:282`).
- **Column pivoting direction.** SciPy's `qr(pivoting=True)` gives `A[:, perm] = Q R`. Here the
  identity is `A P^T = Q R` (`faer/src/linalg/qr/col_pivoting/mod.rs:2`), which is the same statement
  read from the other side.
- **`lstsq` returns one thing.** `solve_lstsq` returns `x`; NumPy also returns the residual, the rank
  and the singular values. Compute the residual with `matmul` (`b - A x`) and get the rank from the
  singular values.
- **No broadcasting, no `einsum`, no stacked arrays.** Every operation is one matrix against one
  matrix, with shapes checked (statically when the shapes are const generics, otherwise by assert).
  A batch of matrices is a loop or a call per item.
- **No mutation flags.** NumPy returns a new array; the in-place functions here overwrite the
  `MatMut` you pass and take a `MemStack` sized by the matching `*_scratch` function.
- **Single precision is single precision.** `f32` input computes in `f32` throughout; there is no
  "upcast internally" step, and tolerances in `np`-style code that assumed `float64` will fail.
- **Threads are not ambient.** There is no `OMP_NUM_THREADS` equivalent. Every parallel call takes a
  `Par`, and `Par::Seq` runs on the calling thread (`faer/src/lib.rs:1070`).
- **Complex conjugation.** `transpose()` does not conjugate, `adjoint()` does. `A = U S V^H`
  (`faer/src/linalg/svd/mod.rs:11`), so a reconstruction is `U * S * V.adjoint()`.

## Numerical rank and truncation

- `scipy.linalg.interpolative` (randomized ID, `interp_decomp`) has **no equivalent**. The nearest
  thing is `col_piv_qr`, whose leading columns of `P()` are the greedy column selection, and a
  truncated SVD you assemble yourself.
- Column-pivoted QR has no rank parameter (`ColPivQrParams` carries only blocking and parallel
  settings, `faer/src/linalg/qr/col_pivoting/factor.rs:95`). Decide the rank from the diagonal of
  `R()`, as you would with NumPy.
- The LU has rank truncation on its in-place entry point: `max_rank`, `abs_tol`, `rel_tol` in
  `PartialPivLuParams` / `FullPivLuParams` (`faer/src/linalg/lu/partial_pivoting/factor.rs:534`),
  reported as `PartialPivLuInfo::rank` (`:568`). The owning `a.partial_piv_lu(par)` does not expose
  it.
- A truncated SVD is the leading `r` columns of `thin_svd`'s `U`, `S` and `V`. `Svd::pseudoinverse`
  (`faer/src/linalg/solvers.rs:1619`) is the built-in rank-aware operation.

## Open gaps

No `interpolative`, no `einsum`/broadcasting, no batched or stacked arrays, no dtype promotion, no
`matrix_power`. `np.linalg.lstsq`'s residual/rank/singular-value return has to be assembled by hand.
The sparse modules (`faer::sparse`) are out of scope for this page: their API is not a translation of
`scipy.sparse`.
