---
name: faer-usage
description: Call this library from a downstream Rust project. Use when writing or reviewing code that factorises, solves, multiplies or decomposes dense matrices with `faer` (package `t4a-faer`): which decomposition and entry point to pick, the borrowed-versus-in-place forms, the error types, scratch sizing, and caller-owned thread teams.
---

# Using the library from a downstream project

This is the usage layer: how a **consumer** of the crate calls it correctly. The optimisation
workflow for developing the fork itself is the separate
[`faer-optimization`](../faer-optimization/SKILL.md) skill.

The package is `t4a-faer`; the Rust library name is `faer` (`faer/Cargo.toml:2`, `faer/Cargo.toml:17`).
Depend on it with the package name aliased:

```toml
[dependencies]
faer = { package = "t4a-faer", version = "0.32" }
```

This is **not** upstream `faer`: upstream examples from a model's training data do not compile here.
The differences are listed in [upstream faer](../../docs/migration/upstream-faer.md); the most visible
is that every parallel entry point takes a `Par` that the caller owns.

## Which decomposition

The methods below are inherent methods on `MatRef` (`faer/src/linalg/solvers.rs:649`). They borrow the
input, copy it into the factor object, and return that object.

| The task | Call | Returns |
| --- | --- | --- |
| Square system `A x = b`, general | `a.partial_piv_lu(par)` (`solvers.rs:408`) | `PartialPivLu`, infallible |
| Square system, symmetric positive definite | `a.llt(Side::Lower, par)` (`solvers.rs:451`) | `Result<Llt, LltError>` |
| Symmetric, indefinite (`L D L^T`) | `a.ldlt(Side::Lower, par)` (`solvers.rs:461`) | `Result<Ldlt, LdltError>` |
| Symmetric, indefinite, Bunch–Kaufman | `a.lblt(Side::Lower, par)` (`solvers.rs:471`) | `Lblt`, infallible |
| Rank and a row/column selection, square | `a.full_piv_lu(par)` (`solvers.rs:414`) | `FullPivLu`, infallible |
| Least squares of a tall full-rank `A` | `a.qr(par)` (`solvers.rs:420`), then `SolveLstsq` | `Qr`, infallible |
| Rank-revealing QR, column selection, ID | `a.col_piv_qr(par)` (`solvers.rs:426`) | `ColPivQr`, infallible |
| Singular values and vectors | `a.svd(par)` (`solvers.rs:434`) or `a.thin_svd(par)` (`solvers.rs:442`) | `Result<Svd, SvdError>` |
| Singular values only (no allocation) | `a.singular_values(par)` (`solvers.rs:587`) | `Result<Vec<Real>, SvdError>` |
| Symmetric eigenpairs | `a.self_adjoint_eigen(Side::Lower, par)` (`solvers.rs:479`) | `Result<SelfAdjointEigen, EvdError>` |
| Symmetric eigenvalues only | `a.self_adjoint_eigenvalues(side, par)` (`solvers.rs:545`) | `Result<Vec<Real>, EvdError>` |
| Generalised symmetric eigenproblem `A x = λ B x` | `a.self_adjoint_generalized_eigen(b, par)` (`solvers.rs:498`) | `Result<SelfAdjointGeneralizedEigen, SelfAdjointGevdError>` |
| Nonsymmetric eigenvalues/eigenvectors | `a.eigen(par)` (`solvers.rs:805`) | `Result<Eigen<Real>, EvdError>` |
| `A x = λ B x`, both general | `a.generalized_eigen(b, par)` (`solvers.rs:795`) | `Result<GeneralizedEigen, GevdError>` |
| `dst = alpha · lhs · rhs` | `matmul(dst, accum, lhs, rhs, alpha, par)` (`faer/src/linalg/matmul/mod.rs:1944`) | `()` |

Read the factors through accessors, not through layout assumptions: `P()`, `L()`, `U()` on the LU types
(`solvers.rs:1231`, `:1236`, `:1241`), `Q_basis()`/`R()`/`compute_thin_Q()` on `Qr`
(`solvers.rs:1361`, `:1371`, `:1403`), `P()`/`R()` on `ColPivQr` (`solvers.rs:1531`, `:1482`),
`U()`/`S()`/`V()` on `Svd` (`solvers.rs:1604`, `:1614`, `:1609`), `U()`/`S()` on
`SelfAdjointEigen` (`solvers.rs:1697`, `:1702`).

## Two entry points per factorization: owning, and in place

Every factorization exists twice, and the choice is the difference between "simple" and "no
allocation".

- **Owning**: `a.partial_piv_lu(par)` copies `A` (`A.to_owned()`, `solvers.rs:1195`), allocates the
  factors, and returns an object that also implements the solve traits. Use this unless a measured
  allocation or copy matters.
- **In place**: `lu_in_place(A, perm, perm_inv, par, stack, params)`
  (`faer/src/linalg/lu/partial_pivoting/factor.rs:625`) overwrites the caller's `MatMut` and writes
  the permutation into the caller's slices. It takes the workspace as a `&mut MemStack` and returns
  `(Info, PermRef)`. The QR, Cholesky and SVD equivalents are `qr_in_place(...)`
  (`faer/src/linalg/qr/no_pivoting/factor.rs:407`), `cholesky_in_place(...)`
  (`faer/src/linalg/cholesky/llt/factor.rs:67`), and `svd(...)`
  (`faer/src/linalg/svd/mod.rs:741`).

The fork keeps faer's `*_in_place` names and adds no `*_into` variants; do not rename them in review
(`docs/review-rules.md:14`). The in-place form is the one that allocates nothing: it is what the
owning wrappers call internally, with a `MemBuffer` they build themselves (`solvers.rs:1205`).

### Sizing the workspace

Ask for the workspace, do not guess it:

```rust
let req = qr_in_place_scratch::<f64>(m, n, block_size, par.shape(), default());
let mut mem = MemBuffer::new(req);
let mut stack = MemStack::new(&mut mem);
```

`*_scratch` functions take a `ParShape` (the width, without the authority to run anything), not a
`Par` (`docs/par-team-migration.md:44`). Combine several requirements with `StackReq::any_of(&[...])`,
as `faer/src/linalg/qr/mod.rs:63` does. `MemStack`, `MemBuffer` and `StackReq` are re-exported from
`faer::dyn_stack` (`faer/src/lib.rs:1431`). A caller-held `MemBuffer` reused across calls is the
route to a steady-state call that allocates nothing; `faer/tests/svd_values_no_alloc.rs` and
`faer/tests/no_step_alloc.rs` are the contracts.

## Solving

Import the trait: `use faer::linalg::solvers::Solve;`. Two families:

| You want | Method | Allocates |
| --- | --- | --- |
| Solve into a new matrix | `solve(&b, par)` (`solvers.rs:231`), also `solve_transpose` (`:259`), `solve_adjoint` | yes, `Rhs::Owned` |
| Solve into a buffer you own | `solve_in_place(&mut x, par)` (`solvers.rs:119`), also the `*_transpose_in_place` / `*_adjoint_in_place` / `rsolve_*_in_place` forms | no |
| Least squares | `solve_lstsq(&b, par)` (`solvers.rs:851`), trait `SolveLstsq` (`:819`) | the owned one |
| Reconstruct, inverse | `reconstruct(par)`, `inverse(par)` (`solvers.rs:55`, `:58`) | yes |

`solve` asserts that the factorised matrix is square (`solvers.rs:2401`), so a rectangular system is a
`solve_lstsq` call, not a `solve` one. `Qr`, `ColPivQr` and `Svd` implement `SolveLstsqCore`
(`solvers.rs:2655`, `:2793`, `:2960`).

## Errors

Which calls can fail is a property of the algorithm, not of the spelling:

- `SvdError::NoConvergence` (`faer/src/linalg/svd/mod.rs:32`)
- `EvdError::NoConvergence` (`faer/src/linalg/evd/mod.rs:35`)
- `LltError::NonPositivePivot { index }` (`faer/src/linalg/cholesky/llt/factor.rs:22`)
- `LdltError::ZeroPivot { index }` (`faer/src/linalg/cholesky/ldlt/factor.rs:691`)
- `SelfAdjointGevdError::{NoConvergence, NonPositiveDefinite}` (`faer/src/linalg/gevd/mod.rs:43`)
- `GevdError::NoConvergence` (`faer/src/linalg/gevd/mod.rs:37`)

LU, QR and LBLT return no `Result`: a singular pivot is a property of the factor, not an error. Read
`rank` from the info of the in-place call for a truncated LU
(`PartialPivLuInfo::rank`, `faer/src/linalg/lu/partial_pivoting/factor.rs:568`).

## Numerical rank and truncation

There is no `truncated_svd` and no rank argument on the owning methods.

- **LU**: rank truncation exists on the in-place entry point only, through `PartialPivLuParams` /
  `FullPivLuParams` (`max_rank`, `abs_tol`, `rel_tol`;
  `faer/src/linalg/lu/partial_pivoting/factor.rs:534`, `:551`–`:556`). Pass
  `PartialPivLuParams { max_rank: Some(r), ..<PartialPivLuParams as Auto<f64>>::auto() }.into()` as
  the `params` argument. The threshold is `max(abs_tol, rel_tol · |U[0,0]|)` and a pivot is rejected
  when `|U[k,k]| <= threshold`; `rel_tol` is measured against the **first** pivot, not the previous
  one (`faer/src/linalg/lu/partial_pivoting/factor.rs:503`–`:530`). A truncated factorization must not
  be handed to the solve, inverse or reconstruct routines of that module.
- **Column-pivoted QR**: no rank parameter yet. Take `R()` and decide the numerical rank from the
  diagonal yourself, or use a truncated SVD.
- **SVD**: `thin_svd` (`solvers.rs:442`) gives `U`, `S`, `V` with `min(m, n)` columns; a rank-`r`
  approximation is the leading `r` columns of `U`, `S` and `V`, which you slice with
  `submatrix(0, 0, .., r)` on the returned views. `Svd::pseudoinverse` (`solvers.rs:1619`) is the
  only built-in rank-aware operation.

## Threads: the caller owns the team

`Par` is the right to run work on a set of threads, and it is neither `Copy` nor `Clone`
(`faer/src/lib.rs:1070`). There is no global thread-pool setting.

| Route | Use |
| --- | --- |
| `Par::Seq` | one call on the calling thread; always correct |
| build a `threadteam::Pool` once, `pool.into_team(w)`, `team.exec(w)` per call | the primary route: a budget held across many calls |
| `faer::team::with_team(w, \|par\| ...)` (`faer/src/utils/team.rs:97`) | a single call, or a script |

Inside a region, lend the token to every call but the last: `par.reborrow()` (as in
`docs/migration/README.md`'s checked example). Nesting does not add threads: split the team first
(`token.split(&[..], |_, child| ...)`) and give each child its own `Par::Team`; do not capture the
parent inside a child, and do not re-acquire the same team recursively — `threadteam::Team::exec`
panics on a thread that already holds a token of that team.

## Layout

Column-major is native (`Mat` is column-major; `MatRef`/`MatMut` may be strided).

- Row-major input is adapted internally by LU, QR and Cholesky, by factoring a column-major copy when
  `nrows · ncols >= 64 · 64` (`faer/src/linalg/mod.rs:71`, `faer/src/linalg/lu/partial_pivoting/factor.rs:649`,
  `faer/src/linalg/qr/col_pivoting/factor.rs:381`). You get the right answer either way; the copy is
  the price.
- SVD, EVD and GEVD warn (with the `perf-warn` feature) when the output vector matrices are row-major
  or generically strided (`faer/src/linalg/svd/mod.rs:771`, `faer/src/linalg/evd/mod.rs:342`,
  `faer/src/linalg/gevd/mod.rs:203`). Allocate eigenvector outputs as ordinary `Mat`s.
- Generic strides are slow everywhere. Copy to a plain `Mat` before a factorisation if a view has an
  awkward stride.
- Build a borrowed row-major view with `MatRef::from_row_major_slice(&s, m, n)`
  (`faer/src/mat/matref.rs:273`); an owned `Mat` from a function with `Mat::from_fn(m, n, |i, j| ..)`
  (`faer/src/mat/matown.rs:381`).

## Pitfalls

- **Package versus library name.** `t4a-faer` in `Cargo.toml`, `faer` in `use`. A plain
  `faer = "0.32"` pulls upstream, not this crate.
- **Upstream code does not compile here.** `set_global_parallelism`, `get_global_parallelism` and
  `Par::Rayon` are gone; the factorization methods take a trailing `Par`.
- **`A = U S V^H`.** `Svd::V()` returns `V`, so a reconstruction is `U · S · V.adjoint()`, not
  `V^T` (`faer/src/linalg/svd/mod.rs:11`). For complex data `adjoint()` conjugates and `transpose()`
  does not.
- **Permutation convention.** `P()` is a `PermRef`; `P · A` has row `i` equal to `A`'s row
  `perm[i]` (`faer/src/perm/mod.rs:282`), and `arrays()` gives the forward and inverse slices
  (`faer/src/perm/permref.rs:143`). Apply it to vectors by indexing, `x[perm[i]]`.
- **`ldlt` has no permutation.** It is unpivoted; a matrix that needs a pivot fails with
  `ZeroPivot`. Use `lblt` for indefinite problems (`solvers.rs:471`).
- **`Side` selects the triangle that is read**, and the other triangle is not checked for symmetry.
- **`matmul` has no `beta`.** `Accum::Replace` or `Accum::Add` only (`faer/src/lib.rs:946`); scale the
  destination first for a general `beta`.
- **`solve` copies.** For a loop over right-hand sides, factor once and call `solve_in_place`, or
  reuse the in-place factorization.

## A typical call sequence

Verified against this tree; the CI-run counterpart is `faer/examples/migration_overview.rs`.

```rust
use faer::linalg::solvers::{Solve, SolveLstsq};
use faer::{Par, mat};

fn main() {
	let a = mat![[4.0_f64, 1.0], [1.0, 3.0]];
	let b = mat![[6.0_f64], [7.0]];
	let lu = a.partial_piv_lu(Par::Seq);
	let x = lu.solve(&b, Par::Seq);
	assert!((x[(0, 0)] - 1.0).abs() < 1e-12);
	assert!((x[(1, 0)] - 2.0).abs() < 1e-12);

	// Tall least squares: the solution of A^T A x = A^T b.
	let tall = mat![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0]];
	let rhs = mat![[1.0_f64], [2.0], [3.0]];
	let qr = tall.qr(Par::Seq);
	let x = qr.solve_lstsq(&rhs, Par::Seq);
	assert!((x[(0, 0)] - 1.0).abs() < 1e-12);
	assert!((x[(1, 0)] - 2.0).abs() < 1e-12);
}
```

## Where to read more

- `docs/migration/README.md`: dependency setup, parallel regions, nested teams, a checked example.
- `docs/migration/`: per-ecosystem guides (NumPy/SciPy, Eigen/BLAS, LAPACK, Julia, nalgebra, ndarray,
  upstream faer) and the flat LAPACK-name table.
- `llms.txt`: the documentation index.
- `faer/examples/migration_*.rs`: compiled and run by `.github/workflows/ci.yml`.
