# From upstream faer

This fork is upstream `faer` with a different parallel API and a set of correctness changes. The
matrix types, the decomposition names and the low-level modules are the same, which is exactly why
upstream examples are the most likely thing to paste in and fail: the code looks right, and the
signatures have moved.

| Upstream faer | This fork |
| --- | --- |
| `faer = "0.2x"` in `Cargo.toml` | `faer = { package = "t4a-faer", version = "0.32" }`; the library name is still `faer` (`faer/Cargo.toml:17`). |
| `faer::get_global_parallelism()` / `set_global_parallelism(..)` | Removed. Every call that runs work takes a trailing `Par` (`docs/par-team-migration.md:94`); `Par::Seq` is the sequential one. |
| `Par::Rayon(n)` | Removed. `Par` has `Seq` and `Team` only (`faer/src/lib.rs:1070`). |
| `faer::get_global_parallelism().degree()` | `par.degree()`, or `ParShape` for a workspace query (`faer/src/lib.rs:1117`, `:1141`). |
| `a.svd()` | `a.svd(par)` (`faer/src/linalg/solvers.rs:434`). |
| `a.llt(Side::Lower)` | `a.llt(Side::Lower, par)` (`faer/src/linalg/solvers.rs:451`). |
| `lu.solve(&b)` | `lu.solve(&b, par)` (`faer/src/linalg/solvers.rs:231`). |
| `lu.reconstruct()` / `lu.inverse()` | `lu.reconstruct(par)` / `lu.inverse(par)` (`faer/src/linalg/solvers.rs:55`, `:58`). |
| `a.determinant()` | `a.determinant(par)` (`faer/src/mat/matref.rs:1718`). |
| `svd_scratch::<T>(m, n, .., par)` | The scratch functions take `ParShape`, not `Par` (`faer/src/linalg/svd/mod.rs:677`). Pass `par.shape()`. |
| `&a * &b` | Still compiles, and runs sequentially: the operators cannot take a `Par` (`docs/par-team-migration.md:110`). |
| `Mat`, `MatRef`, `MatMut`, `mat!`, `Side`, `Accum`, `Conj` | Unchanged in name and role (`faer/src/lib.rs:933`, `:946`, `:954`, `:494`). |
| `faer::linalg::solvers::Solve` | Unchanged, and still needs importing (`faer/src/linalg/solvers.rs:115`). |

## What it takes to port a call site

1. **Add the `par`.** Every high-level method that runs work takes one as the last argument: the
   constructors, `solve*`, `reconstruct`, `inverse`, `pseudoinverse`, `compute_Q`,
   `compute_thin_Q`. A call that only reads a factor (`lu.L()`, `svd.U()`) does not.
2. **Replace the global setting with a team.** Build a `threadteam::Pool` once, turn it into a team
   with `pool.into_team(width)`, and take a token per call inside `Exec::team(|t| ...)`; or use
   `faer::team::with_team(width, |par| ...)` for one call (`faer/src/utils/team.rs:97`).
3. **Pass `par.shape()` to scratch queries.** `ParShape` is `Copy` and carries the width without the
   authority to run (`faer/src/lib.rs:1089`).
4. **Fix the reuse of one `par`.** `Par` is neither `Copy` nor `Clone`
   (`faer/src/lib.rs:1070`): lend it with `par.reborrow()` for every call but the last, and do not
   hand the same token to two concurrent children.
5. **Check the `Result` types where the algorithm can fail.** They are unchanged in shape
   (`Result<Svd, SvdError>` for the SVD, `Result<Llt, LltError>` for Cholesky), but a rank-deficient
   or non-positive-definite input that upstream accepted may now be a documented error.

## Pitfalls

- **Upstream documentation is not a source for these signatures.** The [migration
  overview](README.md#dependency-and-upstream-api-differences) says the same thing from the other
  direction. Do not copy `Par::Rayon(8)` or `set_global_parallelism` from an upstream example.
- **A `par` in the middle of a closure chain.** `par` is consumed by the call; anything after it in
  the same expression that also needs the token has to use `par.reborrow()` first. `par.shape()` is
  cheap and `Copy`, so read it before the lend, not after (`docs/par-team-migration.md:78`).
- **Operators are sequential.** `&a * &b` still exists for convenience and always runs on the calling
  thread (`docs/par-team-migration.md:110`). For a team, call `matmul(.., par)`.
- **SVD vectors.** `Svd::V()` returns `V`, with `A = U S V^H`
  (`faer/src/linalg/svd/mod.rs:11`). Check which side of the old upstream API a stored buffer came
  from before assuming `V^T`.
- **Behavioural fixes, not just API.** The fork carries fixes for defects reproduced in the
  upstream releases (`upstream-reports/README.md`, 40 entries at the time of writing). Code that
  depended on the old numbers, tolerances or `NaN` outputs can see different results; the entries
  name the symptom and the fix.
- **Fork-only additions.** Rank truncation on the LU (`max_rank`, `abs_tol`, `rel_tol` in
  `PartialPivLuParams`, `faer/src/linalg/lu/partial_pivoting/factor.rs:534`) and the public
  Schur/reordering/balancing wrappers are not in upstream's API. Do not assume they exist in an
  upstream build, and do not assume upstream's equivalents exist here.
- **Feature names survive the rename.** `features = ["std", "rayon", "threadteam"]` keeps working;
  `rayon` remains a feature because the team machinery uses it, but it no longer means "the global
  Rayon pool" (`faer/Cargo.toml:116`).

## Open gaps

There is no compatibility shim: no `set_global_parallelism` accepting a width and silently handing
out a team, and no `Par::Rayon` variant. A port is explicit work, and the compiler is the checklist —
every missing `par` is a type error at the call site.
