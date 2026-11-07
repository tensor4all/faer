# LAPACK names: one flat table

One row per LAPACK routine, with the entry point this fork provides. The Fortran guide
([fortran.md](fortran.md)) explains the calling conventions; this page is the lookup table an agent
or a person needs when it knows the LAPACK name and nothing else.

`?` means the four standard precisions (`s`, `d`, `c`, `z`). This fork is generic over the scalar
type, so one Rust function serves all four where the algorithm applies to complex data.

| LAPACK | Operation | faer entry point | Defined at |
| --- | --- | --- | --- |
| `?getrf` | LU with partial (row) pivoting, `P A = L U` | `MatRef::partial_piv_lu(par)`, or `lu_in_place` | `faer/src/linalg/solvers.rs:408`, `faer/src/linalg/lu/partial_pivoting/factor.rs:625` |
| `?getc2` | LU with complete pivoting | `MatRef::full_piv_lu(par)`, or `lu_in_place` | `faer/src/linalg/solvers.rs:414`, `faer/src/linalg/lu/full_pivoting/factor.rs:581` |
| `?getrs`, `?gesv` | Solve from an existing LU / factor and solve | `Solve::solve_in_place`, `Solve::solve` | `faer/src/linalg/solvers.rs:119`, `:231` |
| `?getri` | Inverse from a factorization | `DenseSolve::inverse` | `faer/src/linalg/solvers.rs:58` |
| `?gecon` | Reciprocal condition estimate | **No equivalent** | — |
| `?geqrf` | QR factorization | `MatRef::qr(par)`, or `qr_in_place` | `faer/src/linalg/solvers.rs:420`, `faer/src/linalg/qr/no_pivoting/factor.rs:407` |
| `?orgqr`, `?ungqr`, `?ormqr` | Form or apply `Q` | `Qr::compute_Q`, `Qr::compute_thin_Q` | `faer/src/linalg/solvers.rs:1382`, `:1403` |
| `?geqp3` | Column-pivoted QR, `A P^T = Q R` | `MatRef::col_piv_qr(par)`, or `qr_in_place` (column pivoting) | `faer/src/linalg/solvers.rs:426`, `faer/src/linalg/qr/col_pivoting/factor.rs:372` |
| `?gelqf` | LQ factorization | `lq_in_place` | `faer/src/linalg/lq/no_pivoting/factor.rs:40` |
| `?gels` | Least squares through QR | `Qr::solve_lstsq` (`SolveLstsq`) | `faer/src/linalg/solvers.rs:851` |
| `?gelsd`, `?gelss` | Least squares / minimum norm through SVD | `Svd::solve_lstsq` (`SolveLstsq`) | `faer/src/linalg/solvers.rs:851` |
| `?potrf` | Cholesky, `A = L L^H` | `MatRef::llt(side, par)`, or `cholesky_in_place` | `faer/src/linalg/solvers.rs:451`, `faer/src/linalg/cholesky/llt/factor.rs:67` |
| `?pstrf` | Cholesky with complete pivoting and a rank stop | `cholesky_in_place` (pivoted LLT) | `faer/src/linalg/cholesky/llt_pivoting/factor.rs:47` |
| `?potri` | Inverse from a Cholesky factor | `DenseSolve::inverse` on the `Llt` object | `faer/src/linalg/solvers.rs:58` |
| `?sytrf`, `?hetrf` | Symmetric indefinite, Bunch–Kaufman | `MatRef::lblt(side, par)`, or `cholesky_in_place` (Bunch–Kaufman) | `faer/src/linalg/solvers.rs:471`, `faer/src/linalg/cholesky/bunch_kaufman/factor.rs:1204` |
| `?sytrd`, `?hetrd` | Tridiagonal reduction | `tridiag_in_place` | `faer/src/linalg/evd/tridiag.rs:278` |
| `?gehrd` | Hessenberg reduction | `hessenberg_in_place` | `faer/src/linalg/evd/hessenberg.rs:567` |
| `?gebal` | Balancing | `balance_in_place`, `balance_eigenvectors_in_place` | `faer/src/linalg/evd/balance.rs:86`, `:245` |
| `?syev`, `?heev`, `?syevd`, `?heevd` | Symmetric eigenproblem | `MatRef::self_adjoint_eigen(side, par)`, `self_adjoint_eigenvalues` | `faer/src/linalg/solvers.rs:479`, `:545` |
| `?syevr`, `?heevr` | Symmetric eigenproblem with a selected range | **No equivalent**: no range or Sturm-count interface | `docs/lapack-validation.md:46` |
| `?sygv`, `?hegv` | Generalized symmetric eigenproblem | `MatRef::self_adjoint_generalized_eigen(b, par)` | `faer/src/linalg/solvers.rs:498` |
| `?stebz`, `?stein` | Bisection / inverse iteration on a tridiagonal | **No equivalent** | — |
| `?gesvd`, `?gesdd` | SVD | `MatRef::svd(par)`, `thin_svd`, or `svd` | `faer/src/linalg/solvers.rs:434`, `:442`, `faer/src/linalg/svd/mod.rs:741` |
| `?bdsqr` | SVD of a bidiagonal matrix | `bidiagonal_svd` | `faer/src/linalg/svd/mod.rs:460` |
| `?geev` | Nonsymmetric eigenproblem | `MatRef::eigen(par)` | `faer/src/linalg/solvers.rs:805` |
| `?ggev` | Generalized nonsymmetric eigenproblem | `MatRef::generalized_eigen(b, par)` | `faer/src/linalg/solvers.rs:795` |
| `?gees` | Schur factorization | `multishift_qr` (real and complex) | `faer/src/linalg/evd/schur/real_schur.rs:2427`, `faer/src/linalg/evd/schur/complex_schur.rs:643` |
| `?trevc` | Eigenvectors from a Schur form | `eigenvectors_from_schur_real`, `eigenvectors_from_schur_cplx` | `faer/src/linalg/evd/mod.rs:1797`, `:1866` |
| `?trtri` | Triangular inverse | `invert_lower_triangular`, `invert_upper_triangular` | `faer/src/linalg/triangular_inverse.rs:171`, `:216` |
| `?trsv`, `?trsm` | Triangular solve | `solve_lower_triangular_in_place`, `solve_upper_triangular_in_place` | `faer/src/linalg/triangular_solve.rs:283`, `:385` |
| `?larfg` | Generate a Householder reflector | `make_householder_in_place` | `faer/src/linalg/householder.rs:117` |
| `?lange`, `?lansy` | Matrix norms | `MatRef::norm_l2`, `norm_max`, `norm_l1` | `faer/src/mat/matref.rs:1669`, `:1658`, `:1691` |
| `?laexc`, `?trexc` | Swap blocks in a Schur form | `reorder_schur_real`, `reorder_schur_cplx` | `faer/src/linalg/gevd/reorder.rs:661`, `:722` |
| `?getf2` | Unblocked LU panel | Not separately exposed; `lu_in_place` runs it internally | `faer/src/linalg/lu/partial_pivoting/factor.rs:625` |
| `?gesdd` with `jobz='O'` and a workspace query | Optimal-workspace SVD | `svd_scratch` sizes the workspace; there is no `lwork = -1` query | `faer/src/linalg/svd/mod.rs:677` |

## Storage and leading dimensions

LAPACK takes a column-major buffer and an explicit leading dimension. A `MatRef` carries both strides
and a shape, so there is no `lda` argument: pass the view, and the stride is part of it. A
`MatRef` from `from_column_major_slice(&s, m, n)` is the exact equivalent of a LAPACK array with
`lda = m` (`faer/src/mat/matref.rs:207`).

Row-major input does not need a transposed copy at the call site: LU, QR and Cholesky detect it and
factor a column-major copy internally above `64 * 64` elements (`faer/src/linalg/mod.rs:71`). LAPACK's
own row-major interface does the same copy, so this is the same cost, made invisible.

## In-place, scratch and `Result`

- Every factorisation exists as an owning method (`a.qr(par)`, which copies and allocates) and as an
  in-place function (`qr_in_place(a, h, par, stack, params)`, which writes into the caller's buffers).
  The in-place form is what a LAPACK port wants: it is the one that takes a workspace.
- LAPACK's `lwork` query becomes `*_scratch(..)` returning a `StackReq`, which the caller combines
  and passes as a `MemStack`; a `MemBuffer` the caller keeps across calls is the "reuse the
  workspace" idiom.
- LAPACK reports through `info`. Here the LU and QR calls are infallible and report through their
  info struct (`PartialPivLuInfo { transposition_count, rank }`,
  `faer/src/linalg/lu/partial_pivoting/factor.rs:562`); `llt`, `ldlt`, `svd`, `evd` and the
  generalized routines return `Result` with an enum per family
  (`faer/src/linalg/solvers.rs:7`–`:12`).
- The parallel argument is a `Par`, and it is the caller's budget, not a thread count setting
  (`faer/src/lib.rs:1070`).

## Open gaps

Condition estimation (`gecon`, `trcon`, `pocon`), `?syevr`'s range and Sturm-count interface,
`?stebz`/`?stein`, the expert drivers (`?gesvx`, `?gelsy`, `?ggglm`), packed and banded storage, and
the `lwork = -1` query convention have no equivalent. `docs/lapack-validation.md` lists which LAPACK
test families were ported and which were deliberately omitted.
