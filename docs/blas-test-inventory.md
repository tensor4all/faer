# OpenBLAS BLAS-level test inventory

Bounded Rust adaptation of the checks in OpenBLAS `test/` and `ctest/`, as a
follow-up to [#31](https://github.com/tensor4all/faer/issues/31) (the LAPACK
`lapack-netlib/TESTING` validation in [lapack-validation.md](lapack-validation.md))
and part of [#166](https://github.com/tensor4all/faer/issues/166). Port
completion and numerical success are separate: this file records what was
ported, what has no counterpart, and what stays unported in the later batches.

## Source and licence

OpenBLAS revision `791b02a1a00c996b66a894f6df93421c53b62d80`, directories
[`test/`](https://github.com/OpenMathLib/OpenBLAS/tree/791b02a1a00c996b66a894f6df93421c53b62d80/test)
and
[`ctest/`](https://github.com/OpenMathLib/OpenBLAS/tree/791b02a1a00c996b66a894f6df93421c53b62d80/ctest).
The `test/` programs are the netlib reference BLAS test programs (Jack Dongarra,
Iain Duff, Jeremy Du Croz, Sven Hammarling, 1989-2012); `ctest/` is the CBLAS
translation of the same checks plus row-major wrappers and argument-error tests.
Both directories ship the same licence file, quoted in
[`faer/COPYING.BLAS.NETLIB`](../faer/COPYING.BLAS.NETLIB). It asks for credit
rather than imposing a redistribution condition, and no Fortran is translated,
so the crate's `license` metadata is unchanged and `NOTICE` carries the credit.

Adapted files and the material changes to each are recorded in their headers:

| Rust file | Source check |
| --- | --- |
| `faer/src/linalg/tests/blas_support.rs` | the shared oracles: `DMMCH` (independent product and ratio), `DMAKE` (rogue value in the part a routine must not access), `LDE`/`LDERES` (operand unchanged, referenced half only) |
| `faer/src/linalg/matmul/tests/blas_trmm.rs` | `test/blat3.f` `DCHK3`, `DTRMM` |
| `faer/src/linalg/matmul/tests/blas_rank_update.rs` | `test/blat2.f` `DCHK4` (`DGER`) and `DCHK5` (`DSYR`); `test/blat3.f` `DCHK4` (`DSYRK`) |
| `faer/src/linalg/tests/blas_trsm_cases.rs` | `test/blat3.f` `DCHK3` (`DTRSM`); `test/blat2.f` `DCHK3` (`DTRSV`) |

Material adaptations, common to the three suites: the source's random
`DBEG`/`DMAKE` sequence and its dimension sweep (`N = 0,1,2,3,7,31,63`, `LDA =
N+1` padding, `alpha` 0.0/0.7/1.0) are replaced by bounded shapes with an
exact-integer fixture and exact binary `alpha`, so the reference product is
compared exactly instead of through `DMMCH`'s eps ratio; and the rogue value
(`ROGUE = -1.0D10`) becomes NaN, which is the stronger sentinel. The source's
"operand unchanged" and "leading-dimension padding" checks are kept as
`exact`/`untouched_outside`. The two ports that put a written operand in a
guarded parent are `DTRMM` and `DTRSM`: there the read operand and the written
operand each sit in a rogue-filled parent with one row and one column of padding
on each side, so the operand equality and the padding check both run against the
whole parent, and the source itself only pads the leading dimension of the
arrays it passes in. `DGER`/`DSYR`/`DSYRK` allocate their destination
standalone and check only the entries the routine must not write (the
referenced-half rule) -- no surrounding buffer there. faer's strict-triangle
block structures have no BLAS `DIAG` value; they are covered because faer
exposes them.

## Inventory

"Port?" is this batch only. Rows marked *covered* have a faer unit test that
already exercises the source's check; a reason says which.

### Level 1 — `test/{s,d,c,z}blat1.f` (`CHECK0`-`CHECK3`), `ctest/c_{s,d,c,z}blas1.c`

| OpenBLAS file/check | faer counterpart | Port? | Reason |
| --- | --- | --- | --- |
| `CHECK1` `ISAMAX`/`IDAMAX` (index of the largest modulus) | none (`reductions::norm_max` returns the value, not the index) | no | faer has no index-reduction API; nothing to call |
| `CHECK1` `SASUM`/`DASUM`, `SCASUM`/`DZASUM` | `reductions::norm_l1` (`abs1`, so the complex entries are `\|re\|+\|im\|`) | covered | `norm_l1::tests::test_norm_l1` runs its contiguous `f64` fixtures and, through `check_strided_complex`, a `c64` and a `c32` vector whose row stride is above 1 (the source's `incx = 2` half, which takes the strided fallback rather than the SIMD path). The source's 5-entry lengths and its negative strides are not ported |
| `CHECK2` `SDOT`/`DDOT`/`CDOTU`/`ZDOTU`, `CDOTC`/`ZDOTC` | `matmul::dot::inner_prod` | no | the counterpart exists, but it has no unit test of its own (it is exercised only through the decompositions that call it), so the source's 5-entry dot over `incx`/`incy` in {1, 2, -2, -1} is a candidate for the next batch, not covered |
| `CHECK2`/`CHECK3` `SAXPY`/`SCOPY`/`SSWAP`/`SSCAL`/`SROT`/`SROTG`, `CSSCAL`/`ZDSCAL`/`CROTG`/`CSROT` | none | no | no level-1 vector API in faer (the source sweeps `incx`/`incy` over {1, 2, -2, -1}) |
| `CHECK1` `SNRM2`/`DNRM2`/`SCNRM2`/`DZNRM2` | `reductions::norm_l2`, `MatRef::norm_l2`, `ColRef::norm_l2` | covered | the source check is `n = 0..4` with `incx` 1/2 and small values; `norm_l2::tests::test_norm_l2_strided_range` already reads a stride-2 row view, and the exponent-range half of the property is [#36](https://github.com/tensor4all/faer/issues/36) / [faer-008](../upstream-reports/faer-008-norm-l2-strided-range/README.md) |

### Level 2 — `test/{s,d,c,z}blat2.f` (`DCHK1`-`DCHK6`), `ctest/c_*blat2c.c`

| OpenBLAS file/check | faer counterpart | Port? | Reason |
| --- | --- | --- | --- |
| `DCHK1` `GEMV`/`GBMV` | `matmul::matmul` with one destination column (the `matvec` route) | no | the route is exercised by `matmul`'s own tests; `GBMV`'s banded storage has no counterpart |
| `DCHK2` `SYMV`/`SBMV`/`SPMV` | none | no | faer's structured matmul carries triangular block structures only; a symmetric operand is materialized dense, and banded/packed storage has no API |
| `DCHK3` `TRMV`/`TBMV`/`TPMV` | `matmul::triangular::matmul_with_conj` with a `Rectangular` destination | ported (`blas_trmm.rs`) | the TRMM shapes carry it; the `Rectangular` case with a triangular operand is the level-2 multiply. Banded/packed variants have no counterpart |
| `DCHK3` `TRSV`/`TBSV`/`TPSV` | `triangular_solve::solve_{,unit_}{lower,upper}_triangular_in_place_with_conj` | ported (`blas_trsm.rs`) | the `nrhs = 1` cases are TRSV. Banded/packed variants have no counterpart |
| `DCHK4` `GER` | `matmul::matmul` with `K = 1` (the private `rank_update` module) | ported (`blas_rank_update.rs`) | the exact rank-1 update, including both destination layouts |
| `DCHK5` `SYR`/`SPR` | `matmul_with_conj` with a triangular destination and `K = 1` | ported (`blas_rank_update.rs`) | `SPR`'s packed storage has no counterpart |
| `DCHK6` `SYR2`/`SPR2` | none | no | two rank-1 terms, i.e. two calls; there is no single kernel to test |
| `ZCHK4` `ZGERC`/`ZGERU` (complex only, `zblat2.f:165-166`) | `matmul::matmul` with `K = 1` (`ZGERU`, unconjugated) and `matmul::triangular::matmul_with_conj` with `Conj::Yes` on the row operand (`ZGERC`) | `ZGERU` covered, `ZGERC` no | the complex rank-1 test runs the unconjugated `K = 1` path, which is the `ZGERU` half of the pair. The conjugated form has the same counterpart with `Conj::Yes` but no test, so it is a next-batch candidate, not covered |
| `ZCHK5` `ZHER`/`ZHPR` (complex only, `zblat2.f:167-168`) | `matmul_with_conj` with a triangular destination and `Conj::Yes` on the row operand | no | the `SYR`/`SPR` shape with the row operand conjugated; `ZHPR`'s packed storage has no counterpart. Deliberately unported in this batch |
| `ZCHK6` `ZHER2`/`ZHPR2` (complex only, `zblat2.f:169`) | none | no | two rank-1 terms, i.e. two calls, and `ZHPR2` is packed |
| `ZCHK2` `ZHEMV`/`ZHBMV`/`ZHPMV` (complex only) | none | no | the Hermitian matvec siblings of the `SYMV` row above; the banded and packed storage has no counterpart |

The complex `blat2` file numbers its groups differently from the real one: `ZCHK2`
is the Hermitian matvec, `ZCHK3` carries all six triangular names (as `DCHK3`
does), and the complex-only rank updates are `ZCHK4` (`GERC`/`GERU`), `ZCHK5`
(`HER`/`HPR`) and `ZCHK6` (`HER2`/`HPR2`), from the `SNAMES` list at
`zblat2.f:165-169` and the dispatch at `zblat2.f:343-345`.

### Level 3 — `test/{s,d,c,z}blat3.f` (`DCHK1`-`DCHK5`), `ctest/c_*blat3c.c`

| OpenBLAS file/check | faer counterpart | Port? | Reason |
| --- | --- | --- | --- |
| `DCHK1` `GEMM` | `matmul::matmul`, `matmul::triangular::matmul_with_conj` (both `Rectangular`) | no | `matmul::tests::test_matmul` covers the product but is `#[ignore]`d as too slow; the bounded non-ignored GEMM case is a candidate for the next batch, and this batch's TRMM/rank-k suites already run the dense path as their reference |
| `DCHK2` `SYMM` | none | no | no symmetric structured multiply |
| `DCHK3` `TRMM` | `matmul::triangular::matmul_with_conj` | ported (`blas_trmm.rs`) | both sides, `N`/`T`/`C`, lower/upper, and the generic/strict/unit diagonal; the source's ROGUE diagonal and ROGUE unreferenced triangle are the checked property, and the destination is a padded submatrix of a ROGUE parent |
| `DCHK3` `TRSM` | `triangular_solve::solve_{,unit_}{lower,upper}_triangular_in_place_with_conj` | ported (`blas_trsm.rs`) | both sides and `N`/`T`/`C`; `alpha` has no parameter in faer, so the fixture scales the right-hand side by `alpha` before the solve, which is the same computation; the in-place right-hand side is a padded submatrix of a ROGUE parent |
| `DCHK4` `SYRK` | `matmul_with_conj` with `TriangularLower`/`TriangularUpper` destination | ported (`blas_rank_update.rs`) | the referenced half is computed and the other half must stay untouched (`LDERES 'SY'`); the four TRANS/conjugation arms are `A A^T`, `A^T A`, `A^H A` and `A A^H`, and the `n = 3, k = 1` shape carries `SYR`'s non-diagonal rank-1 update |
| `DCHK5` `SYR2K` | none | no | `A*B^T + B*A^T` is two calls of the same kernel |
| `ZCHK4` `ZHERK` (complex only, `zblat3.f:146`) | `matmul_with_conj` with a triangular destination and `Conj::Yes` on the second operand | covered (`blas_rank_update.rs`) | the `A A^H` arm; `ZSYRK` (`A A^T`) is the unconjugated sibling and the same test covers it in every dtype, so conjugation and its absence are separate cases rather than one case standing in for both |
| `ZCHK5` `ZHER2K` (complex only, `zblat3.f:147`) | none | no | `A*B^H + B*A^H` is two calls of the same kernel |
| `_3m` variants (`cblat3_3m.f`, `zblat3_3m.f`) | none | no | faer has no 3m algorithm |
| `DMAKE`/`DMMCH`/`LDE`/`LDERES` helpers | `tests/blas_support.rs` (`entry`, `product`, `blended`, `exact`, `untouched_outside`) | adapted | reimplemented, not translated: the reference product is explicit, and the comparison is exact |

### `ctest/` specifics and OpenBLAS-only files

| OpenBLAS file/check | faer counterpart | Port? | Reason |
| --- | --- | --- | --- |
| CBLAS row-major wrappers `c_{s,d,c,z}blas{1,2,3}.c` | views: `MatRef::transpose`, `MatMut::transpose_mut` | partly | the rank-1 test also runs a row-major destination (`transpose_mut` on a column-major buffer), which is the second `rank_update` branch; the numerical checks are the `test/` ones |
| argument-error tests (`cblas_test.h`, `c_xerbla.c`, `blat2.f` `DCHKE`, `blat3.f` `DCHKE`) | `assert!`/panic preconditions in `matmul`, `matmul_with_conj`, `triangular_solve` | no | faer rejects a bad shape with a panic and has no XERBLA-style error return or info code |
| `compare_sgemm_bgemm.c`, `compare_sgemm_sbgemm.c`, `compare_sgemm_shgemm.c`, `compare_sgemv_*.c` | none | no | bfloat16/half mixed-precision kernels do not exist in faer |
| `xerbla_test.c`, `shgemm_test.c`, `test_laed3_nan.c`, `get_threading_model.c`, `wasm/` | none | no | OpenBLAS build and runtime specifics (XERBLA, SME kernels, its bundled LAPACK, thread-model query, wasm build) |

## What this batch establishes

Six test functions over four scalar types (`f64`, `f32`, `c64`, `c32`), 24 tests
in the default (non-nightly) run:

| Test | Property |
| --- | --- |
| `matmul::blas::*::trmm_matches_an_independent_product` | the product of a triangular operand equals an independently accumulated product over the bounded shape set, for every lower/upper, generic/strict/unit, `N`/`T`/`C`, left/right and `alpha` combination |
| `matmul::blas::*::trmm_reads_and_writes_nothing_outside_its_operands` | the operand's unreferenced triangle and implicit diagonal (NaN) are never read; the operand and the destination each sit in a padded submatrix of their own ROGUE-filled parent, the destination is checked entry by entry against an independent product computed from the operand's pre-call snapshot, and both whole parents (padding and stored operand alike) must come back unchanged |
| `matmul::blas::*::rank_one_update_matches_the_source_contract` | `A = alpha*x*y^T + beta*A` for `Replace`/`Add`, in a column-major and a row-major destination |
| `matmul::blas::*::symmetric_rank_k_update_writes_only_the_stored_triangle` | the referenced half of `C` is `alpha*op(A)*op(A)^H + beta*C` for all four arms (`A A^T`, `A^T A`, `A^H A`, `A A^H`), the other half (NaN) is neither read nor written, and `A` is unchanged; `(n, k) = (3, 1)` covers `SYR`'s non-diagonal rank-1 update |
| `triangular_solve::blas::*::trsm_recovers_the_exact_solution_with_an_implicit_diagonal` | the exact reference solution is recovered for both sides, `N`/`T`/`C`, lower/upper, generic/unit, with NaN in the unreferenced triangle and in a unit diagonal |
| `triangular_solve::blas::*::trsm_does_not_touch_memory_outside_its_operand` | for `TriangularLower` and `UnitTriangularUpper` on both sides, a known system's exact solution is recovered in place; the coefficient matrix and the in-place right-hand side each sit in a padded submatrix of their own ROGUE-filled parent, and each whole parent (stored operand and padding) must come back unchanged |

Which of these are new, and which overlap existing coverage:

* `matmul::tests::test_triangular` uses random *valid* values in every position,
  so a kernel that reads the unreferenced half or the implicit unit diagonal
  passes it. The weekly aarch64 lane's strict-triangle tests cover the *storage*
  contract of the generic kernel (which entries are written, which guards
  survive), not the numerical value of the product. The rogue/NaN property in
  `blas_trmm.rs` and `blas_rank_update.rs` is therefore new on the default run.
* `triangular_solve`'s own unit tests check subnormal pivots, and
  `linalg::tests::lapack_triangular` does place NaN in the unit diagonal and in
  the unreferenced triangle — but only in `lapack_triangular_cases`, which is
  `#[ignore]`d as opt-in LAPACK validation, and only for the left side. The
  `blas_trsm.rs` port is a bounded version of that property on the default run
  and adds the right side and the exact (not ratio) recovery.

Not covered by this batch, and not claimed: banded and packed storage, symmetric
and Hermitian structured multiply (`SYMM`/`SYMV`), `SYR2K`, the 3m variants,
mixed-precision comparisons, XERBLA-style argument validation, and the
`#[ignore]`d full GEMM sweep. The source's `DMMCH` eps-ratio checker is replaced
by exact comparison, which is stricter for these integer fixtures but says
nothing about accumulation error at scale.
