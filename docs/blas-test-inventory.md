# OpenBLAS BLAS-level test inventory

Bounded Rust adaptation of the checks in OpenBLAS `test/` and `ctest/`, as a
follow-up to [#31](https://github.com/tensor4all/faer/issues/31) (the LAPACK
`lapack-netlib/TESTING` validation in [lapack-validation.md](lapack-validation.md))
and part of [#166](https://github.com/tensor4all/faer/issues/166). Port
completion and numerical success are separate: this file records what was
ported, what has no counterpart, and what stays unported.

Four batches have landed: the triangular, rank-update and `norm_l2` ports
([#349](https://github.com/tensor4all/faer/pull/349)), the `GEMM`/`GEMV` boundary
port ([#373](https://github.com/tensor4all/faer/pull/373)), the remainder port
(`?dot`, `?gerc`, and the argument checks), and the CBLAS row-major layouts.
Which batch a row belongs to is in its `Port?` column.

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
| `faer/src/linalg/matmul/tests/blas_rank_update.rs` | `test/blat2.f` `DCHK4` (`DGER`), `ZCHK4` (`ZGERC`) and `DCHK5` (`DSYR`); `test/blat3.f` `DCHK4` (`DSYRK`) |
| `faer/src/linalg/tests/blas_trsm_cases.rs` | `test/blat3.f` `DCHK3` (`DTRSM`); `test/blat2.f` `DCHK3` (`DTRSV`) |
| `faer/src/linalg/matmul/tests/blas_gemm.rs` | `test/blat3.f` `DCHK1` (`DGEMM`); `test/blat2.f` `DCHK1` (`DGEMV`) |
| `faer/src/linalg/matmul/tests/blas_dot.rs` | `test/blat1.f` `CHECK2` (`?dot`, `?dotc`, `?dotu`) |
| `faer/src/linalg/matmul/tests/blas_row_major.rs` | `ctest/c_{s,d,c,z}blas3.c`, the `CblasRowMajor` wrappers of `GEMM`, `TRMM` and `SYRK`/`HERK` |
| `faer/src/linalg/matmul/tests/blas_arguments.rs` | `test/blat2.f` and `test/blat3.f` `DCHKE`; `test/xerbla_test.c`, `ctest/c_xerbla.c` and `ctest/cblas_test.h` |

Material adaptations, common to these suites: the source's random
`DBEG`/`DMAKE` sequence and its dimension sweep (`N = 0,1,2,3,7,31,63`, `LDA =
N+1` padding, `alpha` 0.0/0.7/1.0) are replaced by bounded shapes with an
exact-integer fixture and exact binary `alpha`, so the reference product is
compared exactly instead of through `DMMCH`'s eps ratio; and the rogue value
(`ROGUE = -1.0D10`) becomes NaN, which is the stronger sentinel. The source's
"operand unchanged" and "leading-dimension padding" checks are kept as
`exact`/`untouched_outside`. The two ports that put a written operand in a
guarded parent are `DTRMM` and `DTRSM`: there the triangular coefficient and
the written operand each sit in a rogue-filled parent with one row and one
column of padding on each side, so the operand equality and the padding check
both run against the whole parent. "Read operand" there means the triangular
one: `DTRMM`'s rectangular `B` is allocated standalone, so what guards it is the
entry-by-entry comparison against the reference, not a sentinel buffer. The
source itself only pads the leading dimension of the
arrays it passes in. `DGER`/`DSYR`/`DSYRK` allocate their destination
standalone and check only the entries the routine must not write (the
referenced-half rule) -- no surrounding buffer there. faer's strict-triangle
block structures have no BLAS `DIAG` value; they are covered because faer
exposes them.

## Inventory

"Port?" covers both batches; the file named in brackets is the port site.

### Level 1 — `test/{s,d,c,z}blat1.f` (`CHECK0`-`CHECK3`), `ctest/c_{s,d,c,z}blas1.c`

| OpenBLAS file/check | faer counterpart | Port? | Reason |
| --- | --- | --- | --- |
| `CHECK1` `ISAMAX`/`IDAMAX` (index of the largest modulus) | none (`reductions::norm_max` returns the value, not the index) | no | faer has no index-reduction API; nothing to call |
| `CHECK1` `SASUM`/`DASUM`, `SCASUM`/`DZASUM` | `reductions::norm_l1` (`abs1`, so the complex entries are `\|re\|+\|im\|`) | covered | `norm_l1::tests::test_norm_l1` runs its contiguous `f64` fixtures and, through `check_strided_complex`, a `c64` and a `c32` vector whose row stride is above 1 (the source's `incx = 2` half, which takes the strided fallback rather than the SIMD path). The source's 5-entry lengths and its negative strides are not ported |
| `CHECK2` `SDOT`/`DDOT`/`CDOTU`/`ZDOTU`, `CDOTC`/`ZDOTC` | `matmul::dot::inner_prod` | ported (`blas_dot.rs`) | the length list `NS = 0 1 2 4` and the increment lists (`INCXS = 1 2 -2 -1`, `INCYS = 1 -2 1 -2`) are the driver's own (`blat1.f:347-350`); the increment is the view's stride and a negative one is the same view built from its last element, and `?dotc` is `Conj::Yes` on the left operand. The source's 5-entry fixture and its scaled tolerance become an exact-integer fixture and an exact sum |
| `CHECK2`/`CHECK3` `SAXPY`/`SCOPY`/`SSWAP`/`SSCAL`/`SROT`/`SROTG`, `CSSCAL`/`ZDSCAL`/`CROTG`/`CSROT` | none | no | no level-1 vector API in faer (the source sweeps `incx`/`incy` over {1, 2, -2, -1}) |
| `CHECK1` `SNRM2`/`DNRM2`/`SCNRM2`/`DZNRM2` | `reductions::norm_l2`, `MatRef::norm_l2`, `ColRef::norm_l2` | covered | the source check is `n = 0..4` with `incx` 1/2 and small values; `norm_l2::tests::test_norm_l2_strided_range` already reads a stride-2 row view, and the exponent-range half of the property is [#36](https://github.com/tensor4all/faer/issues/36) / [faer-008](../upstream-reports/faer-008-norm-l2-strided-range/README.md) |

### Level 2 — `test/{s,d,c,z}blat2.f` (`DCHK1`-`DCHK6`), `ctest/c_*blat2c.c`

| OpenBLAS file/check | faer counterpart | Port? | Reason |
| --- | --- | --- | --- |
| `DCHK1` `GEMV`/`GBMV` | `matmul::matmul` with one destination column (the `matvec` route) | ported (`blas_gemm.rs`), `GBMV` no | the source's `N`/`T` sweep becomes the two matvec routes (a `1`-column destination and a `1`-row one), `m`/`n` = 0 are real cases in both level-2 drivers (`test/dblat2.dat` `0 1 2 3 7 31 63`, CBLAS `ctest/din2` `0 1 2 3 5 9 63`), and `alpha`/`beta` are in their `ALF`/`BET` lists. `GBMV`'s banded storage has no counterpart |
| `DCHK2` `SYMV`/`SBMV`/`SPMV` | none | no | faer's structured matmul carries triangular block structures only; a symmetric operand is materialized dense, and banded/packed storage has no API |
| `DCHK3` `TRMV`/`TBMV`/`TPMV` | `matmul::triangular::matmul_with_conj` with a `Rectangular` destination | ported (`blas_trmm.rs`) | the TRMM shapes carry it; the `Rectangular` case with a triangular operand is the level-2 multiply. Banded/packed variants have no counterpart |
| `DCHK3` `TRSV`/`TBSV`/`TPSV` | `triangular_solve::solve_{,unit_}{lower,upper}_triangular_in_place_with_conj` | ported (`blas_trsm.rs`) | the `nrhs = 1` cases are TRSV. Banded/packed variants have no counterpart |
| `DCHK4` `GER` | `matmul::matmul` with `K = 1` (the private `rank_update` module) | ported (`blas_rank_update.rs`) | the exact rank-1 update, including both destination layouts |
| `DCHK5` `SYR`/`SPR` | `matmul_with_conj` with a triangular destination and `K = 1` | ported (`blas_rank_update.rs`) | `SPR`'s packed storage has no counterpart |
| `DCHK6` `SYR2`/`SPR2` | none | no | two rank-1 terms, i.e. two calls; there is no single kernel to test |
| `ZCHK4` `ZGERC`/`ZGERU` (complex only, `zblat2.f:165-166`) | `matmul::matmul` with `K = 1` (`ZGERU`, unconjugated) and `matmul::triangular::matmul_with_conj` with `Conj::Yes` on the row operand (`ZGERC`) | ported (`blas_rank_update.rs`) | `?geru` is the unconjugated `K = 1` path and `?gerc` the same call with `Conj::Yes` on the row operand; both run, and since conjugation is the identity for the real types the `?gerc` test returns immediately there |
| `ZCHK5` `ZHER`/`ZHPR` (complex only, `zblat2.f:167-168`) | `matmul_with_conj` with a triangular destination and `Conj::Yes` on the row operand | `?her` partly (its `K = 1` arm of `blas_rank_update.rs`), `?hpr` no | the `SYR`/`SPR` shape with the row operand conjugated. It is `?herk` with `k = 1`: `symmetric_rank_k_update_writes_only_the_stored_triangle` already runs `(n, k) = (1, 1)` and `(3, 1)` with `Conj::Yes` on the second operand, so the product, the referenced-half rule and the rogue-value property are covered and this batch adds no separate test. What is *not* asserted is the reference contract faer does not implement: `?her` keeps the diagonal real on exit (its imaginary part dropped), while `matmul_with_conj` computes the algebraic `beta * C_ii + alpha * |x_i|^2`, so the imaginary part of `C_ii` survives `Accum::Add` (the same omission as `ZHERK`). `ZHPR`'s packed storage has no counterpart |
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
| `DCHK1` `GEMM` | `matmul::matmul`, `matmul::triangular::matmul_with_conj` (both `Rectangular`) | ported (`blas_gemm.rs`) | the product, both `TRANS`/`Conj` operands, `alpha` 0/1/-2/0.5 and both `Accum` values over a bounded `(m, n, k)` sweep. The Fortran driver's `test/dblat3.dat` lists `0 1 2 3 7 31 63`, so its `NULL` path for `m <= 0 or n <= 0` and the empty `k = 0` reduction do run there; the CBLAS driver's `ctest/din3` starts at 1. `matmul::tests::test_matmul` stays `#[ignore]`d as too slow |
| `DCHK2` `SYMM` | none | no | no symmetric structured multiply |
| `DCHK3` `TRMM` | `matmul::triangular::matmul_with_conj` | ported (`blas_trmm.rs`) | both sides, `N`/`T`/`C`, lower/upper, and the generic/strict/unit diagonal; the source's ROGUE diagonal and ROGUE unreferenced triangle are the checked property, and the destination is a padded submatrix of a ROGUE parent |
| `DCHK3` `TRSM` | `triangular_solve::solve_{,unit_}{lower,upper}_triangular_in_place_with_conj` | ported (`blas_trsm.rs`) | both sides and `N`/`T`/`C`; `alpha` has no parameter in faer, so the fixture scales the right-hand side by `alpha` before the solve, which is the same computation; the in-place right-hand side is a padded submatrix of a ROGUE parent |
| `DCHK4` `SYRK` | `matmul_with_conj` with `TriangularLower`/`TriangularUpper` destination | ported (`blas_rank_update.rs`) | the referenced half is computed and the other half must stay untouched (`LDERES 'SY'`); the four TRANS/conjugation arms are `A A^T`, `A^T A`, `A^H A` and `A A^H`, and the `n = 3, k = 1` shape carries `SYR`'s non-diagonal rank-1 update |
| `DCHK5` `SYR2K` | none | no | `A*B^T + B*A^T` is two calls of the same kernel |
| `ZCHK4` `ZHERK` (complex only, `zblat3.f:146`) | `matmul_with_conj` with a triangular destination and `Conj::Yes` on the second operand | partly (`blas_rank_update.rs`) | the `A A^H` arm covers the product and the referenced-half rule, and `ZSYRK` (`A A^T`) is the unconjugated sibling the same test covers in every dtype. One part of the reference contract is **not** asserted: the source keeps a complex diagonal and the oracle carries `beta * C_ii` through unchanged, so `?herk`'s convention that the diagonal is real on exit (its imaginary part dropped) is neither implemented by `matmul_with_conj` nor tested here |
| `ZCHK5` `ZHER2K` (complex only, `zblat3.f:147`) | none | no | `A*B^H + B*A^H` is two calls of the same kernel |
| `_3m` variants (`cblat3_3m.f`, `zblat3_3m.f`) | none | no | faer has no 3m algorithm |
| `DMAKE`/`DMMCH`/`LDE`/`LDERES` helpers | `tests/blas_support.rs` (`entry`, `product`, `blended`, `exact`, `untouched_outside`) | adapted | reimplemented, not translated: the reference product is explicit, and the comparison is exact |

### `ctest/` specifics and OpenBLAS-only files

| OpenBLAS file/check | faer counterpart | Port? | Reason |
| --- | --- | --- | --- |
| CBLAS row-major wrappers `c_{s,d,c,z}blas{1,2,3}.c` | views: `MatRef::transpose`, `MatMut::transpose_mut` | ported for the products' destinations (`blas_row_major.rs`); the rank-1 row-major destination by `blas_rank_update.rs`; the level-1 wrappers take no layout parameter, so nothing to port there | `CblasRowMajor` copies its arguments into row-major buffers, so the kernel sees a destination whose row stride is the leading dimension and whose column stride is 1 -- the other way round from a column-major `Mat`. `GEMM` (destination; its operands are row-major views too, a layout the `'T'`/`'C'` arms of `blas_gemm.rs` already reach), `TRMM` (both sides, `N`/`T`/`C`, all six block structures) and `SYRK`/`HERK` (both triangles, four arms, both `Accum` values) now run that destination layout. `TRSM` needed no new arm: the right-side solves of `trsm_recovers_the_exact_solution_with_an_implicit_diagonal` hand the solver a transposed buffer, which is the same layout (a mutation that assumes a column-major right-hand side reddens that test). The level-1 wrappers (`c_*blas1.c`) have no row-major/column-major parameter at all -- they wrap `?dot`, `?asum` and `?nrm2`, whose faer counterparts this file already records -- so there is no layout arm to port |
| argument-error tests (`cblas_test.h`, `c_xerbla.c`, `blat2.f` `DCHKE`, `blat3.f` `DCHKE`) | `assert!`/panic preconditions in `matmul`, `matmul_with_conj`, `triangular_solve` | ported (`blas_arguments.rs`) | faer rejects a bad shape with a panic and has no XERBLA-style error return or info code, so the port asserts the panic and the message naming the violated precondition instead of the argument index: an incompatible inner dimension (`matmul`), a mismatched `?dot` length, and a right-hand side whose row count does not match the triangle. `matmul_with_conj`'s destination-structure check has no arm, because removing it is not a safe mutation to run: the unguarded kernel would receive a non-square triangular destination |
| `compare_sgemm_bgemm.c`, `compare_sgemm_sbgemm.c`, `compare_sgemm_shgemm.c`, `compare_sgemv_*.c` | none | no | bfloat16/half mixed-precision kernels do not exist in faer |
| `xerbla_test.c`, `shgemm_test.c`, `test_laed3_nan.c`, `get_threading_model.c`, `wasm/` | none | no | OpenBLAS build and runtime specifics (XERBLA, SME kernels, its bundled LAPACK, thread-model query, wasm build) |

## What the ports establish

Twenty-one test functions over four scalar types (`f64`, `f32`, `c64`, `c32`),
84 tests in the default (non-nightly) run: eight from the first batch (32 tests,
`trmm`, `trsm`, `trsv`, rank-1 and rank-k), four from the `GEMM`/`GEMV` batch
(16 tests), six from the `?dot`/`?gerc`/argument batch (24 tests), and three from
the row-major batch (12 tests). The `matmul::blas::*` group is 68 of them.

`?her` is the one column-4 name this batch does not add a test for: it is the
`k = 1` arm of `?herk`, which the first batch's rank-k test already runs (see
the `ZCHK5` row).

The row-major batch's three:

| Test | Property |
| --- | --- |
| `matmul::blas::*::gemm_writes_a_row_major_destination` | `?gemm` with the destination *and* both operands in row-major storage: `C = alpha*A*B + beta*C` equals the independently accumulated product for both `Accum` values and `alpha` 0/1/-2/0.5 over the shape set that includes 0 and the non-square ones, with both operands unchanged |
| `matmul::blas::*::trmm_writes_a_row_major_destination` | `?trmm` into a row-major destination, for all six block structures, both sides, `N`/`T`/`C` and four `alpha`, against the independently accumulated product; the triangular operand and the rectangular one are unchanged |
| `matmul::blas::*::rank_k_update_writes_a_row_major_triangle` | `?syrk`/`?herk` with the stored triangle in a row-major square: the four trans/conjugation arms, both triangles, both `Accum` values and four `alpha`, with the unreferenced triangle still holding NaN |

The `?dot`/`?gerc`/argument batch's six:

| Test | Property |
| --- | --- |
| `matmul::blas::*::dot_matches_the_source_stride_sweep` | `?dot` over the driver's length list (`0 1 2 4`, plus 7) and its four increment pairs `(1, 1)`, `(2, -2)`, `(-2, 1)`, `(-1, -2)`, which are `INCXS(KI)` with `INCYS(KI)` in the same iteration; the buffer is NaN-filled between the referred entries and padded on both sides, so a kernel that ignores the increment reads a sentinel instead of the fixture. `?dotc` is the conjugated left-operand arm, and the two mixed arms the source cannot express run as well. Length 0 returns zero |
| `matmul::blas::*::dot_reads_every_referenced_entry` | with the view's stride 1, 2 and -2, a NaN written into any referred entry of *either* operand reaches the sum as a NaN, and an infinity written there leaves it non-finite, so no referred entry can be skipped; the unreferred padding stays a NaN and does not reach the sum. All four conjugation arms run for the complex types (the real types run only the unconjugated arm, where conjugation is the identity), but this is a propagation check only: whether the conjugation is applied correctly is not observable through a NaN or an infinity, and is what `dot_matches_the_source_stride_sweep`'s exact result decides |
| `matmul::blas::*::rank_one_update_conjugates_the_row_operand` | `?gerc` = `alpha * x * conj(y)^T + beta * A` for `Replace`/`Add` and `alpha` 0/1/-2/0.5 over `(m, n)` including 0, with `x` and `y` unchanged; the real types return immediately because the conjugation is the identity there, which is the `?geru` arm the first batch already covers |
| `matmul::blas::*::matmul_panics_on_an_incompatible_inner_dimension` | a product whose inner dimensions disagree panics, and the panic names `lhs_ncols == rhs_nrows` |
| `matmul::blas::*::inner_prod_panics_on_operands_of_different_length` | `?dot` of a length-3 row and a length-4 column panics, and the panic names the length check |
| `matmul::blas::*::triangular_solve_panics_on_a_mismatched_right_hand_side` | a triangular solve whose right-hand side has the wrong row count panics, and the panic names `rhs.nrows() == triangular_lower.ncols()` |

The three argument checks carry the shape rejection that the source's `DCHKE`
reports through XERBLA. They are `should_panic` with the message as part of the
assertion, so a panic anywhere else in the call does not pass them. Each shape
is picked so that the unguarded call reads *less* than the arrays hold, which is
what keeps the non-vacuity runs in bounds: `DCHKE` itself passes one-element
arrays and relies on the routine rejecting the bad argument before touching
them, a path faer does not have once the check is removed.

The four `GEMM`/`GEMV` ones:

| Test | Property |
| --- | --- |
| `matmul::blas::*::gemm_matches_the_source_dimension_sweep` | `C = alpha*op(A)*op(B) + beta*C` equals an independently accumulated product for `'N'`/`'T'`/`'C'` on either operand, both `Accum` values and `alpha` 0/1/-2/0.5, over `(m, n, k)` including `m`, `n` and `k` = 0; each of the four operand storages (the `m x k` and `k x m` forms of `A` and of `B`) still equals a copy taken before the call, so a write to any of them is caught after the call that made it |
| `matmul::blas::*::gemm_reads_the_destination_only_when_it_accumulates` | `Accum::Replace` produces the product from a NaN-filled destination (the source's `beta = 0` "`C` need not be set on entry"), and `Accum::Add` brings the NaN back, so the two arms are distinguishable rather than both silently correct |
| `matmul::blas::*::gemv_matches_the_source_dimension_sweep` | the same `alpha`/`beta`/empty-direction sweep for `y = alpha*op(A)*x + beta*y`, on both matvec routes (`ncols == 1` and `nrows == 1`); `A` and the operand the route reads (`x`, or the `x^T` storage on the transposed route) still equal a copy taken before the call |
| `matmul::blas::*::gemv_zeroes_the_destination_of_an_empty_reduction` | with `k = 0`, `Accum::Replace` zeroes the destination and `Accum::Add` returns it unchanged |

Deliberate deviations from the reference routines, asserted rather than
avoided:

* `?gemv` returns before touching `y` when `m = 0` or `n = 0` (`dgemv.f:218-219`),
  and the source test asserts `y` is unchanged there (`LDE(YS, YY, LY)` on its
  `NULL` path). faer's `matmul` computes the algebraic `beta * dst` instead, so
  `Accum::Replace` with `k = 0` zeroes the destination.
  `gemv_zeroes_the_destination_of_an_empty_reduction` asserts faer's contract
  and says so in its comment, and the empty `m`/`n` dimensions are covered by
  `gemv_matches_the_source_dimension_sweep`.
* `?gemm` with `alpha = 0` need not read `A` or `B`, and `?gemm` with `beta = 0`
  need not read `C`. faer's `Accum::Replace` implements the second (asserted
  above); the first is not asserted, because faer's kernel multiplies by
  `alpha` and a NaN operand would propagate.
* the two drivers' sweeps become bounded shape lists. The empty shapes are in
  both level-2 drivers: `test/dblat2.dat:10` lists `n = 0 1 2 3 7 31 63` and
  `ctest/din2:9` lists `n = 0 1 2 3 5 9 63`, each with `m` set to
  `max(n - n/2 - 1, 0)` and `min(n + n/2 + 1, NMAX)` and with `k = 0 1 2 4`
  (`dblat2.dat:12`, `din2:11`). At level 3 only the Fortran list has them:
  `test/dblat3.dat:10` is `0 1 2 3 7 31 63` while `ctest/din3:9` starts at 1
  (`1 2 3 5 7 9 35`). `alpha`/`beta` `0.0/0.7/1.0` become exact binaries so the
  comparison stays exact.

### Inventory totals

Counting the rows of the four tables above (30 checks):

| Outcome | Checks | Which |
| --- | --- | --- |
| ported to a unit test on the default run | 13 | `GEMV`, `GEMM`, `TRMM`, `TRSM`, `TRMV`, `TRSV`, `GER`, `SYR`, `SYRK`, `DOT`, `ZGERC`/`ZGERU`, the XERBLA argument checks, the CBLAS row-major wrappers |
| partly (one arm of the check ported) | 2 | `ZHERK` and `ZHER`/`ZHPR` (the real-diagonal convention is not implemented; `?hpr` has no counterpart) |
| already covered by a faer unit test that predates this work | 2 | `ASUM`, `NRM2` |
| helper group adapted (not translated) | 1 | `DMAKE`/`DMMCH`/`LDE`/`LDERES` |
| not ported | 12 | `ISAMAX`, `AXPY`/`SCAL`/`ROT`/`COPY`/`SWAP`, `SYMV`/`SBMV`/`SPMV`, `ZHEMV`/`ZHBMV`/`ZHPMV`, `SYR2`/`SPR2`, `ZHER2`/`ZHPR2`, `SYMM`, `SYR2K`, `ZHER2K`, the 3m variants, the mixed-precision comparisons, the OpenBLAS build/runtime tests |

So 13 of the 30 inventoried checks are now covered by a test that runs by
default, 2 partly and 2 were already covered. All 12 that are not ported have no
faer API to call at all — no index reduction (`ISAMAX`), no level-1 vector API,
no symmetric or Hermitian structured multiply, no single kernel for the two-term
updates (`SYR2`/`SYR2K`/`HER2`/`HER2K`), no packed storage, no 3m algorithm, no
half/bfloat16 mixed precision, and OpenBLAS's own build diagnostics. The two
counterparts this batch could reach — `DOT` and the XERBLA argument checks — now
have their tests, and `?gerc`/`?her` joined the rank-1/rank-k tests that already
had the kernel.

### Non-vacuity

Each family was checked by mutating one deliberate defect into the kernel it
tests, confirming the port goes red, and reverting:

| Family | Mutation | Goes red |
| --- | --- | --- |
| `GEMM` (`nano_gemm` path) | `Accum::Replace` applies `beta = 1` instead of resetting the destination | `gemm_reads_the_destination_only_when_it_accumulates`, `gemm_matches_the_source_dimension_sweep` |
| `GEMV` (`matvec_colmajor`) | the `Accum::Replace` arm no longer zeroes the destination | `gemv_matches_the_source_dimension_sweep`, `gemm_reads_the_destination_only_when_it_accumulates` |
| empty reduction (`matmul_imp`'s `K == 0` branch) | the branch returns without applying `beta` | `gemv_zeroes_the_destination_of_an_empty_reduction`, and the `k = 0` arms of both sweeps and of `gemm_reads_the_destination_only_when_it_accumulates` |
| rank-1 update (`rank_update` module) | the `Accum::Replace` arm writes nothing | `rank_one_update_matches_the_source_contract` |
| unit-triangular base case | the sign of the 3x3 subdiagonal is flipped | every `trsm_*` test in all four dtypes |
| `norm_l2` generic-stride fallback | the scaled accumulator is squared unscaled again (the #36 regression) | `test_norm_l2_strided_range` |
| `?dot` strided path (`inner_prod`'s non-contiguous branch) | the operands are read as if they were contiguous from their first logical element | `dot_matches_the_source_stride_sweep`, `dot_reads_every_referenced_entry` |
| `inner_prod_schoolbook`'s conjugation | the `(conj_lhs, conj_rhs)` match is replaced by `(No, No)` | `dot_matches_the_source_stride_sweep` (the `dotc` arm); the unconjugated arm stays green |
| `matmul_with_conj`'s right-operand conjugation | `conj_rhs` reaches the kernel as `Conj::No` | `rank_one_update_conjugates_the_row_operand`; the unconjugated GER test stays green |
| row-major destination (`matmul_imp`'s small-product call and `triangular::matmul_with_conj`) | the kernel is handed the destination's strides as if it were column-major | `gemm_writes_a_row_major_destination`, `trmm_writes_a_row_major_destination`, `rank_k_update_writes_a_row_major_triangle`, and every ported test whose destination has a padded leading dimension (`trmm_reads_and_writes_nothing_outside_its_operands`) |
| shape preconditions (`matmul`'s inner dimension, `ColRef::as_row_shape`, `triangular_solve`'s right-hand side) | each check is relaxed to `<=` or dropped | all three argument tests, one per check: `matmul_panics_on_an_incompatible_inner_dimension` and `triangular_solve_panics_on_a_mismatched_right_hand_side` then panic later with a different message, and `inner_prod_panics_on_operands_of_different_length` does not panic at all |

The first batch's eight, for reference:

| Test | Property |
| --- | --- |
| `matmul::blas::*::trmm_matches_an_independent_product` | the product of a triangular operand equals an independently accumulated product over the bounded shape set, for every lower/upper, generic/strict/unit, `N`/`T`/`C`, left/right and `alpha` combination |
| `matmul::blas::*::trmm_reads_and_writes_nothing_outside_its_operands` | the operand's unreferenced triangle and implicit diagonal hold NaN, so a value taken from them cannot pass through the arithmetic unnoticed (sensitivity to such a value, not proof that no read occurs); the triangular operand and the destination each sit in a padded submatrix of their own ROGUE-filled parent (the rectangular `B` does not: it is standalone, and only its values are compared), the destination is checked entry by entry against an independent product computed from the operand's pre-call snapshot, the coefficient parent comes back unchanged in full (stored operand and padding), and the destination parent changes only inside the written region |
| `matmul::blas::*::rank_one_update_matches_the_source_contract` | `A = alpha*x*y^T + beta*A` for `Replace`/`Add`, in a column-major and a row-major destination |
| `matmul::blas::*::symmetric_rank_k_update_writes_only_the_stored_triangle` | the referenced half of `C` is `alpha * P + beta*C` for the four arms, with `P` = `A A^T`, `A^T A`, `A^H A` or `A A^H` (the first two unconjugated, hence not Hermitian for the complex types; the last two conjugated), the other half (NaN) is neither read nor written, and `A` is unchanged; `(n, k) = (3, 1)` covers `SYR`'s non-diagonal rank-1 update |
| `triangular_solve::blas::*::trsm_recovers_the_exact_solution_with_an_implicit_diagonal` | the exact reference solution is recovered for both sides, `N`/`T`/`C`, lower/upper, generic/unit, with NaN in the unreferenced triangle and in a unit diagonal |
| `triangular_solve::blas::*::trsm_does_not_touch_memory_outside_its_operand` | for `TriangularLower` and `UnitTriangularUpper` on both sides, a known system's exact solution is recovered in place; the coefficient matrix and the in-place right-hand side each sit in a padded submatrix of their own ROGUE-filled parent, the coefficient parent comes back unchanged in full, and the right-hand side's parent changes only inside the written region |
| `triangular_solve::blas::*::trsv_conjugates_an_imaginary_diagonal_exactly` | `conj(A) X = B` and `A X = B` on a 1x1 system with a purely imaginary pivot (`2i`), asserting the exact reciprocal (`i/2` and `-i/2`); a real pivot cannot tell `1/a` from `1/conj(a)`. The real suites return immediately: `conj` is the identity there |
| `triangular_solve::blas::*::trsm_conjugates_wide_imaginary_diagonals_exactly` | the same conjugation on the 2-, 3- and 4-element base cases, which invert their diagonal entries in their own code, with `2i` in every diagonal position and an exact integer solution |

Which of these are new, and which overlap existing coverage:

* `matmul::tests::test_triangular` uses random *valid* values in every position,
  so a kernel that reads the unreferenced half or the implicit unit diagonal
  passes it. The weekly aarch64 lane's strict-triangle tests cover the *storage*
  contract of the generic kernel (which entries are written, which guards
  survive), not the numerical value of the product. The rogue/NaN property in
  `blas_trmm.rs` and `blas_rank_update.rs` is therefore new on the default run.
* `inner_prod` had no unit test of its own before `blas_dot.rs`: it ran only
  inside the decompositions that call it, at their operand's layout — mostly
  contiguous, though `householder.rs` and `svd/bidiag.rs` do call it with one
  conjugated operand. The strided and reversed views, the mixed conjugation arms
  and the sentinel-filled gaps are new on the default run.
* the shape preconditions are reached by other tests only by accident, as a
  panic in a test that expected one; `blas_arguments.rs` is the first place that
  asserts *which* check fires.
* every earlier *product* port handed the kernel a column-major `Mat` as
  destination -- the padded-parent tests pad the leading dimension but keep the
  row stride at 1 -- with two exceptions that are already in the tree: the
  rank-1 update's row-major destination (`blas_rank_update.rs`) and the TRSM
  port's row-major right-hand side (`blas_trsm_cases.rs`). The row-major
  *operands* are not new either: `blas_gemm.rs` reaches them through the
  `'T'`/`'C'` arms of `viewed`. What `blas_row_major.rs` adds is the row-major
  destination for `GEMM`, `TRMM` and the rank-k update.
* `triangular_solve`'s own unit tests check subnormal pivots, and
  `linalg::tests::lapack_triangular` does place NaN in the unit diagonal and in
  the unreferenced triangle — but only in `lapack_triangular_cases`, which is
  `#[ignore]`d as opt-in LAPACK validation, and only for the left side. The
  `blas_trsm.rs` port is a bounded version of that property on the default run
  and adds the right side and the exact (not ratio) recovery.

Not covered by any batch, and not claimed: banded and packed storage, symmetric
and Hermitian structured multiply (`SYMM`/`SYMV`/`HEMV`), `SYR2`/`SYR2K`/
`HER2K`, the 3m variants, mixed-precision comparisons, `?her`'s real-diagonal
convention, `matmul_with_conj`'s destination-structure precondition, the CBLAS
row-major wrappers of the level-1 functions (their routines take no layout
parameter), and the `#[ignore]`d full GEMM sweep. The source's `DMMCH` eps-ratio
checker is replaced by exact comparison, which is stricter for these integer
fixtures but says nothing about accumulation error at scale.
