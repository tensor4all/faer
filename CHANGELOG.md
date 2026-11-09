# Unreleased

Changes since `t4a-faer` 0.32.0 (tag `faer-v0.32.0`). Numbers are tensor4all/faer issues or PRs; each change has its row in `PATCHSET.md`.

## Behaviour change: `norm_max` propagates NaN

- `norm_max` (`MatRef`/`ColRef`/`RowRef::norm_max`, `linalg::reductions::norm_max`) now returns NaN when any entry is NaN, and for a
  complex entry when either its real or its imaginary part is NaN. It used to skip NaN entries: it returned the largest finite magnitude, and 0
  for an all-NaN input. `norm_l1` and `norm_l2` already returned NaN, and LAPACK's `xLANGE('M')` and Julia's `norm(x, Inf)` behave the same way.
  An empty matrix still returns 0. Code that relied on NaN being skipped, for example `norm_max() < tol` as a test that holds for a matrix with
  NaN entries, now sees NaN, and every comparison with NaN is false (#316).

## Correctness

- `gevd_cplx` returned wrong left eigenvectors for some pencils with an infinite eigenvalue (a zero on the diagonal of `B` during the QZ
  iteration, `n < 15`), while the eigenvalues and the right eigenvectors were right. The rotations that chase that zero to the bottom were
  applied to the rows of `Q` instead of its columns. Whether a pencil reaches that branch depends on rounding, so the same input could fail on
  x86-64 with AVX2 and pass on aarch64 (#324).

- `stats::col_mean` and `stats::col_varm` wrote their results in reverse order when the input had a negative row stride (for example
  `mat.reverse_rows()`), and `row_mean`/`row_varm` did the same for a negative column stride (`mat.reverse_cols()`). Both `NanHandling` modes
  were affected, and `varm` also subtracted the mean of the mirrored row. The input rows were reversed to get a positive stride, but `out` and the
  `col_mean` argument were not. They are now reversed together with the input (#317).

- The complex `simd_abs_max` keeps a NaN in either part (#320). The unordered `re > im` select
  returned the finite imaginary magnitude when the real part was NaN, while a NaN imaginary part did
  return NaN. The generic `Complex<T>` SIMD path now also masks the real side, and the
  `ComplexImpl<f32>` / `ComplexImpl<f64>` scalar fallbacks test both parts before the comparison. No
  in-tree type reaches either path (native `c32` / `c64` go through the real coercion in `norm_max`),
  so no existing result changes. No performance claim: two mask ops per SIMD chunk on a path nothing
  in the tree reaches.

# t4a-faer 0.32.0 (tensor4all fork; with t4a-faer-traits 0.24.3)

Changes since `t4a-faer` 0.31.0 (tag `faer-v0.31.0`). Numbers are tensor4all/faer issues or PRs; each change has its row in `PATCHSET.md`.

## Breaking: the `threadteam` pin moves to 0.5

This release requires `threadteam` 0.5 (0.5.1 or newer), not 0.4. `Par::Team` carries a `threadteam::TeamExec`, and two `threadteam` minors are two
distinct types, so a dependency tree that holds both 0.4 and 0.5 will not compile. If you name the pool or team types yourself, add `threadteam` at
the version this crate resolves rather than pinning it separately. Nothing else in the public API changed, and code that only calls `par()` or
passes a team it already obtained needs no edit. This is why the version is 0.32.0 and not a patch: `cargo` cannot see the break in a 0.x minor, so
this paragraph is the only place it is announced (#287).

## Correctness

- The owned dense storage no longer divides by zero for a zero-sized element type. `RawMatUnit::try_with_capacity` padded the row capacity to
  a multiple of `align / size`, which is a division by `size_of::<T>()` and a panic for `T` with no size: `Mat::<()>::with_capacity(1, 1)` panicked.
  A zero-sized element has a byte stride of 0 for every capacity, so the padding is skipped in that case (#286, #288).
- Fixing that made reachable a second, older defect on the same invariant: `DropMat` advances its column pointer in **bytes**, but the unwind
  guard in `init_with` and the column branch of `truncate` passed it an **element** count (`faer/src/mat/matown.rs:359`, `:545`), so an element
  whose alignment exceeds its size was dropped at a misaligned address. That one is undefined behaviour on 0.31.0 independently of zero-sized
  elements -- `#[repr(align(8))] struct S(u32)` reached it through `Mat::<S>::truncate` -- and both sites now pass
  `row_capacity * size_of::<T>()`, as the other two `DropMat` sites already did. Upstream `faer` carries both defects as well (#288).

No performance claim: one branch outside any inner loop and one multiply on the cleanup path. `cargo +nightly miri test -p t4a-faer --test
owned_capacity` is clean, and the two new tests in `faer/tests/owned_capacity.rs` fail before the change and pass after it.

# t4a-faer 0.31.0 (tensor4all fork; with t4a-faer-traits 0.24.3)

Changes since `t4a-faer` 0.30.0 (tag `faer-v0.30.0`). Numbers are tensor4all/faer issues or PRs; each change has its row in `PATCHSET.md`. No public API change.

## Performance

- The partial-pivoting LU's leaf takes 8 columns for f64 (#276). At 1024 with 8 threads LU goes from 4.56 ms to
  3.37 ms against 3.40 ms for MKL, so the ratio moves from 1.34 to 0.99; at 4096 it is 0.55 to 0.66. At 512 the ratio
  is still 1.28: that cell is bound by the leaf's serial section, not by the trailing update, and this change does not
  move it.
- The same leaf is vectorised: scaling, the update and the next pivot search share one pass per column, and the row
  swaps are applied inside the trailing update per column chunk instead of as a separate pass (#276). The SIMD leaf
  produces the same pivots and the same factor entries as the scalar leaf, bit for bit, on random real matrices; that
  test is what makes the second implementation acceptable. **No performance figure is claimed for this part**: it
  landed after the run that produced the numbers above.
- The unit-lower triangular solve stages an aliased right-hand side, and stages only when a set receives 24 lines
  (#276). **This entry claims no figure.** Earlier versions of it quoted a Cholesky ratio of 0.60 against MKL at
  1024 with 8 threads, which cannot be attributed to this change: `Llt` solves its diagonal blocks through the
  non-unit lower solve (`faer/src/linalg/cholesky/ldlt/factor.rs:421`), while the staging is applied only inside
  `solve_unit_lower_triangular_in_place_imp` (`faer/src/linalg/triangular_solve.rs:673`). A column-major Cholesky
  ratio of about 0.58 against MKL is reproducible without the staging, so the improvement seen there has another
  cause or is a property of the baseline; the measurement that would settle it is tracked in #279.

## Correctness

- `faer_traits::math_utils::abs` returns the right modulus at both ends of the range (#274, #275). A nonzero subnormal
  `c32` or `c64` used to come back as 0 because the component squares underflowed, and a component at `f32::MAX`
  overflowed to `inf`; the two outer ranges are now scaled and the inner range is bit-unchanged, so ordinary inputs take
  the same arithmetic as before.
- Still open next to it, recorded in #275 and not fixed here: complex `sqrt` overflows where `re + |z|` is near the
  maximum, and `recip` returns 0 where `|z|^2` overflows.

# t4a-faer 0.30.0 (tensor4all fork; with t4a-faer-traits 0.24.2)

Changes since `t4a-faer` 0.29.0 (tag `faer-v0.29.0`). Numbers are tensor4all/faer issues or PRs; each change has its row in `PATCHSET.md`. No public API change.

- `threadteam` 0.4.0: participants are placed by default, and the leader is placed before every section.

# t4a-faer 0.28.2 (tensor4all fork; with t4a-faer-traits 0.24.2)

Changes since `t4a-faer` 0.28.1 (tag `faer-v0.28.1`). Numbers are tensor4all/faer issues or PRs; each change has its row in `PATCHSET.md`. No public API change.

## Performance and correctness

- `matmul_with_conj_uninit` now sends a one-row destination through the same `matvec` route as
  `matmul(Accum::Replace)` instead of the x86 AVX2 full GEMM kernel. The fresh path was 2.2x slower
  for `m = 1` and produced different last bits from the replace path; it is now at or above the
  replace path's speed and bitwise equal. The O(n) destination initialization is the only added
  work, and the destination's prior contents are still never read (#200, #201).

## Upstream

Codeberg `sarah-quinones/faer` `main` at `0ecd846d` does not contain `matmul_with_conj_uninit`, so
this defect cannot occur upstream and there is nothing to feed back. `n = 1` still takes the GEMM
path in this fork and is recorded in #200 as a separate decision.

# t4a-faer 0.27.1 (tensor4all fork; with t4a-faer-traits 0.24.2)

Changes since `t4a-faer` 0.27.0 (tag `faer-v0.27.0`). Numbers are tensor4all/faer issues or PRs; each change has its row in `PATCHSET.md`. No public API change.

## Performance

- The blocked QR factors the next panel while the current block's trailing update runs, on a split
  team, when the per-step cost model says the overlap pays (#160, #165). Results match the
  sequential factorization (R, rank, Householder factors, Q R = A); scratch does not grow.

## Correctness

- Real Schur reordering: a swap the kernel rejects now leaves T and Q unchanged, and the 2x2
  standardization handles dlanv2's B = 0 case, which fixes a wrong eigenvector of the real EVD for a
  lower triangular 2x2 block such as [[2, 1], [-1, 0]] (#167).

# t4a-faer 0.27.0 (tensor4all fork; with t4a-faer-traits 0.24.2)

Changes since `t4a-faer` 0.26.0 (tag `faer-v0.26.0`). Numbers are tensor4all/faer issues or PRs; each change has its row in `PATCHSET.md`.

## Breaking: threadteam 0.2 (#158)

The public API carries threadteam's types (`Par::Team(threadteam::TeamExec)`,
`ParShape::Team(_, threadteam::DispatchCosts)`), and the requirement is now `threadteam` 0.2. A crate
that builds its `Pool`/`Team` with threadteam 0.1 and lends it to faer must move to threadteam 0.2
together with this release; no faer signature changes otherwise. With threadteam 0.2, `with_team`
called repeatedly at one width reuses the threads of a cached pool instead of spawning and joining a
fresh set (#107).

## Correctness

- The values-only SVD no longer allocates on the heap: dqds takes its work arrays from the caller's
  `MemStack` (included in `svd_scratch`) and sorts without a heap buffer (#156, #157).

# t4a-faer 0.26.0 (tensor4all fork; with t4a-faer-traits 0.24.2)

Changes since `t4a-faer` 0.25.2 (tag `faer-v0.25.2`). Numbers are tensor4all/faer issues or PRs; each change has its row in `PATCHSET.md`.

## Breaking: threadteam is the only parallel route (#97)

`Par::Rayon(NonZeroUsize)`, `Par::rayon(nthreads)`, `ParShape::Rayon(NonZeroUsize)` and
`ParShape::rayon(nthreads)` are removed, together with the spindle path and the Rayon cost model.
`Par::Seq` and `Par::Team` remain. To migrate:

- a one-off parallel call: `faer::team::with_team(n, |par| { /* faer calls with par */ })` replaces
  `Par::rayon(n)`;
- repeated parallel work: build a `threadteam::Pool` and `Team` once and lend `Par::Team(..)` to each
  call, `par.reborrow()` for successive calls (see "parallelism" in `README.md`);
- a workspace query: pass `par.shape()` (or `ParShape::Seq`) where `ParShape::rayon(n)` was used.

A build without the `threadteam` feature has `Par::Seq` only.

## New

- `faer::linalg::matmul::matmul_with_conj_uninit` and `MatUninitMut`: a matrix product into
  `MaybeUninit<T>` storage, returned as an initialized `MatMut`; prior contents are never read
  (#136, requested upstream as sarah-quinones/faer-rs#269).

## Correctness

- Nonsymmetric EVD: the input is scaled into range as `?GEEV` does (#112); the eigenvector back
  substitution clamps relative to the eigenvalue, so residuals no longer grow with n (#112); the real
  Schur deflation-window sort always ends (#144); the 2 x 2 block solves of the real back
  substitution follow `?LALN2` (#145).
- SVD: the rotation path finishes when dqds does not converge (#110); a retained Ritz pair of the
  partial SVD carries its residual estimate through the restart (#108).
- LU: the full-pivoting search breaks ties by column then row, independent of the SIMD width (#105, #106).
- Products with a diagonal read the depth in either direction (#113).
- Sparse LU reports an exact zero pivot as `LuError::NumericSingular` in both backends (#132).
- Krylov solvers: the final residual is reported on `NoConvergence`, and BiCGSTAB stops at the last
  finite iterate after a breakdown (#133, #134).

## Parallelism and performance

- Every region runs at the widest width that pays, on a sub-team of that width (#91); the team
  dispatch cost is read from threadteam (#104).
- No heap allocation inside a parallel region (#111).
- Layout copies split along the destination's major dimension, first touch in parallel (#99).
- SVD: blocked bidiagonalization, dqds shift and a padded work copy (#52).

## Dependencies and toolchain

- `threadteam` 0.1.3 (#107 wait fix) and `t4a-private-gemm-x86` 0.1.27 (#113).
- MSRV is Rust 1.89 (`rust-version = "1.89.0"` in both crates).
- AVX-512 (`x86-v4`) no longer needs the nightly compiler but stays opt-in: enable the `x86-v4`
  feature (`nightly` is kept as its old name).
- `t4a-faer-traits` 0.24.2 only adds `rust-version`.

# 0.24
- add missing scratch space functions for matrix-free eigen/svd solvers
- allow sparse-dense/dense-sparse matmuls to accept both CSC and CSR matrices
- add for-each and triangular solve helpers
- fix uninit read in AMD ordering
- remove serde support
- fix wrong strides in MatMut::from_row_major_slice_with_stride_mut

# 0.23
- generalized eigendecomposition
- self adjoint matrix-free eigensolver
- matrix-free svd solver
- better multithreaded perf

# 0.22
- accelerated matrix multiply backend on `x86_64` targets.
- accelerated column pivoted qr factorization
- accelerated matrix multiply for non primitive (and `Complex<Primitive>`) types.
- implemented an extended precision simd floating point type (exported as `fx128`, complex number as `cxf128`).
- make dense unpivoted qr rank revealing
- removed lblt regularization
- implemented `FromIterator` for `Col` and `Row`.
- stabilized matrix-free solvers.
- implemented matrix-free krylov-schur eigensolver.
- renamed bunch-kaufman to lblt.
- implemented pivoting strategies for the LBLT factorization
- implemented pivoted LLT/LDLT

# 0.20 - 0.21
- Project refactor

# 0.19
- Support matrix-scalar multiplication/division without the `Scale` wrapper for `f32`/`f64`.
- Implemented conjugate gradient, BiCGSTAB, and LSMR iterative solvers (currently gated by the `unstable` feature).
- Implemented Hermitian matrix pseudoinverse implementation. Thanks @lishen_ for the contribution.
- Implemented column and row mean and variance in `faer::stats`.
- Added more iterator and parallel iterator functions (`MatRef::[col|row]_partition`, `MatRef::par_[col|row]_partition`, etc.).
- Added `full` and `zeros` constructors to owned Col, Row, and Matrix ([issue-125](https://github.com/sarah-ek/faer-rs/issues/125)).
- Added `shape` function to return both the row and the column count of a matrix.
- Added several missing associated functions from the mut and owning variants of matrices.
- Implemented `core::iter::{Sum, Product}` for `c32` and `c64`.
- Sparse Cholesky can now be used with user-provided permutations.
- Simplified matrix constructors, adding a variant with a `_generic` prefix for the old behavior.
- LDLT and Bunch-Kaufman decompositions now stores the diagonal blocks instead of their inverses. This helps avoid infinities and NaNs when dealing with singular matrices.
- Integrated `nano-gemm` as a backend for small matrix multiplication.
- Significant performance improvements for small LLT and LDLT decompositions.

# 0.18
- Refactored the project so that `faer` contains all the core and decomposition implementations. `faer-{core,cholesky,lu,qr,svd,evd,sparse}` are now deprecated and will no longer be updated.
- Improved the multithreaded performance of the Eigenvalue decomposition for large matrices.
- Decomposition solve functions now accept column vectors as well as matrices.
- Implemented the L1 norm, and the squared L2 norm.
- Implemented conversions from sparse to dense matrices, by calling `mat.to_dense()`.
- Sparse matrices now support duplicated entries. Note that `faer` will not add duplicated entries to a matrix unless the opposite is explicitly mentioned in the function documentation. `faer` also will deduplicate entries when created with `Sparse{Col,Row}Mat::try_new_from_indices` and other similar functions.
- Implemented conversions from unsorted to sorted sparse matrices by calling `mat.to_sorted()` (or `mat.sort_indices()` for owned matrices).
- Implemented `{Col,Row}::try_as_slice[_mut]` functions that return data as a slice if it is contiguous.
- Implemented `.for_each_with_index` and `.map_with_index` for the matrix zipping API, which passes the matrix row and column indices as well as the values.
- Added `rand` support for randomly generating matrices in the `faer::stats` module, as well as for `faer::complex_native::{c32,c64}`.
- Implemented a pseudoinverse helper for the high level SVD and thin SVD decompositions.

# 0.17
- Implemented sparse matrix arithmetic operators (other than sparse-sparse matrix multiplication), and added mutable sparse views as well as owning sparse matrix containers.
- Implemented `try_from_triplets` for sparse matrices.
- Re-exported subcrates in `faer::modules`.
- Improved performance of the SVD decomposition for small matrices.
- Implemented `col!`, `row!` and `concat!` macros. Thanks @DeliciousHair for the contribution.
- Implemented more `c32/c64` operations. Thanks @edyounis for the contribution.
- Implemented the Kronecker product in `faer_core`. Thanks @edyounis for the contribution.
- Implemented (de)serialization of `Mat`. Thanks @cramt for the contribution.

# 0.16
- Implemented the index operator for row and column structures. Thanks @DeliciousHair for the contribution.
- Exposed a few sparse matrix operations in the high level API.
- Implemented sparse LU and QR, and exposed sparse decompositions in the high level API.
- Better assertion error messages in no_std mode.

# 0.15
- Implemented initial API of `Row`/`RowRef`/`RowMut` and `Col`/`ColRef`/`ColMut` structs for handling matrices with a single row or column.
- Implemented `[Mat|Col|Row]::norm_l2` and `[Mat|Col|Row]::norm_max` for computing the L2 norm of a matrix or its maximum absolute value.
- Fixed several bugs in the eigenvalue decompositions. Special thanks to @AlexMath for tracking down the errors.
- Updated `zipped!` macro API, which now requires a matching `unzipped!` for matching the closure arguments.
- Removed the limitation on the number of matrices that can be passed to `zipped!`.
- Added a `zipped!(...).map(|unzipped!(...)| { ... })` API to allow mapping a zipped pack of matrices and returns the result as a matrix.
- Updated `polars` dependency to 0.34.
- Speed improvements for complex matrix multiplication on AMD cpus.
- New SIMD functions in the Entity trait for aligned loads and stores.
- Renamed multiple methods such as `MatMut::transpose` to `MatMut::transpose_mut`.

# 0.14
- Implemented sparse data structures in `faer_core::sparse`.
- Implemented sparse Cholesky decompositions, simplicial and supernodal. Only the low level API is currently exposed in `faer-sparse`.
- Implemented dynamic regularization for the Bunch-Kaufman Cholesky decomposition.
- Implemented diagonal wrappers that can be used to interpret a matrix as a diagonal matrix, using `{MatRef,MatMut}::diagonal` and `{MatRef,MatMut}::column_vector_as_diagonal`.
- Implemented matrix multiplication syntax sugar for diagonal wrappers, and permutation matrices.
- Implemented `compute_thin_r` and `compute_thin_q` in `faer::solvers::{Qr,ColPivQr}`.
- Implemented initial SIMD support for aarch64.

# 0.13
- Implemented the Bunch-Kaufman Cholesky decomposition for hermitian indefinite matrices.
- Implemented dynamic regularization for the diagonal LDLT.
- Support conversions involving complex values using `IntoFaerComplex`, `IntoNalgebraComplex` and `IntoNdarrayComplex`.
- Refactored the Entity trait for better ergonomics.
- `faer` scalar traits are now prefixed with `faer_` to avoid conflicts with standard library and popular library traits.
- `no_std` and `no_rayon` are now supported, with the optional features `std` and `rayon` (enabled by default).
- Performance improvements in the eigenvalue decomposition and thin matrix multiplication.

# 0.12
- Implemented matrix chunked iterators and parallel chunked iterators.
- Renamed `{Mat,MatMut}::fill_with_zero` to `fill_zeros`
- Renamed `{Mat,MatMut}::fill_with_constant` to `fill`
- More ergonomic `polars` api.
- Refactored Entity and ComplexField SIMD api.
- Switched from DynStack/GlobalMemBuffer to PodStack/GlobalPodBuffer.
- Fixed usize overflow bug in eigenvalue decomposition.

# 0.11
- High level api implemented in `faer`.
- Renamed `Mat::with_dims` to `Mat::from_fn`.
- Renamed `{Mat,MatMut}::set_zeros` to `fill_with_zero`.
- Renamed `{Mat,MatMut}::set_constant` to `fill_with_constant`.

# 0.10
- Performance improvements for small matrices.
- Simpler SVD/EVD API for fixed precision floating point types.
- Simpler math operators (+, -, *). Thanks @geo-ant and @DJDuque.
- More robust pivoted decompositions for rank deficient matrices.
- Better overflow/underflow handling in matrix decompositions, as well as non finite inputs.
- Provide control over global parallelism settings in `faer-core`.
- Various bug fixes for complex number handling.

# 0.9
- Implement the non Hermitian eigenvalue decomposition.
- Improve performance of matrix multiplication.
- Improve performance of LU decomposition with partial pivoting.

# 0.8
- Refactor the core traits for better SIMD support for non native types, using a structure-of-arrays layout.
- Implement the Hermitian eigenvalue decomposition.

# 0.7
- Add `to_owned` function for converting a `MatRef/MatMut` to a `Mat`. Thanks @Tastaturtaste
- Allow comparison of conjugated matrices
- Performance improvements for `f32`, `c32`, and `c64`
- Performance improvements for small/medium matrix decompositions
- Refactor the `ComplexField` trait to allow for non `Copy` types. Note that types other than `f32`, `f64`, `c32`, `c64` are not yet supported.

# 0.6
- Start keeping track of changes.
- Complex SVD support.
- Improve performance for thin SVD.
- Fixed an edge case in complex Householder computation where the input vector is all zeros.

