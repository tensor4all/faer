# faer-039: `norm_max` drops NaN entries

**What happens:** `MatRef`, `ColRef` and `RowRef::norm_max` ignore entries whose magnitude is NaN and return the largest finite magnitude instead, or `0` when every entry is NaN. `norm_l1` and `norm_l2` of the same input return NaN, as IEEE propagation and LAPACK's `xLANGE('M')` do.
**Which input:** any matrix, vector or row with at least one NaN. The program uses a 1x2 matrix, a plain 2-vector, a `c64` vector with the NaN in the real and in the imaginary part, a 2x3 column-major matrix, a strided row, an all-NaN vector, a 100-vector with a NaN at index 57, a 129-vector with a NaN at index 128 (the first length whose column reaches the pairwise split above `LINEAR_IMPL_THRESHOLD = 128`) and the transpose of the 2x3.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-09.

## Reproduce

```sh
cargo run
```

The program calls `norm_max` on each case, prints the expected and the actual value, and exits with status 1 if any result is finite.

| case | route taken inside `norm_max` | expected | `faer` 0.24.4, x86_64 |
|---|---|---|---|
| 1x2 row-major, one NaN | SIMD kernel | NaN | **1** |
| 2-vector, one NaN | SIMD kernel | NaN | **1** |
| 2-vector `c64`, NaN real part | SIMD kernel | NaN | **1** (aarch64: **0**) |
| 2-vector `c64`, NaN imaginary part | SIMD kernel | NaN | **1** |
| 2x3 column-major, one NaN | SIMD kernel | NaN | **1** |
| 2x3, row 1 strided | scalar fallback | NaN | **1** |
| all-NaN vector | SIMD kernel | NaN | **0** |
| 100-vector, NaN at 57 | SIMD kernel | NaN | **99** |
| 129-vector, NaN at 128 | SIMD kernel and pairwise combine | NaN | **127** |
| transpose of the 2x3 | SIMD kernel | NaN | **1** |
| reference: `norm_l1`, `norm_l2` of the 2-vector | — | NaN | NaN (correct) |

The NaN is lost in the contiguous SIMD kernel, in the pairwise split (the 129-element case), in the strided row and in the complex cases, so all three branches of `norm_max` are wrong. The strided row is the case that exercises the scalar fallback, and that branch is reachable from a standalone program: `norm_max` transposes the view when `ncols() > 1 && col_stride().unsigned_abs() == 1` (`:60-62`) and then takes the SIMD route only when `try_as_col_major()` accepts the view, i.e. when its row stride is 1 (`mat/matref.rs:1448`); every other view falls through to the scalar loop at `:118-126`.

`RowRef::norm_max` reaches that loop through its own view: `RowRef::as_mat` is `transpose().as_mat().transpose()` (`row/rowref.rs:422-424`), and `ColRef::as_mat` gives the matrix it builds a column stride of zero (`col/colref.rs:465-477`). A row of the column-major `m` in this program therefore becomes a 1 x 3 view with **row stride 0** and column stride `m.col_stride() = 2`: `:60-62` does not transpose it (its column stride is not 1), and `try_as_col_major` rejects it (its row stride is not 1), so `m.row(1).norm_max()` runs the scalar loop. A row of a *row-major* matrix, whose stride is 1, is transposed at `:60-62` into a contiguous column instead and does take the SIMD route.

The table was recorded on an x86_64 Linux host with `cargo run --locked`; the aarch64 macOS run this entry was first recorded on agrees on every row except the one marked above.

## Cause

`src/linalg/reductions/norm_max.rs` in 0.24.4 drops NaN in three places:

- the SIMD kernel accumulates with `simd.max` (`:21-26`). On aarch64 that lowers to `fmax`, which is IEEE `maxNum` and returns the non-NaN operand; on x86 it lowers to `maxpd`, which returns its **second** operand when either operand is NaN. Neither is NaN-propagating the way `+` is, and which value comes out depends on the instruction and on the operand order the kernel emits;
- the pairwise halves above `LINEAR_IMPL_THRESHOLD` (128) are combined with `fmax` (`:43` and `:56`), which returns the non-NaN operand;
- the scalar fallback combines with `fmax` as well (`:126`).

The result therefore depends on how the reduction is traversed: a value is only kept when nothing propagates it. The two hosts this program was run on lose the NaN in every case, and they differ in exactly one row (`2-vector c64, NaN real part`: `1` on x86_64, `0` on aarch64), which is the `maxpd` operand-order effect showing through: the NaN's fate is decided by the emitted instruction rather than by IEEE propagation.

## How the fork fixed it

The SIMD kernel accumulates a second channel, a sum of the magnitudes (`faer/src/linalg/reductions/norm_max.rs`, `nan_acc`). Every term of that sum is `>= 0`, `+inf` or NaN, so the sum is NaN exactly when an entry is NaN; a NaN there makes the kernel return NaN. The pairwise combine uses `max_propagate_nan` instead of `fmax`, and the scalar loop returns NaN at the first NaN entry. An empty input still returns `0`, and the behaviour change is recorded in `CHANGELOG.md` and in the rustdoc of all four entry points.

Fork tests: `test_norm_max_propagates_nan`, `test_norm_max_finite_and_inf_paths_agree` and `test_norm_max_empty_is_zero` in `faer/src/linalg/reductions/norm_max.rs`; the first covers the same cases as this program and the `norm_l1` / `norm_l2` consistency check.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: `simd.max`, `fmax` and the scalar loop are the same in upstream `main` at `0ecd846` (2026-10-22), `faer/src/linalg/reductions/norm_max.rs:21-56`, `:126`.

## References

* Fork fix: tensor4all/faer#319 (merge commit `c911ff3`).
* Where it was found: tensor4all/faer#316, downstream AtelierArith/Colmat.jl.
