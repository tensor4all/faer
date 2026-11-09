# faer-039: `norm_max` drops NaN entries

**What happens:** `MatRef`, `ColRef` and `RowRef::norm_max` drop entries whose magnitude is NaN and return a finite value instead: the largest finite magnitude in every case this program measures. What it returns depends on the input layout and on the instruction the reduction emits, not only on whether a NaN is present, so the table below is the claim, not a rule about all inputs. `norm_l1` and `norm_l2` of the same input return NaN, as IEEE propagation and LAPACK's `xLANGE('M')` do.
**Which input:** matrices, vectors and rows with at least one NaN, when that NaN is not overwritten before the final combine; which NaN survives depends on the layout, as the table shows. The program uses a 1x2 matrix, a plain 2-vector, a `c64` vector with the NaN in the real and in the imaginary part, a 2x3 column-major matrix, a strided row, an all-NaN vector, a 100-vector with a NaN at index 57, a 129-vector with a NaN at index 128 (the first length whose column reaches the pairwise split above `LINEAR_IMPL_THRESHOLD = 128`) and the transpose of the 2x3.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-09.

## Reproduce

```sh
cargo run
```

The program calls `norm_max` on each case, prints the expected and the actual value, and exits with status 1 if any result is finite.

| case | route taken inside `norm_max` | expected | `faer` 0.24.4, x86_64 |
|---|---|---|---|
| 1x2 column-major, one NaN | SIMD kernel | NaN | **1** |
| 2-vector, one NaN | SIMD kernel | NaN | **1** |
| 2-vector `c64`, NaN real part | SIMD kernel | NaN | **1** (aarch64: **0**) |
| 2-vector `c64`, NaN imaginary part | SIMD kernel | NaN | **1** |
| 2x3 column-major, one NaN | SIMD kernel | NaN | **1** |
| 2x3, row 1 strided | scalar fallback | NaN | **1** |
| all-NaN vector | SIMD kernel | NaN | **0** |
| 100-vector, NaN at 57 | SIMD kernel | NaN | **99** |
| 129-vector, NaN at 128 | SIMD kernel, then the split's combine | NaN | **127** |
| transpose of the 2x3 | SIMD kernel | NaN | **1** |
| reference: `norm_l1`, `norm_l2` of the 2-vector | — | NaN | NaN (correct) |

The NaN is lost in the contiguous SIMD kernel, in the split (the 129-element row shows the split losing it, but not that the combine is the only cause), in the strided row and in the complex cases, so all three branches of `norm_max` are wrong. The strided row is the case that exercises the scalar fallback, and that branch is reachable from a standalone program: `norm_max` transposes the view when `ncols() > 1 && col_stride().unsigned_abs() == 1` (`:60-62`) and then takes the SIMD route only when `try_as_col_major()` accepts the view, i.e. when its row stride is 1 (`mat/matref.rs:1448`); every other view falls through to the scalar loop at `:118-126`.

`RowRef::norm_max` reaches that loop through its own view: `RowRef::as_mat` is `transpose().as_mat().transpose()` (`row/rowref.rs:422-424`), and `ColRef::as_mat` gives the matrix it builds a column stride of zero (`col/colref.rs:465-477`). A row of the `m` in this program therefore becomes a 1 x 3 view with **row stride 0** and column stride **8**. That is the column stride of `m` itself: an owned `Mat` pads its column stride to the 64 byte alignment of its allocation, so the 2 x 3 `m` here has row stride 1 and column stride 8, not 3. The program prints those strides, so its run shows `m: row_stride 1 col_stride 8; m.row(1).as_mat(): row_stride 0 col_stride 8`. `:60-62` does not transpose that view (its column stride is not 1) and `try_as_col_major` rejects it (its row stride is not 1), so `m.row(1).norm_max()` runs the scalar loop. A row whose entries are adjacent does take the SIMD route, and it does so through the same two lines: in the row-major view `m.transpose()` the first row is a 1 x 2 view with row stride 0 and column stride 1, so `:60-62` transposes it into a contiguous 2 x 1 column with row stride 1, which `try_as_col_major` accepts. The strided row of this program's `m` is the case that does not.

The table was recorded on an x86_64 Linux host with `cargo run --locked`; the aarch64 macOS run this entry was first recorded on agrees on every row except the one marked above.

The 129-element row does not isolate the combine: its one-element tail is reduced to `0` on its own, and `fmax(127, 0)` is `127` for a NaN-propagating combine as well, so the row shows the split losing a NaN rather than a combine that is the only cause.

## Cause

`src/linalg/reductions/norm_max.rs` in 0.24.4 drops NaN in three places:

- the SIMD kernel accumulates with `simd.max` (`:21-26`), and `faer-traits` 0.24.0 implements that as `select(gt(lhs, rhs), lhs, rhs)` (`src/lib.rs:654-663`) — not as a machine `max`. A NaN therefore survives only when it arrives as the **right** operand (`max(1.0, NaN)` is NaN, `max(NaN, 0.0)` is 0.0), and the kernel feeds the accumulator as the left operand. An arriving NaN therefore survives that update (`max(0.0, NaN)` is NaN), but the next value of the same lane then arrives as the right operand of an accumulator that is already NaN (`max(NaN, v)` is `v`), so the NaN is dropped unless it is the last value of its lane; otherwise it survives the kernel and is dropped later, in the pairwise combine or in the horizontal reduction. This is the same operand-order effect that `maxpd` has on x86, which is why the two hosts differ only in the row whose NaN reaches the reduction by a different route;
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
