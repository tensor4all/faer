# faer-039: `norm_max` drops NaN entries

**What happens:** `MatRef`, `ColRef` and `RowRef::norm_max` ignore entries whose magnitude is NaN and return the largest finite magnitude instead, or `0` when every entry is NaN. `norm_l1` and `norm_l2` of the same input return NaN, as IEEE propagation and LAPACK's `xLANGE('M')` do.
**Which input:** any matrix, vector or row with at least one NaN. The program uses a 1x2 matrix, a plain 2-vector, a `c64` vector with the NaN in the real and in the imaginary part, a 2x3 column-major matrix, a strided row, an all-NaN vector and a 100-vector with a NaN at index 57.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-09.

## Reproduce

```sh
cargo run
```

The program calls `norm_max` on each case, prints the expected and the actual value, and exits with status 1 if any result is finite.

| case | expected | `faer` 0.24.4 |
|---|---|---|
| 1x2 row-major, one NaN | NaN | **1** |
| 2-vector, one NaN | NaN | **1** |
| 2-vector `c64`, NaN real part | NaN | **0** |
| 2-vector `c64`, NaN imaginary part | NaN | **1** |
| 2x3 column-major, one NaN | NaN | **1** |
| 2x3, row 1 strided | NaN | **1** |
| all-NaN vector | NaN | **0** |
| 100-vector, NaN at 57 | NaN | **99** |
| transpose of the 2x3 | NaN | **1** |
| reference: `norm_l1`, `norm_l2` of the 2-vector | NaN | NaN (correct) |

The NaN is lost on every path: contiguous SIMD, strided rows, the scalar fallback, and the complex cases.

## Cause

`src/linalg/reductions/norm_max.rs` in 0.24.4 drops NaN in three places:

- the SIMD kernel accumulates with `simd.max` (`:21-26`), which is `maxpd` and returns the non-NaN operand;
- the pairwise halves are combined with `fmax` (`:43` and `:56`), which by IEEE 754 `maxNum` returns the non-NaN operand;
- the scalar fallback combines with `fmax` as well (`:126`).

The result therefore depends on the traversal order: `simd.max(x, NaN)` and `simd.max(NaN, x)` both return `x`, so whichever finite value the accumulator holds survives.

## How the fork fixed it

The SIMD kernel accumulates a second channel, a sum of the magnitudes (`faer/src/linalg/reductions/norm_max.rs`, `nan_acc`). Every term of that sum is `>= 0`, `+inf` or NaN, so the sum is NaN exactly when an entry is NaN; a NaN there makes the kernel return NaN. The pairwise combine uses `max_propagate_nan` instead of `fmax`, and the scalar loop returns NaN at the first NaN entry. An empty input still returns `0`, and the behaviour change is recorded in `CHANGELOG.md` and in the rustdoc of all four entry points.

Fork tests: `test_norm_max_propagates_nan`, `test_norm_max_finite_and_inf_paths_agree` and `test_norm_max_empty_is_zero` in `faer/src/linalg/reductions/norm_max.rs`; the first covers the same cases as this program and the `norm_l1` / `norm_l2` consistency check.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: `simd.max`, `fmax` and the scalar loop are the same in upstream `main` at `0ecd846` (2026-10-22), `faer/src/linalg/reductions/norm_max.rs:21-56`, `:126`.

## References

* Fork fix: tensor4all/faer#319 (merge commit `c911ff3`).
* Where it was found: tensor4all/faer#316, downstream AtelierArith/Colmat.jl.
