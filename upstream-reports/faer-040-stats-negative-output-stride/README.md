# faer-040: the `stats` reductions write their output in reverse when the output axis has a negative stride

**What happens:** `stats::row_mean` / `row_varm` return their results in reverse order when the input's column stride is negative (a view such as `mat.reverse_cols()`), and `col_mean` / `col_varm` do the same for a negative row stride (`mat.reverse_rows()`). Both `NanHandling::Propagate` and `NanHandling::Ignore`. For `varm` the reversed means are applied to the reversed columns, so the result is not even a permutation of the correct one.
**Which input:** any matrix with a negative stride on the axis the output is written along. The program uses a 2x4 matrix whose column `j` holds `[0, 2(j+1)]`, compared as a reversed view and as an owned copy of that same view.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-09.

## Reproduce

```sh
cargo run
```

The program runs each reduction on a negative-stride view and on an owned copy of that view, counts the calls where the two disagree, and exits with status 1 when any do. That is 8 calls — 2 operations x 2 axes x 2 `NanHandling` values — and it counts the means and the variances separately, so a fix to one of the two changes the total. On 0.24.4 all 8 disagree and the program prints `WRONG: 8 of 8 calls disagree between a negative-stride view and its owned copy (4 mean, 4 variance)`. In 0.24.4 `row_*` writes one value per column (`out.ncols() == mat.ncols()`) and `col_*` one per row (`out.nrows() == mat.nrows()`), so the axis that must be reversed with the input is the one that carries the output.

| call | expected (owned copy) | negative-stride view, `faer` 0.24.4 |
|---|---|---|
| `row_mean`, reversed columns | `[4, 3, 2, 1]` | **`[1, 2, 3, 4]`** |
| `row_varm`, reversed columns | `[32, 18, 8, 2]` | **`[20, 10, 20, 50]`** |
| `col_mean`, reversed rows | `[5, 0]` | **`[0, 5]`** |
| `col_varm`, reversed rows | `[6.666666666666666, 0]` | **`[33.33333333333333, 40]`** |

The `Propagate` and `Ignore` rows print the same numbers.

## Cause

All four functions in `src/stats/meanvar.rs` (0.24.4) normalise the strides like this, for example `col_mean_ignore` at `:587-591`:

```rust
let mat = if mat.col_stride() >= 0 { mat } else { mat.reverse_cols() };
let mat = if mat.row_stride() >= 0 {
    mat
} else {
    mat.reverse_rows()
};
```

The rows are the output axis of a column reduction. Reversing them on `mat` alone, without reversing `out` (and, for `varm`, `col_mean` as well), writes entry `i` of the result to position `n - 1 - i`. The same block is at `:608-612` (`col_varm_ignore`), `:637-641` (`col_mean_propagate`) and `:661-665` (`col_varm_propagate`). Reversing `mat.reverse_cols()` on the reduced axis is harmless; the defect is only on the output axis. `row_*` delegates to `col_*` on the transposed matrix and inherits it.

## How the fork fixed it

`out` is reversed together with the input in `col_mean_*`, and `out` and `col_mean` together in `col_varm_*` (`faer/src/stats/meanvar.rs`, all four functions). The reduction then computes the same values in the same order as for a positive stride.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the four normalisation blocks are unchanged in upstream `main` at `0ecd846` (2026-10-22), `faer/src/stats/meanvar.rs:585-591`, `:608-612`, `:637-641`, `:661-665`.

## References

* Fork fix: tensor4all/faer#319 (merge commit `c911ff3`).
* Where it was found: tensor4all/faer#317, downstream AtelierArith/Colmat.jl.
