# faer-005: the values only SVD loses a nonzero singular value

**What happens:** `Mat::singular_values` returns all zeros for a matrix whose largest singular value is 1, and reports no error. The full SVD of the same matrix is correct. Any call that does not request `V` is affected.
**Which input:** a matrix of size at least 128 (where the SVD switches to divide and conquer) whose bidiagonal form has a zero block next to a nonzero entry at a split. The program below uses a 256 x 256 zero matrix with `A[128, 128] = 1`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program builds the 256 x 256 matrix with a single entry 1 at (128, 128).
Its singular values are 1, 0, 0, ... .
It compares the leading value from `svd()` and from `singular_values()`.

| | leading singular value |
|---|---|
| expected | 1 |
| `faer` 0.24.4, `svd()` | 1 |
| `faer` 0.24.4, `singular_values()` | **0** |

A smaller case shows the same thing.
An 8 x 8 matrix with `A[4,4] = 1` and `SvdParams::recursion_threshold = 4`, passed to `linalg::svd::svd`, gives `s[0] = 0` when `V` is not requested (values only, or `U` only) and `s[0] = 1` when `V` is requested.

## Cause

Divide and conquer splits the bidiagonal matrix in two halves, solves each half, and merges them.
The bidiagonal solver works on the transposed bidiagonal, so its left factor becomes the caller's `V`.
When `V` is not requested, each half keeps only two rows of its left factor: the first row and the last row (`MatU::TwoRows`).
For a half that is entirely zero, the left factor is the identity.
Its two rows are then row 0 and row n of the identity, so the entry `u[(1, n)]` must be 1.
The code sets it to 0 instead (`src/linalg/svd/bidiag_svd.rs:1075`).
The parent merge reads this entry as `lambda` (lines 1177 to 1180) and uses `alpha * lambda` (line 1185) to couple the halves.
With `lambda = 0` the coupling is lost, and the nonzero singular value at the split disappears.
A call that requests `V` takes the `MatU::Full` branch (lines 1068 to 1071), which writes the whole identity, so it is not affected.

## How the fork fixed it

The fork sets `u[(1, n)] = 1` in the `MatU::TwoRows` branch, a one line change.
With only this line changed in the 0.24.4 source, this program prints 1 for both calls.

Fork regression test: `linalg::svd::tests::test_zero_child_at_split` (`f64` and `c64`, n = 128 and 256, the nonzero entry at the first, middle and last diagonal position, all U and V output choices compared with the QR algorithm).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#39 (commit `ccfaa56`), also part of tensor4all/faer#46.
* Where it was found: tensor4all/faer#33.
