# faer-006: `Lblt::reconstruct` ignores the block diagonal

**What happens:** `Lblt::reconstruct` returns `L L^T` instead of `L B L^T`, so it does not give back the factored matrix.
**Which input:** any matrix. The program below uses `diag(2, 3)`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program factors `A = diag(2, 3)` with `Mat::lblt` (the Bunch Kaufman `L B L^T` factorization) and calls `reconstruct` on the result.
The rebuilt matrix should equal `A`.

| | diagonal of the rebuilt matrix |
|---|---|
| expected | [2, 3] |
| `faer` 0.24.4 | **[1, 1]** |

For `[[2]]` the program in the same way prints `[[1]]`.

## Cause

The low level routine `linalg::cholesky::lblt::reconstruct::reconstruct` takes the block diagonal `B` as a separate argument, `diagonal`.
It only checks the length of `diagonal` (`src/linalg/cholesky/bunch_kaufman/reconstruct.rs:27`).
It then reads the diagonal of `B` from `L` instead: `L[(j, j)]` at line 41 for a 1 x 1 block and lines 48 to 49 for a 2 x 2 block.
The high level `Lblt` stores the diagonal of `B` in `B_diag` and sets the diagonal of `L` to one (`src/linalg/solvers.rs:916-917`).
`Lblt::reconstruct` passes `B_diag` as `diagonal` (line 2113), but it is never read.
So every diagonal entry of `B` is taken as 1.
The low level routine gives the right answer only when the caller passes the packed factor, whose diagonal holds `B`.

## How the fork fixed it

The fork reads `diagonal[j]` and `diagonal[j + 1]` in place of `L[(j, j)]` and `L[(j + 1, j + 1)]`.
With that change this program prints `[2, 3]`.

Fork regression test: `linalg::cholesky::lblt::reconstruct::tests::test_reconstruct_separate_diagonal`. It rebuilds `L B L^T` from a unit `L` and a separate `B` for `f32`, `f64`, `c32` and `c64`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#46 (commit `4e2c241`).
* Where it was found: tensor4all/faer#34.
