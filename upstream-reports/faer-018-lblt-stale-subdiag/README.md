# faer-018: LBL^T factorization leaves old values in the caller's `subdiag` buffer

**What happens:** the low level Bunch Kaufman factorization does not write `subdiag` for some 1 x 1 pivots. A reused buffer keeps its old values there, and the solve then reads the factors wrongly: it panics or returns a wrong answer.
**Which input:** a matrix with a 1 x 1 pivot that has nothing below it (the last column, or a column that is already zero below the diagonal), with a `subdiag` buffer that is not zero on entry. The program uses `diag(1, 2, ..., n)` and a buffer filled with 7.0.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program calls `linalg::cholesky::lblt::factor::cholesky_in_place` on `A = diag(1, 2, ..., n)` for `n` = 1, 3 and 100.
`subdiag` holds the subdiagonal of the 2 x 2 blocks of `B`, and must be 0 for every 1 x 1 block.
A diagonal `A` has only 1 x 1 blocks, so every entry must be 0.
`n = 100` is above the default block size, so the blocked path runs too.
The program then solves `A x = A [1, ..., 1]^T` with `lblt::solve::solve_in_place_with_conj` on the returned factors.
It silences the panic hook and prints "panic" when the solve panics.

| `n` | nonzero `subdiag` entries, expected | `faer` 0.24.4 | solve `max abs(x - 1)`, expected | `faer` 0.24.4 |
|---|---|---|---|---|
| 1 | 0 | **1** | about 1e-16 | **panic** |
| 3 | 0 | **3** | about 1e-16 | **panic** |
| 100 | 0 | **100** | about 1e-16 | **1.0** |

The high level `Mat::lblt` passes a zeroed buffer, so it does not show the problem.

## Cause

Both factorization loops in `src/linalg/cholesky/bunch_kaufman/factor.rs` mark a column as `nothing_to_do` when it is the last one or its largest entry below the diagonal is zero.
This is in the blocked step (lines 543 to 545) and in the unblocked loop (lines 814 to 816).
For such a column they leave the step early (`break 'next_iter`, lines 617 to 619 and 864 to 866).
That jump skips the `subdiag[0] = zero()` that the normal 1 x 1 path writes (lines 624 and 870).
So `subdiag` keeps the caller's value for that column.
The solve treats a nonzero `subdiag[i]` as the start of a 2 x 2 block (`src/linalg/cholesky/bunch_kaufman/solve.rs:67`).
For the last column it then reads row `i + 1`, which is out of bounds and panics.
For other columns it solves the wrong block system.

## How the fork fixed it

The early exit now writes `subdiag[0] = zero()` in both loops.
With that change this program prints no nonzero entries and a solve error of at most 1.1e-16.

Fork regression test: `linalg::cholesky::lblt::reconstruct::tests::test_factor_clears_stale_subdiag` (`n` = 1, 3 and 100, `subdiag` prefilled with 7).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#67 (commit `ff1df37`).
* Where it was found: tensor4all/faer#31, the LAPACK validation of the indefinite factorization with prefilled output buffers.
