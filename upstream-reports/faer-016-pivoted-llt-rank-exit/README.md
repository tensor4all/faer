# faer-016: pivoted Cholesky leaves input entries in the factor after a rank stop

**What happens:** when pivoted Cholesky stops early at numerical rank `r`, the columns of the factor after `r` still hold entries of the input. `reconstruct` then adds them in and returns a matrix with an error of order one.
**Which input:** a positive semidefinite (or nearly so) matrix whose numerical rank is below its size. The program uses `ones(3, 3) + eps * I`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program factors `A = ones(3, 3) + eps * I` with `linalg::cholesky::llt_pivoting::factor::cholesky_in_place`.
It rebuilds `P^T L L^T P` with `llt_pivoting::reconstruct::reconstruct`.
`A` has eigenvalues `eps`, `eps` and `3 + eps`, so rank 1 at the default tolerance is a fair answer.
The rebuilt matrix should then differ from `A` by about `eps`.

| | numerical rank | rebuilt `A[2,2]` |
|---|---|---|
| expected | 1 | 1.0000000000000002 (error about 2e-16) |
| `faer` 0.24.4 | 1 | **3.0000000000000004** (error 2) |

## Cause

In `src/linalg/cholesky/llt_pivoting/factor.rs:120-123`, when the next pivot is below the tolerance, the factorization sets `rank = j`, writes that pivot to `a[(j, j)]`, and leaves the loop.
The columns `j..n` of the lower triangle are not cleared.
They still hold the input entries, partly updated.
`reconstruct` (`src/linalg/cholesky/llt_pivoting/reconstruct.rs:29-40`) multiplies the whole lower triangle as `L L^H`.
It takes no rank argument, so the stale columns are added to the result.

## How the fork fixed it

At the rank exit the fork fills columns `j..n` of the factor with zeros.
The stored factor is then the rank `r` factor, and `reconstruct` returns its `L L^H`.
With that change this program prints rank 1 and a rebuilt `A[2,2]` of 1.

Fork regression test: `linalg::cholesky::llt_pivoting::reconstruct::tests::test_early_rank_reconstruct`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#47 (commit `946ac9f`).
* Where it was found: tensor4all/faer#45.
