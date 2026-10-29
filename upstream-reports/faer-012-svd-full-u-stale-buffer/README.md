# faer-012: the full U of a tall SVD keeps old entries of the output buffer

**What happens:** `linalg::svd::svd` returns a full `U` that is not orthogonal when the input is tall and the caller's `U` buffer is not zero beforehand.
**Which input:** any tall matrix (more rows than columns, so the SVD first takes a QR factorization) with a full `U` requested into a reused buffer. Wide matrices go through the same code after a transpose, with `V` in place of `U`. The program below uses the 4 x 2 identity and a `U` buffer filled with ones.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program fills a 4 x 4 buffer with ones and passes it as the full `U` of a 4 x 2 SVD.
The full `U` must be orthogonal whatever the buffer held before.

| | max \|A - U S V^T\| | max \|U^T U - I\| |
|---|---|---|
| expected | about 1e-16 | about 1e-16 |
| `faer` 0.24.4 | 0 | **2** |

The returned `U` is `[[1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]]`.
Its top right 2 x 2 block is the old buffer content.
The reconstruction only uses the first two columns, so it does not show the problem.

## Cause

For a tall matrix, the public `svd` function first factors `A = Q R`, computes the SVD of the square `R` into the top left `n x n` block of `U`, and then applies `Q` to `U`.
Before applying `Q`, it clears the bottom rows `n..m` of `U` and writes ones on the diagonal of the bottom right block (`src/linalg/svd/mod.rs:645-651`).
It never clears the top right block, rows `0..n` and columns `n..m`.
That block still holds the caller's data when the Householder reflectors are applied (line 653).
So the extra columns of `U` are `Q` times old data, instead of `Q` times `[0; I]`.

## How the fork fixed it

The fork also zeroes the top right block, `u.submatrix_mut(0, n, n, m - n).fill(zero())`, before applying the reflectors.
The extra columns are then `Q [0; I]`, as LAPACK's `dorgqr` produces by starting from the identity.
Run against the fork, the program prints `max |U^T U - I| = 0`.

Fork regression test: `linalg::svd::tests::full_factors_ignore_old_output_entries` (4 x 2 and 2 x 4, buffers filled with ones).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#47 (commit `ee70868`).
* Where it was found: tensor4all/faer#41.
