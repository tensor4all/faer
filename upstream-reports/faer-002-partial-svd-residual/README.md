# faer-002: partial SVD reports a pair as converged when it is not

**What happens:** `partial_svd` returns a singular pair that it counts as converged, but the pair is far from a true singular pair.
**Which input:** any matrix where the algorithm restarts and keeps an unconverged pair in a lower slot. The program below uses a random 200 x 120 matrix.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program asks for the 10 largest singular triplets of a fixed random 200 x 120 matrix.
It uses a small search space (12 to 24 vectors) so the algorithm restarts several times.
For each pair counted as converged, it prints the residual `|A^T A v - s^2 v|`.
A converged pair should give a residual near rounding level.

| | pairs counted as converged | largest residual among them |
|---|---|---|
| expected | 10 | about 1e-11 |
| `faer` 0.24.4 | 10 | **9.3e-2** (pair 4, value 23.218875; the true value is 23.218876) |

Every other pair has a residual near 2e-12.
In a sweep of 1000 runs (200 random seeds, five problem shapes), 46 showed such a pair.

## Cause

The partial SVD restarts by keeping some pairs and dropping others.
Each pair has a convergence estimate stored in `residual`.
Pairs whose estimate is below the tolerance are locked (`src/operator/svd/mod.rs:228`).
The next Lanczos step builds its coupling row from the same array (`H[(i, k)] = residual[i]`, line 318).
So `residual` must be compacted together with `H`, `X` and `Y`.
The `Group::Lock` branch does this (`residual[lo] = residual[hi]`, line 259).
The `Group::Retain` branch (lines 273 to 275) moves `H[(hi, hi)]` and the columns of `X` and `Y` to slot `mi`, but leaves `residual[mi]` unchanged.
After a restart where a dropped pair comes before a kept one, slot `mi` holds the estimate of a different pair.
A pair can then be locked with an estimate below the tolerance while its own residual is large.

## How the fork fixed it

The fork adds the same move to the `Group::Retain` branch: `residual[mi] = residual[hi]`.
With only that line added to the 0.24.4 source, this program prints residuals near 2e-12 for all ten pairs.
The partial self adjoint eigensolver is not affected, because it rebuilds its coupling row from `H Q` instead of from a stored estimate.

Fork regression test: `matrix_free::svd::tests::test_small_cplx` (a 768 x 512 complex matrix, 32 pairs). It checks the residual of every returned pair.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## Note

In this run pairs 5 and 6 print the same value, 22.944167.
That value occurs once in the true spectrum, and the true tenth value 21.652590 is missing.
This also happens with the fix above, so it is a separate behaviour and is not covered by this report.

## References

* Fork fix: tensor4all/faer#108 (commit `1ac38c6`).
* Where it was found: tensor4all/faer#102. With AVX-512 enabled, `test_small_cplx` returned a pair with residual 0.14 to 0.86 that was counted as converged.
