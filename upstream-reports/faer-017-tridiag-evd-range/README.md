# faer-017: tridiagonal EVD fails for finite spectra near the ends of the `f64` range

**What happens:** `tridiagonal_self_adjoint_evd` returns `[-inf, inf]` with `Ok` for one finite spectrum, and `NoConvergence` for another.
**Which input:** a symmetric tridiagonal matrix whose entries are near the overflow threshold, or are subnormal. The program uses two such matrices.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program calls `linalg::evd::tridiagonal_self_adjoint_evd` with eigenvectors on two inputs.
Both spectra are finite and representable.

| input `(diag, offdiag)` | expected | `faer` 0.24.4 |
|---|---|---|
| `([-1e308, 1e308], [0])` | `Ok`, [-1e308, 1e308] | **`Ok`, [-inf, inf]** |
| `([2s, 2s, 2s], [-s, -s])`, `s = 1e-310` | `Ok`, `s * [2 - sqrt(2), 2, 2 + sqrt(2)]` | **`Err(NoConvergence)`** |

The first matrix is already diagonal.

## Cause

Both cases are in `qr_algorithm` (`src/linalg/evd/tridiag_evd.rs:9`).

1. For `n = 2` it uses a closed form, before any scaling.
   It computes `t0 = (a - d).hypot(2 b) / 2` (lines 32 to 33).
   Here `a - d = -2e308` overflows to `-inf`, so `t0 = inf`.
   The eigenvalues `t1 - t0` and `t1 + t0` (lines 35 to 36) become `-inf` and `inf`.
2. For `n > 2` it divides the matrix by its largest entry `max` by multiplying with `max_inv = max.recip()` (lines 85 to 95).
   For `max = 2e-310`, `1 / max` is larger than the largest finite value, so `max_inv = inf`.
   All entries become `inf`, and the iteration never converges.

The divide and conquer backend, used from size 128 on by default, does not scale the input at all.
The fork found that it also returns `inf` or `NaN` near the top of the range (sizes 129 and 300).
This program does not test that backend.

## How the fork fixed it

Both backends now rescale `(diag, offdiag)` before they start, as LAPACK `dsteqr` does.
When the largest entry is outside `[sqrt(min_positive) / eps, sqrt(max_positive)]`, every entry is divided by it.
Dividing, not multiplying by the reciprocal, also works for a subnormal largest entry.
The computed eigenvalues are multiplied back at the end.
The scaling runs before the `n = 2` closed form too.
With that change this program prints the exact spectra for both inputs.

Fork regression tests, in `faer/src/linalg/evd/mod.rs`: `tridiagonal_range_tests::two_by_two_at_the_range_limit`, `tridiagonal_range_tests::subnormal_three_by_three` and `tridiagonal_range_tests::spectrum_at_both_range_limits` (both backends).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## Note

The same `n = 2` closed form also overflows in `(a + d) * half` (line 34) when both diagonal entries are large with the same sign.
That case is covered by a separate report, [faer-013](../faer-013-evd-2x2-overflow/).

## References

* Fork fix: tensor4all/faer#62 (commit `f59395a`).
* Where it was found: tensor4all/faer#57.
