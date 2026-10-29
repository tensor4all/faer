# faer-013: 2 x 2 self adjoint eigenvalues overflow for a finite spectrum

**What happens:** the self adjoint eigensolver returns `[inf, inf]` for a 2 x 2 matrix whose eigenvalues are finite. The call returns `Ok`.
**Which input:** a 2 x 2 self adjoint matrix whose two diagonal entries add up to more than the largest finite value. The program uses `1e308 * I`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program calls `self_adjoint_eigenvalues` on the diagonal matrix `[[1e308, 0], [0, 1e308]]`.
Both exact eigenvalues are `1e308`.

| | eigenvalues |
|---|---|
| expected | [1e308, 1e308] |
| `faer` 0.24.4 | **[inf, inf]** |

## Cause

The tridiagonal QR algorithm has a closed form for `n = 2` (`src/linalg/evd/tridiag_evd.rs:27-80`).
It computes the mean of the two diagonal entries as `t1 = (a + d) * half` (line 34).
The sum `a + d = 2e308` overflows to `inf` before the halving.
Both eigenvalues `t1 - t0` and `t1 + t0` (lines 35 to 36) are then `inf`.
This branch runs before the scaling by the largest entry that the general case applies (line 85 onward), so nothing keeps the sum in range.

## How the fork fixed it

The fork keeps `(a + d) / 2` when the sum is finite.
When the sum is not finite, it uses `a / 2 + d / 2` instead.
Halving first in every case was not chosen, because it would round a subnormal diagonal to zero.
With that change this program prints `[1e308, 1e308]`.

Fork regression tests, in `faer/src/linalg/evd/mod.rs`:
* `scale_and_empty_tests::two_by_two_finite_spectrum_at_the_range_limit` (`f64` and `f32`);
* `scale_and_empty_tests::two_by_two_subnormal_diagonal`, which checks that a subnormal diagonal is not rounded to zero.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## Note

The same closed form also computes `(a - d).hypot(2 b) / 2` (lines 32 to 33), which overflows when `a` and `d` have opposite signs.
That case is covered by a separate report, [faer-017](../faer-017-tridiag-evd-range/).

## References

* Fork fix: tensor4all/faer#47 (commit `946ac9f`).
* Where it was found: tensor4all/faer#42.
