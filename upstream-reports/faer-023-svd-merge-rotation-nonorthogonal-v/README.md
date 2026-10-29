# faer-023: the divide and conquer SVD returns a non orthogonal V

**What happens:** `Mat::svd` returns a `V` whose orthogonality error is 262 times LAPACK's bound, while `U` and the reconstruction are fine.
**Which input:** a bidiagonal matrix whose entries span a very wide range, so that deep merges of divide and conquer work with tiny numbers. The program below uses a 128 x 128 `f32` lower bidiagonal matrix with entries `eps^u`, `u` uniform in [-2, 2] (LAPACK's xCHKBD test type 16), and default parameters.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The entries are drawn from a fixed xorshift generator, so the input is the same on every machine.
The program prints three ratios in the style of LAPACK's ORT01 and BDT01 checks, all computed in `f64`.
Each must be below 50.

| | \|A - U S V^T\|_F / (\|A\|_F n eps) | \|I - U^T U\|_1 / (n eps) | \|I - V^T V\|_1 / (n eps) |
|---|---|---|---|
| expected | below 50 | below 50 | below 50 |
| `faer` 0.24.4 | 0.045 | 0.092 | **262** |

The reconstruction only checks `U S V^T`, so it does not show the problem.
A scan of 39 seeds at n = 126, 127, 128 with `recursion_threshold` 8 and 128 (234 calls) gave 57 V ratios above 50, up to 2.8e4.
No U ratio in the scan exceeded 50; among the 57 failing calls the largest U ratio was 0.14.

## Cause

Each merge of divide and conquer builds a two entry rotation from `al = alpha * lambda` and `bp = beta * phi` (`src/linalg/svd/bidiag_svd.rs:1185-1186`).
It forms `r0 = (al.abs2() + bp.abs2()).sqrt()` (line 1187) and `c0, s0 = al / r0, bp / r0` (lines 1188 to 1192).
The rotation is applied to the first and last column of the merged left factor (lines 1196 to 1243).
`alpha` and `beta` come from a subproblem that has already been rescaled, and for this input deep merges see `al` and `bp` near `1e-22`.
Their squares are then below the smallest normal `f32` number, or zero.
So `r0` keeps only a few bits, and `c0^2 + s0^2` is no longer 1.
A traced merge on an input of this kind had `al = 1.0e-24`, `bp = -4.0e-22` and `c0^2 + s0^2 = 1.0022`.
The "rotation" then scales the two columns instead of rotating them, and the left factor loses orthogonality.
The solver works on the transposed bidiagonal, so this left factor is the caller's `V`.

## How the fork fixed it

The fork keeps the old arithmetic when `al^2 + bp^2` is a normal number.
When it is subnormal or not finite, the fork computes `r0` as a scaled hypot, `w sqrt(1 + (z / w)^2)` with `w = max(|al|, |bp|)` and `z = min(|al|, |bp|)`.
It takes `c0, s0` from a Givens rotation normalised by a ratio, so `c0^2 + s0^2` is 1 to rounding.
Run against the fork, the program prints a V ratio of 0.016.

Fork regression test: `linalg::svd::tests::lapack_svd_f32::lapack::bidiagonal_merge_boundary_rotation_is_orthogonal` (also for `f64`, `c32`, `c64`).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#87 (commit `b1ceefe`).
* Where it was found: tensor4all/faer#75.
