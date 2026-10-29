# faer-008: `norm_l2` of a strided vector overflows or underflows

**What happens:** `norm_l2` returns `inf` or `0` for a vector with a finite, representable norm when the vector is not contiguous in memory.
**Which input:** a row of a column major matrix (or any view with a general stride) whose entries are near the square root of the overflow or underflow threshold. The program uses the row `[s, s/2]`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program takes the first row of `[[s, s/2], [0, s]]`, which is strided in memory, and prints its `norm_l2`.
It compares this with the norm of a contiguous copy and with the exact value `sqrt(1.25) * s`.

| s | expected | strided row, `faer` 0.24.4 | contiguous copy, `faer` 0.24.4 |
|---|---|---|---|
| 1 | 1.118033988749895 | 1.118033988749895 | 1.118033988749895 |
| 1e160 | 1.1180339887498949e160 | **inf** | 1.1180339887498947e160 |
| 1e-300 | 1.118033988749895e-300 | **0** | 1.1180339887498948e-300 |

## Cause

`norm_l2` asks `norm_l2_x3` for three sums of squares: one scaled down, one unscaled, one scaled up.
It then picks the one that is in range (`src/linalg/reductions/norm_l2.rs:156-166`).
For a view with a general stride, `norm_l2_x3` takes a fallback (lines 144 to 153).
The fallback first builds the norm safely with `hypot` in `acc`.
It then squares `acc` without scaling (`acc = acc.abs2()`, line 152).
It forms the three sums from that square: `min_positive * acc`, `acc` and `max_positive * acc` (line 153).
For `s = 1e160`, `acc` squared is `1.25e320`, which overflows to `inf`.
For `s = 1e-300`, it is `1.25e-600`, which underflows to `0`.
The scaling that follows cannot recover either value.
The SIMD path for contiguous data scales each entry before squaring, so it is correct.

The SVD bidiagonalization calls `norm_l2` on a matrix row (`src/linalg/svd/bidiag.rs:163`), so the SVD can reach this fallback too.
This program does not test that path.

## How the fork fixed it

The fork scales `acc` before squaring it, as the SIMD path does: the three sums become `(sqrt_min_positive * acc)^2`, `acc^2` and `(sqrt_max_positive * acc)^2`.
With that change this program prints the exact norm for all three scales.

Fork regression test: `linalg::reductions::norm_l2::tests::test_norm_l2_strided_range`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#46 (commit `4e2c241`).
* Where it was found: tensor4all/faer#36.
