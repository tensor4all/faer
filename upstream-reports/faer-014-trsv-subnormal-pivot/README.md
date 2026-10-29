# faer-014: triangular solve returns inf or NaN for a subnormal pivot

**What happens:** the lower triangular solve returns `inf` or `NaN` for a system whose exact solution is all ones.
**Which input:** a triangular matrix with a nonzero subnormal diagonal entry (a pivot), real or complex. The program uses diagonal systems `d x = d`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program calls `solve_lower_triangular_in_place` on three diagonal systems `L x = b` with `b` equal to the diagonal of `L`.
A subnormal number is a nonzero value below `f64::MIN_POSITIVE`.

| system | expected `x` | `faer` 0.24.4 |
|---|---|---|
| `f64`, 1 x 1, `d = 1e-310` | [1] | **[inf]** |
| `f64`, 2 x 2, `d = 2^-1074` on both diagonal entries | [1, 1] | **[inf, NaN]** |
| `c64`, 1 x 1, `d = i * 2^-1074` | 1 + 0i | **NaN + NaNi** |

Each system is nonsingular, has condition number 1, and its solution is exactly representable.

## Cause

The small base cases of the triangular solve multiply by the reciprocal of each pivot instead of dividing by the pivot.
In `src/linalg/triangular_solve.rs`, the 1 x 1 case computes `inv = recip(l00)` (line 113) and then `x0 = x0 * inv` (line 116).
The 2 x 2 and larger cases do the same with `l00_inv`, `l11_inv` and so on (lines 121 to 122, 128, and onward).
For a subnormal pivot the reciprocal is larger than the largest finite value, so it is `inf`.
`inf` times the right hand side is `inf`.
In the 2 x 2 case the coupling coefficient `-l10 * l11_inv` (line 123) is `0 * inf`, which is `NaN`, so the second entry is `NaN`.
For a complex pivot, the reciprocal from `faer-traits` is `NaN` already, so the result is `NaN`.

## How the fork fixed it

The fork first switched the real 1 x 1 case to a division when the reciprocal is not finite.
It then replaced this with a general rule.
A base case block whose diagonal contains a nonzero subnormal pivot now forms each residual first and then divides it by the pivot.
A real residual is divided directly.
A complex residual is multiplied by the reciprocal when that is finite.
Otherwise the residual and the pivot are both scaled by `1 / (4 * MIN_POSITIVE)` before the division, so the divisor is a normal number.
Blocks without a subnormal pivot keep the faster reciprocal arithmetic.
With that change this program prints `x = 1` in all three cases.

Fork regression tests, in `faer/src/linalg/triangular_solve.rs`:
* `tests::tiny_pivot_scalar_solve`;
* `tests::subnormal_real_diagonal_solve`, `tests::subnormal_complex_diagonal_solve` and `tests::subnormal_coupled_solve` (all four scalar types, lower and upper, both conjugations, sizes 1 to 129).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fixes: tensor4all/faer#47 (commit `946ac9f`) for the real 1 x 1 case, and tensor4all/faer#83 (commit `017d504`) for the general rule.
* Where it was found: tensor4all/faer#43 (real 1 x 1) and tensor4all/faer#74 (real `n >= 2` and complex).
* The complex reciprocal of a subnormal value is a related report, [faer-020](../faer-020-complex-recip-subnormal/).
