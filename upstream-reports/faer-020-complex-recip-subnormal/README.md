# faer-020: complex reciprocal of a subnormal value is `NaN - inf i`

**What happens:** the complex reciprocal returns `NaN - inf i` for a subnormal input whose exact reciprocal is finite.
**Which input:** a nonzero complex value whose magnitude is at most about `MIN_POSITIVE / 2`, so a subnormal value. The program uses `z = i * MIN_POSITIVE / 2`.
**Which version:** `faer` 0.24.4, which uses `faer-traits` 0.24.0.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program calls `faer::traits::math_utils::recip` on the `c64` value `z = i * f64::MIN_POSITIVE / 2`.
The exact reciprocal is `-i * 2 / MIN_POSITIVE = -i * 2^1023`, which is a finite `f64`.

| | `recip(z)` |
|---|---|
| expected | `0 - 8.98846567431158e307 i` |
| `faer` 0.24.4 | **`NaN - inf i`** |

Every complex `recip` in `faer` goes through this function, for example the 1 x 1 pivot of the complex triangular solve.

## Cause

`recip_impl` in `faer-traits-0.24.0/src/lib.rs:353-384` scales the input before it forms `1 / (re^2 + im^2)`.
For an input whose parts are at most one in magnitude it multiplies both parts by `big = sqrt_max_positive`, which is `2^511` for `f64` (lines 378 to 382).
For a subnormal input that is not enough.
Here `im * big` is `2^-512`, and its square `2^-1024` is below `1 / max_positive`.
The reciprocal of that square overflows to `inf` (line 381).
The result is then `re * inf = 0 * inf = NaN` for the real part and `-im * inf = -inf` for the imaginary part.

## How the fork fixed it

When both parts are below `sqrt_min_positive` after the first scaling, the fork scales them by `big` a second time.
It then multiplies the result by `big` twice to undo this.
Only inputs below `min_positive` in magnitude take the new branch.
The result is finite whenever the exact reciprocal is, and `1 / 2^-1074` still gives `inf`.
With that change this program prints `0 - 8.98846567431158e307 i`.

Fork regression tests:
* `recip_of_subnormal`, in `faer-traits/src/lib.rs` (`f64` and `f32`, real and imaginary subnormal inputs);
* `linalg::triangular_solve::tests::tiny_imaginary_pivot_scalar_solve` (`c64` and `c32`).

## Affected versions

Confirmed by running this program: `faer` 0.24.4 with `faer-traits` 0.24.0.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#68 (commit `57a06de`).
* Where it was found: tensor4all/faer#31, the LAPACK validation of the triangular solve.
