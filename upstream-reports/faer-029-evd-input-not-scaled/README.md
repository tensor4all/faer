# faer-029: the nonsymmetric EVD loses every eigenvector digit when the entries are tiny

**What happens:** `Mat::eigen` returns eigenvectors whose residual `|A U - U S|` is about `1 / eps` times larger than LAPACK's, so they have no correct digit, or, further down, eigenvectors with NaN or infinite entries, still with `Ok`.
**Which input:** a nontrivial matrix (random, Grcar and near-Jordan were tested; a diagonal one is not affected) whose largest entry is below about `2^-100` (`f32`, `c32`) or `2^-960` (`f64`, `c64`). The program uses the 64 x 64 Grcar matrix scaled by a power of two.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program prints LAPACK's GET22 residual ratio (NaN when `S` or `U` has a nonfinite entry), `|A U - U S|_1 / (|A|_1 |U|_1 eps)`, measured in `f64` on the matrix divided by its scale (exact for a power of two).
LAPACK's CHKHS test accepts a ratio below 20.

| type | scale | expected (LAPACK `?geev`, scipy/OpenBLAS) | `faer` 0.24.4 |
|---|---|---|---|
| f64 | 1 | 16.4 | 10.6 |
| f64 | `2^-960` | 16.4 | **1.1e15** |
| f64 | `2^-1000` | 16.4 | **`U` not finite, returned with `Ok`** |
| f32 | 1 | 15.8 | 10.4 |
| f32 | `2^-100` | 15.8 | **`U` not finite, returned with `Ok`** |
| f32 | `2^-110` | 15.8 | **`U` not finite, returned with `Ok`** |

A sweep with this fork before its fix (which turns nonfinite eigenvectors into `NoConvergence`, so every cell has a finite ratio) over `f32`, `f64`, `c32` and `c64`, n = 16 to 257, random, Grcar and near-Jordan matrices, gives the same picture: the ratio grows from about `2^-90` (32-bit), and from `2^-100` (32-bit) and `2^-960` (64-bit) down it is `1e4` to `2e15`, while LAPACK stays at 2 to 70.
Near the top of the range the loss is smaller: on LAPACK's own `c32` test matrices scaled to `sqrt(MAX) / 16`, the fork before its fix gives a ratio of 42 to 59 where `cgeev` gives 29 to 40.

## Cause

`evd_imp` (`src/linalg/evd/mod.rs`) copies the matrix and runs the Hessenberg reduction and the Schur iteration on it as given.
LAPACK's `?GEEV` first computes the largest entry modulus `anrm` and, when it is below `smlnum = sqrt(safmin) / eps` or above `bignum = 1 / smlnum`, scales the matrix to that bound; it scales the eigenvalues back at the end.
Without that step the reduction and the iteration of a tiny matrix work in subnormals, where the relative precision is gone.
Measured stage by stage on a `c32` test matrix near the top, the Hessenberg backward error is the same with and without scaling (16 to 20), and the Schur backward error is 226 to 258 without and 67 to 93 with it.

## How the fork fixed it

`evd_imp` scales its input as `?GEEV` does, with a power of two instead of `smlnum / anrm`, so the scaling itself does not round (an entry far below the largest can still underflow when the matrix is scaled down, as with LAPACK). The largest entry is taken as the largest `|re|` or `|im|`: faer's complex modulus overflows for `2^1023 + 0i` and underflows for `2^-1074 + 0i`.
The eigenvectors do not change; the eigenvalues are multiplied back.
With that change this program prints 17.7 (`f64`) and 15.2 (`f32`) at every scale.

Fork regression test: `geev_scaling_tests::evd_scales_its_input_into_range` in `faer/src/linalg/evd/mod.rs` (Grcar and pseudo-random matrices, all four types, scales to both ends of the range).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: upstream `main` at `0ecd846` (2026-10-22) has the same `evd_imp` without input scaling.

## References

* Fork fix: tensor4all/faer#112.
* Found by: the bounded LAPACK validation's nonsymmetric EVD family (`c32`) and the stage-by-stage comparison with `cgeev` asked for in tensor4all/faer#112.
