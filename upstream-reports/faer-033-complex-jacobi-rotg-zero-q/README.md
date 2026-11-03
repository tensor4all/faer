# faer-033: complex `JacobiRotation::rotg(p, 0)` returns `r = 1` instead of `p`

**What happens:** `JacobiRotation::rotg(p, q)` returns the rotation `J` and the value `r` with `J [p, q]^T = [r, 0]^T`. For complex `p` and `q = 0` it returns the identity rotation, which maps `[p, 0]^T` to itself, together with `r = 1`. The returned `r` is therefore not the value the rotation carries unless `p = 1`.
**Which input:** any complex `p` other than 1 with `q = 0`. The program uses `p = 3 + 2i`.
**Which version:** `faer` 0.24.4, the latest release on crates.io.
**Added to this list:** 2026-11-03.

## Reproduce

```sh
cargo run
```

The program calls `JacobiRotation::rotg(c64::new(3.0, 2.0), c64::new(0.0, 0.0))` and prints the rotation, the returned `r`, and `J [p, 0]^T`, which it forms from `c` and `s` (`J = [[c, -conj(s)], [s, c]]`).

| | value |
|---|---|
| expected | `r = 3 + 2i`, `J [p, 0]^T = [3 + 2i, 0]` |
| `faer` 0.24.4 | **`r = 1 + 0i`**, `J [p, 0]^T = [3 + 2i, 0]` |

The rotation is the identity and its action is right, so only the returned `r` is wrong; a caller that uses `r` as the entry the rotation carries gets 1.

## Cause

`rotg` is in `src/linalg/jacobi.rs`. Its real branch handles `q = 0` by returning the identity rotation with `r = p` (lines 62 to 67).
Its complex branch handles the same case first (lines 94 to 101) and returns the identity rotation with `r = one::<T>()`, the constant 1, instead of `p`.
The rest of the complex branch computes `r` with the magnitude of `(p, q)` and the phase of `p`, as LAPACK's `zrotg` does, so `r` is the value the rotation carries everywhere except in this branch.

## How the fork fixed it

The complex zero-`q` branch returns `a` (the input `p`). The rotation is unchanged, so `(J, r)` is consistent again.

Fork regression test: `complex_rotg_with_zero_q_returns_p` in `faer/src/linalg/jacobi.rs`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#208 (merge commit `7cdf90f`).
* Where it was found: tensor4all/faer#205.
