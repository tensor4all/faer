# faer-034: `JacobiRotation::apply_on_the_right_2x2` applies `conj(J)`, not `J`

**What happens:** `JacobiRotation::apply_on_the_right_2x2` is documented to compute `M J` for the rotation `J = [[c, -conj(s)], [s, c]]`. For complex coefficients with a non-real `s` it computes `M conj(J)` instead, so even `M = I` returns `conj(J)`.
**Which input:** any rotation with a non-real `s`, applied to any block. The program uses `c = 0.6`, `s = 0.8i` and the blocks `M = I` and `M = [[1, 2i], [3, 4]]`.
**Which version:** `faer` 0.24.4, the latest release on crates.io.
**Added to this list:** 2026-11-03.

## Reproduce

```sh
cargo run
```

The program applies the rotation from the right to two 2 x 2 complex blocks and compares the result with `M J` formed directly from `c` and `s`.

| block | expected | `faer` 0.24.4 |
|---|---|---|
| `M = I` | `J` | **off by 1.6** in the largest entry, `2.26` in the Frobenius norm |
| `M = [[1, 2i], [3, 4]]` | `M J` | **off by 6.4** in the largest entry |

The real path is unaffected, because there `conj(J) = J` for a real `s`.

## Cause

`apply_on_the_right_2x2` (`src/linalg/jacobi.rs:205` to `:215`) delegates to the left 2 x 2 helper, transposing the block around it. It passes `self.adjoint()` as the rotation (line 213).
`J^H M^T` transposed is `M conj(J)`, not `M J`, so every block that is multiplied from the right with a non-real `s` receives the conjugated rotation. `apply_on_the_left_2x2` itself is correct.

## How the fork fixed it

The delegation uses `self.transpose()` (line 213), which gives `M J` by the convention of `apply_on_the_left_2x2`. With that change this program reproduces `M J` exactly for both blocks.
The real path keeps its bits, because `adjoint()` and `transpose()` differ only in the sign of `conj(s)`.

Fork regression tests: `complex_apply_on_the_right_2x2_matches_explicit_product` and `real_apply_on_the_right_2x2_matches_explicit_product` in `faer/src/linalg/jacobi.rs`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#208 (merge commit `7cdf90f`).
* Where it was found: tensor4all/faer#206.
