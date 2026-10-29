# gemm-x86-001: a product with a diagonal gives a wrong value when the left matrix is stored backwards

**What happens:** `gemm` returns a wrong value for `A * diag(d) * B`.
**Which input:** the left matrix `A` has a negative column stride (its columns are stored in reverse order).
**Which version:** `private-gemm-x86` 0.1.22, the latest release on crates.io, with AVX2.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program multiplies a 1 x 3 row `a = [1, 0, 0]`, the diagonal `d = [1, 2, 3]` and a 3 x 1 column of ones.
The result must be `d[0] = 1`.
It runs the product twice.
The first time `a` is stored forwards.
The second time the same `a` is stored backwards, with column stride `-1`.

| `a` stored | expected | `private-gemm-x86` 0.1.22 |
|---|---|---|
| forwards (column stride +1) | 1 | 1 |
| backwards (column stride -1) | 1 | **3** |

In the backwards case the product used `d[2]` where it should use `d[0]`.

## Cause

`gemm_impl` in `src/lib.rs:1244` handles a negative column stride of the left matrix by walking the shared dimension (the depth) backwards.
It moves the `lhs` pointer to the last column and the `rhs` pointer to the last row, then flips the sign of `lhs_cs` and `rhs_rs`.
The diagonal also runs along the depth, but `real_diag` and `real_diag_stride` are left as they were.
So depth index `t` of the left and right matrices meets `d[depth - 1 - t]` instead of `d[t]`.
The pointer advance between depth blocks (`src/lib.rs:1504`) keeps the same mismatch.
All instruction sets and all four element types go through this one function, so AVX2 and AVX-512, real and complex, are all affected.

The left matrix inside `gemm` is not always the caller's left matrix.
When the destination's row stride is larger than its column stride, `gemm` swaps the operands and works on the transpose.
Then a negative row stride of the caller's right matrix reaches the same branch.

## How the fork fixed it

In the same `lhs_cs < 0` branch, the fork also moves the diagonal pointer to its last depth entry and flips the sign of its stride.
A null diagonal pointer stays null.
With that change this program prints 1 in both cases.

Fork regression test: `tests/diag_reversed_depth.rs` in `t4a-private-gemm-x86` (`avx256_diag_follows_reversed_depth`, `avx512_diag_follows_reversed_depth`).
It checks every sign of the depth strides of `A`, `B` and `d`, both destination layouts, `f32`, `f64`, `c32`, `c64`, and both instruction sets.

## Affected versions

Confirmed by running this program: `private-gemm-x86` 0.1.22.
Not fixed upstream: 0.1.22 is also the head of the upstream repository (`71872b9`).

## References

* Fork fix: tensor4all/t4a-private-gemm-x86#5 (commit `84ca6db`), released as `t4a-private-gemm-x86` 0.1.27.
* Where it was found: tensor4all/faer#113. faer's unpivoted LDL multiplies `L D L^H` through this path, so a matrix stored with reversed rows and columns factored wrongly at n >= 65.
* faer side: tensor4all/faer#117 (temporary workaround), tensor4all/faer#122 (workaround removed after the fixed release).
