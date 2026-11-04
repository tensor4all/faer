# faer-038: generalized eigenvectors are NaN for a repeated complex-conjugate pair

**What happens:** `generalized_eigen` and `linalg::gevd::gevd_real` return NaN eigenvectors for a real pencil whose spectrum contains a complex-conjugate pair twice. The eigenvalues are correct and the call returns `Ok`.
**Which input:** `A = diag(R, R, 3)` with `R = [[2, 1], [-1, 2]]` (eigenvalues `2 +- i`), with or without a coupling `A[1, 3] = 0.75` between the two copies of `R`, and `B = I`.
**Which version:** `faer` 0.24.4; also `faer` `main` at `0ecd846`.
**Added to this list:** 2026-11-04.

## Reproduce

```sh
cargo run
```

The program calls `generalized_eigen` on the coupled pencil, then `gevd_real` for both eigenvector sides on the coupled and the uncoupled pencil, and lists the columns that hold a non-finite entry.

| call | expected | `faer` 0.24.4 and `main` `0ecd846` |
|---|---|---|
| `generalized_eigen`, coupling 0.75: `U` finite | `true` | **`false`** |
| `gevd_real`, coupling 0.75: non-finite left / right columns | `[]` / `[]` | **`[0, 1]` / `[2, 3]`** |
| `gevd_real`, coupling 0: non-finite left / right columns | `[]` / `[]` | **`[0, 1]` / `[2, 3]`** |

The right vectors of the second copy of the pair and the left vectors of the first copy are NaN; the other columns are correct.

`gevd_real`'s buffer in the program is larger than `gevd_scratch` asks for. With exactly the requested size, 0.24.4 panics inside the 2 x 2 solve's `lu_in_place` (`buffer is not large enough`) before it reaches the NaN; `main` at `0ecd846` does not.

## Cause

The eigenvector substitution of `qz_to_gevd_real` solves, for every 2 x 2 diagonal block it meets, the shifted system `C = ca*A(block) - w*diag(d0, d1)` with `solve_complex_shifted_2x2`, a full-pivot LU:

```text
src/linalg/gevd/mod.rs:373-490   (faer 0.24.4)
        let cmax = C.norm_max();
        if cmax < *smin {
            let ref smin_inv = smin.recip();
            zip!(B.rb_mut()).for_each(|unzip!(x)| *x *= smin_inv);
        }
        let (_, row_perm, col_perm) =
            linalg::lu::full_pivoting::factor::lu_in_place(...);
        linalg::lu::full_pivoting::solve::solve_in_place(...);
```

When the block is another copy of the eigenvalue being computed, `C` is exactly singular while its norm is above `smin`. The LU's second pivot is zero, and the solve divides the right hand side by it; for the uncoupled pencil the right hand side is zero as well, so the result is `0 / 0`. The NaN then reaches the whole column through the back transformation.

The same helper also scales the right hand side by `1 / smin` when `cmax < smin` and then still factors and divides by `C` (`mod.rs:402` and `:451`), so a tiny block is divided by twice.

LAPACK `dtgevc` solves the same systems with `dlaln2`, which replaces a second pivot below `smin` by `smin` (`dlaln2.f:416-419`, `:541-545`) and, when `cmax < smin`, uses `smin * I` for `C` and returns (`dlaln2.f:390-401`, `:478-492`); Reference-LAPACK v3.12.1.

## How the fork fixed it

`solve_complex_shifted_2x2` follows `dlaln2` in both places: it returns after the `cmax < smin` scaling, and after the factorization it replaces a second pivot below `smin` by `smin` (`|re| + |im|` for the complex system).

With the coupling the pair is defective, so the second eigenvector of `2 + i` is close to the first one, as LAPACK `dggev` also returns (OpenBLAS 0.3.30, LAPACK 3.12.0, through Julia 1.13.3); without it the two vectors are independent.

Regression in the fork: `faer/tests/gevd_repeated_complex_pair.rs` (f64 and f32, left, right and both sides, coupled and uncoupled), and `linalg::gevd::tests::complex_shifted_2x2_*` for the helper.

Longer coupled chains also need overflow control after protecting the pivots. The fork now scales large right-hand sides before solving, propagates that scale through the active eigenvector, and normalizes between substitution steps. The additional regressions use eight repeated blocks in `f32` and 22 in `f64`, with both left and right vector requests. See the [decision record](../../docs/worklogs/2026-11-04-gevd-repeated-complex-pairs.md).
