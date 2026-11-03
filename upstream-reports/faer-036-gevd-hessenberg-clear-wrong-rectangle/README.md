# faer-036: the generalized Hessenberg reduction clears the reduced columns when no right transformation is requested

**What happens:** `linalg::gevd::gevd_real` returns an unusable spectrum when the right eigenvectors (and therefore the right transformation `Z`) are not requested: `beta` stays exactly zero for all but the first four eigenvalues, so `alpha/beta` is not a number. The call itself returns `Ok`.
**Which input:** any regular pencil whose generalized Hessenberg reduction reaches more than one block (order above `gen_hessenberg`'s `block_size`, 32). The program uses `A = B*C` with `B = X X^T + I` (symmetric positive definite) and `C` the tridiagonal 1-D Laplacian at n = 40.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-11-03.

## Reproduce

```sh
cargo run
```

The program calls `gevd_real` four times on the same pencil, once per pair of `(left, right)` eigenvector requests, and counts the eigenvalues whose `beta` is zero.
The pencil is regular, so every `alpha/beta` is finite and every `beta` is nonzero.

| request | expected zeros | `faer` 0.24.4 |
|---|---|---|
| eigenvalues only | 0 | **36 of 40** |
| left only | 0 | **36 of 40** |
| right only | 0 | 0 |
| both | 0 | 0 |

The right-only and both-sides requests are correct, which is what hid the defect: they are the only ones that ask for `Z`.

## Cause

The blocked loop of `generalized_hessenberg` stores the rotations of each block by writing them below the subdiagonal of the block's columns, then applies them to the trailing columns.
When no right transformation is requested (`Z == None`) it clears them again, but with the rectangle `A[jcol+2.., jcol..n-jcol-2]` instead of the block's own columns:

```text
src/linalg/gevd/gen_hessenberg/mod.rs:685-690   (faer 0.24.4)
        } else {
            zip!(A.rb_mut().get_mut(jcol + 2.., jcol..n - jcol - 2))
                .for_each(|unzip!(x)| *x = zero());
            zip!(B.rb_mut().get_mut(jcol + 2.., jcol..n - jcol - 2))
                .for_each(|unzip!(x)| *x = zero());
        }
```

`n - jcol - 2` columns reach past the block (`nnb` columns), so the clear also erases

* the subdiagonal of the block's later columns (column `j`'s subdiagonal entry is at row `j + 1`), and
* every entry below the subdiagonal of the columns that come **after** the block, i.e. the part of `A` and `B` that the same block's trailing update has just written.

The QZ iteration therefore starts from a destroyed pair and never deflates; `alpha`/`beta` keep their initial zeros.

## How the fork fixed it

One entry per block column:

```text
src/linalg/gevd/gen_hessenberg/mod.rs:686-694   (tensor4all/faer, #224)
            for j in jcol..jcol + nnb {
                zip!(A.rb_mut().get_mut(j + 2.., j))
                    .for_each(|unzip!(x)| *x = zero());
                zip!(B.rb_mut().get_mut(j + 2.., j))
                    .for_each(|unzip!(x)| *x = zero());
            }
```

That is exactly what the right-transformation branch consumes. The subdiagonal and the columns after the block are left alone.
With it the four requests above agree and the analytic spectrum `2 - 2*cos(k*pi/(n+1))` is reproduced to `1e-8 * (||A|| + ||B||)`.

Regression in the fork: `faer/tests/gevd_repeated_eigen.rs`, `spectrum_does_not_depend_on_which_eigenvectors_are_requested*` at `f32`/`f64`/`c32`/`c64`.
