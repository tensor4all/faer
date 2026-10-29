# faer-025: nonsymmetric EVD returns a wrong eigenvector when the Schur norm sum overflows

**What happens:** `Mat::eigen` returns `Ok` with a vector that is not an eigenvector, for a matrix whose entries and eigenvalues are all finite.
**Which input:** a matrix near the top of the floating point range, where the sum of the entry magnitudes of its Schur form overflows. The program uses `2^1023 * [[1, 1], [0, 1/2]]`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program calls `Mat::eigen` on `A = c * [[1, 1], [0, 1/2]]` for `c = 1` and `c = 2^1023`.
For every `c`, the eigenvector for the eigenvalue `c / 2` is proportional to `[-2, 1]`.
The program checks that pair on `A / c`, which is exact for a power of two.
It prints the relative residual `|(A / c - 1/2) x| / |x|`.

| `c` | eigenvalue | eigenvector | relative residual |
|---|---|---|---|
| expected, any `c` | `c / 2` | proportional to [-2, 1] | about 1e-16 |
| `faer` 0.24.4, `c = 1` | 0.5 | [-2, 1] | 0 |
| `faer` 0.24.4, `c = 2^1023` | 4.49e307 | **[0, 1]** | **1.0** |

The eigenvalues are right in both cases.
Only the vector is wrong.

## Cause

The eigenvectors are computed from the Schur form `T` by back substitution with the shifts `t_ii - lambda`.
When a shift is smaller than `eps * norm`, it is replaced by `eps * norm` (`src/linalg/evd/mod.rs:701-702`, and also at `:794`, `:828` and `:917`).
`norm` is the plain sum of `|t_ij|` over the Schur form (`:567-572` for the real form, `:632-637` for the complex form).
Here that sum is `2.5 * 2^1023`, which overflows to `inf`.
Every shift is then smaller than `eps * inf = inf`, and is replaced by `inf`.
Dividing by `inf` gives 0 for the first component of the vector, so the result is `[0, 1]`.
The caller sees a finite unit vector and `Ok`.

## How the fork fixed it

The fork's clamp is now `eps * max(|lambda|, max |t_ij|)`, relative to the eigenvalue and the largest entry instead of the sum.
`|re| + |im|` is scaled by `eps` before it is added, so it does not overflow.
A Schur form whose largest entry is outside `[2^-20, 2^20]` is first scaled by a power of two, as LAPACK `?GEEV` scales its input.
With that change this program prints `[-0.894, 0.447]` and a residual of 0 for both values of `c`.

Fork regression tests, in `faer/src/linalg/evd/mod.rs`: `range_limit_vector_tests::two_by_two_at_the_range_limit` and `range_limit_vector_tests::complex_triangular_at_the_range_limit`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## Note

The same clamp in the same lines also makes the eigenvector residual grow with `n` for a matrix with a clustered spectrum.
That is covered by a separate report, [faer-003](../faer-003-evd-clamp-grows-with-n/).
One fix to the clamp addresses both.

## References

* Fork fixes: tensor4all/faer#87 (fix commit `79d0293`), which made the norm sum overflow safe; it was later replaced by tensor4all/faer#121 (commit `0c806f1`), the clamp described above.
* Where it was found: tensor4all/faer#77.
