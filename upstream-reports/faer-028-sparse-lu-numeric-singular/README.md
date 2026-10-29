# faer-028: sparse LU factors a numerically singular matrix without an error

**What happens:** the numeric sparse LU factorization returns `Ok` for a matrix whose pivot is exactly zero, and the solve then returns NaN.
**Which input:** a structurally full but singular matrix, in both the simplicial and the supernodal backend. The program uses `A = [[1, 1], [1, 1]]`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program forces each backend in turn (`SupernodalThreshold::FORCE_SIMPLICIAL`, then `FORCE_SUPERNODAL`), factors `A`, and, if the factorization succeeds, solves `A x = [1, 1]`.

| backend | expected | `faer` 0.24.4 |
|---|---|---|
| simplicial | an error | **`Ok`, then `x = [NaN, NaN]`** |
| supernodal | an error | **`Ok`, then `x = [NaN, NaN]`** |

## Cause

- **Simplicial.** The pivot search starts from `pivot_val = -1` (`src/sparse/linalg/lu.rs:1729`), so any candidate wins, including an exact zero. `SymbolicSingular` is raised only when a column has no candidate at all (line 1748). Nothing rejects the zero pivot: it is stored in `U`, and its reciprocal (line 1752) scales whatever rows remain below it.
- **Supernodal.** Each supernode's panel is factored by the dense `lu_in_place_recursion` (line 940), which gives no singular signal.

In this program the zero pivot is the last one, so no rows remain below it, and the factors stay finite with a zero on `U`'s diagonal. The solve divides by that zero, which gives the NaN. When rows do remain below a zero pivot, the dense kernel also scales them by its infinite reciprocal (faer-010).

## How the fork fixed it

The fork adds `LuError::NumericSingular { index }`. This is the variant the upstream maintainer asked for in upstream PR #332, which was closed unmerged.
- The simplicial backend returns it when the chosen pivot is exactly zero.
- The supernodal backend returns it when a diagonal entry of a factored panel is exactly zero.

`index` counts elimination steps in the fill-reducing column order, as `SymbolicSingular`'s does. Tolerance-based detection is not part of the fix; it is tracked separately. Run against the fork, both backends report `NumericSingular`.

Fork regression tests: `sparse::linalg::lu::tests::codeberg_339` (`f32`, `f64`, `c32`, `c64`, both backends) and `sparse::linalg::lu::tests::numeric_singular_after_an_elimination_step`, and `sparse::linalg::lu::tests::nonzero_subnormal_pivot_is_not_singular` (a nonzero subnormal complex pivot is not reported).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Upstream report: sarah-quinones/faer#339 (simplicial only). Upstream PR: sarah-quinones/faer#332.
* Fork fix: tensor4all/faer#132.
