# faer-011: the SVD least squares solve returns NaN for a rank deficient matrix

**What happens:** `Svd::solve_lstsq` returns NaN when a singular value is exactly zero, although the system has an exact finite solution.
**Which input:** any tall matrix with an exactly zero singular value. The program below uses `A = [[1, 0], [0, 0], [0, 0]]` and `b = [2, 0, 0]`.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program computes the SVD of `A` and calls `solve_lstsq` with `b`.
`x = [2, 0]` solves `A x = b` exactly and has the smallest norm.
The same SVD's `pseudoinverse()` gives this answer.

| | `x` from `solve_lstsq` |
|---|---|
| expected (`pseudoinverse() * b`) | `[2, 0]` |
| `faer` 0.24.4 | **`[NaN, NaN]`** |

## Cause

The least squares solve computes `tmp = U^H b`, divides each row by its singular value, and multiplies by `V`.
The division is `s = recip(S[i])` followed by `tmp[i] * s` (`src/linalg/solvers.rs:2702-2707`).
For a zero singular value, `s` is infinity.
The matching entry of `U^H b` is 0 for a consistent system, so the product is `0 * inf`, which is NaN.
The final product with `V` spreads the NaN to every entry of `x`.
`pseudoinverse()` handles a zero singular value, so the two answers from the same SVD disagree.

## How the fork fixed it

The fork sets the entry to zero when the singular value is exactly zero, and divides otherwise.
This drops only exactly zero directions.
A rank tolerance was considered and rejected, because it would also drop small but representable directions such as `diag(1, 2^-54)`.
Run against the fork, the program returns `[2, 0]`.

Fork regression test: `linalg::solvers::tests::svd_lstsq_rank_deficient`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#47 (commit `ee70868`).
* Where it was found: tensor4all/faer#40.
