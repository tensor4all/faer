# faer-004: the divide and conquer SVD returns wrong factors for a 128 x 128 bidiagonal matrix

**What happens:** `Mat::thin_svd` returns `U`, `S`, `V` with `|A - U S V^T|_F / |A|_F = 1.9e-3`, and reports no error. `U` and `V` are orthonormal, but one singular value is wrong by eight orders of magnitude.
**Which input:** a bidiagonal matrix with a tiny diagonal entry at the row where divide and conquer splits. The program below uses a 128 x 128 upper bidiagonal matrix.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The matrix is upper bidiagonal, with diagonal `d` and superdiagonal `e`.
`d_k = 1` and `e_k = 1` for `k < 64`, `d_64 = 1e-13`, and every other entry is 0.
128 is the default `recursion_threshold`, the size where the SVD switches from the QR algorithm to divide and conquer.
The reference values come from LAPACK `gesdd` and from faer's own QR algorithm (`recursion_threshold = usize::MAX`), which agree.

| | \|A - U S V^T\|_F / \|A\|_F | `s[63]` | `s[64]` |
|---|---|---|---|
| expected | about 1e-15 | 4.8327e-2 | 1.2403e-14 |
| `faer` 0.24.4 | **1.9e-3** | **4.8333e-2** | **5.8994e-6** |

## Cause

The merge step of divide and conquer finds the new singular values as roots of the secular equation, a scalar function with one root between each pair of poles.
`secular_eq_root_finder` refines each root with a secant iteration.
The loop runs while the step is large and `|f_cur - f_prev| > epsilon` (`src/linalg/svd/bidiag_svd.rs:76-80`).
For this matrix the function is flat near the smallest root: two successive values differ by less than `epsilon` while the estimate is still far from the root.
The loop then stops, but `use_bisection` stays false, so the unconverged secant estimate is returned as the root.
The merge rebuilds its update column from the computed roots (`perturb_col0`), so a wrong root also makes `U S V^T` differ from `A` while `U` and `V` stay orthonormal.
LAPACK's `dlasd4` does not treat flat function values as convergence.

## How the fork fixed it

The fork keeps the other stopping tests and moves the flat value test into the loop body.
When `|f_cur - f_prev| <= epsilon`, it sets `use_bisection = true` and leaves the loop, so the existing bracketed bisection finds the root.
The same change is made in the root finder that the self adjoint eigensolver shares.
Run against the fork, the program prints `2.8e-15`, `s[63] = 4.8327e-2` and `s[64] = 1.2403e-14`.

Fork regression tests: `linalg::svd::tests::test_tiny_diagonal_at_split` (this matrix through the public SVD) and `linalg::svd::bidiag_svd::tests::test_secular_eq_flat_away_from_root` (the root finder on a flat function).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#29 (commit `65dd42f`).
* Where it was found: tensor4all/faer#28.
