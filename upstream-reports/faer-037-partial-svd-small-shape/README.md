# faer-037: `partial_svd` panics for small matrices and for a full projection

**What happens:** the public `partial_svd` panics at an internal assertion. The driver clips the projection dimension down to `min(m, n)` and then hands it to `partial_svd_imp`, which asserts that the dimension is strictly smaller than `min(m, n)`, so the two disagree whenever the clip is active.
**Which input:** any nonempty matrix with `min(m, n) <= 2 * MIN_DIM` (`MIN_DIM` is 32) under the default parameters, and any explicit `max_dim >= min(m, n)`. The program uses a 4x4 identity, one requested singular triplet and default parameters.
**Which version:** `faer` 0.24.4, the latest release on crates.io.
**Added to this list:** 2026-11-04.

## Reproduce

```sh
cargo run
```

The program calls `matrix_free::eigen::partial_svd` with the shape above and reports whether it returned the triplet or panicked.

| | result |
|---|---|
| expected | `n_converged = 1`, `s[0] = 1` |
| `faer` 0.24.4 | **panic** at `operator/svd/mod.rs:144`: `Assertion failed: max_dim < Ord::min(m, n)` (`max_dim = 4`, `Ord::min(m, n) = 4`) |

## Cause

The driver computes `max_dim = min(max(params.max_dim, max(2 * MIN_DIM, 2 * n_sval)), min(m, n))` (`faer/src/operator/eigen/mod.rs:1430` in 0.24.4), so for every nonempty matrix with `min(m, n) <= 64` the clip leaves `max_dim == min(m, n)`, and lowering `params.max_dim` does not help because of the `2 * MIN_DIM` floor.
`partial_svd_imp` then asserts the strict inequality `max_dim < Ord::min(m, n)` (`faer/src/operator/svd/mod.rs:144` in 0.24.4, the same file and line on upstream `main` at `0ecd846`), so the public entry point violates its own callee's precondition. The projection needs room for one Lanczos column beyond the subspace it keeps, which the full-subspace case cannot provide.

## How the fork fixed it

A full-projection branch (`faer/src/operator/svd/full.rs`) materialises the operator one column at a time through `BiLinOp::apply`, runs the dense `svd`, and copies the leading triplets out; the iterative branch is unchanged for `max_dim < min(m, n)`. `partial_svd_scratch` returns the dense-path workspace in the same case. Outputs are written only after the decomposition succeeds, and a non-finite operator or a failed SVD returns a zero count.

Fork regression tests: `mwe_identity_default_params`, `projection_clip_boundary` (both sides of the `2 * MIN_DIM` boundary) and `empty_shape_is_reported` in `faer/src/operator/svd/full.rs`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same driver clip and the same assert are in upstream `main` at `0ecd846` (2026-10-22), `faer/src/operator/svd/mod.rs:144`.

## References

* Fork fix: tensor4all/faer#216 (merge commit `ea09256`).
* Where it was found: tensor4all/faer#213.
