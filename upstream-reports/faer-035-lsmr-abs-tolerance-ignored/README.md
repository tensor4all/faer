# faer-035: LSMR never reads `abs_tolerance`

**What happens:** `LsmrParams::abs_tolerance` is ignored. The convergence test compares the normal residual with `norm_ref * rel_tolerance` only, so an initial residual below `abs_tolerance` is reported as `NoConvergence` instead of converging at the initial guess.
**Which input:** a consistent system whose initial normal residual is below the absolute tolerance. The program uses `A = I`, `b = [1, 1]`, the zero initial guess, `abs_tolerance = 10`, `rel_tolerance = 0` and `max_iters = 0`; the initial normal residual is `sqrt(2) = 1.41`.
**Which version:** `faer` 0.24.4, the latest release on crates.io.
**Added to this list:** 2026-11-03.

## Reproduce

```sh
cargo run
```

The program calls `matrix_free::lsmr::lsmr` with the parameters above and prints the initial residual, the expectation and the result.

| | result |
|---|---|
| expected | `Ok`, converged at the initial guess, because `1.41 <= abs_tolerance` |
| `faer` 0.24.4 | **`Err(NoConvergence { abs_residual: 1.41, rel_residual: 1.0 })`** |

## Cause

The convergence test uses the relative tolerance only: `threshold = norm_ref * params.rel_tolerance` (`src/operator/lsmr.rs:371`), compared with the normal residual at lines 381 and 612.
`abs_tolerance` is read nowhere in the LSMR implementation; it appears only in the struct and in `Default` (`:11` and `:24`), so a caller who sets it alone asks for a bound that is never tested.
The other iterative solvers of the same family do read theirs (`src/operator/conjugate_gradient.rs:143`, `src/operator/bicgstab.rs:153`).
The two LSMR fields have different meanings: `rel_tolerance` scales the residual with the problem, `abs_tolerance` is an absolute bound on the same normal residual that `LsmrInfo::abs_residual` reports.

## How the fork fixed it

The threshold is `max(abs_tolerance, norm_ref * rel_tolerance)`, the same scale as the absolute residual that the relative test already uses, at both comparison points.
With `rel_tolerance = 0` the absolute bound now decides, and this program returns `Ok` with `iter_count = 0`.

Fork regression test: `test_abs_tolerance_is_honoured` in `faer/src/operator/lsmr.rs`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#208 (merge commit `7cdf90f`).
* Where it was found: tensor4all/faer#207.
