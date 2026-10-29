# faer-021: complex tridiagonal EVD of a 0 x 0 matrix panics

**What happens:** `tridiagonal_self_adjoint_evd` panics with an index out of bounds for an empty complex input when eigenvectors are requested.
**Which input:** a 0 x 0 tridiagonal matrix of a complex type (`c32` or `c64`), with eigenvectors requested.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program calls `linalg::evd::tridiagonal_self_adjoint_evd::<c64>` with empty `diag`, `subdiag` and `s`, and a 0 x 0 eigenvector output, inside `catch_unwind`.
The workspace is sized by hand: 0.24.4 has no scratch query for this entry point, and the dense query `self_adjoint_evd_scratch` overflows on `n = 0` in a debug build.

| | result |
|---|---|
| expected | `Ok(())` |
| `faer` 0.24.4 | **panic** at `src/linalg/evd/mod.rs:246`, "Assertion failed: row < self.nrows()", `row = 0`, `self.nrows() = 0` |

The real types and the values only path return `Ok(())` for the same input.

## Cause

`tridiagonal_self_adjoint_evd` has no early return for `n = 0`, unlike `self_adjoint_evd` (`src/linalg/evd/mod.rs:284`).
For a complex type with eigenvectors requested, it builds a phase vector `scale` of length `n` to turn the real eigenvectors into complex ones.
It writes `scale[0] = one()` unconditionally (line 246).
For `n = 0` that index is out of bounds.

## How the fork fixed it

The fork returns `Ok(())` at the start of the function when `n == 0`.
With that change this program prints `Ok(())`.

Fork regression test: `linalg::evd::scale_and_empty_tests::empty_tridiagonal_with_vectors` (`f64`, `c64` and `c32`).

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#69 (commit `8a199b0`).
* Where it was found: tensor4all/faer#31, the LAPACK validation of the tridiagonal EVD.
