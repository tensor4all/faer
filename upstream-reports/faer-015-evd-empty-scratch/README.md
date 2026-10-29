# faer-015: self adjoint EVD of a 0 x 0 matrix panics in a debug build

**What happens:** `self_adjoint_eigen` on an empty matrix panics with "attempt to subtract with overflow" while it sizes its workspace.
**Which input:** any 0 x 0 matrix, in a build with overflow checks (the default `dev` profile).
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program calls `self_adjoint_eigen` on `Mat::<f64>::zeros(0, 0)` inside `catch_unwind`.

| | result |
|---|---|
| expected | `Ok`, with a 0 x 0 eigenvector matrix |
| `faer` 0.24.4, `cargo run` | **panic** at `src/linalg/evd/mod.rs:142:92`, "attempt to subtract with overflow" |
| `faer` 0.24.4, `cargo run --release` | `Ok((0, 0))` |

## Cause

`self_adjoint_evd_scratch` requests workspace for the Householder step with `apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(n - 1, bs, n)` (`src/linalg/evd/mod.rs:142`).
For `n = 0`, `n - 1` is a `usize` subtraction below zero.
With overflow checks on, this panics.
Without them it wraps to `usize::MAX`.
The release run of this program still returns `Ok`, but the request is computed from a wrapped value.
The high level `self_adjoint_eigen` calls this scratch query, so any empty input reaches it.

## How the fork fixed it

The fork uses `n.saturating_sub(1)` in this request.
With that change this program prints `Ok((0, 0))` with the default `dev` profile.

Fork regression test: `linalg::evd::scale_and_empty_tests::empty_matrix`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4 (panic with the default `dev` profile).
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#47 (commit `946ac9f`).
* Where it was found: tensor4all/faer#44.
