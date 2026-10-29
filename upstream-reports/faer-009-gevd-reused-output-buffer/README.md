# faer-009: generalized eigenvectors depend on the old contents of the output buffer

**What happens:** `gevd_real` returns wrong eigenvectors when the eigenvector output buffer is not zero on entry. The eigenvalues are correct and the call returns `Ok`.
**Which input:** any matrix pair, with an eigenvector buffer that holds nonzero values off the diagonal (for example a reused buffer). The program uses `A = [[2, 1], [0, 3]]`, `B = I` and a buffer filled with 2.0.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The program calls the low level `linalg::gevd::gevd_real` for the right eigenvectors only.
It passes a 2 x 2 output buffer filled with 2.0.
It then prints the residual `max |A U diag(beta) - B U diag(alpha)|`, which should be near rounding level.

| | first right eigenvector | residual |
|---|---|---|
| expected | proportional to [1, 0] (eigenvalue 2) | below 1e-12 |
| `faer` 0.24.4 | **[0.4472, 0.8944]** | **0.894** |

With a buffer filled with zeros the same call is correct.
The high level wrapper allocates zeroed buffers, so it does not show the problem.

## Cause

The QZ step accumulates its transformations into the eigenvector outputs, so the outputs must start as the identity.
In `src/linalg/gevd/mod.rs:244-248` the initialization only writes ones on the diagonal (`u.diagonal_mut().column_vector_mut().fill(one())`).
The entries off the diagonal keep whatever the caller's buffer held.
The transformations are then applied to that matrix instead of to the identity.

## How the fork fixed it

The fork fills each requested output with zeros before it writes the diagonal, so the output starts as the full identity.
LAPACK `dgges` does the same with `dlaset` before `dhgeqz` accumulates into it.
With that change this program prints the eigenvector `[1, 0]` and a residual of 0.

Fork regression test: `linalg::gevd::t::reused_eigenvector_buffers`. It prefills left, right and both outputs with nonzero values.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#46 (commit `4e2c241`).
* Where it was found: tensor4all/faer#37.
