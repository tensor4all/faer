# Six small issues in one change (#303, #304, #135, #283, #222, #245)

## Decisions

- **QR rank truncation mirrors the dense LUs (#303).** `ColPivQrParams` gains
  `max_rank`/`abs_tol`/`rel_tol` (`Option`, all `None` from `auto()`), `ColPivQrInfo` gains `rank`,
  and the stopping rule is the LU one (`|R[k,k]| > max(abs_tol, rel_tol * |R[0,0]|)`, relative to the
  **first** entry) through the same `linalg::lu::rank_threshold`/`rank_rejects` helpers, so the two
  factorizations cannot drift apart. `max_rank = r` runs the first `r` steps and stops after step
  `r - 1`'s trailing update, which is the last one whose output the rank-`r` factor needs: rows
  `0..r` of the stored factor are the full factorization's first `r` rows bit for bit, and the block
  below is the Schur complement after those steps. (Stopping *before* that update was the first
  attempt and is wrong: the update is what finalizes row `r - 1`.)
  - `ColPivQrInfo` had no hidden non-exhaustive field, so `#[non_exhaustive]` is added to it **and**
    to `PartialPivLuInfo` in the same change: the next field on either is then additive.
  - `ColPivQrParams` keeps its hidden `non_exhaustive` field, so it is additive already.
  - The compact-WY upgrade loop in `qr_in_place` now stops at the computed reflectors; the columns
    past them would otherwise be read as if they held reflectors.
  - No high-level `ColPivQr::rank()` method: the LU side has none either (the issue's second bullet
    asks for one, but mirroring the LUs is the scope here, and a method that recomputes the diagonal
    test on the stored `R` is a separate API decision).
- **Balancing job selection (#304).** `BalanceJob::{None, Permute, Scale, Both}` maps `xGEBAL`'s
  `N`/`P`/`S`/`B`; `balance_in_place_with` runs only the named steps; `balance_in_place` is the
  `Both` wrapper, which is what it always computed (the existing doc says `job=B`). `None` and
  `Scale` return `0..n` as the active range, `Permute`/`Both` the isolated one. The non-finite check
  stays for every job, including `None` (a caller that relies on it does not get a silent pass).
  The issue's second part -- an optional balancing step inside the EVD driver -- is **not** here: it
  is a driver design decision (left/right back transformation, default `None`) that this batch does
  not take.
- **BiCGSTAB breakdown (#135).** `BicgError::Breakdown { abs_residual, rel_residual, iter_count }`
  as the upstream request (Codeberg #334) words it, with the iteration count it suggests as
  "possibly useful". This implementation factors the block `r̃ᵀ v`, so the breakdown conditions are
  its rank test, a half step that is not finite, and a new residual that is not finite; the doc of
  the variant names them. `BicgError` becomes `#[non_exhaustive]` (the upstream issue proposes it in
  the same breath as the variant), so the *next* variant is additive even though this one is not.
- **Pivoted Cholesky block size (#283).** `assert!(block_size > 0)` in `cholesky_in_place` and in
  `cholesky_in_place_scratch`, documented on the params and as `# Panics` on both functions. An
  assertion rather than a typed error: this crate's parameter contract is asserted (`qr`'s
  `block_size > 0` is an `assert!`, and the LLT shape checks are too), and the scratch query returns
  `StackReq`, not `Result`.
- **LSMR scale invariance (#245).** The bidiagonalisation coefficients carry the scale of `A`, and
  `zetabar` (their product) and the triangular solves carry its square or its inverse, so an `A` at
  either end of the range left the range although the solution fits. The coefficient matrices are
  divided by the power of two nearest `|A|` (taken from the first `alpha`), the two products that
  feed the next coefficients are brought to the same scale, and the solution update is multiplied by
  the inverse: exact arithmetic for `A / scale`, whose solution and residual are the same objects.
  `beta`'s first block is `|b|` and is left alone. The convergence test compares the estimate with
  the initial preconditioned normal residual as a **ratio** (the product
  `rel_tolerance * norm_ref` underflowed at a small `|A|`), and a convergence claim additionally
  requires a finite iterate. The estimate is the normal residual of the scaled problem, so it is
  multiplied by `scale` for `LsmrInfo::abs_residual` and for the comparison with the caller's
  `abs_tolerance`; `rel_residual` is the ratio, which is the same in both frames. Consequences: a
  problem with `||A||` inside `[1/sqrt(2), sqrt(2))` keeps `scale = 1` and therefore bit-identical
  recurrences and reported numbers.
- **PATCHSET (#222).** A row for #217 records the protected denominator and that upstream 0.24.4 and
  `main` (`0ecd846`) carry it at `faer/src/linalg/gevd/mod.rs:321-338` / `:342-360` (read, not run),
  plus the `1.0 + 1e-16` note: it rounds to `1.0` in f64, so the unit test now uses `A = 1e-300`
  with `w = 0`, a representable denominator below `smin` that the pre-fix code scales twice into
  `inf`. No `upstream-reports/` entry: that is the deferred-feedback decision of `AGENTS.md`, and a
  runnable report against upstream was not built here.

## Verification

- `cargo test -p t4a-faer --lib`: `test result: ok. 679 passed; 0 failed; 162 ignored`.
- Focused: `qr_rank_truncation` (5), `lsmr_scale` (5), `piv_llt_block_size` (5), `public_balance` (5,
  four inherited), `matrix_free::bicgstab::tests` (9, four now asserting `Breakdown`).
- Mutation checks: disabling the LSMR scale (coefficient `1`) fails `lsmr_scale`; removing the
  block-size assertion fails three `piv_llt_block_size` tests; reverting the `#217` unit-test line
  to `1.0 + 1e-16` leaves it passing on the fixed code, which is exactly why it is not a regression,
  and the new line fails on the pre-fix `solve_shifted_1x1`.
- `cargo clippy -p t4a-faer-traits --all-targets -- -D warnings` is clean (unchanged crate; the gate
  covers it only). `cargo fmt` is checked on the changed files only: `--all --check` already fails on
  the base with 666 `Diff` lines.
- Not verified: any x86-only path (the host is aarch64), and the QR truncation on a team beyond the
  4-participant cases in the test.
