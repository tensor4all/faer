# Rank truncation for the partial- and full-pivoting LU (#294)

## Decisions

- **Opt-in fields, defaults unchanged.** `FullPivLuParams` and `PartialPivLuParams` gain
  `max_rank: Option<usize>`, `abs_tol: Option<f64>`, `rel_tol: Option<f64>` (all `None` from
  `auto()`); `FullPivLuInfo` and `PartialPivLuInfo` gain `rank: usize`. With every option `None` the
  code paths and their arithmetic are the ones before the change and `rank = min(m, n)`. The issue
  sketched `Real<T>` tolerances with `0` as "off"; `Option<f64>` was the contract given for the
  implementation, and it keeps the params non-generic as they are today.
- **Stopping rule.** A pivot is accepted while `|pivot_k| > max(abs_tol, rel_tol * |pivot_0|)`
  (modulus; an absent tolerance counts as 0), and the first rejected pivot ends the factorization.
  `rel_tol` is relative to the **first** pivot, which is the rrLU contract tensor4all-core already
  uses; relative-to-previous is a different algorithm. The definition is written in the params'
  doc comment, and a test (`rel_tol_is_relative_to_the_first_pivot`) uses pivots 1, 1e-3, 1e-6 so
  that the two definitions give different ranks (2 against 3).
- **Full pivoting stops in the loop.** The next pivot is already known at the top of step `k` (the
  previous step's fused rank-one update searched it), so the check costs one modulus. On a stop the
  remaining transpositions are the identity, no further rank-one update runs (`O(m n r)` instead of
  `O(m n min(m, n))`), and the trailing block holds the Schur complement after `r` steps. `max_rank`
  stops at the top of step `r`, after step `r - 1`'s update, so the trailing block is the Schur
  complement in both cases; that last update is one pass over the trailing block, the cost of any
  single step. The pre-existing stop at an exactly zero or subnormal Schur complement is reported as
  `rank = k` when an option is set and leaves `rank = min(m, n)` otherwise (compatibility).
- **Partial pivoting: `max_rank` shortens, tolerances do not.** `max_rank = r` runs the existing
  recursion on the `m x r` panel (its swap stage already swaps the columns past the panel), then
  computes `U[:r, r:]` with one unit-lower solve -- the same call the wide-matrix branch made, which
  is the `r = min(m, n)` case of it, so the default path is the same call on the same views. The
  Schur complement update is skipped; the trailing block holds the row-permuted input. The
  tolerances are checked on `|U[k, k]|` after the factorization: partial pivoting is recursive and
  blocked, a stop inside it would need a right-looking restructuring, and the issue accepts a
  post-hoc rank as the first step. Partial-pivoting pivots are not rank-revealing in general; the
  doc says so.
- **Non-finite input.** The pivot searches ignore NaN (`>` against NaN is false), so a NaN spreading
  through the Schur complement could end the full pivoting loop as an "exactly zero" block, and an
  infinite first pivot makes `rel_tol * |pivot_0|` infinite and rejects everything after it. Both
  would be a silently truncated rank. Choice: when any option is set the input is scanned once
  (`is_all_finite`, one pass, before the factorization); a non-finite input switches the tolerances
  off and `rank` is `min(m, n)` capped by `max_rank`, so the NaN/Inf reaches the factors as in the
  untruncated run. Independently, a non-finite threshold switches the tolerance off and a NaN pivot
  modulus is never rejected.
- **Solve/inverse/reconstruct** take the packed factor without the info, so they cannot reject a
  truncated one; the params' doc says they assume the full factorization.

## Verification conclusions and constraints

- `faer/tests/lu_rank_truncation.rs`: golden digests of the default factorizations (f64 and c64, six
  shapes, `Par::Seq` and a team of 4) recorded on `1d36fbb`, the commit before the change, compared
  bit for bit on aarch64 only (the SIMD kernels round differently per architecture); options that
  never fire leave every bit unchanged on all architectures; exact rank `r` for f32/f64/c32/c64,
  square/tall/wide; `max_rank` reproduces the untruncated pivots (full pivoting: factors bit-equal;
  partial pivoting: rows equal, factors to rounding, since the `r`-column recursion splits its blocks
  differently); NaN/±Inf inputs.
- Not verified: an x86 golden; non-finite values produced by overflow during the elimination of a
  finite input (the scan sees only the input); the parallel full-pivoting path at a truncation stop
  beyond the team-of-4 digests.
- No campaign measurement. A release-mode scan on the development Mac (single run per cell, not a
  paired comparison) is in the PR description; the faer-benchmark harness has no rank-truncated
  case.
