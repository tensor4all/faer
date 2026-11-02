# Cap the equal-area triangle split at one strip per part

`utils::partition::lower_area_parts` and the width clamp in the product into a
lower triangle (`matmul/triangular.rs`), for the residual of #170's equal-area
split: 1024/64T Cholesky +7.4 % (#168).

## Decisions

- **The cap is `total / (align * m)`, applied as the region width.** Part `i`
  holds `total / parts` area and a part of `dc` columns from column `c` holds at
  most `dc * (m - c) <= dc * m`, so `total / parts >= align * m` forces every
  exact boundary to advance by at least `align`. Rounding to the nearest column
  commutes with an integer shift and the round-up of two columns `align` apart
  differs by `align`, so the parts stay non-empty at or below the cap. It is
  applied where the width is chosen (not inside `lower_area_split`, whose `i`
  and `parts` must agree with the caller's loop), so the participants a
  collapsed part would have woken stay parked.
- **The mechanism is pinned by a unit test, not by a timer.** At 64 parts on the
  128x128 trailing block of the 1024 Cholesky the exact parts are one column
  wide, every boundary rounds up to 8, seven of 64 parts are empty, and the cap
  is 8. `lower_area_parts(1024, 1024, 8) = 64`, so the full trailing block keeps
  every part; only the small blocks lose them.

## Verification conclusions and constraints

Correctness gates pass: `cargo test -p t4a-faer --lib` 595 passed / 138 ignored
(3 new partition tests), nightly `rustfmt` on the touched regions, `cargo clippy
--all-targets -D warnings` clean for both files, `cargo +1.89.0 build` (MSRV).

**No performance conclusion.** The acceptance cell moved from 1024/64T to
1024/48T (grid limit 48T = the user's decision; the re-read of the acceptance
criterion = approved by the audit, declared in #168 before measuring), and 48T
turned out to be unmeasurable on this host: the strict idle guard needs 48 of 64
cores below 5 % for 3 s while `md0_resync` (root), another user's
`cursor-server` nodes and `codex_run_monitor` are resident. No 48T or 64T cell
was ever accepted; they are recorded as missing with the measured busy values,
not filled in. The 64T cells accepted under a relaxed guard are kept in
`relaxed-guard-deviation/` and excluded from acceptance: one `fork` sample is
41.216 ms against 0.815 ms in the same build's other rep.

Over the cells that were accepted (1-32T) the paired `head/base` ratios are
0.973-1.032 for the arm the cap affects and 0.932-1.132 for the control arm that
it does not touch, so the cap's effect is **inside this host's noise**: neither
"no regression" nor "fixed" is established. One MKL reference rep is excluded
under a criterion stated before it was applied (a `potrf` 3x off the cell's
other reps; the whole process was 2.6x slow while both guard checks passed -
the guard cannot see interference inside a run).

Evidence: `/home/shinaoka/runs/faer-168-partition-cap` (`README.md`, raw CSVs,
`guard.log`, `guard.err`, `exit.log`); tables and the exclusion criterion in
[#168](https://github.com/tensor4all/faer/issues/168#issuecomment-6571401294)
and [#160](https://github.com/tensor4all/faer/issues/160#issuecomment-6571430032).

## What remains unverified

The +7.4 % cell itself. It is a 1024/64T cell, so this host cannot re-verify it
and the 48T proxy is missing too. A quiet window (weekend night) for 48T alone,
or a decision to judge small-size high-T cells only at <= 4T (one L3 domain),
are the two options left; neither was attempted here.
