# Team dispatch cost read from threadteam (threadteam-rs#6)

## Decisions
- The extrapolation of the team dispatch cost above the measured widths lives once, in threadteam
  (`WidthCosts`, 0.1.1: linear above a row's largest anchor; 0.1.2: a measured Zen2 team row for
  `x86_64-avx2` with anchors 2 to 64). faer's `utils::cost::team_section_cost` and
  `MEASURED_MAX_WIDTH` (#97, provisional) are deleted; the four readers ask the table directly:
  `ParShape::dispatch_cost` reads `DispatchCosts::cost(Team, w)`, `par_pays_time`,
  `team::kernel::kernel_width` and the tiled team product's floor ask `TeamExec::dispatch_cost(w)`.
  The requirement moves to `threadteam = "0.1.2"`, so a resolver cannot pick 0.1.0 (flat above 6).
- `kernel_width` asked the widths in increasing order and stopped at the first that did not pay,
  assuming a cost that never falls with the width. Measured rows fall (`aarch64-apple` team 3.0,
  2.3, 1.4 us at widths 2, 4, 6), so a product that pays at 6 but not at 2 got no width. The widths
  are now searched from the full width down and the first that pays is taken.

## Verification conclusions and constraints
- Decisions change, by design, also at widths 8 and below, so the "bit-identical up to 8" argument of
  #97 does not carry over:
  - rows whose largest anchor is 6 (generic, `aarch64-apple`) grow from width 7 on (threadteam is
    linear from its largest anchor; faer held the table flat to 8): generic width 8 is 4 us instead
    of 3 us, so a 10 us one-section region at width 8 now stays serial;
  - `x86_64-avx512` team reads its measured width-8 anchor (0.98 us instead of 0.9);
  - `x86_64-avx2` (Zen2 row) is cheaper than what faer charged that class before at every width
    (1.42 us against 3 us at width 2, 9.33 us against `3 * 64 / 8` = 24 us at width 64), and measured
    up to 64 instead of extrapolated, so regions on such hosts go parallel more readily.
- Tests: `utils::cost::tests::the_team_cost_is_threadteams_table_at_every_width` (shape cost equals
  threadteam's table at widths 2 to 128 for five rows; integer `cost(w_m) * w / w_m` above each
  row's largest anchor, including width 128 on the Zen2 row; a region that does not pay at 8 does not
  pay at 64), `utils::team::tests::kernel_width_finds_the_widest_paying_width_on_a_falling_row`.
- Not measured: the adoption protocol's blast-radius acceptance run (every family whose gate reads
  the team cost). This change moves gate inputs, not kernels; per the maintainer's rule against a long
  campaign per change it is left to the maintainer before merge.
