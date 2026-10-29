# Every region at the widest width that pays (tensor4all/faer#91)

## Decisions
- **One rule, one helper.** The maintainer and monitor decided the rule: a region runs at the widest width in
  `2..=degree` that pays, `pays_at(estimate, sections, dispatch_cost(w))`, on a sub-team of that width.
  - `utils::cost::par_width` (units at a rate) and `par_width_time` (a serial time) search the widths from
    the full degree down and stop at the first width that pays. Every width is considered, so a falling row
    (`aarch64-apple`) is safe. The cheap rejection against the cheapest width of the table comes first.
  - `par_width_upto` and `par_width_time_upto` search inside a cap, for splits whose task count is bounded
    by their shape.
  - `ParThreshold::region_width` adds the parameter policy: `Never` gives 1, and `Fixed(floor)` gives the
    whole degree when `units >= floor`.
  - `kernel::kernel_width` (the x86 GEMM kernel's two sections) is the same function.
- **Narrowing is a single-child split.** `utils::thread::with_width(par, w, f)` gives:
  - `f(par)` when `w >= degree`;
  - `f(Par::Seq)` when `w < 2`;
  - otherwise `team.split(&[w])`. A one-child split dispatches no worker share: the child is led by the
    caller, and threadteam's `section` still runs its join guard and its panic handling.

  The helper lives in `utils::thread`, so callers compile without `threadteam`.
- **Only the region narrows.** Each site wraps only its own region in `with_width`, and the caller keeps its
  whole `par` for later, larger work.
- **Per-step factorizations.** Hessenberg, tridiagonal, unblocked bidiagonal, full pivoting LU, Bunch-Kaufman
  and column pivoting QR set `par = Par::Seq` the first time a step did not pay and kept it for the rest of
  the call. They now decide per step.
  - `Never` still makes the whole call sequential, through an explicit `par = Seq` at the top.
  - Column pivoting QR's non-delayed product at the head of step k uses the width decided at step k-1, which
    is the decision its comment says covers it.
  - **Behaviour change:** the post-loop `upgrade_householder_factor` of the Hessenberg and tridiagonal
    reductions ran under the last step's latch, which was always sequential by then. It now asks its own
    gate.
- **Capped splits.** A split whose task count has an upper bound searches inside the cap; it does not clamp
  the answer afterwards, which a non-monotone row would make wrong.
  - Triangular solve: jobs of 64 columns. The region's width is searched separately and shared out among the
    jobs, which keeps the nested parallelism the old code gave each job.
  - The block apply's column split: `K` tasks, likewise.
  - `upgrade_householder_factor`: four rows per participant, at both gates. A narrower outer width re-runs
    the call on a sub-team of that width.
- **Blocked bidiagonal panel.** The gate stays a floor: sequential, or the whole `par`. Its estimate is the
  remaining area, and narrowing the whole panel to it would cap the panel's two trailing products, which
  choose their own widths.
- **The tiled product (the non-x86 route) takes the same rule.** `gemm_tiled` chose between the whole
  team and one thread. It now asks `par_width_team`. When the tile grid uses fewer participants than the
  chosen width, that smaller width must pay as well, because a row need not be monotone.
- **The `V^H V` reduction splits into exactly the chosen number of row ranges.** It uses balanced
  `par_split_indices`, where ceiling-sized chunks could produce fewer tasks than the width the gate chose.
- **`Never` at the public boundaries.** Column pivoting QR and the bidiagonalization apply `Never` before
  forwarding `par` to their compact-factor upgrades. Otherwise the upgrade's own gate could open a split
  under `Never`, now that it caps by rows per participant instead of rejecting.
- **New gate on the Bunch-Kaufman blocked swap loop.** It split over the whole degree with no gate at all,
  and the family test caught it. It now uses `par_width` at the LU swap rate.
- **Removed:**
  - `par_pays`, `region_pays` and `region_pays_impl`;
  - the GEMM route's full-width predicate at `matmul/mod.rs`. It was reached only after the team arm
    returns, so it was always 1.

  `ParShape::pays_full` stays as public API.
- **Trace.** `par-trace` records the chosen width (a new `width` field), and `trace_pub` takes it too.
- **Counter.** `utils::team::stats` gains `width`: the regions narrowed to a sub-team, by width, up to 256.
  The `run` histogram saturates at 16.

## Verification conclusions and constraints
- **Tests, no timing:**
  - `par_width_takes_the_widest_width_that_pays`: a rising row, a falling row, a cap inside a falling row.
  - `with_width_runs_on_a_sub_team_of_that_width`: the participant count of a section run on the narrowed
    token, and the parent's width afterwards.
  - `the_families_narrow_their_regions_and_match_sequential`, on a team of 8 whose row makes width 4 the
    widest that pays. Families: full and partial pivoting LU, column pivoting QR, Hessenberg, tridiagonal,
    bidiagonal, Bunch-Kaufman, triangular solve and matvecs. For each:
    - regions narrowed to 4, and some of the family's own splits ran at 4;
    - no split wider than 4;
    - the outputs equal the sequential ones within `1e-10` relative.
  - The family test runs every converted region in its own counter window: the triangular solve and both
    matvecs separately, Bunch-Kaufman with full pivoting (the default `PartialDiag` does not reach the
    converted steps), the block apply, the compact-factor upgrade, the sparse product and the tiled product.
  - `never_reaches_the_compact_factor_upgrades`: a 33 x 9 column pivoting QR at block 9 and a
    bidiagonalization under `Never`, on a cheap row, open no section. It fails without the boundary guards.
  - The rewritten gate tests: `region_rules_follow_the_serial_time` (partial pivoting LU),
    `fixed_and_never_keep_their_meaning`, `par_width_reads_the_degree_of_the_token`.
- **Builds:** default, `par-trace`, and `--no-default-features --features std`. The last has the same
  warnings as the base.
- **Not changed (follow-ups):**
  - The Hessenberg step estimate is `(n-k-1)^2`. The work it gates after the first step is the `A02` matvecs,
    `k (n-k-1)`, which grows while the estimate shrinks. Re-modelling it needs calibration.
  - The regions with no gate or a degree-wide split that #94 owns: `rank_update`, the SIMD GEMM loops, the
    schoolbook product, native `K == 1`.
  - The section counts passed by some gates are inherited models, not dispatch counts.
- **Performance evidence is the monitor's:** broccoli at 1/8/16/32/64 threads, and evo.
