# Full-pivoting LU: small square row-major inputs skip the copy (tensor4all/faer#314)

## Decisions

- `full_pivoting::factor::lu_in_place` used to take the shared row-major copy
  (`linalg::row_major_copy_pays`, at least `64 * 64` elements, tall or square) before
  its own transpose dual (`lu_in_place_unblocked(.., transpose = true)`) was even tried.
  It now skips the copy for **square matrices below 128 whose rows are contiguous**
  (`col_stride == 1`, any row stride, so a padded or reversed row-major view counts) and
  keeps it everywhere else (`row_major_takes_the_copy`, which takes the strides).
- The stride condition came from the #342 review: the first version tested only the
  shape, so a generic-stride view such as 96x96 with strides `(2, 192)` also skipped the
  copy and, in place, fell to the separate matrix update and pivot search instead of
  the fused SIMD kernel (that kernel needs the unit column stride the dual sees as a
  unit row stride). Generic strides, and a column stride of `-1`, keep the copy as on
  `main`. That path was not timed; the allocation test pins the dispatch.
- The rule is the issue's acceptance criterion applied shape by shape: the dual has to
  run within 1.05x of a column-major input. **Timed sizes**: on Apple M5 Max, square
  64, 80, 96, 112, 120 and 127 in f64 and c64 at one and four threads (the paired run
  below: per-repetition median C/A 0.888-1.017 over those 24 cells); on Zen 5, the
  issue's earlier comment, n = 64 and 96 at 0.94-1.04 over f64/c64/f32 at 1T and 8T.
  The sizes in between (65-79, 81-95, ...) and every size other than 64 and 96 on
  Zen 5 were **not timed**; they are taken on the strength of their neighbours.
  Square `128` is mixed on Zen 5 (f64 1T 1.114) and from 256 up the dual leaves the band
  on at least one machine (M5 four threads 1.087 at 512, 1.089 at 1024; Zen 5
  1.08-1.27), so the boundary sits below 128. On M5 one thread the square dual stays
  at 1.03-1.04 up to 1024, so a single-thread-only rule could go further; not taken.
- Off the square the band ends early (`80 x 64` f64 1.06x, `128 x 64` 1.11-1.14x on
  M5, an ungated scan run before measurement 1 at a 1-min load of about 7), so
  near-square shapes keep the copy.
- No change to the dual kernel in this slice, and no change to `row_major_copy_pays`
  or to the other families that share it.
- **Pivot ties depend on the layout.** The search runs on the input as it is when
  `|row_stride| < |col_stride|` (or on its column-major copy when the copy is taken),
  and on its transpose otherwise, and keeps the smallest column, then the smallest row,
  of that matrix. So a row-major input that is not copied breaks a tie by row first. The
  square exclusion is not the only such case: wide row-major matrices and those below
  `64 * 64` elements are never copied and were searched transposed before this change
  too (#342 review: 8x16 row-major picks `(0, 4)`, column-major `(1, 2)`). Example
  (`faer/tests/full_piv_lu_row_major_no_copy.rs`): zero 64x64 f64 with
  `A[(1, 2)] = A[(0, 4)] = 2`, `max_rank = Some(1)`, `Par::Seq` pivots on `(1, 2)` with
  two transpositions column-major and on `(0, 4)` with one row-major. #314 allows a
  different pivot order on ties if it is documented; it is, under "Pivot ties" on
  `lu_in_place`, which now describes that dispatch rather than the copy exclusion alone.
  On the adaptation test's fixtures, which hit no such tie, the pivots, permutations and
  transposition counts agree; that is all the test shows.
- **The SIMD search did not keep the smaller row on a tie within one column**
  (inherited from upstream, found by the second #342 review). Each column is searched
  with four interleaved accumulators (three in the fused update), and they were merged
  with a strict `>`, so an equal maximum in the accumulator merged first won even with
  the larger row: zero 64x64 row-major with `A[(2, 6)] = A[(2, 8)] = 2` pivoted on
  `(2, 8)` on Apple M5 (two f64 lanes), and column-major `A[(6, 2)] = A[(8, 2)]` on
  `(8, 2)`. `best_score` now takes the other accumulator on a strictly larger score or
  on an equal one with a smaller row. Red/green: `pivot_search_keeps_the_smaller_row_across_accumulators`
  (every row pair of a 64-row column, both searches, three alignments) failed at rows
  `(0, 2)` with row 2 returned, and the integration test returned `(2, 8, 2)`, before the
  fix; both pass after it. This changes which pivot is taken on such ties for every
  layout, including column-major; it is the order the scalar search and the existing
  doc comment in `reduce_2d` already described. Not timed: the merge runs once per
  column per step, against an O(m) column walk. The parallel step's merge across
  column blocks keeps the earlier block with a strict `>`, which is the smaller column;
  no tie test runs on a team.

## Where the dual loses (stage breakdown, M5 Max, f64, one thread)

Per-stage timers inside `lu_in_place_unblocked` (local patch, not committed):

- **Square, n >= 256**: the rank-one update with the fused pivot search costs the same
  through both layouts (1024: 104.4 ms column-major, 103.7 ms dual). The whole gap is
  the dual's row scaling `A[(k, j)] *= inv`, which walks a strided row every step,
  one cache line per element, O(n^2) in total: 0.22 -> 3.26 ms at 1024,
  0.06 -> 0.60 ms at 512, 0.016 -> 0.069 ms at 256. The pivot search is fused into the
  update and is not where the time goes.
- These stage timers ran before the process gate below existed (load 5-7, no competing
  build checked); they compare stages inside one call, not cells across processes.
- **Tall**: three things at once. The scaling again (4096x256: 0.39 -> 5.04 ms), the
  long swap turning strided (1.25 -> 7.63 ms), and the trailing block turning wide
  (update 36.1 -> 40.2 ms), the last one being the issue comment's "wide trailing block"
  effect.
- So a dual that scaled the pivot row contiguously (fused into the update's column
  walk, or staged through a contiguous buffer) would plausibly close the square gap;
  that is a kernel change for a later slice and is **not measured**.

## Verification conclusions and constraints

- Tests: `faer/tests/full_piv_lu_row_major_no_copy.rs` records the largest single
  allocation of the calling thread during the call; on 64, 96 and 127 square, and on
  a padded 96x96 row-major view (strides `(100, 1)`), it stays below `m * n * 8`; on
  128x128 and 256x32 (copy kept) and on 96x96 views with strides `(2, 192)` and
  `(192, 2)` it reaches it, so the counter is shown to see the copy. It is an
  integration test because the lib's test binary already has a `#[global_allocator]`
  (`sparse::linalg::amd`). Red checks: with the square exclusion removed the test
  fails at `64x64` (an allocation of 32768 bytes, the whole matrix); with only the
  `col_stride == 1` condition removed it fails at `96x96, strides (2, 192)` (largest
  allocation 96 bytes). A second test pins the tie example above. The full-pivoting
  arm of `test_row_major_adaptation_matches_the_column_major_one` now asks the new
  guard whether the two orders must agree bit for bit (copy) or up to rounding (dual),
  and covers 127x127.
- Whole crate suite (`cargo test -p t4a-faer --no-fail-fast`, this host): the one
  failure at this change's heads is `lq_allocates_nothing_60x300` (allocation counts
  `[13, 12, 17, 17]`); the same command on `b16d6b8` (run once, 2026-11-09, separate
  target directory) failed on that test alone, with 955 passing. The counts are in the
  #342 description. It is one of the host failures listed in
  `2026-11-04-gevd-repeated-complex-pairs.md`.
- Padded and reversed row-major views (`(100, 1)`, `(-100, 1)`) go through the dual by
  the guard; the allocation test shows that for `(100, 1)` and the adaptation test
  covers reversed rows for correctness. Their speed was **not measured**; the timed
  cells are contiguous row-major only.
- **Measurement 2 (the one that covers the merged code)**: the source of `32c615f`
  (clean committed tree; `faer/src` tree `38ce727`, unchanged by the later doc-only
  commits) against `main` `b16d6b8`, both built into fbench `01a1cb8` with the same
  one-line local `set_threads` patch (returns Ok for `NAME == "none"`), family
  `full_piv_lu`, row `faer-fork`, the `in_place` line, `--reps 5 --prime-ms 250
  --statistic median`. Cells: square 64, 80, 96, 112, 120, 127 (changed) and 128, 256
  (unchanged code, the control), f64 and c64, 1T and 4T, three repetitions; per cell
  and repetition the two binaries run back to back (head first on odd repetitions,
  base first on even). Each process reports its own column-major A and row-major C (or
  B), so the ratio is taken within a process and the median over the three
  repetitions is reported. Gate as below (no `rustc`, busy `cargo`, or other
  `target/` binary before and after); 11 cells were discarded and rerun, 96 kept, 1-min
  load before a kept cell 7.0-26.4, 18:41-19:01 JST. `fbench verify` passed all 64
  cells per thread count (1T, 4T) on the head binary. The full table is in the #342
  description. Result: C/A 0.888-1.017 on all 24 changed cells, base B/A 0.978-1.343
  (median per cell); the unchanged 128/256 controls read C/A 0.976-1.181 at head and
  B/A 1.005-1.150 at base. Single repetitions swing far more (one control cell 0.650,
  another 1.616; the two binaries' A arms of one repetition differ by up to 1.8x), so
  per-repetition medians are the only statistic here, and they are a scan, not a
  campaign `candidate_compare.py` result (that was not run).
- Measurement 1 (the large-shape and boundary evidence above, 64-1024 square and the
  tall shapes): tensor4all/faer#314 comment (fbench `01a1cb8`, fork row only, fork
  `b16d6b8` plus a local, never committed `FAER_FLU_COPY=off` environment switch that
  skipped the copy branch, i.e. the dual on every shape; that switch is in no published
  revision, so arm C of that run cannot be rebuilt from a checkout. f64, `--reps 5`, three
  repetitions per cell, the arms run back to back within each repetition and the cells
  cycled per repetition rather than grouped by arm).
- Gate: a 1-minute load below 4 is not reachable on this host (its idle baseline is
  about 6 from WindowServer and sysmond), so a cell was kept only if no competing
  `rustc`, busy `cargo`, or other `target/` or bench binary was running before and
  after it. Two of 44 cells were discarded and rerun. Load averages recorded over the
  kept cells (1 / 5 / 15 min): 7.58-49.01 / 29.23-54.65 / 26.63-32.71, decaying from
  two other agents' builds just before; the per-cell values are in the issue comment.
- macOS cannot pin threads, so the 4T cells mix core types: the two column-major
  controls of a 4T cell differ by up to 15% (1024x64), against at most 1% at 1T. The 4T
  ratios on large shapes are the least reliable numbers here. Measurement 2 shows the
  same at small sizes.
- Upstream faer was not measured.
