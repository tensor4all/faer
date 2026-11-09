# Full-pivoting LU: small square row-major inputs skip the copy (tensor4all/faer#314)

## Decisions

- `full_pivoting::factor::lu_in_place` used to take the shared row-major copy
  (`linalg::row_major_copy_pays`, at least `64 * 64` elements, tall or square) before
  its own transpose dual (`lu_in_place_unblocked(.., transpose = true)`) was even tried.
  It now skips the copy for **square matrices below 128** and keeps it everywhere else
  (`row_major_takes_the_copy`).
- The rule is the issue's acceptance criterion applied shape by shape: the dual has to
  run within 1.05x of a column-major input. Square `64..=127` passes on both machines
  measured (Apple M5 Max, this change: f64 C/A 1.009 / 1.039 at 64 and 1.007 / 1.005
  at 128, one / four threads; in an ungated scan, 64-120 square at 0.98-1.04 in f64 and c64; Zen 5, the
  issue's earlier comment: n = 64 and 96 at 0.94-1.04 over f64/c64/f32 at 1T and 8T).
  Square `128` is mixed on Zen 5 (f64 1T 1.114) and from 256 up the dual leaves the band
  on at least one machine (M5 four threads 1.087 at 512, 1.089 at 1024; Zen 5
  1.08-1.27), so the boundary sits below 128. On M5 one thread the square dual stays
  at 1.03-1.04 up to 1024, so a single-thread-only rule could go further; not taken.
- Off the square the band ends early (`80 x 64` f64 1.06x, `128 x 64` 1.11-1.14x on
  M5, the same ungated scan), so near-square shapes keep the copy.
- No change to the dual kernel in this slice, and no change to `row_major_copy_pays`
  or to the other families that share it.

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
  allocation of the calling thread during the call; on 64, 96 and 127 square it stays
  below `m * n * 8`, and on 128x128 and 256x32 (copy kept) it reaches it, so the
  counter is shown to see the copy. It is an integration test because the lib's test
  binary already has a `#[global_allocator]` (`sparse::linalg::amd`). With the
  square exclusion removed from the guard the test fails at `64x64` (an allocation of
  32768 bytes, the whole matrix). The full-pivoting arm of
  `test_row_major_adaptation_matches_the_column_major_one` now asks the new guard
  whether the two orders must agree bit for bit (copy) or up to rounding (dual), and
  covers 127x127.
- Measurement: tensor4all/faer#314 comment (fbench `01a1cb8`, fork row only, fork
  `b16d6b8` plus a local switch that bypasses the copy, f64, `--reps 5`, three
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
  ratios on large shapes are the least reliable numbers here.
- Upstream faer was not measured.
