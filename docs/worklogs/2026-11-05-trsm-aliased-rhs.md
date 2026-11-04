# Staged right hand side in the unit-lower solve at an aliasing stride

No upstream revision was measured for this change. Host: evo2, AMD Ryzen AI MAX+ 395 (Zen 5, 16 cores in
two 8-core L3 domains, AVX-512), boost on (it cannot be disabled there).

## Decisions

- **Cause, from the breakdown.** The evo2 baseline (lcbench, protocol v2: prime 250 ms, 5 calls, 3 rotated
  process sets) has faer's partial pivoting LU losing to MKL `dgetrf` from 2 threads at n = 256-512 and
  from 4-9 threads at 1024-2048, with faer's rate flat while MKL's grows (n = 512: 66 GF/s from 5T on,
  MKL 128 at 8T). Temporary stage timers in `lu_in_place_recursion` (n = 512): the `U` panel solve is
  1.10 ms of 2.61 at 1T (42 %) for about a quarter of the flops, and still 0.55 ms at 8T. In isolation the
  same solves summed to 0.54 ms (1T): the difference is the matrix's 4 KiB column stride.
- **The stride of the right hand side, not of the triangle.** Unit lower `b x b` against `b` columns,
  1T: `b = 256` 658 us at stride 512 for both, 268 us with only the right hand side at 520, 593 us with
  only the triangle at 520; `b = 64`: 30.3 / 6.8 / 26.8 us. Below 128 rows the solve runs on chunks of at
  most 64 columns whose rows its base case and short products walk; at a 1 KiB-multiple stride those 64
  lines share one L1 set (12 ways), the PERFORMANCE_TIPS lesson from the bidiagonalisation.
- **Rejected: a narrower column chunk.** Chunks of 32/16/8 columns at the aliasing stride gave 613/504/454
  us at `b = 256` (against 652 and a padded 260) and slowed the padded case to 275/312/352 us: the short
  products' destination rows alias too, so the chunk width is not the lever.
- **Chosen: stage the node.** A node of 16-128 rows whose columns put at least 24 lines into one L1 set
  (`k * gcd(stride, 4096) / 4096`, twice the 12 ways) copies its right hand side into a per-thread buffer whose
  stride does not alias (`n`, or `n` plus 64 bytes when `n` itself is a 1 KiB multiple, as
  `svd::work_rows`), runs the unchanged recursion on it and copies back. Same operations, same order:
  bitwise the same result. `O(n k)` copies against `O(n^2 k)` work. Per node, so each participant of the
  existing column split stages its own chunk on its own thread; a re-entrant use of the buffer on one
  thread falls back to the in-place solve.
- **Working memory.** Per-thread, grown once to at most `128 x 64` elements, reused (AGENTS.md, working
  memory in parallel code); native scalars only (the buffer is `u64` words), `std` only. The public
  signatures and the LU's scratch query are unchanged.
- Scope: only the unit-lower solve, which is the LU's. The other three triangular solves keep their path;
  the Cholesky's solve has its right hand side transposed (unit column stride), so its rows are
  contiguous and it does not have this defect.

- **The staging floor, from the campaign.** The first revision (`e114eeb`) staged every node of at least 13
  columns at any 1 KiB-multiple stride. The full evo2 grid measured the n = 128 LU (1 KiB stride) 6-8 %
  slower at every budget from 1T to 8T. Staged against in place at the same stride (`f64`, `b = k`, 1T):
  1 KiB: 1.8/1.4 us at 32, 8.0/6.7 at 64 (8 and 16 lines per set); 2 KiB: 1.9/1.9 at 32, 8.2/12.8 at 64,
  47/78 at 128; 4 KiB: 2.0/6.1, 8.5/30.7, 50/138. The rule is therefore the number of lines one set
  receives, with the floor at 24 (between the 16 where the copy loses and the 32 where it wins); the
  table sits next to the constant.
- **Not an effect of this change: Cholesky at 4096.** The candidate grid showed the unchanged LLT path
  13-18 % faster at n = 4096 at every budget while the MKL control held (median 0.997). An interleaved
  A/B of the two binaries gave the same times (342-345 ms 1T, 51 ms 8T for both): the baseline run's
  4096 Cholesky was slow, a between-run shift of one cell family that this change does not reach (LLT
  uses the non-unit solve). Recorded, not claimed.

## Verification conclusions and constraints

- `triangular_solve::aliased_rhs::staged_solve_is_bitwise_the_in_place_one`: f32/f64/c32/c64, shapes that
  cross the 128-row node and leave a column tail, both conjugations, `Par::Seq` and a 3-member team;
  bitwise equality against the same data at a stride that does not alias, and the staging count proves
  the path ran on the 4 KiB stride and not on the padded one nor on an odd 1 KiB multiple.
- `no_step_alloc` now covers a partial pivoting LU at a 2 KiB stride on teams of 4, 8 and 70: no
  allocation after the warm-up.
- The full lib suite passes.
- Campaign numbers: `~/work/lu-chol-opt/` on evo2 (raw logs, tables, the before/after page).
- **Unverified.** Only evo2 (Zen 5). The 1 KiB rule and the 13-column floor come from the L1 geometry
  (48 KiB, 12 ways) shared by the Zen generations measured in this fork, not from a scan on other hosts;
  a many-core host (broccoli) was not run.
