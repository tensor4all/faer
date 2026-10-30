# Look-ahead in the blocked QR (faer#160)

Revision measured: `main` `81f3b1d`/`294e432`. No upstream revision was measured for this change.

## Decisions

- **Cause.** The probes on broccoli (EPYC 7702P, 64 cores) at 4096 located the cause. At 16T `qr_in_place` is 371 ms against MKL `dgeqrf`'s 328 ms. The top-level panel takes 104-121 ms, and its share does not shrink with the thread count (0.67 of its 1T time at 4-32T, 0.96 at 64T). The trailing apply is 240 ms, already below MKL's whole call. faer#160 has the stage tables.
- **Rejected candidates** (faer#160, faer#90):
  - Whole-team GEMMs in the block apply: slower, and they allocate per call (the kernel's job arrays).
  - A 4-task cap on the block apply: slower.
  - Owner-packed shared V and T (OpenBLAS `level3_thread.c` buffer roles): wins of 0.45-0.97 on large shapes at 8-32T. It regresses at small n, collapses at 64T and slows the panel stage, so it was not adopted under the maintainer's "adopt if it helps without regressions".
  - A row-split apply for the panel's narrow targets: it engaged, but the panel moved by 1 ms.

  The panel's time is a fixed cost per recursion node, not flops. Block 128 → 32 cuts its arithmetic about 4×, but the panel only drops from 104 to 86 ms at 16T. At block 32 it already runs at serial `dgeqr2`'s speed: 106 ms in MKL's composed stages.
- **Look-ahead.** The deficit is that the panel is not hidden. Each step therefore applies block `j` to the next block's columns, then runs a `join_sized` of two branches:
  - a panel group factors panel `j + 1`, in its own coordinates, on a disjoint submatrix and disjoint factor columns;
  - the rest of the team applies block `j` to the far columns.

  Constraints from the pre-review:
  - Only full-rank blocks (`row == col`) look ahead. Rank loss keeps the existing compaction path, so no factor merge runs concurrently.
  - The two branches split the scratch envelope that `qr_in_place_scratch` already gives, with `can_hold` checked. There is no new buffer and no scratch growth.
- **Per-step decision** (`lookahead_width`). Sequential costs `P(p) + F/p`. Overlapped costs `max(P(a), F/(p − a))` plus one split. F and P come from `cost::block_apply_estimate`. The panel is estimated as a block apply of its own width on itself, which matches its measured serial time at block 128. `a` is taken from {1, 4}.

  The panel's team speed-up is a constant, `PANEL_TEAM = 0.68`, measured in panel2 on broccoli: 185 ms on one participant against 124-131 ms on 4-32. At 4T the model overlaps only the last blocks, where the far apply is between roughly 1.3 and 8 panels' work.
- **Block width unchanged.** `Q_coeff`'s height is the compact-WY layout that the Q formation, solves and sequence applies read. With the panel hidden, the block-width lever shrinks: the model gives block 128 the edge at 16T, because the apply rate falls with the width.

## Verification conclusions and constraints

- `test_lookahead_matches_sequential` compares against `Par::Seq`: R and the rank to rounding, the compact factors on full-rank input, and `Q R = A` through `reconstruct`, which reads the factor layout the sequence consumers use. It covers:
  - f32, f64 and c64;
  - tall, square and wide shapes, and a block that does not divide the width;
  - the step model's own choice on 4 participants;
  - every eligible step forced, with panel groups of 1 and 4;
  - rank loss inside prefactored panels: a zero column, and a whole zero panel.

  It asserts that the path ran.
- The full `t4a-faer` lib suite passes.
- Performance: the broccoli 1-64T and evo 1-8T acceptance runs (fork against `main` and against MKL/OpenBLAS `dgeqrf`, n = 256-4096) are requested through the monitor. Their numbers go into the PR's Parallel Scaling section.
- **Unverified.**
  - `PANEL_TEAM` and the panel estimate come from one host (Zen 2) and one size. Other hosts may need their own row.
  - The panel group's width of 2 was not measured, so it is not offered.
  - The whole QR call is not covered by `no_step_alloc`: `upgrade_householder_factor`'s deep split allocates its partials. That is inherited, not introduced here.
