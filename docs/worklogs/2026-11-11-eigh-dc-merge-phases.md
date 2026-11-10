# Where the time goes inside the tridiagonal divide and conquer at n = 64..128

Issue: [tensor4all/faer#391](https://github.com/tensor4all/faer/issues/391), increment 1 (measurement only;
nothing in the library changes). Follows
[`2026-11-10-eigh-vs-mkl-small-sizes.md`](2026-11-10-eigh-vs-mkl-small-sizes.md), whose MKL numbers are used
here as the comparison point and were **not** re-measured.

Measured revision: fork `main` `859feef7`, built `--no-default-features --features std,rayon,threadteam`,
rustc 1.99.1. The routine measured is stage 2, the public `tridiagonal_self_adjoint_evd` with vectors, on the
tridiagonal that faer's own `tridiag_in_place` makes from a seeded random Hermitian matrix (the same stage as
table B of the previous record). The instrumented build times every phase of every recursion level of
`divide_and_conquer_recurse` with `Instant` into process-wide atomics; an unmodified build of the same
revision gives the reference call time. The instrumentation is not part of this change; the patch used for
the instrumented build is kept as
[`evidence/2026-11-11-eigh-merge-phases/instrumentation.patch`](evidence/2026-11-11-eigh-merge-phases/instrumentation.patch)
so that the phase boundaries and the allocation counters behind every table can be read. The probe binary
that installs the counting global allocator is in
[`evidence/2026-11-11-eigh-merge-phases/probe/`](evidence/2026-11-11-eigh-merge-phases/probe/); it increments
the counter from `alloc`, `alloc_zeroed` and `realloc`, so allocations on every thread are counted, and it
takes the counts from that counter around the timed window.

All numbers are recomputed from
[`evidence/2026-11-11-eigh-merge-phases/`](evidence/2026-11-11-eigh-merge-phases/) by its `aggregate.py`
(`python3 aggregate.py phases-raw.csv counts-raw.csv revision-raw.csv`). Every timing value is the median of
five processes, one cell per process, each process reporting its per-call mean over a 1 s window after 250 ms
of priming. The phase tables add one such median per phase and level, so their totals are a **sum of medians**
and can differ from the median of the per-process totals printed in the coverage table: f64 n = 128 at 8T is
1.923 ms there against 1.927 ms in the coverage row, and c64 n = 128 at 1T 1.650 against 1.651. The allocation
counts are medians of the same five processes; the count-only table (C) is a **single observation per
counter**, not five.

## Decisions

- **The 1.94x at 1T is carried by the secular-equation solve: 1.064 ms of the 1.574 ms D&C at f64 n = 128
  (67%).** The 1.94x is today's QR (1.773 ms) against MKL's `?stedc` (0.915 ms, previous record). Switching
  that size to our D&C (`recursion_threshold = 16`) only reaches 1.574 ms because the secular solve alone costs
  more than the whole gap that remains (1.574 - 0.915 = 0.659 ms). Every other phase of the D&C together is
  0.517 ms, which is already below MKL's entire `?stedc`: rotations 0.233 (Löwner product 0.146, eigenvector
  columns of the rank-one update 0.087, Householder deflation apply 0.000), leaf QR 0.152, the two GEMMs
  0.092, copy-out 0.030, deflation and permutation 0.008. c64 is the same picture (1.072 of 1.650 ms, 65%),
  because a complex input runs the same real tridiagonal solve; its extra 0.066 ms is the complex rescale
  after the solve. The share is 59-71% in every thr = 16 cell measured (n = 64, 96, 128; table A).
- **The 8T gap is carried by the same phase: 0.905 ms of the 1.199 ms critical path at f64 n = 128 (75%),
  against MKL's 0.610 ms for the whole `?stedc` at 8T.** The root loop of the top-level merge runs on one
  participant for 0.577 ms - by itself 95% of MKL's whole 8T call. The levels below do run their halves in
  parallel, but their secular solve is slower per node than at 1T (level 1: 0.262 vs 0.157 ms per node,
  1.67x; level 2: 0.066 vs 0.044, 1.50x). The critical-path estimate (table B) reproduces the measured 8T
  call within 2% in every cell (1.0% at f64 n = 128), so the decomposition holds at 8T too.
- **Why the secular solve costs that much: 12-13 full O(k) passes per root, at about 2 ns per term, in a
  serial loop over the roots.** Counted in a separate build (table C), each root costs 12.0-13.0 passes over
  the k poles (two of them 4-wide batched passes, so about 19 single-term evaluations per pole per root:
  307,456 term evaluations for the 128 roots of the top merge at n = 128). The full-bisection fallback is not
  the cause: it fires for 7-25 of the roots per level and costs at most 124 of about 1,650 passes. The time
  per term is 1.86 ns at the top merge of n = 128 and 2.04 / 2.39 ns at levels 1 / 2 (0.572 ms / 307,456
  terms, and so on). The root finder is `secular_eq_root_finder` in `svd/bidiag_svd.rs`, shared with the SVD.
- **Ruled out by number, so not the next work:**
  - the GEMM: 0.092 ms (6%) at 1T and 0.037 ms (3%) of the 8T path. Turning the update into one rank-k
    GEMM cannot move a phase this small;
  - deflation and permutation, the `dlaed2` analogue: 0.008 ms (under 1%);
  - allocation: **zero heap allocations per call** at every level and phase, in both builds, at 1T and 8T,
    for every cell. The merge takes all its workspace from the `MemStack`;
  - the leaf: 0.152 ms (10%) at 1T and 0.022 ms of the 8T path. A leaf costs the same inside the recursion as
    alone (19.1 µs per n = 16 leaf in place, 18.5 µs standalone), so the recursion adds no overhead to it.
    QR alone costs 69-77 ns per n^2 from n = 8 to 48 and rises to 108 ns per n^2 at n = 128 (table D).
- **The threshold knob still holds at 1T, and is not neutral at small n.** Unmodified, `recursion_threshold
  = 16` against the default gives 1.574 vs 1.773 ms at f64 n = 128 (-11%), 0.894 vs 0.926 at n = 96 (-3%) and
  0.382 vs 0.376 at n = 64 (+2%).
- **Next (increment 3), all aimed at the one phase the numbers name.** The issue's targets turn into budgets
  for the secular solve, given the rest of the D&C as measured: at 1T, stage 2 <= 1.26 ms needs the secular
  solve <= 0.74 ms (from 1.064, -30%); at 8T, stage 2 <= 0.70 ms needs the secular solve on the critical path
  <= 0.41 ms (from 0.905, 2.2x).
  1. Split the root loop of `compute_eigenvalues` across the team at the levels where it is serial. Each root
     is independent and writes only its own `mus[i]` / `shifts[i]`, so the results stay bit-identical for any
     thread count. It is the only lever on the top level's 0.577 ms at 8T. About 1-2 days with the gate and
     tests. The 8T budget may not be reachable by this alone, because the parallel levels already slow down
     1.5-1.7x per node.
  2. Fewer passes per root: a `dlaed4`-style rational step, which LAPACK uses for this, in place of the
     current two-batch start, geometric bisection and secant. This is the only lever on the 1T budget: -30% is
     18.8 term evaluations per pole per root (12.8 passes, two of them 4-wide; 307456/(128x128)) down to at most
     about 13.2, which is about 13 single-term passes, or 7.2 passes if the two batched passes are kept. The root
     finder is shared with
     `bidiag_svd`, so either give the EVD its own finder or carry the SVD's accuracy suite through the change.
     About 3-5 days with the accuracy work.
  3. A cheaper pass: the pass is a scalar loop with one division per term. Evaluating several roots per pass
     (the batch kernel already has that shape) is the vectorised form. About 2-3 days, and worth measuring only
     after 2, because it multiplies with the pass count.

  The 8-12% 8T regression of `main` against the record's revision (below) should be bisected before 1 is
  measured, because it sits on the same path (about half a day).

## Verification conclusions and constraints

- **The instrumentation covers the call.** At 1T the sum of all phases is within +0.0% to +0.6% of the
  unmodified call in every cell, and **+0.4% at f64 n = 128 thr = 16** (1.581 vs 1.574 ms; table E). The
  instrumented call itself is within 2% of the unmodified one in every cell but one: **+2.5% at c64 n =
  64, 8T, thr = 16** (0.327 vs 0.319 ms), at most +1.8% elsewhere (f64 n = 64, 1T: 0.389 vs 0.382). At 8T the phase sum is
  thread-time across concurrent nodes and exceeds the wall time by construction (+47-62%). The 8T statement is
  therefore made from the critical-path estimate (per-node time per level, multiplied by how many nodes of that
  level run one after another), which matches the measured call within 1.0-1.9%.
- **`main` is 8-12% slower than the record's revision at 8T under `recursion_threshold = 16`, and identical at
  1T** (table F, both unmodified, same probe, same session). This is why this record's 8T stage 2 at f64 n = 128
  is 1.202 ms where the previous record has 1.065: the record's revision measures 1.075 here. Between the two
  revisions `threadteam` moved from 0.1.2 to 0.5.1 and `utils/team.rs` / `utils/cost.rs` were rewritten. Which
  change is responsible is **not** bisected.
- **Not measured:**
  - MKL. Its `?stedc` numbers are the previous record's, and how MKL splits its own time between the secular
    solve and the rest is unknown. So "carried by" means that our secular solve alone is larger than the whole
    gap, not that MKL's secular solve is cheaper by a known amount.
  - Why the parallel levels' secular solve is 1.5-1.7x slower per node than at 1T. The 8T CPU set is 4 cores
    plus their SMT siblings, and two halves landing on siblings of one core would share its divider; that is
    a hypothesis, the placement was not recorded.
  - Deflation-heavy inputs. The input is a random Hermitian matrix, and nothing deflates: the mean k equals the
    node size at every level. A clustered spectrum shrinks k and shifts time from the secular solve towards the
    GEMM and the permutations; these shares do not describe that case.
  - The default threshold's D&C. At n <= 128 with the default, the D&C is never reached (n < 128 goes to QR,
    n = 128 falls back at once), so the default cells are 99-100% leaf QR and have no merge to decompose.
- Host: AMD Ryzen AI 9 HX 470 (`zen5-cpu` development host), SMT on, boost off, governor `performance`, kernel
  7.0.0-38-generic. 1T on CPU 0, 8T on `0-3,12-15`, runner pinned to `4-11,16-23`, no `RAYON_*`/`OMP_*`
  variables, the team built by `faer::team::with_team(8, ..)`. Before each run nothing else was using CPU (the
  only `cargo` match on the process list was an idle devcontainer whose mount names contain "cargo").
  rustc 1.99.1, where the previous record used 1.99.0. The 1T stage-2 numbers reproduce that record within
  1.5% (1.773 vs 1.772 default, 1.574 vs 1.552 thr = 16).

## Evidence

ms per call, median of five processes. `thr` is `recursion_threshold`. "Thread-time" sums every node of a
level, so at 8T a level's sum is not wall time (table B is).

Phase groups: (a) `compute_eigenvalues`, the secular root finder over all k roots; (b) building z from the
children's boundary rows, the merge permutation `pl_before`, the deflation test with its Householder
construction and compaction (`pr` / `pl_after` / `run_info`), and the eigenvalue sort; (c) the Löwner
recomputation of z, the eigenvector columns of the rank-one update and the Householder apply / un-permute;
(d) the two `matmul` calls; (d') `u.copy_from(tmp)` and the eigenvalue write-back; (e) `qr_algorithm` at a
leaf; other: the split before the join, `normalize_range` and workspace setup, and the complex rescale.

### A. phases per recursion level, `recursion_threshold = 16`

Thread-time per call. `nodes x k`: merges at that level x mean non-deflated size. The default threshold has no
merge at these sizes: its cells are one QR call (0.376 / 0.926 / 1.773 ms for f64 n = 64 / 96 / 128 at 1T, the
same at 8T); the per-level tables of every cell, the default included, are in the script's output.

| dtype | n | T | level | nodes x k | (a) secular | (b) deflation+perm | (c) rotations | (d) GEMM | (d') copy-out | (e) leaf QR | other | level sum |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 64 | 1 | 0 | 1 x 64 | 0.151 | 0.001 | 0.033 | 0.009 | 0.004 | 0.000 | 0.001 | 0.199 |
| f64 | 64 | 1 | 1 | 2 x 32 | 0.088 | 0.002 | 0.016 | 0.003 | 0.002 | 0.000 | 0.000 | 0.111 |
| f64 | 64 | 1 | 2 | 4 leaves of 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.075 | 0.000 | 0.075 |
| f64 | 64 | 1 | **all** | | **0.239** (62%) | **0.003** (1%) | **0.049** (13%) | **0.012** (3%) | **0.007** (2%) | **0.075** (19%) | **0.001** (0%) | **0.384** |
| f64 | 64 | 8 | 0 | 1 x 64 | 0.152 | 0.002 | 0.038 | 0.008 | 0.003 | 0.000 | 0.001 | 0.205 |
| f64 | 64 | 8 | 1 | 2 x 32 | 0.135 | 0.002 | 0.023 | 0.005 | 0.002 | 0.000 | 0.000 | 0.168 |
| f64 | 64 | 8 | 2 | 4 leaves of 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.087 | 0.000 | 0.087 |
| f64 | 64 | 8 | **all** | | **0.287** (62%) | **0.004** (1%) | **0.061** (13%) | **0.013** (3%) | **0.006** (1%) | **0.087** (19%) | **0.001** (0%) | **0.459** |
| f64 | 96 | 1 | 0 | 1 x 96 | 0.324 | 0.002 | 0.075 | 0.029 | 0.010 | 0.000 | 0.001 | 0.440 |
| f64 | 96 | 1 | 1 | 2 x 48 | 0.184 | 0.002 | 0.036 | 0.008 | 0.005 | 0.000 | 0.000 | 0.235 |
| f64 | 96 | 1 | 2 | 4 x 24 | 0.113 | 0.003 | 0.018 | 0.003 | 0.001 | 0.000 | 0.000 | 0.139 |
| f64 | 96 | 1 | 3 | 8 leaves of 12 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.084 | 0.000 | 0.084 |
| f64 | 96 | 1 | **all** | | **0.621** (69%) | **0.006** (1%) | **0.130** (14%) | **0.040** (5%) | **0.016** (2%) | **0.084** (9%) | **0.001** (0%) | **0.899** |
| f64 | 96 | 8 | 0 | 1 x 96 | 0.327 | 0.002 | 0.084 | 0.012 | 0.009 | 0.000 | 0.001 | 0.437 |
| f64 | 96 | 8 | 1 | 2 x 48 | 0.294 | 0.003 | 0.049 | 0.007 | 0.005 | 0.000 | 0.000 | 0.360 |
| f64 | 96 | 8 | 2 | 4 x 24 | 0.164 | 0.004 | 0.027 | 0.006 | 0.002 | 0.000 | 0.001 | 0.204 |
| f64 | 96 | 8 | 3 | 8 leaves of 12 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.115 | 0.000 | 0.115 |
| f64 | 96 | 8 | **all** | | **0.785** (70%) | **0.010** (1%) | **0.161** (14%) | **0.025** (2%) | **0.017** (1%) | **0.115** (10%) | **0.002** (0%) | **1.116** |
| f64 | 128 | 1 | 0 | 1 x 128 | 0.572 | 0.002 | 0.136 | 0.068 | 0.017 | 0.000 | 0.001 | 0.796 |
| f64 | 128 | 1 | 1 | 2 x 64 | 0.315 | 0.003 | 0.065 | 0.018 | 0.009 | 0.000 | 0.000 | 0.410 |
| f64 | 128 | 1 | 2 | 4 x 32 | 0.177 | 0.003 | 0.032 | 0.006 | 0.004 | 0.000 | 0.000 | 0.223 |
| f64 | 128 | 1 | 3 | 8 leaves of 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.152 | 0.000 | 0.152 |
| f64 | 128 | 1 | **all** | | **1.064** (67%) | **0.008** (1%) | **0.233** (15%) | **0.092** (6%) | **0.030** (2%) | **0.152** (10%) | **0.001** (0%) | **1.581** |
| f64 | 128 | 8 | 0 | 1 x 128 | 0.577 | 0.003 | 0.149 | 0.027 | 0.017 | 0.000 | 0.001 | 0.774 |
| f64 | 128 | 8 | 1 | 2 x 64 | 0.524 | 0.004 | 0.085 | 0.015 | 0.010 | 0.000 | 0.000 | 0.637 |
| f64 | 128 | 8 | 2 | 4 x 32 | 0.265 | 0.005 | 0.047 | 0.010 | 0.005 | 0.000 | 0.001 | 0.332 |
| f64 | 128 | 8 | 3 | 8 leaves of 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.179 | 0.000 | 0.179 |
| f64 | 128 | 8 | **all** | | **1.366** (71%) | **0.013** (1%) | **0.280** (15%) | **0.052** (3%) | **0.031** (2%) | **0.179** (9%) | **0.002** (0%) | **1.923** |
| c64 | 64 | 1 | 0 | 1 x 64 | 0.152 | 0.001 | 0.033 | 0.009 | 0.004 | 0.000 | 0.018 | 0.218 |
| c64 | 64 | 1 | 1 | 2 x 32 | 0.087 | 0.002 | 0.016 | 0.003 | 0.002 | 0.000 | 0.000 | 0.110 |
| c64 | 64 | 1 | 2 | 4 leaves of 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.075 | 0.000 | 0.075 |
| c64 | 64 | 1 | **all** | | **0.239** (59%) | **0.003** (1%) | **0.049** (12%) | **0.012** (3%) | **0.007** (2%) | **0.075** (19%) | **0.019** (5%) | **0.402** |
| c64 | 64 | 8 | 0 | 1 x 64 | 0.154 | 0.002 | 0.038 | 0.009 | 0.003 | 0.000 | 0.011 | 0.217 |
| c64 | 64 | 8 | 1 | 2 x 32 | 0.131 | 0.002 | 0.023 | 0.004 | 0.002 | 0.000 | 0.000 | 0.164 |
| c64 | 64 | 8 | 2 | 4 leaves of 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.087 | 0.000 | 0.087 |
| c64 | 64 | 8 | **all** | | **0.285** (61%) | **0.004** (1%) | **0.061** (13%) | **0.013** (3%) | **0.006** (1%) | **0.087** (19%) | **0.012** (3%) | **0.468** |
| c64 | 96 | 1 | 0 | 1 x 96 | 0.317 | 0.002 | 0.075 | 0.029 | 0.010 | 0.000 | 0.038 | 0.471 |
| c64 | 96 | 1 | 1 | 2 x 48 | 0.184 | 0.002 | 0.036 | 0.008 | 0.005 | 0.000 | 0.000 | 0.235 |
| c64 | 96 | 1 | 2 | 4 x 24 | 0.107 | 0.003 | 0.018 | 0.003 | 0.001 | 0.000 | 0.000 | 0.133 |
| c64 | 96 | 1 | 3 | 8 leaves of 12 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.085 | 0.000 | 0.085 |
| c64 | 96 | 1 | **all** | | **0.608** (66%) | **0.006** (1%) | **0.130** (14%) | **0.040** (4%) | **0.016** (2%) | **0.085** (9%) | **0.039** (4%) | **0.924** |
| c64 | 96 | 8 | 0 | 1 x 96 | 0.320 | 0.002 | 0.085 | 0.014 | 0.009 | 0.000 | 0.038 | 0.468 |
| c64 | 96 | 8 | 1 | 2 x 48 | 0.296 | 0.003 | 0.049 | 0.009 | 0.006 | 0.000 | 0.000 | 0.363 |
| c64 | 96 | 8 | 2 | 4 x 24 | 0.155 | 0.004 | 0.027 | 0.006 | 0.002 | 0.000 | 0.001 | 0.196 |
| c64 | 96 | 8 | 3 | 8 leaves of 12 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.116 | 0.000 | 0.116 |
| c64 | 96 | 8 | **all** | | **0.771** (68%) | **0.010** (1%) | **0.161** (14%) | **0.028** (2%) | **0.017** (1%) | **0.116** (10%) | **0.039** (3%) | **1.142** |
| c64 | 128 | 1 | 0 | 1 x 128 | 0.577 | 0.002 | 0.136 | 0.068 | 0.017 | 0.000 | 0.066 | 0.865 |
| c64 | 128 | 1 | 1 | 2 x 64 | 0.316 | 0.003 | 0.066 | 0.018 | 0.009 | 0.000 | 0.000 | 0.411 |
| c64 | 128 | 1 | 2 | 4 x 32 | 0.179 | 0.003 | 0.032 | 0.006 | 0.004 | 0.000 | 0.000 | 0.226 |
| c64 | 128 | 1 | 3 | 8 leaves of 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.148 | 0.000 | 0.148 |
| c64 | 128 | 1 | **all** | | **1.072** (65%) | **0.008** (0%) | **0.234** (14%) | **0.092** (6%) | **0.030** (2%) | **0.148** (9%) | **0.066** (4%) | **1.650** |
| c64 | 128 | 8 | 0 | 1 x 128 | 0.584 | 0.003 | 0.150 | 0.026 | 0.017 | 0.000 | 0.066 | 0.845 |
| c64 | 128 | 8 | 1 | 2 x 64 | 0.526 | 0.004 | 0.085 | 0.012 | 0.009 | 0.000 | 0.000 | 0.637 |
| c64 | 128 | 8 | 2 | 4 x 32 | 0.271 | 0.005 | 0.047 | 0.010 | 0.005 | 0.000 | 0.001 | 0.339 |
| c64 | 128 | 8 | 3 | 8 leaves of 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.174 | 0.000 | 0.174 |
| c64 | 128 | 8 | **all** | | **1.381** (69%) | **0.013** (1%) | **0.282** (14%) | **0.048** (2%) | **0.031** (2%) | **0.174** (9%) | **0.067** (3%) | **1.995** |

### B. 8T critical path, `recursion_threshold = 16`

For each level, the per-node time multiplied by how many nodes of that level run one after another (1 here:
every split at 8T ran its halves in parallel), summed over the levels.

| dtype | n | thr | (a) | (b) | (c) | (d) | (d') | (e) | other | estimate | measured call |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 64 | 16 | 0.220 | 0.003 | 0.050 | 0.011 | 0.004 | 0.022 | 0.001 | 0.310 | 0.316 |
| f64 | 96 | 16 | 0.515 | 0.005 | 0.116 | 0.017 | 0.012 | 0.014 | 0.001 | 0.682 | 0.692 |
| f64 | 128 | 16 | 0.905 | 0.007 | 0.203 | 0.037 | 0.023 | 0.022 | 0.001 | 1.199 | 1.211 |
| c64 | 64 | 16 | 0.219 | 0.003 | 0.050 | 0.011 | 0.005 | 0.022 | 0.012 | 0.321 | 0.327 |
| c64 | 96 | 16 | 0.507 | 0.005 | 0.116 | 0.019 | 0.013 | 0.014 | 0.038 | 0.713 | 0.721 |
| c64 | 128 | 16 | 0.914 | 0.007 | 0.204 | 0.034 | 0.023 | 0.022 | 0.066 | 1.270 | 1.284 |

### C. secular-equation work per merge level, `recursion_threshold = 16`, count-only build

A pass is one sweep over the k poles; a 4-wide batched pass counts as one pass and as 4k terms. Counts do not
depend on the thread count.

| dtype | n | thr | level | roots | O(k) passes | passes/root | bisection roots | bisection passes | terms (k x passes) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 64 | 16 | 0 | 64 | 781 | 12.2 | 10 | 13 | 74560 |
| f64 | 64 | 16 | 1 | 64 | 786 | 12.3 | 7 | 26 | 37440 |
| f64 | 96 | 16 | 0 | 96 | 1189 | 12.4 | 16 | 45 | 169440 |
| f64 | 96 | 16 | 1 | 96 | 1199 | 12.5 | 18 | 49 | 85200 |
| f64 | 96 | 16 | 2 | 96 | 1244 | 13.0 | 15 | 90 | 43680 |
| f64 | 128 | 16 | 0 | 128 | 1634 | 12.8 | 25 | 108 | 307456 |
| f64 | 128 | 16 | 1 | 128 | 1640 | 12.8 | 18 | 121 | 154112 |
| f64 | 128 | 16 | 2 | 128 | 1547 | 12.1 | 15 | 17 | 74080 |
| c64 | 64 | 16 | 0 | 64 | 789 | 12.3 | 9 | 27 | 75072 |
| c64 | 64 | 16 | 1 | 64 | 766 | 12.0 | 7 | 6 | 36800 |
| c64 | 96 | 16 | 0 | 96 | 1156 | 12.0 | 10 | 7 | 166272 |
| c64 | 96 | 16 | 1 | 96 | 1206 | 12.6 | 14 | 58 | 85536 |
| c64 | 96 | 16 | 2 | 96 | 1159 | 12.1 | 11 | 4 | 41640 |
| c64 | 128 | 16 | 0 | 128 | 1650 | 12.9 | 18 | 118 | 309504 |
| c64 | 128 | 16 | 1 | 128 | 1652 | 12.9 | 20 | 124 | 154880 |
| c64 | 128 | 16 | 2 | 128 | 1580 | 12.3 | 20 | 39 | 75136 |

### D. QR alone (`recursion_threshold = usize::MAX`), f64, 1T

| n | pristine call | instr leaf phase | per n^2 (ns) | per n^3 (ns) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0049 | 0.0049 | 76.3 | 9.53 |
| 12 | 0.0103 | 0.0103 | 71.7 | 5.97 |
| 16 | 0.0185 | 0.0185 | 72.4 | 4.52 |
| 24 | 0.0398 | 0.0399 | 69.1 | 2.88 |
| 32 | 0.0770 | 0.0773 | 75.2 | 2.35 |
| 48 | 0.1786 | 0.1799 | 77.5 | 1.62 |
| 64 | 0.3760 | 0.3784 | 91.8 | 1.43 |
| 96 | 0.9257 | 0.9301 | 100.4 | 1.05 |
| 128 | 1.7728 | 1.7787 | 108.2 | 0.85 |

### E. coverage: all phases of the instrumented build against the unmodified call

| dtype | n | T | thr | phase sum (instr) | call (instr) | call (pristine) | sum/pristine - 1 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| f64 | 64 | 1 | 16 | 0.384 | 0.389 | 0.382 | +0.6% |
| f64 | 64 | 1 | default | 0.378 | 0.379 | 0.376 | +0.6% |
| f64 | 64 | 8 | 16 | 0.460 | 0.316 | 0.311 | +48.1% |
| f64 | 64 | 8 | default | 0.378 | 0.379 | 0.376 | +0.6% |
| f64 | 96 | 1 | 16 | 0.898 | 0.908 | 0.894 | +0.4% |
| f64 | 96 | 1 | default | 0.931 | 0.931 | 0.926 | +0.5% |
| f64 | 96 | 8 | 16 | 1.116 | 0.692 | 0.689 | +62.0% |
| f64 | 96 | 8 | default | 0.931 | 0.931 | 0.925 | +0.6% |
| f64 | 128 | 1 | 16 | 1.581 | 1.590 | 1.574 | +0.4% |
| f64 | 128 | 1 | default | 1.778 | 1.779 | 1.773 | +0.3% |
| f64 | 128 | 8 | 16 | 1.927 | 1.211 | 1.202 | +60.3% |
| f64 | 128 | 8 | default | 1.781 | 1.782 | 1.774 | +0.4% |
| c64 | 64 | 1 | 16 | 0.402 | 0.407 | 0.400 | +0.5% |
| c64 | 64 | 1 | default | 0.374 | 0.374 | 0.374 | +0.0% |
| c64 | 64 | 8 | 16 | 0.469 | 0.327 | 0.319 | +47.1% |
| c64 | 64 | 8 | default | 0.374 | 0.374 | 0.373 | +0.1% |
| c64 | 96 | 1 | 16 | 0.924 | 0.933 | 0.920 | +0.4% |
| c64 | 96 | 1 | default | 0.929 | 0.929 | 0.929 | -0.0% |
| c64 | 96 | 8 | 16 | 1.142 | 0.721 | 0.715 | +59.6% |
| c64 | 96 | 8 | default | 0.931 | 0.932 | 0.928 | +0.3% |
| c64 | 128 | 1 | 16 | 1.651 | 1.661 | 1.648 | +0.2% |
| c64 | 128 | 1 | default | 1.799 | 1.800 | 1.797 | +0.1% |
| c64 | 128 | 8 | 16 | 1.995 | 1.284 | 1.275 | +56.5% |
| c64 | 128 | 8 | default | 1.800 | 1.801 | 1.798 | +0.1% |

### F. record revision against `main`, both unmodified, `recursion_threshold = 16`

| dtype | n | T | 1ac38c6d | 859feef7 | main / record rev |
| --- | ---: | ---: | ---: | ---: | ---: |
| f64 | 64 | 1 | 0.382 | 0.382 | 1.002 |
| f64 | 64 | 8 | 0.285 | 0.310 | 1.088 |
| f64 | 96 | 1 | 0.894 | 0.895 | 1.001 |
| f64 | 96 | 8 | 0.629 | 0.690 | 1.097 |
| f64 | 128 | 1 | 1.576 | 1.578 | 1.001 |
| f64 | 128 | 8 | 1.075 | 1.199 | 1.116 |
| c64 | 64 | 1 | 0.400 | 0.401 | 1.001 |
| c64 | 64 | 8 | 0.295 | 0.318 | 1.080 |
| c64 | 96 | 1 | 0.920 | 0.920 | 1.000 |
| c64 | 96 | 8 | 0.650 | 0.710 | 1.093 |
| c64 | 128 | 1 | 1.645 | 1.646 | 1.001 |
| c64 | 128 | 8 | 1.167 | 1.275 | 1.093 |
