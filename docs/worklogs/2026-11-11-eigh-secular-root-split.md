# The secular-equation roots of a D&C merge, split across the team

Issue: [tensor4all/faer#391](https://github.com/tensor4all/faer/issues/391), increment 3, step 1. Follows
[`2026-11-11-eigh-dc-merge-phases.md`](2026-11-11-eigh-dc-merge-phases.md) (PR #393), which named the
secular-equation solve as the phase that carries the 8T gap: 0.905 ms of a 1.199 ms critical path at f64
n = 128 on the Zen 5 host, 0.577 ms of it in the serial root loop of the top merge.

Measured revisions: `main` `03329d59` against this branch's `51347b7b` (the library change; the commit that
adds this record changes no source). Built `--no-default-features --features std,rayon,threadteam`, rustc
1.99.0, release with `debug = 1`.

## Decisions

- **The change.** `compute_eigenvalues` takes the merge's `Par`. The root loop is cut into contiguous blocks
  of roots, one block per participant, through `utils::thread::{with_width, for_each}`. The width is
  `cost::par_width_upto` of `k^2` units at a new rate `secular_roots`, one section, at most `k`. With
  `Par::Seq`, a one-thread team, or a width that does not pay, the serial loop runs as before. The serial
  root code is one function, `secular_root`, and both paths call it.
- **The rate.** `secular_roots` is in ns per `k^2` for all `k` roots of one merge. The Zen 5 value, 35.2, is
  the median of the six top-level 1T cells of the previous record's table A (34.4 to 37.1). The M5 value,
  11.2, is **derived, not measured**: the Zen 5 value scaled by the two rows' `tridiag_evd_dc` ratio
  (31.0 / 97.4). `generic`, which this host (`x86_64-avx2`) uses, is the per-entry minimum, 11.2. On this
  host the gate opened at every merge of every measured cell. Even the smallest, `k` = 24 on a two-thread
  sub-team (f64 n = 96, level 2), ran its secular phase in 0.54x the serial time.
- **Determinism: what is guaranteed, and which option was chosen.** Each root is computed by
  `secular_root` alone, from read-only `d`, `z` and `rho`, and writes only its own `(shift, mu)`. No
  arithmetic is shared between roots, so a root's bits do not depend on which participant computes it or
  on where the blocks are cut. The roots are therefore bitwise identical for every split, and so for every
  thread count, `Par::Seq` included.

  The width follows the cost model, so it does depend on `par.degree()` and on the host's cost table. It
  was **not** pinned to a size-derived count, because pinning would buy nothing here: the result is already
  independent of the split. The split itself is a function of `(k, degree, host class)` only, never of
  scheduling.

  The tests pin both halves of the claim:
  - `secular_split_tests::split_roots_are_bitwise_identical_at_every_width`: `k` = 128, 96, 64 and 32, each
    with a deflated tail, on teams of 2, 3 and 8. The same count run twice gives the same bits, and every
    team count gives the bits of `Par::Seq`.
  - `divide_and_conquer_on_a_team_is_reproducible`: the whole D&C on a team of 8, twice, same bits; against
    `Par::Seq` it is compared by its residual only.

  The across-thread-count claim is scoped to the root loop. `docs/reproducibility.md` asks for no
  across-count bitwise test of a public routine, and none is added: the two `matmul`s of the merge are free
  to differ. On this host's inputs the whole call happened to be bitwise identical at 1T and 8T, on both
  `main` and the branch (table B). That is an observation, not a promise.
- **Result on this host, f64 n = 128, `recursion_threshold = 16`:**
  - The 8T call goes from 1.329 to 0.589 ms (0.443x).
  - The 8T critical path estimate goes from 1.326 to 0.591 ms. The difference is in the secular phase:
    0.916 to 0.181 ms, of which the top merge is 0.678 to 0.101 ms, a 6.7x speedup on 8 threads.
  - Every other phase group moves by at most 0.001 ms.
  - The twelve cells are 0.44x to 0.49x of `main` at 8T (table A).
- **1T is not strictly unchanged.** The serial path does the same arithmetic as before, and output bits
  match `main`. In the pristine builds, though, the branch's 1T call is +0.2% to +0.6% in both runs, with
  min-max ranges mostly not overlapping. In the instrumented builds the same cells are -0.7% to +0.1%, and
  the secular phase per node is 0.986 to 1.002 of `main` at every level (table C). The pristine difference
  lands in no phase and has the opposite sign under instrumentation. That is consistent with code layout,
  but not shown to be: it is reported as a measured +0.2% to +0.6%, not as "unchanged".

## Verification conclusions and constraints

- The crate's lib tests (release): 796 passed, 0 failed, 162 ignored (the 4 new tests included).
- The opt-in LAPACK groups for `tridiagonal_evd` and `self_adjoint_evd`:
  - every `lapack_tridiagonal_evd_*` group passes for f32/f64/c32/c64, including `_cases` (thresholds 8
    and 128, `Par::Seq`) and `_parallel` (n = 257 on teams of 2 and 4, which takes the split);
  - every `lapack_self_adjoint_evd_*` group passes except `lapack_self_adjoint_evd_cases`.
- **`lapack_self_adjoint_evd_cases` fails on `main` too.** It fails for all four scalar types on unmodified
  `main` `03329d59`, with the same 228 diagnostic lines as on this branch (identical by `diff`). They are
  all `kind=7`, with vectors, mostly at threshold 8 (n = 9, 10, 16, 127, 128, 129), with ratios around
  1e14 against 60. It runs at `Par::Seq`, where this change takes the old serial loop. It is outside this
  change and is not tracked by an open issue at the time of writing.
- Accuracy (table B): the output bits are identical to `main` in every cell, at 1T and at 8T. The residual
  `max|T U - U Λ| / (n ε max|T|)` and the orthogonality `max|UᴴU - I| / (n ε)` are therefore identical
  too: 0.040 to 0.088 and 0.063 to 0.156. Neither record had accuracy cells at these sizes, so `main` is
  measured beside the branch here rather than quoted. There are zero heap allocations per call in every
  process.
- **Measurement conditions, with a deviation from the measurement rules.** Another agent session on this
  host kept starting `cargo test` builds during both runs. The rule's remedy, confining every other process
  to CPUs off the measured set, was not available to this session. Instead:
  - each run started after two minutes with no cargo or rustc process;
  - the runner checked CPUs 0-15 (the measured CCXs and their two neighbours) idle (< 3% busy over 300 ms)
    before every process;
  - run 2 also checked after every process and drops a process that saw >= 3% busy afterwards: 7 of 120
    pristine and 22 of 120 instrumented processes, the cell keeping its remaining processes (at least 2).
  
  Run 1 had no after-check and is kept complete. The two runs agree within 1% in every cell of every
  table, so the numbers are not carried by the interference. They are still not a quiet-host acceptance
  run.
- **Not measured:**
  - MKL. The record's MKL numbers are from the Zen 5 host and are not comparable with this host's times,
    so no parity claim is made.
  - The Zen 5 host. The previous record's cells were not re-measured there.
  - More than 8 threads. Cost of the split as a function of `p`: one section per merge, each participant
    reads all of `d` and `z` (`2k` scalars, at most 2 KiB at `k` = 128), with no combine, so the traffic is
    `p·2k` reads and nothing serial. A 64-thread run on this host was not made.
  - Deflation-heavy spectra (small `k`), where the gate would stay closed for small merges.

Host: `broccoli`, AMD EPYC 7702P (Zen 2, `x86_64-avx2` class), 64 cores, SMT off, boost off, governor
`performance`, kernel 5.4.0-216. 1T on CPU 0, 8T on CPUs 0-7 (two 4-core CCXs), runner on 16-63, no
`RAYON_*`, `OMP_*` or `MKL_*` variable, team from `faer::team::with_team(8, ..)`.

## Evidence

The summaries of both runs, which hold every table below and every cell left out of it, are in
[`evidence/2026-11-11-eigh-secular-root-split/`](evidence/2026-11-11-eigh-secular-root-split/) as
`run1-aggregate.md` and `run2-aggregate.md`. The rest of the evidence lands in the same directory in a
separate pull request that follows this one, so that each stays reviewable:
- the pristine raw CSVs (`pristine.csv`, `pristine2.csv`);
- `run.py` (the runner, with the gates) and `aggregate391.py`, which recomputes every table from the raw CSVs;
- the probe;
- `instrumentation.patch`, the previous record's patch carried onto `51347b7b`, with the count-only hunk in
  `compute_eigenvalues` dropped and the `SECULAR` lap after the call, as before. It is not part of the
  library.

Timing values: ms per call, median over the kept processes (five per cell before drops), each process
reporting its per-call mean over 1 s after 250 ms of priming. The tables below are run 2.

### A. call time, pristine builds

| dtype | n | T | main | branch | branch / main |
| --- | ---: | ---: | ---: | ---: | ---: |
| f64 | 64 | 1 | 0.449 | 0.452 | 1.005 |
| f64 | 64 | 8 | 0.359 | 0.172 | 0.481 |
| f64 | 96 | 1 | 1.049 | 1.054 | 1.004 |
| f64 | 96 | 8 | 0.782 | 0.357 | 0.456 |
| f64 | 128 | 1 | 1.840 | 1.844 | 1.002 |
| f64 | 128 | 8 | 1.329 | 0.589 | 0.443 |
| c64 | 64 | 1 | 0.457 | 0.458 | 1.003 |
| c64 | 64 | 8 | 0.365 | 0.178 | 0.489 |
| c64 | 96 | 1 | 1.046 | 1.051 | 1.005 |
| c64 | 96 | 8 | 0.788 | 0.362 | 0.459 |
| c64 | 128 | 1 | 1.864 | 1.868 | 1.002 |
| c64 | 128 | 8 | 1.346 | 0.610 | 0.453 |

### B. accuracy and output bits (identical in every process of a cell, both runs)

| dtype | n | residual / (n ε max\|T\|) | orthogonality / (n ε) | digest (main = branch, 1T = 8T) |
| --- | ---: | ---: | ---: | --- |
| f64 | 64 | 0.0878 | 0.1562 | `0917282574b49e5f` |
| f64 | 96 | 0.0696 | 0.0833 | `c60a93e406d8f864` |
| f64 | 128 | 0.0413 | 0.0820 | `48c092d996371b8d` |
| c64 | 64 | 0.0842 | 0.1250 | `29d16988490003b0` |
| c64 | 96 | 0.0462 | 0.0729 | `aa46d7d2ad79b4ae` |
| c64 | 128 | 0.0402 | 0.0625 | `a78223421e34139b` |

### C. critical path by phase group, instrumented builds

n = 128 only; every cell, and the per-level secular time per node (top merge 0.13x-0.16x of `main` at 8T, level 1 0.27x-0.29x, level 2 0.51x-0.54x), is in the aggregates. The phase groups are those of the previous record. At 1T the estimate is the phase sum.

| dtype | n | T | | (a) secular | (b) | (c) | (d) | (d') | (e) | other | estimate | instr call | top merge (a) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 128 | 1 | main | 1.257 | 0.012 | 0.291 | 0.117 | 0.018 | 0.175 | 0.002 | 1.872 | 1.882 | 0.679 |
| f64 | 128 | 1 | branch | 1.245 | 0.012 | 0.291 | 0.117 | 0.018 | 0.178 | 0.002 | 1.861 | 1.870 | 0.674 |
| f64 | 128 | 8 | main | 0.916 | 0.012 | 0.276 | 0.049 | 0.048 | 0.023 | 0.002 | 1.326 | 1.341 | 0.678 |
| f64 | 128 | 8 | branch | 0.181 | 0.012 | 0.276 | 0.048 | 0.048 | 0.023 | 0.003 | 0.591 | 0.610 | 0.101 |
| c64 | 128 | 1 | main | 1.266 | 0.012 | 0.292 | 0.117 | 0.018 | 0.169 | 0.020 | 1.894 | 1.903 | 0.683 |
| c64 | 128 | 1 | branch | 1.263 | 0.012 | 0.292 | 0.118 | 0.018 | 0.171 | 0.020 | 1.894 | 1.904 | 0.684 |
| c64 | 128 | 8 | main | 0.922 | 0.013 | 0.274 | 0.049 | 0.051 | 0.022 | 0.021 | 1.352 | 1.366 | 0.683 |
| c64 | 128 | 8 | branch | 0.187 | 0.014 | 0.275 | 0.048 | 0.051 | 0.023 | 0.021 | 0.618 | 0.633 | 0.106 |

The raw CSVs of the instrumented builds (`instr.csv`, run 1; `instr2.csv`, run 2; 28,048 lines) are not
committed, so the record stays reviewable. They are kept on the measurement host, `broccoli`, under
`~/work/eigh391-secular-root-split/`, with their SHA-256 sums, and can be regenerated from the patches:
`regenerate.sh` there applies the previous record's patch to `main` `03329d59` and this record's
`instrumentation.patch` to `51347b7b`, builds the probe against each, and reruns the instrumented grid with
`run.py`. A regenerated file is a new measurement, so it reproduces the method, not the bytes.
