# The column-slab branch on current main (#71)

## Decisions

- **Adopt.** The column-slab branch (`perf/tall-skinny-column-slab`, original head `b4a4825`, based on
  `50c2b98`) was rebased onto current main and re-measured on the cell #71 names. At 8T, order c,
  it is 2.70x faster than main for f64 (`0.0674` against `0.1819` ms) and 1.27x for c64, and every
  other order/dtype at 4T and 8T is 1.26-2.00x faster. On this host that is 2.4x faster than MKL in
  the same cell, where main was 1.13x slower than MKL. The rebased branch is
  `perf/tall-skinny-column-slab-rebased` (head `ffc4c89f`), offered as a draft PR off this one.
- The issue's predeclared target (3x in both orders and both dtypes) is **still not met**: 2.70x
  (c) and 2.00x (f) for f64, 1.27x and 1.54x for c64. A slab of a row-major operand still misses the
  kernel's fused path, which is why the f-order gain is smaller than the c-order one at 8T and why
  c64 gains least.
- 0.25.0 did not move this cell: `main` (`5c2f4b70`) and `0e15729` agree within 2% in all eight
  cells. Order c also still does not scale from 4T to 8T at 0.25.0 here (0.1848 -> 0.1856), which is
  the shape of the issue's own table; only the absolute times differ, because the issue's numbers
  come from a different host (`EVO-X1Pro`, Ryzen AI 9 HX 470: 0.429 -> 0.430). So the branch is the
  only change that moves this cell, and the issue's premise that 0.25.0 had already taken most of the
  gain is a statement about a different machine.
- The rebase conflicts and how they were resolved:
  - `faer/src/linalg/matmul/mod.rs`, the thread count handed to `private_gemm_x86::gemm`: the branch
    passed its `budget` there; main passes `1`, because `Par::Team` (`gemm_team`) is now the only
    parallel route and the kernel must never thread. **Main's `1` kept**; the branch's floor now only
    decides the slab width, and its comment says so.
  - `faer/src/linalg/matmul/mod.rs`, `mod tests` imports: main's superset kept (`fx128`,
    `MaybeUninit`).
  - `faer/src/linalg/matmul/mod.rs`, the split region: the branch used `spindle::for_each`, which
    main no longer has; it is `crate::utils::thread::for_each(par, n_jobs, iter, f)` now, so the
    slabs run on the team instead of on an outside runtime.
  - `faer/src/linalg/matmul/mod.rs`, the branch's own test: `Par::rayon(8)` and a raw Rayon pool are
    gone from main, so all four call sites (including the f32 case and the c32 block that the last
    branch commit added) run through `crate::team::with_team(8, ..)`.
  - `PATCHSET.md`, the ledger row: main had filled the insertion point with newer rows; both kept.
    The row now carries the #71 re-measurement and drops its claim that
    `qr_factor f64 c 128^2` 8T is still a regression, which the branch head had already shown to be
    inside the spread.
  - `faer/src/utils/team.rs`, `the_team_floor_is_per_dtype`: c64 `64^3` on a team of six reaches the
    slab split before the team route, so `engaged` also counts `st.split`. The per-dtype floor
    property (c64 `64^3` is not left sequential) is what the test states, and it is still asserted;
    the same test's f64 `192^3` case (a destination above the 128-row cap, so no slab) still
    exercises `gemm_tiled + gemm_team_kernel` directly.
- Not chosen: replacing the branch's work floor with main's `utils::cost` rule
  (`par_width_time_upto`). It is the more consistent structure, and it was implemented and tried: a
  region-shaped cost gate admits 80-128-wide square trailing updates, so LLT's `640^2`
  factorization stopped reaching the team kernel and
  `utils::team::tests::team_kernel_triangular_indexed_and_factorizations_match_sequential` failed on
  the route it asserts. Which of the two the fork should carry is a decision that needs its own
  measurement of those trailing updates; it is left as a review finding rather than folded into a
  rebase.

## Verification conclusions and constraints

Cell: `gemm` f64/c64, m = 16, n = 240, k = 4096 -- the issue's `16x240x4096`, which `fbench` spells
`--n 16x4096x240`. Values are the median of three process-set medians, ms, three complete process
sets per condition, per condition 3000 ms priming and 30 calls per row, rows rotated by set.
Host: EVO-X2, AMD Ryzen AI MAX+ 395, 32 logical CPUs, two L3 domains (0-7 with SMT siblings 16-23,
8-15 with 24-31), measured set 0-3 (4T) and 0-7 (8T), observer pinned to CPU 27, 4-7 and 16-31 left
free for a parallel lane. **Boost was on** (`cpufreq_boost=1`): this host has no passwordless sudo,
so the protocol's boost-off requirement could not be met. Every condition was measured under the
same boost-on state, so the ratios stand, but these absolute numbers are not comparable with the
published boost-off pages.

| cell | order | 4T slab | 4T main | 4T 0.25.0 | 8T slab | 8T main | 8T 0.25.0 | 8T slab/main |
|---|---|---|---|---|---|---|---|---|
| f64 | c | 0.1333 | 0.1827 | 0.1848 | **0.0674** | **0.1819** | 0.1856 | **0.37** |
| f64 | f | 0.0950 | 0.1337 | 0.1362 | **0.0495** | **0.0992** | 0.1058 | **0.50** |
| c64 | c | 0.3897 | 0.5541 | 0.5558 | 0.2199 | 0.2804 | 0.2858 | 0.78 |
| c64 | f | 0.3435 | 0.4340 | 0.4361 | 0.1785 | 0.2755 | 0.2831 | 0.65 |

MKL, the same binaries and runs (ms): f64 4T c 0.2519 / f 0.4007, 8T c 0.1606 / f 0.2223; c64 4T
c 1.2302 / f 12.23, 8T c 0.5404 / f 1.8856. The MKL row moved by at most 0.6% across the three
conditions, so the machine and the harness were the same in every arm.

Two collateral cells of families the change can reach, f64 8T (slab / main / 0.25.0, ms), both
orders:

| cell | order | slab | main | 0.25.0 |
|---|---|---|---|---|
| `gemm` 4096x16x240 (the write-back leg, destination above the 128-row cap) | c | 0.1379 | 0.1381 | 0.1388 |
| | f | 0.1665 | 0.1480 | 0.1662 |
| `gemm` 64x64x64 (inside the cap, below the flop floor at 64^3 f64) | c | 0.0039 | 0.0039 | 0.0047 |
| | f | 0.0039 | 0.0039 | 0.0049 |

The write-back leg's f-order column is bimodal on this host in every condition
(0.147-0.168 ms, one set high and two low in a different order per condition), so it shows no
effect of the change; the 64x64x64 cell is identical to main's.

The three process-set medians of one condition are within 1.8% of each other (largest: c64 8T
order c on the slab, 0.2165/0.2199/0.2204), so the branch/main gap is far outside this machine's
spread. Reproducing the frozen campaign policy instead (250 ms priming, 5 calls) changes the 8T
f64 readings by under 1% (slab c 0.0670 / f 0.0497, main c 0.1813 / f 0.0978), so the priming
policy is not what produces the gap.

`cargo test -p t4a-faer --lib` at the rebased head: 751 passed, 0 failed, 162 ignored; on main: 749
passed, 0 failed, 162 ignored (the two extra are the branch's). `fbench verify` reports `ok` for
every row of the cell at 4T and 8T for f64 and c64.

Reproduction recipe, `faer-benchmark` at `bde6b46` with `fbench` built for each fork revision
(`scripts/build_for_faer_rev.sh --vendor mkl`) and `extern` upstream at `faer-v0.24.4`:

```
benchmarks/scripts/pinned.sh 0-7 -- <fbench> run --threads 8 --n 16x4096x240 --family gemm \
  --dtype f64 --batch 1 --regime gemm --csv <out>.csv --statistic median --prime-ms 3000 --reps 30 \
  --rotate-rows 0
```

What remains unverified, and what a merge would still need:

- Only two collateral cells were measured, not the declared guard set. The change reaches every
  `matmul` with a destination of at most 128 rows, so LLT/LU/QR trailing updates and the
  `householder_apply` legs are in its blast radius; `docs/merge-procedure.md` asks for the declared
  guard grid at 4T and 8T, and a candidate touching the split of work across threads also takes the
  1024 tier. None of that was run.
- The OpenBLAS arm could not be built on this host (no `gfortran`, no oneAPI `ifx`), so the yardstick
  is MKL only, contrary to the rule that a vendor comparison uses OpenBLAS.
- The gate keeps the branch's own work floor beside main's `utils::cost` rule (see "Not chosen"
  above): the fork carries two rules for one question until that is settled and measured.
- Order c is still 1.36x slower than order f on the slab at 8T (0.0674 against 0.0495), the deficit
  the original ledger row named: a slab whose operands are row-major misses the fused kernel.
- Nothing here says the branch is correct beyond the two added tests, the crate's own suite and
  `fbench verify` on this cell.
