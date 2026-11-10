# #186: does the M5 Max gap table still hold, and on what conditions

No upstream revision was measured; the comparison is against `Accelerate` on the same host and
against this fork's own revision `325e482`, which is the revision tensor4all/faer#186 recorded.

## Decisions

- **Measure two revisions, not one.** #186 measured `325e482` (`t4a-faer` 0.28.0, 2026-11-01 13:19
  +0900). `origin/main` here is `5c2f4b70`, 114 commits later. The fork took the `gemm` fork with the
  Apple SME microkernels in `6e7cfcc6` (#189) ten hours *after* the measured revision, so a single
  reading of `main` cannot say whether #186's numbers were right — it can only say they are stale.
  Both revisions are therefore measured back to back on this host with the same harness, and every
  claim below is attributed to the revision that shows it.
- **The harness is in the tree as an archive, not as a `benchmarks/` directory.** The repository
  forbids a benchmark harness of its own (`docs/adoption-protocol.md`), so the program that produced
  these numbers lives in `docs/worklogs/evidence/2026-11-11-faer-186-harness.tar.gz`: one `main.rs`
  per revision (same source, different `path` dependency), one `accel.c` for the reference, `sweep.sh`
  for the grid and `width-sweep.sh` for the width scan, each recording a cell as `FAILED` and exiting
  non-zero when a binary is missing or a run fails, so a partial session cannot pass for a complete one.
  The data files are `results.txt` (the grid), `width-sweep.txt` (both passes) and `accel-cholesky.txt`
  (the reference's repeated Cholesky passes). It also carries the two `Cargo.lock` files the numbers were
  measured with, one per arm, because `Cargo.toml` states ranges (`dyn-stack = "0.13"`, and faer's own
  manifest ranges for `gemm`, `threadteam` and `sysctl`): without them a fresh extraction resolves
  whatever is current today and nothing in the output would say so. The worklog README normally leaves commands
  out; they are here because the request was to make the measurement re-runnable, and a number whose
  command is missing is not.
- **Work counts.** GEMM `2 m n k` real FLOP, complex GEMM `8 m n k` (four real multiplies and four
  real additions per complex multiply-add), LU `2/3 n^3`, Cholesky `1/3 n^3`, QR `4/3 n^3`, GEMV counts
  matrix elements (Gelem/s). The complex convention is not a detail, and it is checkable against #186
  rather than asserted: at `n = 2048`, `n^3 = 8.5899e9`, so the archive's 1395 ms complex GEMM at
  `325e482` is `8 n^3 / t = 49.2 GF/s` and `2 n^3 / t = 12.3 GF/s`. #186 records 49.2 for that cell, so
  #186 counted `8 n^3`; the harness read 12.2 for it until the convention was corrected, and that
  disagreement is what identified the mistake.
- **The clock is an assumption.** The NEON ceiling quoted below (74 GF/s for one core) is
  `4 lanes x 2 FMA x 2 flop/FMA x 4.6 GHz = 73.6 GF/s`; the 4.6 GHz and the count of two FMA pipes per
  core are the nominal figures for this core, not something measured here.
  The argument it supports is one-sided: a lower clock makes the single-thread GEMM result *further*
  above the NEON range, so the conclusion (single-thread GEMM on this host is not a NEON kernel) does not
  depend on the exact number.
- **1T first.** #186 says its own `n >= 2048` thread sweeps scatter, and they do (below), so the 1T
  column carries the conclusions and the team column only supports them.

## Host and conditions

| item | value |
| --- | --- |
| CPU | `Apple M5 Max`, `hw.ncpu` 18, `hw.perflevel0.physicalcpu` 6, `hw.perflevel1.physicalcpu` 12 |
| OS | macOS 26.7.2 (build 25G332), `sw_vers` |
| compiler | `rustc 1.96.0 (ac68faa20 2026-05-25)`, `cargo 1.96.0`, matching `rust-toolchain.toml` |
| power | `pmset -g ps`: AC power, battery 100% charged |
| thermal | `pmset -g therm`: no thermal and no performance warning level recorded, before and after |
| uptime | 13 days 21 h, no reboot in the window, so the same kernel and the same P/E assignment |
| background | `pi`, `Hermes` renderer, `WindowServer`, `loginwindow`, `python3` and `WallpaperAerials` each held 9-14 % CPU; 2 login sessions |
| load average | 4.96-5.90 (1 min) across the three recording sessions; up to 10.98 around the width sweeps |
| pinning | none: macOS has no `taskset`, so no measurement here is CPU-pinned. #186 says the same. |

The host is the one #186 was written on, and #186's own caveat applies unchanged: these runs shared the
host with other processes. Three 4-rep sessions were recorded per arm (start and end load average in
the `## session` lines of `sweep.sh`), and the table reports the best of the three session-bests. The
spread column quoted below is defined as `(max - min) / median` over those three session-bests. On that
definition every faer 1T cell is 0.5-3.5 %, every faer team cell 1.8-15.2 %, and the `Accelerate`
reference 2.6-12.2 % (GEMM f64 2.6 %, c64 2.9 %, LU 5.4 %, QR 2.9 %, GEMV 2.8 %; Cholesky 12.2 %, from
session-bests 121.19, 124.58 and 136.36 GF/s — 11.1 % of the max and 12.5 % of the min, since the
divisor is the median). The reference has no settled Cholesky value at all; the two archived files quoted
further down carry that caveat.

One session was **discarded**: the first `325e482` pass read `lu` 1T as 36.79 GF/s, where two later
passes read 50.22 and 50.76. The load average was 10.98 immediately after it (against 5.06 before),
and no other cell in that pass moved by more than 1.6 %, which is the signature of a short excursion
rather than of the code. It is the only result in this record that was thrown away.

## Reproduction of #186's table

`bash sweep.sh <session>` from the archive, one cell per `faer186-harness <family> 2048 <1|6> 4` and
`./accel <family> 2048 4`, best of 4 repetitions per process, best of 3 sessions. GF/s.

`325e482` column against #186:

| family | faer 1T | #186 | delta | faer 6T | #186 | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gemm f64 n=2048 | 62.76 | 57.8 | +8.6 % | 311.19 | 294.3 | +5.7 % |
| gemm c64 n=2048 | 49.23 | 49.2 | +0.1 % | 272.81 | 209.4 | +30 % |
| lu n=2048 | 50.76 | 47.8 | +6.2 % | 195.99 | 164.4 | +19 % |
| cholesky n=2048 | 48.73 | 43.7 | +11.5 % | 159.90 | 138.7 | +15 % |
| qr n=2048 | 50.44 | 50.9 | -0.9 % | 186.08 | 212.1 | -12 % |
| gemv 4096^2 (Gelem/s) | 8.89 | 8.88 | +0.1 % | - | - | - |

`Accelerate` (one thread, as #186 found; the evidence for that is process CPU time over elapsed time,
0.999-1.000 in 15 of the 18 recorded cells and 1.095-1.103 on QR, where ~8 ms of `sys` for the 90 ms
call dominates the ratio. It is consistent with one active thread and does not prove it: this harness
does not set `VECLIB_MAXIMUM_THREADS`, and #186 reports that setting has no effect here):

| family | measured | #186 | delta |
| --- | ---: | ---: | ---: |
| gemm f64 n=2048 | 439.75 | 439.5 | +0.1 % |
| gemm c64 n=2048 | 410.09 | 401.1 | +2.2 % |
| lu n=2048 | 172.04 | 169.7 | +1.4 % |
| cholesky n=2048 | 136.36 | 170.6 | -20 % |
| qr n=2048 | 128.02 | 126.8 | +1.0 % |
| gemv 4096^2 (Gelem/s) | 15.21 | 15.49 | -1.8 % |

Verdict per cell: **reproduced** at `325e482` for every 1T cell (+8.6 %, +0.1 %, +6.2 %, +11.5 %,
-0.9 %, +0.1 % in the order of the table); **reproduced within the scatter #186 itself warns about**
for the 6T cells (+5.7 % to +30 %, and QR 12 % *below* the recorded value); **reproduced** for the whole
`Accelerate` reference except Cholesky.

Two cells need their caveats stated rather than a verdict:

- **`Accelerate` Cholesky does not pin down.** Six consecutive passes of `dpotrf_` on the same SPD 2048
  matrix read 117.6-131.8 GF/s (`accel-cholesky.txt`), while the grid's three session-bests read 121.19,
  124.58 and 136.36, and the same program rebuilt against the current LAPACK headers
  (`-DACCELERATE_NEW_LAPACK`, `accel-cholesky-newlapack.txt`) reads 72.3-119.2 with three of its four
  passes at 113.0-119.2. Six passes of `dgetrf_` in the grid move 5.4 %, so this routine is the least
  settled number in the record and the Cholesky ratios inherit it.
- `lu` 6T at `325e482` moved 174.4-196.0 across its three sessions (11.2 %), which is exactly #186's
  "the thread sweeps scatter".

## The current tree

`main` = `5c2f4b70`, harness built against `faer/`, default features (which now include
`gemm/apple-sme`, `faer/Cargo.toml:92`).

| family | `main` 1T | `325e482` 1T | `main` 6T | Accelerate 1T | 1T ratio to Accelerate | 1T gain since `325e482` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gemm f64 | 420.50 | 62.76 | 457.90 | 439.75 | 1.05x | 6.70x |
| gemm c64 | 261.86 | 49.23 | 517.18 | 410.09 | 1.57x | 5.32x |
| lu | 154.17 | 50.76 | 225.44 | 172.04 | 1.12x | 3.04x |
| cholesky | 90.75 | 48.73 | 233.24 | 136.36 | 1.50x* | 1.86x |
| qr | 100.39 | 50.44 | 195.73 | 128.02 | 1.28x | 1.99x |
| gemv 4096^2 (Gelem/s) | 9.05 | 8.89 | - | 15.21 | 1.68x | 1.02x |

`*` the Cholesky ratio uses the best reference pass in this window; against the six-pass median (120.6
GF/s) it is 1.33x, and against the `-DACCELERATE_NEW_LAPACK` build 1.25x — see the caveat below.

So #186's headline — a 2.5x to 8.1x single-thread gap to Accelerate, with GEMM worst — no longer
describes `main`. Every 1T cell except complex GEMM, Cholesky and GEMV is within 1.3x of the reference,
and at a team of six every family measured at that width is *ahead* of one Accelerate thread (GEMM f64
1.04x, c64 1.26x, lu 1.31x, Cholesky 1.71x, QR 1.53x; GEMV has no 6T cell in this record). The cause is
not a measurement artefact: `1T = 420.5 GF/s` is 5.7x the NEON ceiling of one M5 core (4 lanes x 2 FMA x
2 flop/FMA x 4.6 GHz = 73.6 GF/s), and the archive's source of the microkernel (`t4a-gemm-f64` 0.20.1) is
Apple SME assembly.

## The six causes, one by one

| # | claim from #186 | holds at `5c2f4b70`? | evidence |
| --- | --- | --- | --- |
| 1 | no native aarch64 GEMM, and `gemm-common`'s `has_amx_impl` rejects M5 | **code fact holds, consequence obsolete** | `t4a-gemm-common-0.20.4/src/cache.rs:197-217` still matches `Some("M1" \| "M2" \| "M3")` at `:211`, so M5 still never takes the AMX path. But AMX is not the route that matters here: the Apple SME probe is separate (`cache.rs:52-53`, `:59-96`: `FEAT_SME` + `FEAT_SME_F64F64`/`FEAT_SME_F32F32`, `svl == 64`, macOS >= 26) and the f64 kernel is `t4a-gemm-f64-0.20.1/src/microkernel.rs:444-497`. The measured 1T GEMM above the NEON ceiling says the probe answers yes on this host. |
| 2 | aarch64 team GEMM packs B per member; x86 shares one pack | **holds, and is now the largest measured loss** | `faer/src/utils/team.rs:693` `gemm_tiled` is the only team route off x86_64; its own doc comment says "one `gemm::gemm` per member with `Parallelism::None`" (`:714-716`), it calls `gemm::gemm` per tile at `:718` and hands out tiles at `:794`. The x86 route is `gemm_team` (`:602`) through `private_gemm_x86::gemm_with(fork)` (`available()` at `:558`). See the recommendation. |
| 3 | no performance-core cap on the width, and the cost table stops at 6 | **partly holds** | `par_width_team` (`faer/src/utils/cost.rs:678`) walks `2..=team.size().min(max)` with no cap, and threadteam 0.5.1's `aarch64-apple` row is measured at widths 2, 4, 6 only (`threadteam-0.5.1/src/cost.rs:460`, 3.0/2.3/1.4 us, extrapolated above). But a cap does exist since, just not here and not for every family: `max_region_width` (`cost.rs:644-661`, `degree.min(largest_measured_width() * 2)` = 12 on this class, from a measurement taken on this same host) is applied at exactly two places, `faer/src/linalg/lu/partial_pivoting/factor.rs:321` and `:351`. GEMM is deliberately exempt (`cost.rs:641-643`), and so are QR and Cholesky in code — which is why their 18-participant collapse in the sweep below is the uncapped tail, not a capped region. Also, `AARCH64_APPLE.gemm_mac` (`cost.rs:407`) is still `0.037 ns` per real MAC, the pre-SME single-thread rate; one SME thread measures 0.0048 ns. |
| 4 | `PERFORMANCE_TIPS.md` recommends a `faer::spindle::with_lock` that does not exist | **does not hold for the guide; the stale advice survives in the ledger** | `PERFORMANCE_TIPS.md:40-71` now states the opposite contract ("the team is the caller's, and it is held across the work"), with the pool-and-team snippet at `:57-63` and `faer::team::with_team` named as the one-off form at `:67`; the only `spindle` left in it is a historical parenthetical at `:48`. `grep -rn spindle faer/src` returns 0 matches, so there is no `faer::spindle` re-export either. But `PATCHSET.md:118` still calls `spindle::with_lock` "how a caller with a thread budget is meant to call faer", and `PATCHSET.md:145` still asserts that the library "re-exports `faer::spindle` and documents that a caller with a thread budget holds it". Those are ledger rows about past experiments, yet the second one describes a symbol the tree does not contain, which is the same defect one file over. `docs/worklogs/2026-10-23-blocked-bidiag.md:137` and `docs/worklogs/2026-10-28-householder-apply-dispatch-and-row-split.md:7` are historical records and are not up for rewriting. |
| 5 | GEMV is a row-wise inner product, 71 GB/s against 124 GB/s | **holds, unchanged** | 4096x4096 f64, 1T: 9.05 Gelem/s (72.4 GB/s) at `main` and 8.89 (71.1 GB/s) at `325e482`, i.e. this candidate has not moved at all in 114 commits; the reference reads 15.21 (121.7 GB/s). The dispatch picks `matvec_colmajor::matvec` for this shape (`faer/src/linalg/matmul/mod.rs:1289`), which is the per-row inner-product form #186 points at. |
| 6a | `cholesky/ldlt/factor.rs` guards the left-looking path with `if true \|\|` | **present, but no longer untended — and the pattern is not unique to it** | `faer/src/linalg/cholesky/ldlt/factor.rs:538` still reads `if true \|\| n < right_looking_threshold`, and `:529-540` now carries the measurement that justifies it: on this host, at f64 2048, the left-looking path is never faster (LLT 8T 14.4 -> 17.1 ms, 1T 67.2 -> 71.1 ms, same residual). The dead branch is deliberate. A second instance #186 did not list sits at `faer/src/linalg/evd/schur/real_schur.rs:1151` (`let infqr = if true \|\| jw < params.blocking_threshold`), there without a note; `grep -rn "if true \|\|" faer/src` returns those 2 matches and no others. |
| 6b | `svd/mod.rs` pads rows on a 1 KiB cache-line assumption while the line is 128 B | **code present, rationale in #186 is wrong** | `faer/src/linalg/svd/mod.rs:70` still tests `(m * size) % 1024 == 0`, but `:62-67` records the reason and it is not the line size: a column stride that is a 1 KiB multiple maps one row onto a few L1 *sets*, which is a power-of-two stride conflict at any line size. The pad is 64 B and it is measured (8T 18.6 -> 15.3 ms at 512). |
| 6c | `qr/no_pivoting/factor.rs` uses a panel ratio measured on an EPYC | **holds** | `faer/src/linalg/qr/no_pivoting/factor.rs:464` still sets `PANEL_TEAM { from: 4, to: 32, ratio: 0.68 }` from the `:461-463` note "4096 at block 128 on broccoli (EPYC 7702P)". |

Nothing here was changed; this record changes no code.

## What the width sweep says on this host today

`width-sweep.sh` on `main`, GF/s, two passes of 3 reps per cell, at load average 16.1-20.9 (a busier
window than the grid's 5-6, which is why this is a scan and not a paired comparison). The raw output of
both passes is `width-sweep.txt` in the archive.

| family | pass | 6 | 12 | 14 | 18 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemm f64 | 1 | 448.1 | 633.8 | 609.8 | 503.8 |
| gemm f64 | 2 | 454.4 | 605.2 | 610.0 | 550.5 |
| gemm c64 | 1 | 451.2 | 567.8 | 551.0 | 546.5 |
| gemm c64 | 2 | 455.1 | 578.9 | 538.5 | 553.4 |
| lu | 1 | 255.9 | 231.1 | 238.2 | 251.0 |
| lu | 2 | 248.1 | 256.1 | 226.9 | 254.4 |
| cholesky | 1 | 210.8 | 255.5 | 222.0 | **39.0** |
| cholesky | 2 | 204.9 | 224.4 | 266.8 | 184.1 |
| qr | 1 | 211.5 | 258.3 | 257.5 | **103.5** |
| qr | 2 | 212.7 | 257.2 | 243.7 | **104.2** |

What repeats: GEMM is 1.3-1.4x faster at 12 participants than at 6 (f64 633.8/605.2 against
448.1/454.4; c64 567.8/578.9 against 451.2/455.1); QR at 18 loses 2.5x in both passes (103.5 and 104.2
against ~258); Cholesky at 18 loses in both, catastrophically in one (39.0). What does not: **lu at 18 is within 2 % of its best width in both archived
passes** (251.0 and 254.4), and an earlier unarchived pass that read 183.4 there is what a load excursion
looks like — it is not repeated, and the "lu collapses at w=18" line in #186 does not reproduce here.

So the 18-participant tail is real for QR and Cholesky, absent for LU in these passes, and the mechanism
is written down in `cost.rs:626-661` (a region's tail, not its median).

## Recommendation: one next item

**Team GEMM is the only cell that a team of six does not improve.** At `main`, `gemm f64` n=2048 runs at
420.50 GF/s on one thread and 457.90 GF/s on six — 1.09x for 5.2x the CPU time (process `user` time
40.77 ms against 212.48 ms; the three sessions read 5.18x, 5.34x and 5.21x) — while one Accelerate thread
is 439.75. The per-busy-core rate in the team run is 80.9 GF/s (`457.90 / (212.48 / 37.52)`, i.e. 5.66 busy cores),
which is the NEON range, against 420 GF/s for a single SME thread, and
`faer/src/utils/team.rs:693-718` shows the team route off x86_64 is `gemm::gemm` per tile with
`Parallelism::None`, the route #186's candidate 2 describes.

The next step is a **probe, not a change**: establish whether per-tile `gemm::gemm` calls reach the
SME microkernel at all, or whether the cost is streaming-mode and `ZA` state around every tile
(`t4a-gemm-f64-0.20.1/src/microkernel.rs:494-496` enters and leaves streaming mode per kernel call, and
macOS saves `ZA` on context switch). Either answer is a bounded change to `gemm_tiled`, and neither is
attempted here. Cholesky already reaches 2.57x from the same six participants, which says a team of six
on this host is capable of more than 1.09x; it does not identify where GEMM's loss comes from, and kernel
selection, tile distribution and the streaming-mode cost are all still open until the probe runs.

Second, if that is fixed: GEMV 1T (1.68x, candidate 5) is the largest remaining single-thread gap and
the only cell that has not moved since #186. Complex GEMM (1.57x) and Cholesky (1.50x) are the next two by
ratio, and both already beat one Accelerate thread at six participants, so GEMV is the largest gap that
is also a single-thread loss end to end. A two-line documentation repair (the
`spindle::with_lock` assertions in `PATCHSET.md:118` and `:145`) is not a measurement and is left to
whoever owns the ledger.

## Verification conclusions and constraints

- Every number here is from this host in this window, unpinned and beside 5-20 % background load, and
  every one is backed by a file in the archive: the grid and its three sessions by `results.txt`, the
  width scan by `width-sweep.txt`, the reference's Cholesky repeats by `accel-cholesky.txt`. No cell was
  measured on a quiet machine, so differences below about 5 % are not resolved by it; the faer 1T column
  is the part that repeats to 0.5-3.5 %, and the reference to 2.6-12.2 %. The one measurement that is
  **not** in the archive is the discarded session described above, which is why its reason is stated
  there instead of a number.
- The archive reproduces the whole grid as two copies of itself that differ in two lines of
  `Cargo.toml` (the faer `path`, and the second copy's binary `name`) plus a copy of the matching
  `Cargo.lock`, which `README.txt` spells out; `cargo build --release --locked` on both copies from a
  fresh extraction puts the binaries at the paths `sweep.sh` defaults to, and resolves the dependencies
  to the versions measured. Those lock files are also the sharpest statement of the revision gap: arm 2
  pins `gemm`/`gemm-common` 0.19.0 and `threadteam` 0.3.0, arm 1 pins `t4a-gemm` 0.20.1,
  `t4a-gemm-common` 0.20.4 and `threadteam` 0.5.1.
  The preflight check and the `FAILED` record were exercised both ways: a missing arm exits 2 before any
  cell runs, and an arm that runs and fails leaves 11 `FAILED` cells and exit 1.
  `accel.c` compiles against `Accelerate` with `-framework Accelerate`; the `-DACCELERATE_NEW_LAPACK`
  variant is a separate arm (`accel-cholesky-newlapack.txt`, 72.3-119.2 GF/s on Cholesky) and is not the
  arm used for the table.
- **Unverified:** why the `Accelerate` Cholesky reference has no settled value — see the two archived
  files above — so every Cholesky ratio in this record inherits it; whether the team GEMM
  loss is kernel selection or streaming-mode cost (the recommendation's probe); the E-core contribution
  at widths 8-18 (no QoS or CPU affinity control was used, so the split is unknown).
- The `#186` table's `Accelerate` column is a fair reference for four of five families and a 20 % outlier
  for Cholesky; a future comparison should re-measure it rather than quote it, and a future Cholesky
  claim should come from a median of several passes there rather than a best.
