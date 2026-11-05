# Upstream performance reports

This list is updated over time. Last updated: 2026-11-06.

This directory is the performance counterpart of [`upstream-reports/`](../README.md): slow paths that we found while working on a port of `faer` and that also occur in the published upstream crates.
The symptom of every entry is a number (a time or a ratio of times), not a wrong value.
The discipline is the parent directory's: each entry is a standalone Cargo project that depends only on the upstream crate from crates.io, pinned to the version in the table, it prints the expected against the actual, and a reader can run it.
Slow paths that exist only in this fork are not listed here.

Run an entry with `cargo run --release` inside its subdirectory.
The program prints a table of times and ratios and exits with status 1 when the slow path shows.
Where the entry is about copying or layout, the program also asserts that the faster route gives the same data bit for bit; a ratio between two routes that compute different results would not be a report.

| # | Title | Upstream crate, version confirmed | Kind of symptom | Fix in this fork | Fixed upstream | Added |
|---|---|---|---|---|---|---|
| [perf-faer-001](perf-faer-001-qr-row-major/) | QR of a row-major matrix is up to 31 times slower than copying it to column major first | `faer` 0.24.4 | time, layout | commits `044f947`, `ae92684`, `10ee1c9` | not yet | 2026-11-06 |
| [perf-faer-002](perf-faer-002-trsm-aliased-rhs/) | The unit lower triangular solve is up to 2.6 times slower at a power-of-two stride | `faer` 0.24.4 | time, layout | tensor4all/faer#276 (`e114eebc`, `17f3d73b`) | not yet | 2026-11-06 |
| [perf-gemm-001](perf-gemm-001-apple-sme-unused/) | `gemm` does not use the matrix unit of an Apple M4 or M5 | `gemm` 0.19.0 | time, host-specific (needs SME) | tensor4all/faer#189, #190 (`t4a-gemm` 0.20.1) | not yet | 2026-11-06 |

"Fixed upstream" was checked against the latest releases on crates.io on 2026-11-06: `faer` 0.24.4 and `gemm` 0.19.0.

## The measurement contract

A performance report is worthless without the conditions it was measured under, so every entry states the following, and its program enforces what it can.

* **Host.** CPU model, core counts by kind, operating system, compiler. Every entry names its host in the README, and numbers from different hosts are never put in one table or compared with each other. The three entries above were measured on an **Apple M5 Max** (6 performance and 12 efficiency cores, macOS 26.7.2, rustc 1.96.0), not on the Ryzen host of the fork's benchmark campaign.
* **Core set and pinning.** Linux entries pin the process to a named CPU set (`taskset`) and say which. macOS has no thread affinity: an entry there asks for the performance cores (`QOS_CLASS_USER_INTERACTIVE`), says that it is not pinned, and records the host's load during the run.
* **Threads.** The thread count of every timed call is fixed and stated. The three entries above are single-threaded (`Par::Seq`, `Parallelism::None`).
* **Priming.** At least one untimed call of every route before timing, so that buffers are paged in and one-time dispatch (CPU feature probes, cache-size queries) is paid outside the samples.
* **Repetitions.** A sample is a run of back-to-back calls lasting a stated minimum time (or one call when a call is longer), and each route gets the same number of samples. The routes alternate sample by sample, so a drift of the host affects all of them alike. Input that a call overwrites is reloaded before every call, outside the timed region.
* **Medians.** The reported time is the median of the samples, and the program prints enough to judge the spread (the max/min sample ratio, or the per-route times).
* **The ratio is computed inside one process.** The slow and the fast route run in the same process, on the same data, alternating; the README reports their ratio. A ratio across two processes or two runs is not a report.
* **Correctness before speed.** The program checks the results of the routes it times (against each other, bit for bit where the claim is about copying or layout, and against an independent computation where it is not) before it prints a time.
* **The threshold.** The README states the ratio at which the program exits with status 1, chosen well above the spread that the run showed.

## How an entry is laid out

Each subdirectory holds `Cargo.toml` (the pinned upstream crate and, where the comparison needs one, the published comparator; a `[workspace]` table so it stays out of the `t4a-faer` workspace), `Cargo.lock`, `src/main.rs`, and a `README.md` that states, in its first three lines, what is slow, for which input and in which version.
The README then gives the command, the host, the table of times and ratios with the measurement conditions, the cause with file and line in the upstream source, how this fork fixed it, the affected versions and references.

## Inventory of performance claims in this fork (an inventory, not a set of finished reports)

Built from a sweep, on 2026-11-06, of the 143 merged pull requests of tensor4all/faer (titles with `perf(`, and bodies stating a measured before and after), of the work logs under `docs/worklogs/`, and of `PATCHSET.md`, at `main` `1125802`.
"Before -> after as stated" quotes the source; nothing in this column was measured or recomputed for the sweep.
"Trace" says whether the number can be found in this repository (a work log, `PATCHSET.md`, a table in the source); a number that lives only in a pull request body or in another repository is a claim without a trace.
Status is one of: `built` (an entry above), `entry-ready` (upstream has the same slow path and a single-process repro against the upstream crate is straightforward), `needs a standalone repro` (upstream has it, but a repro needs real work: pinned multi-thread runs, a particular host, a comparator), `fork-only, not reportable` (the code or the cost is the fork's own), `claim without a trace`.

| PR or record | What changed | Before -> after as stated | Trace | Upstream (0.24.4 / gemm 0.19.0) | Status |
|---|---|---|---|---|---|
| commits `044f947`, `ae92684` (no PR) | unpivoted `qr_in_place` factors a row-major matrix through a column-major copy | `4096x256` f64 1T 47.83 -> 19.44 ms (column major 15.77) | `docs/worklogs/2026-10-15-qr-row-major-layout.md` | same panel, `qr/no_pivoting/factor.rs:11-80`; `perf-warn` only (:270) | **built**: perf-faer-001 |
| commit `10ee1c9` (no PR) | column-pivoted QR through the same copy | `256x256` 1T 58.96 -> 1.52 ms (column major 1.45) | `docs/worklogs/2026-10-15-row-major-adaptation-lu-colpiv.md` | `qr/col_pivoting/factor.rs:170-192` | **built**: perf-faer-001 |
| tensor4all/faer#276 (`00c6b5aa`), staged solve | unit-lower solve stages a right hand side at an aliasing stride | `b = 256` 1T: 658 us at stride 512 -> 268 us with the right hand side at 520 | `docs/worklogs/2026-11-05-trsm-aliased-rhs.md`; campaign data outside the repository | `linalg/triangular_solve.rs:420-500` | **built**: perf-faer-002 |
| tensor4all/faer#189 (`6e7cfcc6`) | `gemm` replaced by `t4a-gemm` with Apple SME kernels | M5 Max `gemm_f64` n = 1024 1T 58.9 -> 216.2 GFLOP/s | none in this repository (driver in `faer-mac`) | `gemm` 0.19.0 has NEON only on aarch64 | **built**: perf-gemm-001 (re-measured; the PR's number itself is a claim without a trace) |
| commits `10ee1c9`, `d132011` (no PR) | partial pivoting LU through a column-major copy | `512x512` 1T 12.14 -> 3.37 ms (column major 3.04) | `docs/worklogs/2026-10-15-row-major-adaptation-lu-colpiv.md` | `lu/partial_pivoting/factor.rs:234-300`, no adaptation | entry-ready |
| commit `d132011` (no PR) | `MatRef::to_owned` of a row-major source copies in tiles | `512x512` 1T 1.006 -> 0.275 ms (column-major source 0.057) | same work log | `mat/matref.rs:1608-1626`; reached from `partial_piv_lu()` and `qr()` | entry-ready |
| work log `2026-10-26-qr128-and-join-gates` (part of #61) | gates on the triangular inverse's join and the sparse product | M5 Max 6T, time over serial: triangular inverse n = 128 6.62 -> 1.00; sparse product 500 15.0 -> 1.02 | the work log (raw data outside the repository; the log calls itself "rough") | no gate: `linalg/triangular_inverse.rs:43-77`, `sparse/linalg/matmul.rs:164-175` | needs a standalone repro (threaded, and the fork's before had other gates) |
| tensor4all/faer#9 (`04542fae`) | `upgrade_householder_factor` opens no region below its own floor | QR f64 `Par::rayon(8)` 128x128 1.05 -> 0.235 ms | `docs/worklogs/2026-10-14-compact-factor-region-floor.md` | no floor, `linalg/householder.rs:132-260`; upstream loss unmeasured | needs a standalone repro |
| tensor4all/faer#14 (`88fec8d6`) | LU trailing update and panel solve sequential below a floor | f64 4T n = 128 0.0980 -> 0.0582 ms | `docs/worklogs/2026-10-14-lu-trailing-update-floor.md` | ungated, `lu/partial_pivoting/factor.rs:104-117` | needs a standalone repro |
| commit `0627828` (no PR) | block apply's column split uses the whole thread budget | probe `512^2` 8T 5.27 -> 4.74 ms | `docs/worklogs/2026-10-18-qr-block-apply-splits.md` | cap of 4 tasks, `linalg/householder.rs:511-533` | needs a standalone repro |
| commit `2000dd9` (no PR) | the native `gemm` gets threads only above its own threshold | f64 `32^3` 8T 12.02 -> 2.30 us | `docs/worklogs/2026-10-18-small-gemm-thread-floor.md` | x86_64 only, `linalg/matmul/mod.rs:17`, `:1356-1410` | needs a standalone repro (x86, threaded) |
| tensor4all/faer#52 (`5e45c574`) | blocked bidiagonalisation, dqds for values only, padded work copy | `svd_values` 1T against the fork's pin 0.51 at n = 64; f64 n = 64 1T 0.239 ms against MKL 0.255 | `docs/worklogs/2026-10-23-*.md`, `PATCHSET.md` | no dqds and an unblocked bidiagonalisation (`svd/bidiag.rs:74`); the denominators are the fork's, not upstream's | needs a standalone repro (an algorithm addition rather than a slow path) |
| tensor4all/faer#276, leaf width | f64 LU leaf 16 -> 8 | 1024 8T 4.56 -> 3.37 ms (MKL 3.40) | the work log has a scan (4-11 % at 1024) but not these numbers | `recursion_threshold: 16`, `lu/partial_pivoting/factor.rs:216` | claim without a trace |
| tensor4all/faer#276, SIMD leaf | SIMD leaf and swaps inside the trailing update | not measured (the PR says so) | none | — | claim without a trace |
| tensor4all/faer#165 (`7ff08ed2`) | look-ahead in the blocked QR on a team | 4096 16T 372.4 -> 292.4 ms (EPYC 7702P) | PR body only; driver on a probe branch | no look-ahead upstream, but the fix and its baseline are the fork's team route | claim without a trace |
| tensor4all/faer#23 (`50c2b989`) | row-major copy only from 64 x 64 elements | `qr_factor` 32x32 8T 0.0420 -> 0.0238 ms | `faer-benchmark` review, not this repository | the copy is the fork's | claim without a trace; fork-only |
| tensor4all/faer#188 (`072483a1`) | cap a derived region width at twice the measured anchors | M5 Max w = 18 n = 2048 38.2 / 57.2 -> 127.3 / 132.5 GFLOP/s | a code comment in `faer/src/utils/cost.rs` only | threadteam width search | claim without a trace; fork-only |
| tensor4all/faer#170 (`4923d8f4`) | Cholesky trailing update in equal-area parts on a team | 4096 4T 772.281 -> 505.531 ms | ratios in `docs/worklogs/2026-10-30-cholesky-area-split.md`; the ms table in the PR only | the cost measured was the fork's team executor | fork-only, not reportable |
| tensor4all/faer#196 (`b3093380`) | cap the equal-area split at one strip per part | 1024/48T median 1.021 ("inside the resolution") | the work log has 1-32T only | — | fork-only, not reportable |
| tensor4all/faer#88 (`f5d5338d`) | regions on the caller's worker group; row split dropped | tall `householder_apply` 2.0-3.8x behind upstream -> 0.63-1.07 | `docs/worklogs/2026-10-28-householder-apply-dispatch-and-row-split.md` | the regions were the fork's own | fork-only, not reportable |
| tensor4all/faer#201 (`3262086b`) | one-row uninitialised product takes the replace path | replace/fresh f64 1T 0.451 -> 1.405 | `docs/worklogs/2026-11-03-matmul-uninit-thin.md` (ranges) | `matmul_with_conj_uninit` does not exist upstream | fork-only, not reportable |
| tensor4all/faer#177, #193 (Pfaffian) | Pfaffian setup, panel and tiles | n = 1024 f64 Seq 111.34 -> 37.3221 ms; n = 256 Team8 0.5000 -> 0.3005 ms | `docs/worklogs/2026-10-31-pfaffian-176.md`, `2026-11-02-pfaffian-supply.md` | no Pfaffian upstream | fork-only, not reportable |
| tensor4all/faer#218 (LQ) | packed LQ | f64 256x256 8T 5.48 ms against OpenBLAS 22.84 | `docs/worklogs/2026-11-03-lq-null.md` | no LQ upstream | fork-only, not reportable |
| tensor4all/faer#61, #97 | `Par<'team>`; threadteam as the only route | team/held 0.087 (gemm 2048x2048x64, M5 6T); non-regression 1.0001 / 0.9978 | #61: outside the repository; #97: `docs/worklogs/2026-10-29-threadteam-only.md` | threadteam | fork-only, not reportable |
| work logs `2026-10-18-svd-values-only`, `2026-10-14-bidiag-parallel-hint`, `2026-10-14-evd-svd-floor-sweep`, `2026-10-14-tall-panel-region-cost` | candidates measured and dropped | e.g. values-only through implicit QR 512 8T 48.079 -> 54.199 ms | the work logs | — | not reportable (dropped, negative results) |

No measured performance claim: tensor4all/faer#104, #109, #174, #181, #190 (a same-tree check), #227 (an allocation count, with a time spot check the PR calls no claim).
tensor4all/faer#274 is a correctness fix whose measured speed is unchanged.

## Claims that do not survive the sweep

Each item names the place in this repository that contradicts or fails to support a stated performance claim.

* **The full pivoting LU copy is "dropped" but shipped.** `docs/worklogs/2026-10-15-row-major-adaptation-lu-colpiv.md:54-64` says the hunk was measured as a no-op and taken out, and the ledger row says the same; `faer/src/linalg/lu/full_pivoting/factor.rs:525-545` carries the row-major copy, guarded by `row_major_copy_pays`.
* **A C/F ratio that is not a C/F ratio.** The same log (:94-96) gives "C/F ratio is 38.8x -> 1.05" for the column-pivoted QR; 38.8 is the row-major before over after (58.96 / 1.52, :76), not row major over column major.
* **The values-only SVD baseline table holds the candidate's numbers.** `docs/worklogs/2026-10-18-svd-values-only.md:32-36` labels its `svd_values` column as the baseline; its values (0.53, 2.20, 11.50, 54.20, 63.10) are the implicit-QR candidate's (:58-63), not the divide and conquer baseline's (0.526, 2.214, 10.848, 48.079, 65.737).
* **#14's floor is not what runs.** The partial pivoting LU's default is now `ParThreshold::Derived` (`faer/src/linalg/lu/partial_pivoting/factor.rs:590`); the 16384 floor the PR measured applies only when a caller asks for `Fixed`. #9 and #14 are also "pre-protocol" adoptions that `PATCHSET.md:35-37` says no claim may cite as a gated result.
* **#23's ledger row was not updated.** `PATCHSET.md:109` still says "The copy has no cost gate yet"; #23 added that gate (`row_major_copy_pays`, `faer/src/linalg/mod.rs:71`). The PR's own numbers live in another repository.
* **#188's numbers disagree with the code comment that is their only record.** The comment (`faer/src/utils/cost.rs:629-643`) gives 66.9, 87.3 and 116.9 GFLOP/s at width 14; the PR's first table gives 63.4, 113.1 and 222.1 for the same width. Both were taken with 1 to 6 busy cores of background load, and the PR's own baseline moved from 24.8 to 45.4 GFLOP/s between two runs of one cell.
* **#193's headline is a different run from its record.** The PR gives 0.5000 -> 0.3005 ms (2.499 -> 4.159 times PfaPack); `docs/worklogs/2026-11-02-pfaffian-supply.md:96-97` and the ledger give 0.5373 -> 0.2995 (2.325 -> 4.170). The ledger still reads "acceptance pending" where the PR reads "criterion met".
* **#196's work log says the accepted cell was never measured.** `docs/worklogs/2026-11-02-equal-area-partition-cap.md:30-36`: "48T turned out to be unmeasurable on this host … No 48T or 64T cell was ever accepted"; the PR presents three accepted 48T runs. Neither claims an improvement.
* **#276's headline is not in the repository and is not the shipped code.** "4.56 -> 3.37 ms" and "0.60 against MKL" appear nowhere in the repository; the leaf-width log (`docs/worklogs/2026-11-05-lu-leaf-width.md:15-17`) records 4-11 % at 1024 against a baseline of 4.24 ms (:9). The SIMD leaf and the swap fusion were merged in the same PR without a measurement, so what ships is not what was measured.
* **#189's factorization numbers predate the gate that changed them.** The QR and LU losses (QR 512 6T 0.60x) were measured before `t4a-gemm` 0.20.1's per-type size gate (#190) and describe no current code; there is no record of them in the repository. Separately, this sweep's perf-gemm-001 measured between 323 and 372 GFLOP/s, over its runs, for `t4a-gemm` 0.20.1 at f64 n = 1024 1T on the same model of host, against the PR's 216.2 for the git revision it used; the two are different versions, so neither number contradicts the other, but the PR's is not the current one.
* **#52 and #88 measured routes that no longer exist.** Both campaigns ran with the caller holding faer's spindle worker group, and #88's dispatch fix was to that route; #97 (`b0262938`) removed spindle and `Par::Rayon`. The dqds that #52 introduced later failed LAPACK's type 16 bidiagonal and gained a fallback (`PATCHSET.md:128`), and the ledger rows for it (`PATCHSET.md:126-127`) still say EXPERIMENT and that the target "is **not** met (0.47)" after the merge and the later 0.239 ms.
* **#201's ranges disagree.** `docs/worklogs/2026-11-03-matmul-uninit-thin.md:26-27` gives the one-row before as 0.45-0.70 "at every thread count"; the PR's c64 4T and 8T befores are 0.228 and 0.232.
* **The LU work-floor units row is KEEP without a measurement.** `docs/worklogs/2026-10-19-lu-work-floor-units.md:47` says "Not a measurement"; `PATCHSET.md:114` keeps it, and the floor has since been replaced by `trailing_update_pays`.
* **#177 and #218 are diagnostics, not results.** #177 calls itself a "Shared-host, ungated, unpaired diagnostic" and misses its own target (Best1T/Team8 3.713 against at least 4). #218's 8T wins are against OpenBLAS 0.3.8, a distribution build, and its record says "No cell passed the campaign's 5% idle gate" (`docs/worklogs/2026-11-03-lq-null.md:80`); the campaign's current OpenBLAS was not measured.

## The backlog

Everything in the inventory that is not `built` is the backlog.
The two `entry-ready` rows (partial pivoting LU and `MatRef::to_owned` on a row-major matrix) are the next ones to build; both are single-threaded layout comparisons of the same shape as perf-faer-001.
