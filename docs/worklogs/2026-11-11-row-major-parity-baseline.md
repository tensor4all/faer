# Row-major parity baseline: every family, both layouts, one table (tensor4all/faer#314, step 1)

This record changes no dispatch and no kernel, so `PATCHSET.md` is unchanged. It is the
"measure first" step of tensor4all/faer#314's plan: for every dense factorization family, what a
row-major call costs against the same call on a column-major matrix, measured at `2154ba17`
(`origin/main` on 2026-11-11, the commit this branch is cut from).

Raw samples, the summary and the harness are in
[`evidence/2026-11-11-row-major-parity/`](evidence/2026-11-11-row-major-parity/).

## Decisions

- **A cell is one family, one shape, one layout, one thread count.** The input matrix *and* every
  output matrix the call takes -- the QR/LQ coefficient buffer, the eigenvector and singular-vector
  matrices, the GEVD's `A`, `B` and `U` -- are allocated in the cell's layout, so a row-major cell
  is the call a row-major caller actually makes, not a row-major input wired into a column-major
  call.
- **The in-place free functions are the measured surface**, not the `PartialPivLu` / `Qr` /
  `Llt` / `SelfAdjointEigen` structs. The constructors that own a factor copy the input into a
  column-major matrix first (`PartialPivLu::new` is `A.to_owned()`; `Llt::new` fills `Mat::zeros`
  through `copy_from_triangular_lower`), so they never reach the row-major dispatch and their
  caller pays about the `col` column plus one copy that this record does not measure.
  `SelfAdjointEigen::new` is the exception: it lends the input straight to `self_adjoint_evd`
  (for `Side::Lower`; `Side::Upper` passes `A.adjoint()`, which is column-major for a row-major
  input) and allocates only a column-major `U`, so a row-major `Side::Lower` caller pays the input
  staging of the `row` cell together with the eigenvector write of the `col` cell -- a combination
  no cell here measures.
- **f64 only.** The issue's numbers and the fork's layout guards are all f64-shaped; `c64`/`f32`
  are not in this table and are not implied by it.
- **Shapes**: `32, 64, 128, 512, 2048` square, plus the tall `4096x256` and `1024x64`. Three
  families are square-only (`LLT`, `LBL^T`, Hermitian `EVD`): the decomposition is defined on an
  `n x n` matrix and a tall cell does not exist. `GEVD` is the one family whose cells the grid's
  cost forced apart: the QZ is O(n^3) and does not scale on this host (512 costs 384-420 ms at one
  thread and 362-390 ms at eight), and a 2048 call takes 25 s column-major and 110-124 s
  row-major. Its 32-512 cells are from the main pass; the two 2048 pairs are from a dedicated
  later run of the same harness and protocol, rebuilt with the GEVD-only family selection, rather
  than from a re-run of all 148 cells. That run
  re-read the 512 cells too: the column-major median was 419.8 ms against 384.5 ms in the main
  pass (9.2% of the reported value, `|a - b| / b`) and the ratio 2.69 against 2.86 (5.9% of the
  reported ratio) -- the largest process-to-process variation in this record, so a GEVD cell
  carries about that much between processes.
- **Threads**: `Par::Seq` (one participant) and `faer::team::with_team(8, ...)` (eight
  participants: the calling thread and seven team members). Nothing above eight.
- **Five repetitions, median reported.** There are 148 cells per thread count (74 family-and-shape
  cells times two layouts), 296 in total, five samples each. Each cell runs one discarded warm-up
  call and then five calls, each on a freshly refilled input matrix with the same values in both
  layouts; the warm-up
  is what pays a family's first-call costs (dispatch, thread-local setup), and the refill happens
  outside the timed region. The `min_us`/`max_us` columns of
  [`314-f64-summary.csv`](evidence/2026-11-11-row-major-parity/314-f64-summary.csv) are the
  spread of those five; nothing in this record is a single sample.
- **Allocation counting is a global allocator in the harness** (a `GlobalAlloc` over `System`
  that adds the requested size to a byte count and tracks the largest single request while a flag
  is set). It is process-wide rather than thread-local so an eight-thread call's allocations are
  counted too. The counted call is the *second* one, not the first: the first GEMM a thread runs
  allocates its thread-local workspace in `t4a-private-gemm-x86` (128 MiB, one per thread; found
  with a backtrace, `private_gemm_x86::gemm_impl` -> `std::thread_local::...::get_or_init_slow`),
  and counting the first call charges that one-time buffer to whichever family happens to run
  first.
- **Two passes per thread count, the second is reported.** The first eight-thread pass overlapped
  a competing `cargo` build pinned to CPU 27 (load average 11 at its start) and left 12 cells with
  a >1.5x spread among their five samples, with medians up to 1.8x slower than the second pass on
  the LU cells (eight-thread `lu_partial` 2048: median 28.9 ms against 15.4 ms; `lu_full` 2048:
  323 ms against 178 ms). The second passes ran at a load average below 0.5 with no competing build
  and left 2 cells (eight threads) and 3 cells (one thread) above 1.5x, all at `n = 32`, where the
  first sample after the warm-up is the slow one. No number in this record comes from the first
  passes; they are kept as `314-f64-time-1T-first-pass.csv` and `314-f64-time-8T-first-pass.csv`
  for audit. Relative differences in this record are `|a - b| / b` against the reported value
  `b`: over the 146 cells the two one-thread passes share, the largest is a factor 1.062
  (`evd_vals` at `n = 32`: 50.688 us in the first pass, 47.709 us reported -- 6.24% of the
  reported value, and 5.88% if the first pass is the denominator instead).
- **Pinned to CPUs 0-7** with `taskset`, which is one L3 domain on this host (`0-7,16-23`), with
  the SMT siblings of those cores left out of the mask. The observing side was **not** pinned
  outside that domain, which the benchmark rule asks for: the shell and the harness's own output
  run wherever the scheduler puts them, and `kache-engine` idles at about 16% of one core on an
  unpinned CPU. The two one-thread passes agree to within that factor 1.062 on the 146 cells they
  share, which is the repeatability of the grid, not a bound on a systematic observer effect (two
  runs with the same unpinned observer can share the same bias). These numbers are not a
  pinned-observer-compliant baseline; see Limits.

## The host

```
uname -a        Linux EVO-X2 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC \
                Mon Oct 19 14:43:01 UTC 2 x86_64
/proc/cpuinfo   AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 32 logical CPUs, 30 GiB RAM
L3 domains      0-7,16-23 / 8-15,24-31 (two 32 MiB domains, 8 physical cores each)
governor        performance
rustc -V        1.96.0 (ac68faa20 2026-05-25)   <- rust-toolchain.toml pins 1.96.0
cargo -V        1.96.0 (30a34c682 2026-05-25)
RUSTFLAGS       (empty)
git rev-parse   HEAD = 2154ba17 (origin/main)
build           cargo build --release --example rm314 --features perf-warn
```

The harness is built with the default feature set plus `perf-warn`, which only compiles the
`log::warn!` calls behind the layout warnings; no dispatch condition or kernel changes with it.
Outside the repository the host's default toolchain is 1.99.1; the pinned 1.96.0 above is what
built the measured binary.

## Timing: median of five calls, milliseconds

Column-major (`col`) and row-major (`row`) are the same call on the same values; `row/col` is the
ratio of the two medians. A value below 1.00 means the row-major call is the faster one.

| family | shape | 1T col | 1T row | 1T row/col | 8T col | 8T row | 8T row/col |
|---|---|---|---|---|---|---|---|
| LU partial pivot | 32 | 0.00421 | 0.00444 | 1.05 | 0.00427 | 0.00457 | 1.07 |
| LU partial pivot | 64 | 0.0127 | 0.0137 | 1.08 | 0.0135 | 0.0168 | 1.25 |
| LU partial pivot | 128 | 0.0521 | 0.0539 | 1.03 | 0.0483 | 0.0737 | 1.53 |
| LU partial pivot | 512 | 1.65 | 1.63 | 0.99 | 0.701 | 0.79 | 1.13 |
| LU partial pivot | 2048 | 79.22 | 77.82 | 0.98 | 15.38 | 24.57 | 1.60 |
| LU partial pivot | 4096x256 | 4.41 | 5.56 | 1.26 | 1.81 | 2.27 | 1.26 |
| LU partial pivot | 1024x64 | 0.111 | 0.171 | 1.53 | 0.114 | 0.165 | 1.45 |
| LU full pivot | 32 | 0.00724 | 0.00686 | 0.95 | 0.00834 | 0.00769 | 0.92 |
| LU full pivot | 64 | 0.0307 | 0.0307 | 1.00 | 0.032 | 0.0335 | 1.05 |
| LU full pivot | 128 | 0.191 | 0.244 | 1.28 | 0.21 | 0.221 | 1.05 |
| LU full pivot | 512 | 13.44 | 15.40 | 1.15 | 7 | 6.96 | 0.99 |
| LU full pivot | 2048 | 926.9 | 984.2 | 1.06 | 178.2 | 189.2 | 1.06 |
| LU full pivot | 4096x256 | 30.52 | 29.13 | 0.95 | 6.23 | 6.15 | 0.99 |
| LU full pivot | 1024x64 | 0.434 | 0.67 | 1.55 | 0.295 | 0.338 | 1.15 |
| QR (no pivoting) | 32 | 0.0117 | 0.0173 | 1.48 | 0.0124 | 0.0178 | 1.44 |
| QR (no pivoting) | 64 | 0.0439 | 0.0424 | 0.97 | 0.0456 | 0.0501 | 1.10 |
| QR (no pivoting) | 128 | 0.146 | 0.14 | 0.96 | 0.128 | 0.136 | 1.06 |
| QR (no pivoting) | 512 | 3.42 | 3.92 | 1.15 | 1.07 | 1.12 | 1.05 |
| QR (no pivoting) | 2048 | 160.4 | 188.6 | 1.18 | 26.75 | 35.87 | 1.34 |
| QR (no pivoting) | 4096x256 | 10.23 | 13.18 | 1.29 | 4.56 | 5 | 1.10 |
| QR (no pivoting) | 1024x64 | 0.327 | 0.52 | 1.59 | 0.279 | 0.34 | 1.22 |
| QR (column pivoting) | 32 | 0.0124 | 0.179 | 14.44 | 0.0139 | 0.176 | 12.66 |
| QR (column pivoting) | 64 | 0.0403 | 0.0525 | 1.30 | 0.0419 | 0.0482 | 1.15 |
| QR (column pivoting) | 128 | 0.188 | 0.222 | 1.18 | 0.193 | 0.195 | 1.01 |
| QR (column pivoting) | 512 | 8.55 | 12.29 | 1.44 | 2.39 | 3.29 | 1.37 |
| QR (column pivoting) | 2048 | 860.0 | 894.5 | 1.04 | 115.2 | 131.7 | 1.14 |
| QR (column pivoting) | 4096x256 | 36.89 | 40.33 | 1.09 | 5.63 | 6.82 | 1.21 |
| QR (column pivoting) | 1024x64 | 0.439 | 0.601 | 1.37 | 0.276 | 0.325 | 1.18 |
| LQ (no pivoting) | 32 | 0.0146 | 0.0112 | 0.76 | 0.0169 | 0.0116 | 0.68 |
| LQ (no pivoting) | 64 | 0.0431 | 0.0431 | 1.00 | 0.0564 | 0.0452 | 0.80 |
| LQ (no pivoting) | 128 | 0.135 | 0.143 | 1.06 | 0.159 | 0.127 | 0.80 |
| LQ (no pivoting) | 512 | 3.67 | 3.16 | 0.86 | 1.19 | 1.02 | 0.86 |
| LQ (no pivoting) | 2048 | 183.1 | 160.3 | 0.88 | 30.83 | 26.70 | 0.87 |
| LQ (no pivoting) | 4096x256 | 9.58 | 10.03 | 1.05 | 1.7 | 1.71 | 1.00 |
| LQ (no pivoting) | 1024x64 | 0.257 | 0.257 | 1.00 | 0.0761 | 0.0858 | 1.13 |
| Cholesky (LLT) | 32 | 0.00167 | 0.00415 | 2.49 | 0.00166 | 0.0041 | 2.47 |
| Cholesky (LLT) | 64 | 0.00865 | 0.0308 | 3.56 | 0.00854 | 0.0307 | 3.59 |
| Cholesky (LLT) | 128 | 0.0287 | 0.129 | 4.48 | 0.028 | 0.126 | 4.49 |
| Cholesky (LLT) | 512 | 0.827 | 1.86 | 2.25 | 0.342 | 1.09 | 3.20 |
| Cholesky (LLT) | 2048 | 43.66 | 51.66 | 1.18 | 7.24 | 10.47 | 1.45 |
| Cholesky (LBL^T, Bunch-Kaufman) | 32 | 0.00574 | 0.00508 | 0.89 | 0.00579 | 0.00501 | 0.87 |
| Cholesky (LBL^T, Bunch-Kaufman) | 64 | 0.0195 | 0.0271 | 1.39 | 0.0202 | 0.0267 | 1.32 |
| Cholesky (LBL^T, Bunch-Kaufman) | 128 | 0.0986 | 0.206 | 2.09 | 0.095 | 0.19 | 2.00 |
| Cholesky (LBL^T, Bunch-Kaufman) | 512 | 3.31 | 11.31 | 3.42 | 2.91 | 7.41 | 2.55 |
| Cholesky (LBL^T, Bunch-Kaufman) | 2048 | 91.26 | 241.7 | 2.65 | 49.96 | 129.5 | 2.59 |
| Hermitian EVD, values only | 32 | 0.0477 | 0.0454 | 0.95 | 0.0485 | 0.0461 | 0.95 |
| Hermitian EVD, values only | 64 | 0.162 | 0.16 | 0.99 | 0.163 | 0.161 | 0.99 |
| Hermitian EVD, values only | 128 | 0.632 | 0.627 | 0.99 | 0.62 | 0.621 | 1.00 |
| Hermitian EVD, values only | 512 | 13.64 | 13.75 | 1.01 | 8.65 | 8.85 | 1.02 |
| Hermitian EVD, values only | 2048 | 555.9 | 557.3 | 1.00 | 166.4 | 169.4 | 1.02 |
| Hermitian EVD, vectors | 32 | 0.06 | 0.062 | 1.03 | 0.0604 | 0.0629 | 1.04 |
| Hermitian EVD, vectors | 64 | 0.242 | 0.251 | 1.04 | 0.241 | 0.244 | 1.01 |
| Hermitian EVD, vectors | 128 | 1.1 | 1.14 | 1.04 | 1.04 | 1.05 | 1.01 |
| Hermitian EVD, vectors | 512 | 24.52 | 24.82 | 1.01 | 10.20 | 10.38 | 1.02 |
| Hermitian EVD, vectors | 2048 | 984.2 | 998.5 | 1.01 | 220.1 | 227.2 | 1.03 |
| SVD, values only | 32 | 0.0471 | 0.045 | 0.96 | 0.0477 | 0.0454 | 0.95 |
| SVD, values only | 64 | 0.161 | 0.16 | 1.00 | 0.164 | 0.163 | 0.99 |
| SVD, values only | 128 | 0.658 | 0.668 | 1.01 | 0.71 | 0.718 | 1.01 |
| SVD, values only | 512 | 24.67 | 26.98 | 1.09 | 12.47 | 14.54 | 1.17 |
| SVD, values only | 2048 | 1497 | 1525 | 1.02 | 338.9 | 375.3 | 1.11 |
| SVD, values only | 4096x256 | 14.76 | 18.79 | 1.27 | 8.6 | 12.52 | 1.46 |
| SVD, values only | 1024x64 | 0.549 | 0.594 | 1.08 | 0.519 | 0.555 | 1.07 |
| SVD, vectors | 32 | 0.114 | 0.115 | 1.01 | 0.115 | 0.116 | 1.01 |
| SVD, vectors | 64 | 0.514 | 0.521 | 1.01 | 0.512 | 0.516 | 1.01 |
| SVD, vectors | 128 | 1.97 | 1.99 | 1.01 | 1.52 | 1.59 | 1.04 |
| SVD, vectors | 512 | 49.96 | 57.31 | 1.15 | 20.56 | 25.10 | 1.22 |
| SVD, vectors | 2048 | 2381 | 2668 | 1.12 | 514.6 | 627.5 | 1.22 |
| SVD, vectors | 4096x256 | 32.59 | 38.45 | 1.18 | 13.54 | 19.58 | 1.45 |
| SVD, vectors | 1024x64 | 1.14 | 1.23 | 1.08 | 0.925 | 0.979 | 1.06 |
| GEVD (QZ, right vectors) | 32 | 0.213 | 0.233 | 1.09 | 0.212 | 0.233 | 1.10 |
| GEVD (QZ, right vectors) | 64 | 1.54 | 1.66 | 1.07 | 1.48 | 1.59 | 1.07 |
| GEVD (QZ, right vectors) | 128 | 9.07 | 10.98 | 1.21 | 8.98 | 10.91 | 1.22 |
| GEVD (QZ, right vectors) | 512 | 384.5 | 1100 | 2.86 | 361.9 | 1062 | 2.93 |
| GEVD (QZ, right vectors) | 2048 | 25357 | 124029 | 4.89 | 22913 | 110407 | 4.82 |


## Allocations during one call

The counter sees heap allocations only: scratch that comes from the caller's `MemStack` is outside
it, and so is the one-time per-thread 128 MiB GEMM workspace described above. `m·n·8` is the size
of the input matrix in bytes, so a `largest B` equal to it is exactly "a whole-matrix buffer".

| family | shape | threads | col allocs | col bytes | col largest B | row allocs | row bytes | row largest B | m·n·8 B |
|---|---|---|---|---|---|---|---|---|---|
| LU partial pivot | 32 | 1 | 0 | 0 | 0 | 1 | 2048 | 2048 | 8192 |
| LU partial pivot | 32 | 8 | 0 | 0 | 0 | 1 | 2048 | 2048 | 8192 |
| LU partial pivot | 64 | 1 | 0 | 0 | 0 | 1 | 4096 | 4096 | 32768 |
| LU partial pivot | 64 | 8 | 0 | 0 | 0 | 1 | 4096 | 4096 | 32768 |
| LU partial pivot | 128 | 1 | 0 | 0 | 0 | 1 | 8192 | 8192 | 131072 |
| LU partial pivot | 128 | 8 | 0 | 0 | 0 | 1 | 8192 | 8192 | 131072 |
| LU partial pivot | 512 | 1 | 0 | 0 | 0 | 1 | 32768 | 32768 | 2097152 |
| LU partial pivot | 512 | 8 | 0 | 0 | 0 | 1 | 2097152 | 2097152 | 2097152 |
| LU partial pivot | 2048 | 1 | 0 | 0 | 0 | 1 | 131072 | 131072 | 33554432 |
| LU partial pivot | 2048 | 8 | 0 | 0 | 0 | 1 | 33554432 | 33554432 | 33554432 |
| LU partial pivot | 4096x256 | 1 | 0 | 0 | 0 | 1 | 262144 | 262144 | 8388608 |
| LU partial pivot | 4096x256 | 8 | 0 | 0 | 0 | 1 | 8388608 | 8388608 | 8388608 |
| LU partial pivot | 1024x64 | 1 | 0 | 0 | 0 | 1 | 65536 | 65536 | 524288 |
| LU partial pivot | 1024x64 | 8 | 0 | 0 | 0 | 1 | 524288 | 524288 | 524288 |
| LU full pivot | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| LU full pivot | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| LU full pivot | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| LU full pivot | 64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| LU full pivot | 128 | 1 | 0 | 0 | 0 | 1 | 131072 | 131072 | 131072 |
| LU full pivot | 128 | 8 | 0 | 0 | 0 | 1 | 131072 | 131072 | 131072 |
| LU full pivot | 512 | 1 | 0 | 0 | 0 | 1 | 2097152 | 2097152 | 2097152 |
| LU full pivot | 512 | 8 | 0 | 0 | 0 | 1 | 2097152 | 2097152 | 2097152 |
| LU full pivot | 2048 | 1 | 0 | 0 | 0 | 1 | 33554432 | 33554432 | 33554432 |
| LU full pivot | 2048 | 8 | 0 | 0 | 0 | 1 | 33554432 | 33554432 | 33554432 |
| LU full pivot | 4096x256 | 1 | 0 | 0 | 0 | 1 | 8388608 | 8388608 | 8388608 |
| LU full pivot | 4096x256 | 8 | 0 | 0 | 0 | 1 | 8388608 | 8388608 | 8388608 |
| LU full pivot | 1024x64 | 1 | 0 | 0 | 0 | 1 | 524288 | 524288 | 524288 |
| LU full pivot | 1024x64 | 8 | 0 | 0 | 0 | 1 | 524288 | 524288 | 524288 |
| QR (no pivoting) | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| QR (no pivoting) | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| QR (no pivoting) | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| QR (no pivoting) | 64 | 8 | 0 | 0 | 0 | 1 | 32768 | 32768 | 32768 |
| QR (no pivoting) | 128 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| QR (no pivoting) | 128 | 8 | 0 | 0 | 0 | 1 | 131072 | 131072 | 131072 |
| QR (no pivoting) | 512 | 1 | 0 | 0 | 0 | 1 | 2097152 | 2097152 | 2097152 |
| QR (no pivoting) | 512 | 8 | 0 | 0 | 0 | 1 | 2097152 | 2097152 | 2097152 |
| QR (no pivoting) | 2048 | 1 | 0 | 0 | 0 | 1 | 33554432 | 33554432 | 33554432 |
| QR (no pivoting) | 2048 | 8 | 0 | 0 | 0 | 1 | 33554432 | 33554432 | 33554432 |
| QR (no pivoting) | 4096x256 | 1 | 0 | 0 | 0 | 1 | 8388608 | 8388608 | 8388608 |
| QR (no pivoting) | 4096x256 | 8 | 20 | 110592 | 9216 | 21 | 8499200 | 8388608 | 8388608 |
| QR (no pivoting) | 1024x64 | 1 | 0 | 0 | 0 | 1 | 524288 | 524288 | 524288 |
| QR (no pivoting) | 1024x64 | 8 | 0 | 0 | 0 | 1 | 524288 | 524288 | 524288 |
| QR (column pivoting) | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| QR (column pivoting) | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| QR (column pivoting) | 64 | 1 | 0 | 0 | 0 | 1 | 32768 | 32768 | 32768 |
| QR (column pivoting) | 64 | 8 | 0 | 0 | 0 | 1 | 32768 | 32768 | 32768 |
| QR (column pivoting) | 128 | 1 | 0 | 0 | 0 | 1 | 131072 | 131072 | 131072 |
| QR (column pivoting) | 128 | 8 | 0 | 0 | 0 | 1 | 131072 | 131072 | 131072 |
| QR (column pivoting) | 512 | 1 | 0 | 0 | 0 | 1 | 2097152 | 2097152 | 2097152 |
| QR (column pivoting) | 512 | 8 | 14 | 917504 | 65536 | 15 | 3014656 | 2097152 | 2097152 |
| QR (column pivoting) | 2048 | 1 | 0 | 0 | 0 | 1 | 33554432 | 33554432 | 33554432 |
| QR (column pivoting) | 2048 | 8 | 30 | 7864320 | 262144 | 31 | 41418752 | 33554432 | 33554432 |
| QR (column pivoting) | 4096x256 | 1 | 0 | 0 | 0 | 1 | 8388608 | 8388608 | 8388608 |
| QR (column pivoting) | 4096x256 | 8 | 6 | 753664 | 147456 | 7 | 9142272 | 8388608 | 8388608 |
| QR (column pivoting) | 1024x64 | 1 | 0 | 0 | 0 | 1 | 524288 | 524288 | 524288 |
| QR (column pivoting) | 1024x64 | 8 | 2 | 131072 | 65536 | 3 | 655360 | 524288 | 524288 |
| LQ (no pivoting) | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| LQ (no pivoting) | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| LQ (no pivoting) | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| LQ (no pivoting) | 64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| LQ (no pivoting) | 128 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| LQ (no pivoting) | 128 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| LQ (no pivoting) | 512 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2097152 |
| LQ (no pivoting) | 512 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 2097152 |
| LQ (no pivoting) | 2048 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 33554432 |
| LQ (no pivoting) | 2048 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 33554432 |
| LQ (no pivoting) | 4096x256 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8388608 |
| LQ (no pivoting) | 4096x256 | 8 | 5 | 17280 | 4608 | 5 | 17280 | 4608 | 8388608 |
| LQ (no pivoting) | 1024x64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 524288 |
| LQ (no pivoting) | 1024x64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 524288 |
| Cholesky (LLT) | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| Cholesky (LLT) | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| Cholesky (LLT) | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| Cholesky (LLT) | 64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| Cholesky (LLT) | 128 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| Cholesky (LLT) | 128 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| Cholesky (LLT) | 512 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2097152 |
| Cholesky (LLT) | 512 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 2097152 |
| Cholesky (LLT) | 2048 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 33554432 |
| Cholesky (LLT) | 2048 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 33554432 |
| Cholesky (LBL^T, Bunch-Kaufman) | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| Cholesky (LBL^T, Bunch-Kaufman) | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| Cholesky (LBL^T, Bunch-Kaufman) | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| Cholesky (LBL^T, Bunch-Kaufman) | 64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| Cholesky (LBL^T, Bunch-Kaufman) | 128 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| Cholesky (LBL^T, Bunch-Kaufman) | 128 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| Cholesky (LBL^T, Bunch-Kaufman) | 512 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2097152 |
| Cholesky (LBL^T, Bunch-Kaufman) | 512 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 2097152 |
| Cholesky (LBL^T, Bunch-Kaufman) | 2048 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 33554432 |
| Cholesky (LBL^T, Bunch-Kaufman) | 2048 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 33554432 |
| Hermitian EVD, values only | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| Hermitian EVD, values only | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| Hermitian EVD, values only | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| Hermitian EVD, values only | 64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| Hermitian EVD, values only | 128 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| Hermitian EVD, values only | 128 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| Hermitian EVD, values only | 512 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2097152 |
| Hermitian EVD, values only | 512 | 8 | 13 | 851968 | 65536 | 13 | 851968 | 65536 | 2097152 |
| Hermitian EVD, values only | 2048 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 33554432 |
| Hermitian EVD, values only | 2048 | 8 | 29 | 7602176 | 262144 | 29 | 7602176 | 262144 | 33554432 |
| Hermitian EVD, vectors | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| Hermitian EVD, vectors | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| Hermitian EVD, vectors | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| Hermitian EVD, vectors | 64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| Hermitian EVD, vectors | 128 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| Hermitian EVD, vectors | 128 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| Hermitian EVD, vectors | 512 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2097152 |
| Hermitian EVD, vectors | 512 | 8 | 13 | 851968 | 65536 | 13 | 851968 | 65536 | 2097152 |
| Hermitian EVD, vectors | 2048 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 33554432 |
| Hermitian EVD, vectors | 2048 | 8 | 29 | 7602176 | 262144 | 29 | 7602176 | 262144 | 33554432 |
| SVD, values only | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| SVD, values only | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| SVD, values only | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| SVD, values only | 64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| SVD, values only | 128 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| SVD, values only | 128 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| SVD, values only | 512 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2097152 |
| SVD, values only | 512 | 8 | 27 | 1769472 | 65536 | 27 | 1769472 | 65536 | 2097152 |
| SVD, values only | 2048 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 33554432 |
| SVD, values only | 2048 | 8 | 59 | 15466496 | 262144 | 59 | 15466496 | 262144 | 33554432 |
| SVD, values only | 4096x256 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8388608 |
| SVD, values only | 4096x256 | 8 | 31 | 831488 | 65536 | 31 | 831488 | 65536 | 8388608 |
| SVD, values only | 1024x64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 524288 |
| SVD, values only | 1024x64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 524288 |
| SVD, vectors | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| SVD, vectors | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| SVD, vectors | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| SVD, vectors | 64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| SVD, vectors | 128 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| SVD, vectors | 128 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| SVD, vectors | 512 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2097152 |
| SVD, vectors | 512 | 8 | 27 | 1769472 | 65536 | 27 | 1769472 | 65536 | 2097152 |
| SVD, vectors | 2048 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 33554432 |
| SVD, vectors | 2048 | 8 | 59 | 15466496 | 262144 | 59 | 15466496 | 262144 | 33554432 |
| SVD, vectors | 4096x256 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8388608 |
| SVD, vectors | 4096x256 | 8 | 31 | 831488 | 65536 | 31 | 831488 | 65536 | 8388608 |
| SVD, vectors | 1024x64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 524288 |
| SVD, vectors | 1024x64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 524288 |
| GEVD (QZ, right vectors) | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| GEVD (QZ, right vectors) | 32 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8192 |
| GEVD (QZ, right vectors) | 64 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32768 |
| GEVD (QZ, right vectors) | 64 | 8 | 0 | 0 | 0 | 1 | 32768 | 32768 | 32768 |
| GEVD (QZ, right vectors) | 128 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| GEVD (QZ, right vectors) | 128 | 8 | 0 | 0 | 0 | 1 | 131072 | 131072 | 131072 |
| GEVD (QZ, right vectors) | 512 | 1 | 0 | 0 | 0 | 1 | 2097152 | 2097152 | 2097152 |
| GEVD (QZ, right vectors) | 512 | 8 | 0 | 0 | 0 | 1 | 2097152 | 2097152 | 2097152 |
| GEVD (QZ, right vectors) | 2048 | 1 | 0 | 0 | 0 | 1 | 33554432 | 33554432 | 33554432 |
| GEVD (QZ, right vectors) | 2048 | 8 | 0 | 0 | 0 | 1 | 33554432 | 33554432 | 33554432 |


## What the table says

Read as "the row-major call is N times the column-major call"; the worst square cell of each
family at each thread count, and the tall shapes:

| family | 1T, worst square (n) | 8T, worst square (n) | 1T tall 4096x256 / 1024x64 | 8T tall |
|---|---|---|---|---|
| LU, partial pivoting | 1.08 (64) | 1.60 (2048) | 1.26 / 1.53 | 1.26 / 1.45 |
| LU, full pivoting | 1.28 (128) | 1.06 (2048) | 0.95 / 1.55 | 0.99 / 1.15 |
| QR, no pivoting | 1.48 (32) | 1.44 (32) | 1.29 / 1.59 | 1.10 / 1.22 |
| QR, column pivoting | 14.44 (32) | 12.66 (32) | 1.09 / 1.37 | 1.21 / 1.18 |
| LQ, no pivoting | 1.06 (128) | 0.87 (2048) | 1.05 / 1.00 | 1.00 / 1.13 |
| Cholesky, LLT | 4.48 (128) | 4.49 (128) | - | - |
| Cholesky, LBL^T (Bunch-Kaufman) | 3.42 (512) | 2.59 (2048) | - | - |
| Hermitian EVD, values only | 1.01 (512) | 1.02 (512) | - | - |
| Hermitian EVD, vectors | 1.04 (64) | 1.04 (32) | - | - |
| SVD, values only | 1.09 (512) | 1.17 (512) | 1.27 / 1.08 | 1.46 / 1.07 |
| SVD, vectors | 1.15 (512) | 1.22 (512) | 1.18 / 1.08 | 1.45 / 1.06 |
| GEVD (QZ, right vectors) | 4.89 (2048) | 4.82 (2048) | - | - |

Family by family:

- **LU, partial pivoting** is at parity on the square shapes at one thread (0.98-1.08 from 32 to
  2048) and pays 1.13-1.60 at eight threads from 512 up, where `lu_in_place` takes the copy again
  (`takes_the_copy`); the tall shapes cost 1.26-1.53 in both settings. The one buffer it allocates
  is `m * min(n, recursion_threshold)` elements, i.e. `64 * m` bytes at f64 and
  `recursion_threshold = 8` (`row_major_leaf_buffer`), except where the copy is taken and the
  counter then sees exactly one `m * n * 8`.
- **LU, full pivoting** is at parity wherever the transpose dual runs (0.95 at 32, 1.00 at 64),
  pays 1.28 at 128 one thread (that shape keeps the copy: `row_major_takes_the_copy` stops below
  128), 1.15 at 512, 1.06 at 2048, and 1.55 on `1024x64`.
- **QR, no pivoting** is at parity on square 64-128 (0.96-0.97, the dual below 192 at one thread),
  1.48 at 32 (below the dual's floor), 1.15-1.18 at 512-2048 and 1.29-1.59 on the tall shapes, with
  one `m * n * 8` buffer from 512 up.
- **QR, column pivoting** is 12.7-14.4x at `n = 32`. That is the shape where the copy is *not*
  taken: `row_major_copy_pays` needs 4096 elements and 32x32 has 1024, so the call stays on the
  direct path and the pivot search walks a row-major matrix. From 64 up the ratio is 1.0-1.4
  (1.30 at 64, 1.18 at 128, 1.44 at 512, 1.04 at 2048), and the counter shows the `m * n * 8`
  copy from 64 up.
- **LQ** is the one family where a row-major input is the faster one on most square cells:
  0.68-0.87 from 32 to 2048 at eight threads and 0.76-0.88 at 32, 512 and 2048 at one, with 64 and
  128 at one thread the two exceptions (1.00 and 1.06). `lq_in_place` factors the transposed view,
  so a row-major input hands the QR panel its contiguous direction. No square cell allocates in
  either layout; the tall `4096x256` at eight threads makes the same five small allocations
  (17 KB) in both, and `1024x64` at eight threads is 1.13, outside the range above.
- **Cholesky (LLT)** has no row-major path at all: 2.49, 3.56, 4.48, 2.25 and 1.18 at 32-2048 one
  thread, 2.47-4.49 up to 128, 3.20 at 512 and 1.45 at 2048 on eight. No copy is involved (0 allocations in both layouts): this is the
  lower-triangular kernel itself walking a row-major matrix.
- **Cholesky (LBL^T)** costs 1.39 at 64, 2.09 at 128, 3.42 at 512 and 2.65 at 2048 one thread
  (2.00-2.59 at eight), again with no allocation in either layout. It is the family the issue's
  spec lists first for the dual (upper LBL^T).
- **Hermitian EVD** is at parity: 0.95-1.02 for the values and 1.01-1.04 with the eigenvectors, at
  both thread counts and every size. The input is staged into the scratch tridiagonal buffer in
  both layouts and a row-major `u` is written through an `n x n` column-major stand-in
  (`stage_u`), so the two layouts make the *same* heap allocations -- none at one thread, and the
  parallel regions' per-thread buffers at eight (29 calls, 7.6 MB at 2048, in both layouts).
  Nothing here is a layout buffer that only the row-major call makes.
- **SVD** is at parity up to 128 (0.95-1.04) and 1.02-1.22 at 512-2048 (values-only 2048 at one
  thread is 1.02, the vector cells 1.12-1.22), with the tall `4096x256` at 1.18-1.46. The input goes into the scratch bidiagonal buffer for a square-ish shape and
  through a scratch QR buffer for a tall one, in both layouts, and the two layouts make identical
  heap allocations (none at one thread; 59 calls, 15.5 MB at 2048, and 31 calls, 0.83 MB at
  `4096x256`, at eight) -- but those copies read a row-major source in one case and a
  column-major one in the other, so their cost is not thereby shown to be equal. Which loop
  carries the 2-22% is not measured. The values-only tall cell is already 1.27 at one thread and
  1.46 at eight with no singular-vector output, so the vector matrices are not the whole story
  there.
- **GEVD** is 1.07-1.22 up to 128, 2.86 at 512 and 4.89 at 2048 (2.93 and 4.82 at eight threads).
  From 512 up the row-major call also allocates one `m * n * 8` buffer, and a backtrace puts it in
  `qr_in_place`'s column-major copy, reached from `gevd_real`: the row-major QZ therefore runs a
  different factorization path there (a column-major copy of one matrix), so this measurement
  cannot split the 2 MB (512) or 32 MB (2048) buffer from the 715 ms and 98 s gaps, and no split
  is claimed. At eight threads the same copy is already taken at 64 (32 KB) and 128 (131 KB).

## The one paragraph this record does not support

The shapes where row-major loses are the ones whose kernel touches the matrix a row or a column at
a time in the direction the layout does not store contiguously: the LLT/LBL^T lower update, the
column-pivoted QR pivot search, the QZ's Householder work on `A`/`B`, and LU's pivot column. That
is consistent with the pattern in the table (the GEMM-dominated parts are at parity, the panel
parts are not), and consistent with the reasons recorded when the existing duals were built, but
this record measures times and allocations only: it does not attribute any gap to a specific loop,
and no profile below the call level was taken.

## Limits

- f64 only; nothing here speaks for `c64`, `c32` or `f32`.
- The tall shapes are `4096x256` and `1024x64` only. `m >> n` and `n >> m` extremes (one column,
  one row, `256x4096`) are not in the table.
- No non-Hermitian EVD (`evd_real`) cell is included: this record
  covers the families the issue's plan lists, not every decomposition in the crate.
- Reference BLAS/LAPACK is not part of the comparison: every number is faer against faer.
- The eight-thread cells are one team of eight on eight physical cores of one L3 domain; wider
  teams and the second L3 domain are not measured.
- The GEVD 2048 pairs come from the dedicated run described in Decisions, and the GEVD family is
  the least reproducible one here: its 512 cell read 419.8 ms against 384.5 ms and a ratio of 2.69
  against 2.86 between the two runs (see Decisions).
- The observing side was not pinned outside the measured L3 domain (see Decisions). Two passes
  agreeing to within a factor 1.062 (`|a - b| / b`, 146 shared cells) show repeatability, not the
  size of a systematic observer effect, so this is not a compliant baseline.
- The `LBL^T` fixture is symmetric positive definite: a diagonal sign flip is a congruence, so it
  stays positive definite and Bunch-Kaufman takes no 2x2 pivots. Those cells measure the layout
  cost of the 1x1 path only, which is not the case the factorization exists for.

## Reproduction

From a checkout of this commit, with `docs/worklogs/evidence/2026-11-11-row-major-parity/rm314.rs`
copied to `faer/examples/rm314.rs` (it is an example so that it links the crate, the workspace's
`Cargo.lock` and the pinned toolchain; it is not part of the crate and is not compiled by any
gate):

```
cp docs/worklogs/evidence/2026-11-11-row-major-parity/rm314.rs faer/examples/rm314.rs
cargo build --release --example rm314 --features perf-warn

taskset -c 0-7 target/release/examples/rm314 time  docs/worklogs/evidence/2026-11-11-row-major-parity/314-f64-time-1T.csv 1
taskset -c 0-7 target/release/examples/rm314 time  docs/worklogs/evidence/2026-11-11-row-major-parity/314-f64-time-8T.csv 8
taskset -c 0-7 target/release/examples/rm314 alloc docs/worklogs/evidence/2026-11-11-row-major-parity/314-f64-alloc-1T.csv 1
taskset -c 0-7 target/release/examples/rm314 alloc docs/worklogs/evidence/2026-11-11-row-major-parity/314-f64-alloc-8T.csv 8

rm faer/examples/rm314.rs
```

`time` writes one row per repetition (`rep` = 0..4, `micros`) and prints each cell's five samples
to stderr as it runs, so a run leaves its own log; `alloc` writes one row per cell
(`rep` = `alloc`, with `alloc_count`, `alloc_bytes`, `alloc_largest` and `matrix_bytes`) and runs
no timed repetitions, so it prints nothing per cell.
`RM314_MAX=<n>` restricts a run to the square cells up to `n` for a quick check.

A pass as the harness ships covers all 148 cells per thread count; a full one-thread pass takes
about 20 minutes, most of it the two GEVD 2048 cells (25 s and 124 s per call). The GEVD 2048 rows
in the evidence were taken with `FAMILIES` reduced to `["gevd"]` in that file -- a one-line edit --
so that the other 138 cells were not re-run (the GEVD-only run covers 10 cells per thread count).

The summary CSV is the table in machine-readable form: one row per cell and layout with
`reps, median_us, min_us, max_us` and the allocation columns, plus one `row/col` row per cell.

## What this record corrects in the issue text

The issue's "Current state (main)" section is three slices behind `2154ba17`: at this commit
partial-pivot LU factors a row-major matrix in place (`2154ba17`), full-pivot LU skips the copy for
a square row-major matrix below 128 (`828fd9e4`), QR takes the LQ dual for a sequential square
row-major matrix below 192 (`4f111fcd`), LQ stages its panels (`3cdfaf34`) and the EVD writes a
row-major `u` through a column-major stand-in (`c7470bdf`) so that it no longer warns for a
row-major eigenvector matrix -- the warnings that remain for a row-major caller are `LBL^T`'s and
SVD's, plus GEVD's for a row-major input or eigenvectors. The table above is the state of this
commit, not the state the issue describes.
