# faer #176 — evo timing handoff

This is a **timing comparison**, not an adoption claim. b=32 is provisional. The script builds b=16 and b=32 from the same candidate commit and compares them with #177 and PfaPack Fortran 1T. Its declared protocol is custom: three independent process sessions; each has at least 0.5 s priming and three warmup calls, then ten ≥0.5 s timed batches. Each sample is elapsed time divided by calls; report the median of each process's samples, then the median of those three medians. The same input matrix is used for each implementation. Fortran's matrix copy and faer's full public call are timed; scratch and team creation are outside the timed loop.

## Preconditions

- A clean checked-out candidate commit with this runbook; no concurrent benchmarks, builds or edits to measured sources. evo must remain idle for the run. This command does **not** change boost or pin other processes.
- Linux `zen5-cpu` (AMD Ryzen AI 9 HX 470), boost disabled (`/sys/devices/system/cpu/cpufreq/boost` is `0`), `taskset`, `git`, `tar`, `python3`, Rust/Cargo, `gfortran`, OpenBLAS development library and network access to `https://github.com/basnijholt/pfapack`. PfaPack is checked out at `5ad8a9b11020db5ac6acc9705408213ee67a1294`. By default `perf` and hardware-counter permission are also required.
- Existing checkout `~/projects/tensor4all/faer-benchmark` provides `benchmarks/scripts/pinned.sh` and the campaign lock. Do not edit it while a run is active. The script records its revision and guard hashes.
- At runtime the script checks that CPUs **4–11** are eight distinct physical cores in one L3, that its observer affinity excludes the entire L3 (including SMT siblings), and boost is off. `pinned.sh` checks the whole measured L3 before and after every timing cell. A rejected or failed process stops the run, with evidence retained; do not selectively retry.
- Fortran verifies OpenBLAS's effective thread budget is 1. The wrapper removes `MKLROOT` and selected build/BLAS overrides and sets `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=GOTO_NUM_THREADS=1`. Ensure no other hidden environment or program contends with the domain. The parent shell and any active agent/observer processes must also stay off the measured L3.

## Launch

If `perf stat` is unavailable, explicitly set `T4A_ALLOW_NO_PERF=1` on the launch command.
The script records the preflight result and `perf_event_paranoid` in the log and manifest,
continues with the unchanged wall-clock correctness/timing guards, and skips profiling.
`REPORT.md` marks both profile and bandwidth **not measured**; inspect `summary.csv` for timing gates.
Without the opt-in, a failed perf preflight still stops the run. A later profiling failure still stops it.

The requested handoff response supplies the exact immutable candidate SHA and the one-line worktree/`nohup` command. It checks out a new detached worktree and runs `scripts/run_acceptance.sh` with a new output directory, from observer CPUs 0–3,12–15; measured Seq/PfaPack CPU is 4, Team8 CPUs are 4–11. Do not substitute a different checkout or overwrite prior output. The output is under `~/.local/state/faer-measurements/pfaffian-181-<SHA>/`; the separate nohup log is next to that directory.

The script holds `~/projects/tensor4all/faer-benchmark/target/campaign.lock` during build, verification, timing and any available profiling. It builds base `71b25ffb7ae3ce038ec25b35c91e526be41b9a63` and the candidate in isolated archives and changes only `PANEL` to 16 in the b16 archive. It generates identical f64/c64 inputs and checks all 48 sizes/types/modes for each of base, b16 and b32 against PfaPack before timing. Seq and Team1/4/8 must agree exactly.

## Outputs and interpretation

- `manifest.json`: candidate/base/PfaPack and guard identities, binary/source hashes, host affinity, boost, compiler and library metadata.
- `correctness.json`: all 144 base/b16/b32 numerical cases; raw fixtures and manifest under `matrices/`.
- `samples.csv`, per-process `*.stdout`/`*.stderr` and rejected archives: chronological sample values, batch iteration counts and idle evidence.
- `summary.csv`, `summary.json`: process-median summaries, Fortran/Team8 and Seq/Fortran ratios, n64/n1024 Seq/base regression guards and f64 useful GFLOP/s using `n³/3`.
- `profile-{base,b16,b32}.{data,txt}`: **separate** profiled runs when perf is available; inspect for old swap costs. Profile failures stop the script after preserving timing evidence and must be reported independently.
- `REPORT.md`: status and interpretation. DRAM bandwidth is **not measured**; 12/b bytes/useful-flop is a matrix-interface payload estimate, not a DRAM counter. No AVX-512 or performance-adoption claim follows from it. With unavailable perf and explicit opt-in, a completed wall-clock run exits zero; this signals completion, not acceptance. When perf is available, the existing nonzero completion status remains because bandwidth evidence and manual acceptance review are outstanding. Read the saved `summary.csv` and `REPORT.md` in either case.

For every f64/c64 row at n=256,512,1024, the requested timing thresholds are **Seq / PfaPack-1T ≤ 1.05** and **PfaPack-1T / Team8 ≥ 4.0**. n64 and n1024 Seq/base are additional regression guards; their numeric cutoff has not been specified, so inspect the ratios rather than silently inventing one. A threshold table is not a final acceptance decision until numerical checks, guard evidence, width choice, regression guards and outstanding profiling/bandwidth evidence are reviewed.

Estimated **45–90 minutes**, excluding lock wait, slow downloads and rejected attempts; not an evo measurement. The 150 timed processes alone take at least 28.75 minutes (ten 0.5 s batches, 0.5 s priming and 6 s before/after guards each). Failures stop the run; inspect logs and retain the incomplete directory.
