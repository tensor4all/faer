# faer#173: the Pfaffian Team path at the widest width this quiet host has, and AVX-512 execution

Follow-up to [#173](https://github.com/tensor4all/faer/issues/173), the hardware
homework left by [#171](https://github.com/tensor4all/faer/pull/174): the
Pfaffian Team path and the AVX-512 build had only been measured on an EPYC 7702P,
which has no AVX-512 at all, so `x86-v4` compiling there was never evidence that
AVX-512 execution was *correct*, let alone useful.

This record answers that on **evo**, the quiet Zen 5 host: an AMD Ryzen AI 9 HX
470 w/ Radeon 890M, 24 online hardware threads, boost off, all governors
`performance`. Measured tree: this fork at `2c2bfad1`, both build configurations,
widths 1/4/8/16, `f64` and `c64`, n = 64/128/256/512/1024. **32 threads were not
measured because they do not exist on this host**, and the widest width the
protocol allows here is 16 — both stated with numbers below rather than rounded
off.

The work is a validation of an existing code path, not a change: no library
source moved for it. Evidence: `evidence/2026-11-11-pfaffian-quiet-host.tar.gz`
(topology, the binary hashes in `binaries.sha256`, the harness sources, every
run's stdout and guard log, and the build and test logs; the executable
binaries, the matrix inputs and `pinned.sh` are not in the archive).

## Decisions

- **Widths 4/8/16, and the width cap stated instead of guessed.** 8 cores carry
  two threads each here. A width above 16 needs both L3 domains, which leaves no
  CPU outside the measured set for the observer, so 24 is not measurable under
  the quiet-host protocol and 32 is not a width this host has. The widest width
  measured is therefore 16, which is 8 physical cores with both SMT siblings.
- **The measured set is the 8-core domain, not the 4-core one**, so that width 16
  exists at all. That choice costs cache: the two domains are *asymmetric*, and
  the 8-core one has the smaller L3 (below).
- **`x86-v4` is the crate's own feature**, not `RUSTFLAGS`: no `target-cpu=native`
  appears anywhere in this campaign. The lane each binary actually selects is
  read back from the binary and from every run rather than inferred from the
  flags.
- **One-thread Fortran PFAPACK in the same window.** The Team numbers are only
  interesting against a 1-thread implementation, and reusing a two-day-old
  baseline would have made every ratio a cross-window one.

## Host, measured before the first run

| fact | value |
|---|---|
| CPU model | AMD Ryzen AI 9 HX 470 w/ Radeon 890M (Zen 5), kernel 7.0.0-38-generic |
| online CPUs | 24 (`0-23`), 12 cores, 2 threads/core, siblings `i` and `i+12` |
| `nproc` seen by the campaign | 8 — the agent is affinity-pinned to the observer set, not a cgroup limit (`cpuset.cpus.effective` = `0-23`) |
| L3 | two instances, **asymmetric**: `0-3,12-15` = 4 cores / 8 threads / **16 MiB**, `4-11,16-23` = 8 cores / 16 threads / **8 MiB** |
| reserved CPUs | none: `isolated` empty, no `isolcpus`, no `cpu.max` quota |
| frequency | `boost` = 0, every governor `performance`, max 2000 MHz |
| AVX-512 in `/proc/cpuinfo` | `avx512f dq ifma cd bw vl vbmi vbmi2 vnni bitalg vpopcntdq bf16 vp2intersect` |

`lscpu`'s summary line prints 24 MiB of L3 in two instances, which reads like two
equal 12 MiB halves; the per-domain `size` file says 16 MiB and 8 MiB. The
consequence is real: at n = 1024 the `f64` matrix (8 MiB) exactly fills the
measured domain's cache and `c64` (16 MiB) is twice it, so the large-n Team
numbers are *not* cache-resident, while the small-n ones are.

## Selected SIMD lane, per build

| build | runtime lane | f64 lanes | c64 lanes |
|---|---|---:|---:|
| default | `pulp::x86::v3::V3` | 4 | 2 |
| `x86-v4` | `pulp::x86::v4::V4` | 8 | 4 |

Read from `pulp::Arch::dispatch` inside the harness (printed by every run) and
alongside the crate test `runtime_selected_lane`: all 80 default runs reported
`V3` and all 80 `x86-v4` runs reported `V4`, with no exception. `cargo tree -e
features -i pulp` shows `pulp feature "x86-v4"` present only in the second build.

`x86-v4` remains a non-default feature for the reason in `faer/Cargo.toml`: with
`pulp/x86-v4` the library returned non-finite factors in full-pivot LU / QZ /
matrix-free SVD on some host, tracked by
[#86](https://github.com/tensor4all/faer/issues/86). Nothing here changes that;
it is why the AVX-512 numbers below are a measurement of the Pfaffian path, not a
recommendation to enable the feature.

## Times, milliseconds: median of the two session medians, each session a median of ten single-call samples

Copy included, scratch allocated outside the clock, team held across the calls.
Each cell ran through the harness's `pinned.sh` (3 s idle window, ≤ 5 % busy on
the measured CPUs and their SMT siblings, before and after the run); 180 runs,
**0 guard rejections**, largest session-to-session spread 2.80 %. Observer
pinned to the other L3 domain (`0-3,12-15`) for the whole campaign; one
measurement at a time under the global bench lock. The Fortran arm asserts its
BLAS thread count is 1.

**default (`V3`, 4 `f64` / 2 `c64` lanes)**

| n | dtype | Fortran 1T | Seq | Team4 | Team8 | Team16 | Seq/F | T8/Seq | T16/Seq | F/T8 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | f64 | 0.0175 | 0.0204 | 0.0177 | 0.0173 | 0.0239 | 1.168 | 1.18 | 0.85 | 1.01 |
| 64 | c64 | 0.0659 | 0.0442 | 0.0366 | 0.0356 | 0.0461 | 0.671 | 1.24 | 0.96 | 1.85 |
| 128 | f64 | 0.1173 | 0.1196 | 0.0818 | 0.0731 | 0.0942 | 1.020 | 1.64 | 1.27 | 1.60 |
| 128 | c64 | 0.5545 | 0.2658 | 0.1738 | 0.1515 | 0.1879 | 0.479 | 1.75 | 1.41 | 3.66 |
| 256 | f64 | 1.2862 | 0.8269 | 0.3710 | 0.2888 | 0.3379 | 0.643 | 2.86 | 2.45 | 4.45 |
| 256 | c64 | 3.3791 | 1.7901 | 0.8084 | 0.6259 | 0.7324 | 0.530 | 2.86 | 2.44 | 5.40 |
| 512 | f64 | 10.0452 | 6.2009 | 2.2664 | 1.5297 | 1.5095 | 0.617 | 4.05 | 4.11 | 6.57 |
| 512 | c64 | 26.7524 | 13.2286 | 4.8706 | 3.2218 | 3.3308 | 0.494 | 4.11 | 3.97 | 8.30 |
| 1024 | f64 | 63.4826 | 52.1024 | 15.3337 | 8.7156 | 8.1720 | 0.821 | 5.98 | 6.38 | 7.28 |
| 1024 | c64 | 222.1802 | 119.3995 | 35.9379 | 20.6856 | 19.7072 | 0.537 | 5.77 | 6.06 | 10.74 |

**`x86-v4` (`V4`, 8 `f64` / 4 `c64` lanes)**

| n | dtype | Fortran 1T | Seq | Team4 | Team8 | Team16 | Seq/F | T8/Seq | T16/Seq | F/T8 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | f64 | 0.0175 | 0.0197 | 0.0169 | 0.0169 | 0.0231 | 1.125 | 1.17 | 0.85 | 1.04 |
| 64 | c64 | 0.0659 | 0.0411 | 0.0336 | 0.0326 | 0.0423 | 0.624 | 1.26 | 0.97 | 2.02 |
| 128 | f64 | 0.1173 | 0.1194 | 0.0813 | 0.0739 | 0.0965 | 1.018 | 1.62 | 1.24 | 1.59 |
| 128 | c64 | 0.5545 | 0.2480 | 0.1613 | 0.1389 | 0.1749 | 0.447 | 1.78 | 1.42 | 3.99 |
| 256 | f64 | 1.2862 | 0.8254 | 0.3696 | 0.2896 | 0.3523 | 0.642 | 2.85 | 2.34 | 4.44 |
| 256 | c64 | 3.3791 | 1.7364 | 0.7575 | 0.5793 | 0.6696 | 0.514 | 3.00 | 2.59 | 5.83 |
| 512 | f64 | 10.0452 | 6.1453 | 2.2827 | 1.5157 | 1.6012 | 0.612 | 4.05 | 3.84 | 6.63 |
| 512 | c64 | 26.7524 | 12.9108 | 4.6881 | 3.0359 | 3.0733 | 0.483 | 4.25 | 4.20 | 8.81 |
| 1024 | f64 | 63.4826 | 51.6846 | 15.2908 | 9.0024 | 8.5951 | 0.814 | 5.74 | 6.01 | 7.05 |
| 1024 | c64 | 222.1802 | 118.4343 | 35.1445 | 19.9706 | 18.6064 | 0.533 | 5.93 | 6.37 | 11.13 |

**`x86-v4` against default, same cell — ratio > 1 means AVX-512 is faster**

| n | dtype | Seq | Team8 | Team16 |
|---:|---|---:|---:|---:|
| 64 | f64 | 1.039 | 1.027 | 1.033 |
| 64 | c64 | 1.076 | 1.093 | 1.089 |
| 128 | f64 | 1.002 | 0.989 | 0.976 |
| 128 | c64 | 1.072 | 1.091 | 1.074 |
| 256 | f64 | 1.002 | 0.997 | 0.959 |
| 256 | c64 | 1.031 | 1.081 | 1.094 |
| 512 | f64 | 1.009 | 1.009 | 0.943 |
| 512 | c64 | 1.025 | 1.061 | 1.084 |
| 1024 | f64 | 1.008 | 0.968 | 0.951 |
| 1024 | c64 | 1.008 | 1.036 | 1.059 |

**Team16 against Team8 — the same 8 cores, second thread each (ratio > 1 means
the SMT twin helps)**

| n | dtype | default | `x86-v4` |
|---:|---|---:|---:|
| 64 | f64 | 0.725 | 0.729 |
| 64 | c64 | 0.772 | 0.770 |
| 128 | f64 | 0.776 | 0.766 |
| 128 | c64 | 0.806 | 0.794 |
| 256 | f64 | 0.855 | 0.822 |
| 256 | c64 | 0.855 | 0.865 |
| 512 | f64 | 1.013 | 0.947 |
| 512 | c64 | 0.967 | 0.988 |
| 1024 | f64 | 1.067 | 1.047 |
| 1024 | c64 | 1.050 | 1.073 |

## Verification conclusions and constraints

- **AVX-512 execution is confirmed, and it is a small, size-dependent win.** The
  `V4` lane runs and returns the right answer (below); against the `V3` build it
  buys 2.5–7.6 % on `c64` Seq and 6.1–9.4 % on the `c64` Team cells from n = 64 to
  512, and 3.6–5.9 % on the `c64` Team cells at n = 1024, while `f64` is neutral
  to −6 %: +3.9 % at n = 64 Seq and slower in seven of the eight Team cells
  (0.989 at 128 Team8, 0.976 at 128 Team16, 0.997 at 256 Team8, 0.959 at 256
  Team16, 0.943 at 512 Team16, 0.968 at 1024 Team8, 0.951 at 1024 Team16).
  Doubling the register width does not move the factorization much. **The reason
  is a hypothesis, not a measurement**: the tables below record elapsed times
  only, and no performance counters or DRAM-bandwidth measurements were taken
  (see the limits), so they do not separate cache capacity from dependency
  latency, instruction throughput or dispatch. What is measured is the effect
  itself — the direction and size in each cell. This is the execution evidence
  that `2026-10-31-pfaffian.md` recorded as unperformed.
- **The Team path scales to the width the host has.** Team8 over Seq is
  4.05 / 4.11 (`f64`/`c64`) at n = 512 and 5.98 / 5.77 at n = 1024 (default), so
  51 % and 72–75 % parallel efficiency on 8 threads. Team4 alone is 2.2x at
  n = 256 and 2.7x at 512. At n = 64 and 128 the team is nearly free relative to
  Seq (1.18–1.75x) because Seq is only 0.02–0.27 ms there.
- **Team16 is not extra compute.** Sixteen threads are the SMT siblings of the
  same eight cores. It *loses* 16–38 % to Team8 at n ≤ 256, ties at 512 (0.947 –
  1.013), and wins 4.7–7.3 % at n = 1024 only, where the memory-bound cells have
  enough idle issue slots for a second thread. So on this host the useful width
  is 8; 16 is a 5 % large-n effect, not a second machine.
- **The #176 4x target still misses at n = 256 and meets at n ≥ 512.**
  `best 1T / Team8` with the best measured 1-thread arm (Fortran PFAPACK or
  faer's own Seq, whichever is faster in that cell) is 1.01–1.26 at n = 64,
  1.59–1.78 at 128, **2.85–3.00 at 256**, 4.05–4.25 at 512 and 5.74–5.98 at
  1024. At n = 256 faer's own Seq is the fastest 1-thread implementation, so the
  Team8 factor there is the parallel factor, not a serial-deficit artefact.
- **The small-size serial criterion is still missed at n = 64 `f64`**:
  Seq/Fortran is 1.168 (default) and 1.125 (`x86-v4`), against the ≤ 1.05
  criterion; n = 128 `f64` passes at 1.020 / 1.018 and every other cell is
  0.45–0.82. Unchanged from the pre-#173 records — small sizes are dispatch- and
  call-bound, and neither the lane width nor the width budget addresses that.
- **Correctness, both builds:** `cargo test --release -p t4a-faer --test
  pfaffian` is **12 passed, 0 failed** in each configuration, including
  `runtime_selected_lane` (which prints the lane above), the extended-precision
  and extreme-scale cases and the team-agreement/row-dispatch cases; the held-team
  no-allocation test `tests/no_step_alloc.rs` is **1 passed** in each. All four
  runs are in the evidence.
- **Cross-build agreement is exact.** Over the 10 (n, dtype) cells × 4 widths
  (40 input/width combinations per build), the `V4` build returns a Pfaffian
  identical to the `V3` build in every one of the 18 printed significant digits
  (max relative difference 0.0 on the printed values, not merely below
  tolerance), and both agree with the Fortran PFAPACK reference inside 5.8e-13
  relative.
- **The one named `#86` symptom that also runs here passes**: the full-pivot unit
  test `linalg::lu::full_pivoting::factor::tests::test_gh238` passes in the
  `x86-v4` build on this host (and in the default build). That is one test on one
  host; it does not close #86 for QZ or the matrix-free SVD.

## What remains unverified

- **Widths above 16 on this host.** 24 threads need both L3 domains, and then no
  CPU is left for the observer, so a 24-thread cell is unmeasurable here rather
  than unmeasured; 32 threads do not exist (24 online, 12 cores). The claim this
  record makes is about widths 1–16 on 8 cores.
- **A cross-domain team.** The two-domain stall recorded in
  `2026-11-05-cross-ccd-team-stall.md` was not revisited; nothing here says how a
  width of 17–24 would behave on this host if it could be observed.
- **The 8 MiB measured domain is the smaller cache.** A run on the 16 MiB
  4-core domain is a different experiment (width cap 8, larger cache) and was not
  made, so the large-n `f64` numbers are "out of cache" rather than "in the best
  available cache".
- **#86 itself.** The `x86-v4` feature still returns non-finite factors in other
  paths on some host; the Pfaffian path being correct under it does not make the
  feature shippable.
- No upstream (non-fork) faer revision was measured. No `perf` counters, no DRAM
  bandwidth (`perf_event_paranoid`), no GPU, no Raspberry Pi, no aarch64.
