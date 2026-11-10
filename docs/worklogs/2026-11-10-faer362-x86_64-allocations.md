# faer#362: the three warm-call allocations on x86_64 CI (#271)

This record includes the reproduction command and its output, which the worklog README normally
leaves out: the request for this investigation asked for them explicitly.

## Verdict

None of (a), (b) or (c). The three allocations are made by **libtest's harness thread**, not by
faer. In the CI job (`cargo llvm-cov nextest`) the instrumented test runs for about 260 s. At 60 s
libtest's `main` thread prints `test ... has been running for over 60 seconds`. That warning
allocates three times, and the test's allocator counts every allocation in the process while
`COUNTING` is set (`faer/tests/schur_eigenvectors_team_no_alloc.rs:31-63`). The allocations land in
whichever cell's counted warm call is running at t = 60 s. On the CI runner that was f64 n = 1024 F.
On this host it was f32 n = 1024 F.

The x86_64 vs aarch64 split is a **coverage vs no-coverage** split. The aarch64 numbers in the PR
were taken without instrumentation (the run finishes in well under 60 s). Uninstrumented on this
x86_64 host, every row is 0 as well (below). The column-major team matvec path from #271 allocates
nothing on a warm call on either platform, as far as these runs reach.

## The three sites

All three are on thread `main` (libtest's console loop). The test body runs on its own thread,
`schur_eigenvectors_on_a_team_allocate_nothing`. The frames are from rustc 1.99.0 (`b940084d7`,
`library/test/src`):

| # | kind | size / align | site | owner |
|---|---|---|---|---|
| 1 | `alloc` | 512 B / 8 | `Vec<TestDesc>::push` in `test::run_tests::get_timed_out_tests`, `library/test/src/lib.rs:425:27` (called from `lib.rs:480:33`) | libtest: list of tests past the warning timeout |
| 2 | `alloc` | 82 B / 1 | `format!` → `String::with_capacity` in `PrettyFormatter::write_timeout`, `library/test/src/formatters/pretty.rs:259:26` (from `console.rs:268:47`, `lib.rs:482:25`) | libtest: the warning text |
| 3 | `realloc` | 82 B / 1 (old layout) | `String::push_str` growing the same `format!` buffer, `pretty.rs:259:26` | libtest: the warning text |

No faer, threadteam, gemm or `t4a-private-gemm-x86` frame appears in any of the three backtraces.
The faer frames in the run's output belong to the panic backtrace of the assertion, not to an
allocation.

## How it was established

Host: AMD EPYC 7702P (Zen 2, AVX2 + FMA, no AVX-512, so `HostClass::X86_64Avx2`), Linux 5.4, a
4-CPU affinity mask, `CARGO_BUILD_JOBS=8`. Branch `fix/271-schur-team-scratch` at `7c61133c`.
The backtraces came from a copy of the test whose allocator calls
`std::backtrace::Backtrace::force_capture()` for each counted allocation, with a thread-local
reentrancy guard so the capture is not counted, and records the thread's name. That copy was kept
outside the tree and is not part of the branch.

1. **Uninstrumented, pinned toolchain (1.96.0), `cargo test`**: all 48 rows are 0, in both tests.
2. **Uninstrumented, stable (1.99.0), `cargo nextest run`**: all rows are 0 in both tests.
3. **Uninstrumented, test pinned to one CPU (`taskset -c 60`)**, which oversubscribes the team of
   2, n = 1024: all 0. So scheduler contention alone does not produce it.
4. **Uninstrumented, cost tables forced to `X86_64Avx512`** in faer and threadteam (a local
   env-gated patch, since reverted), n = 1024: all 0. The CI runner's dispatch decisions alone do
   not produce it either. This could not exercise the AVX-512 *kernels*, which this CPU cannot run.
5. **Instrumented (`-C instrument-coverage`, what `cargo llvm-cov` injects), stable, `cargo nextest
   run`**: **reproduced**. The output is below.
6. **Same instrumented build, backtrace copy, n = 1024**: 3 allocations at f32 n = 1024 F, with the
   three backtraces in the table above.
7. **Control: same instrumented build, allocations on thread `main` not counted**, all n: the 60 s
   warning fires as before, and **every row is 0** (output below).

The CI log shows the same sequence. `test schur_eigenvectors_on_a_team_allocate_nothing has been
running for over 60 seconds` appears in the test's captured output immediately before the
`n = 1024 F: [... ("f64", 3) ...]` row, in both failing tests.

### Reproduction (step 5)

```
cd faer
LLVM_PROFILE_FILE=<scratch>/prof/%p-%m.profraw \
RUSTFLAGS="-C instrument-coverage --cfg=coverage" RUSTUP_TOOLCHAIN=stable CARGO_BUILD_JOBS=8 \
  cargo nextest run --features=unstable --test schur_eigenvectors_team_no_alloc --no-capture
```

```
n = 64 F: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
n = 64 C: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
n = 257 F: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
n = 257 C: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
        SLOW [> 60.000s] (───) t4a-faer::schur_eigenvectors_team_no_alloc schur_eigenvectors_on_a_team_allocate_nothing
test schur_eigenvectors_on_a_team_allocate_nothing has been running for over 60 seconds
n = 1024 F: [("f32", 3), ("f64", 0), ("c32", 0), ("c64", 0)]
n = 1024 C: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
thread 'schur_eigenvectors_on_a_team_allocate_nothing' (283347) panicked at faer/tests/schur_eigenvectors_team_no_alloc.rs:253:9:
warm calls allocated: [
    "f32 n = 1024 F: 3 allocations",
]
        FAIL [ 839.498s] (1/4) t4a-faer::schur_eigenvectors_team_no_alloc schur_eigenvectors_on_a_team_allocate_nothing
```

(nextest cancelled the remaining tests after this failure, its default fail-fast.)

### Control (step 7)

```
n = 64 F: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
n = 64 C: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
n = 257 F: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
n = 257 C: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
test schur_eigenvectors_on_a_team_allocate_nothing has been running for over 60 seconds
n = 1024 F: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
n = 1024 C: [("f32", 0), ("f64", 0), ("c32", 0), ("c64", 0)]
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 844.86s
```

## Why (a), (b) and (c) are ruled out

- **(a) an uncovered per-call buffer in the column-major path**: such a buffer would be allocated on
  every warm call, instrumented or not. Runs 1 to 4 show 0. In the instrumented run it appears in
  exactly one cell, the one running at t = 60 s, and the cell changes with the host's speed
  (f64 on CI, f32 here). The backtraces contain no faer frame.
- **(b) a platform-specific packing step on x86_64**: the no-`q` case makes no full `matmul`. The
  only product in `evd_from_real_schur_imp` / `evd_from_cplx_schur_imp` is the one-column
  `matmul` in the recursive solve (`faer/src/linalg/evd/mod.rs:1037`, `:1263`, `:1356`), which goes
  to `matvec_colmajor::matvec` (`faer/src/linalg/matmul/mod.rs:1289`). Uninstrumented x86_64
  runs are 0, so the claim in the test is not aarch64-only.
- **(c) the thread-local buffer taken by someone else**: a lent-out `PARTIALS`
  (`faer/src/linalg/matmul/mod.rs:906-932`, fallback at `:930`) would allocate through `MemBuffer::new` in
  `with_partials`, on the calling thread. All three allocations are on `main`, in libtest.

## What this changes, and what it does not establish

- The changelog can say that the column-major team matvec allocates nothing on a warm call on both
  x86_64 (AVX2 class, measured here) and aarch64 (the PR's measurement).
- The test was wrong under coverage. It counted the whole process, and libtest's harness allocates
  when a test passes 60 s, so no faer change could make it pass in the CI job.
- Chosen: the test's allocator now leaves the harness thread out of the count, as the control
  does. `count` in `faer/tests/schur_eigenvectors_team_no_alloc.rs` skips the increment when
  `std::thread::current().name() == Some("main")`, and it does so only while `COUNTING` is set.
  A thread-local reentrancy guard (`IN_COUNT`) is needed because `thread::current()` can allocate
  on its first use; the guard keeps that allocation from recursing or being counted.
  `check_all` asserts that the test body is not on `main`. If it were, its own allocations would
  be dropped and the test would pass without measuring anything. libtest spawns a named thread
  per test even with `--test-threads=1`, and the assertion holds there too.
- With the filter, the instrumented run of step 5 passes both tests on this host: every row is 0,
  and the 60 s warning fired in each test (874.9 s and 879.6 s). The uninstrumented run passes
  both as well, with all rows 0.
- Not chosen: keeping the instrumented run under 60 s, for example by moving n = 1024 into its own
  test. That is fragile, because it depends on the runner's speed.
- Eight of the nine `faer/tests` files with a `#[global_allocator]` do not filter by thread. Any of
  them that runs for more than 60 s under `llvm-cov` can fail the same way. That was not checked
  test by test.
- Not established here: the with-`q` case was not run under instrumentation with backtraces on this
  host. CI failed it at the same cell with the same warning line before the row, so the same
  mechanism is inferred, not observed. The AVX-512 kernels (`InstrSet::Avx512` in
  `t4a-private-gemm-x86`, chosen at `faer/src/linalg/matmul/mod.rs:1465` and `:1752`) were not
  executed, because this CPU has no AVX-512. The CI runner's CPU model is not in its log.
