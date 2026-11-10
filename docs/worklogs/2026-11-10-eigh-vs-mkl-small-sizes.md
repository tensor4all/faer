# `eigh` at n = 64..128 against MKL: which stage carries the loss

Issue: [tensor4all/faer#354](https://github.com/tensor4all/faer/issues/354).

Measured revision: `1ac38c6d7bab168c4478c9e2379a5bd9e69a2272` (fork `main`, clean tree), built
`--no-default-features --features std,rayon,threadteam` (the campaign's feature set: AVX2
kernels, `x86-v4` off). MKL 2026.1 (`libmkl_rt.so.3`, Intel OpenMP), driver links the fork by
path and MKL in one process.

Every number below is recomputed from
[`evidence/2026-11-10-eigh-raw-samples.csv`](evidence/2026-11-10-eigh-raw-samples.csv) - 420
cell rows, each with its `samples_ms` column - by the script at the end of this record
(reproduction A/B/C/layout). Table A is the **pooled median of all 15 samples** of a cell (3
sessions x 5 reps); tables B and C are the median of the 5 samples of one session, in which that
cell ran in its own process.

## Decisions

- **The loss is one stage, not the whole call.** `self_adjoint_evd` splits into the
  tridiagonalization (`tridiag::tridiag_in_place`), the tridiagonal eigenproblem with vectors
  (`tridiagonal_self_adjoint_evd`) and the block-Householder apply onto `U`. At n = 64..128 the
  middle one is **76% of the f64 call** (53-57% for c64, where the complex apply is itself
  large) and is the only stage slower than MKL's counterpart: f64 n = 128 at 1T is 1.772 ms
  against MKL's `?stedc` 0.915 ms (**1.94x**), and c64 is 1.754 against 0.920 (**1.91x**).
  Stage 1 is at parity with MKL's `?sytrd` there (0.335 vs 0.338 ms) and stage 3 costs 0.199 ms.
- **The rest of MKL's `?syevd` is unattributed, not measured.** `?syevd` exceeds its own
  `?sytrd` plus `?stedc` by 0.084 / 0.195 / 0.544 / 3.460 / 22.296 ms at f64 n = 64..512 1T
  (table B, `MKL remainder`). That time contains whatever else the driver does - the reflector
  apply, the copy, the scaling - but this record did not probe which, so it is **not** called
  the reflector apply and it is not compared with faer's stage 3. What the arithmetic does
  establish is the direction of the whole-call gap: faer's stage 2 is 0.857 ms slower than
  MKL's `?stedc` at f64 n = 128 1T, yet the whole call is only 0.525 ms slower (2.321 vs
  1.796, 1.29x), because MKL's unattributed remainder (0.544 ms) is larger than faer's stage 3
  (0.199 ms).
- **Below n = 128 the tridiagonal eigenproblem has no parallel route at all**, which is why faer
  gains nothing from 8 threads in that band while MKL does. `qr_algorithm` takes no `Par`; the
  entry gate is `n < params.recursion_threshold` (default 128), and at exactly n = 128
  `divide_and_conquer` is entered and immediately falls back to QR because its first test is
  `n <= qr_fallback_threshold` with the same 128. The fork's own gate trace (`cost::trace`,
  built with `par-trace`) shows no `tridiag_evd` decision being asked at all at n = 128, so the
  D&C fork is not declined - it is never reached. MKL's `?syevd` is expected to take its D&C route
  at these sizes - its own `?stedc` scales 1.22-1.50x from 1T to 8T at n = 64..128 - but the
  driver internal branch was not traced here, since table B times `?sytrd` and `?stedc` as
  separate calls.
- **The measured consequence of that threshold, not a hypothesis:** `recursion_threshold = 16`
  (D&C at every size, the only knob, no code change) takes 16-32% off the f64 call and 13-26% off
  the c64 call at 8T for n = 64..128, turns the complex cells from a small loss into a win, and is
  worth 8% at f64 n = 256, 3% at c64 n = 256 and 1% at n = 512. At 1T it is neutral at n = 64 and 96 and 9% (f64) /
  4% (c64) better at n = 128. **Table C is the measurement**; the per-cell numbers are not repeated
  here. So the default is not wrong at 1T; it is wrong at 8T, where D&C's merge is cheaper than a
  serial QR of the same tridiagonal, and slightly wrong at 8T above the band as well.
- **Hypothesis (2) (thread start-up) is real but an order of magnitude too small to explain
  2x.** f64 stage 1 at 8T is 0.71x (n = 96) and 0.79x (n = 128) of its own 1T time: the parallel
  gate opens a region that does not pay, costing 20-29% of a stage that is 15% of the call.
  MKL's `?sytrd` loses the same way (0.71x, 0.77x), so this is not a fork-specific defect.
- **Hypothesis (3) (row-major vs column-major) is refuted at this size, within measured
  bounds.** faer's c and f cells differ by at most **0.68% at 1T and 1.70% at 8T** (worst cell
  f64 n = 256 both times; median 0.28% / 0.49%); MKL's differ by up to 2.62% at 1T (f64 n = 96,
  c 0.898 vs f 0.875) and 5.03% at 8T (f64 n = 64, c 0.336 vs f 0.320). Those are the bounds over
  the 20 cells measured; nothing outside them is claimed. faer's order difference is two orders
  of magnitude below the 77-94% loss at 8T, and the mechanism is in the source:
  `self_adjoint_evd` copies the lower triangle into a contiguous temporary
  (`trid.copy_from_triangular_lower(A)`) before any kernel sees it, so the caller's order never
  reaches a kernel.
- **Not implemented.** The knob is worth 13-32% at 8T, but after it faer is still 1.3-1.5x behind
  MKL on the f64 cells at 8T (table C, thr=16 column), so a threshold change alone does not reach parity
  and the remaining question is a design one (a parallel QR leaf, a threaded D&C merge, or MKL's
  two-stage reduction), not a tuning one. The options, their risk and their expected effect are
  below for the maintainer's decision.

## Verification conclusions and constraints

- What the tables establish: (A) the faer/MKL ratio per dtype, order, size and thread budget;
  (B) the stage split, with MKL's own `?sytrd`/`?stedc` measured on the *same* tridiagonal, the
  sum of faer's three stages closing with `self_adjoint_evd` within 2.6% at 1T and 4.6% at 8T
  except f64 n = 64 at 8T (+11.0%) and f64 n = 256 at 8T (-8.4%). The residual is **consistent
  with** the lower-triangle copy, the team dispatch and each region's fill/drain, none of which a
  stage row contains — but the tables do not isolate those, so the residual is not attributed to
  them. At f64 n = 256 8T the three rows already sum to more than the whole call (7.208 against
  6.649 ms), where no extra work can explain a negative remainder: that difference is unexplained,
  and each stage row takes its own timestamps;
  (C) the threshold
  knob; the samples behind them are in the CSV.
- **What remains unverified.**
  - Whether a parallel QR leaf (deflated blocks as tasks) beats lowering the threshold. The
    threshold knob is measured; a threaded QR is not, and it is the route that can move the 1T
    gap at n = 64/96 too, where the knob is neutral. At n = 128 the knob already gives 9% (f64)
    and 4% (c64) at 1T, so it is not the only 1T improvement available.
  - How much of MKL's `?stedc` advantage is its merge implementation rather than its parallelism.
    At 1T MKL is 1.55-1.94x faster than faer's QR at n = 64..128 (f64), so part of the gap
    survives any parallelisation of faer's current algorithm.
  - What MKL's `?syevd` does with the 0.084-22.296 ms its `?sytrd` and `?stedc` do not account
    for (table B). Not probed, so not attributed.
  - The complex path's trailing work (the `scale` normalisation and the complex combine after the
    real tridiagonal solve) is not a separate row: table B's c64 stage 2 is the real solve, which
    is what `self_adjoint_evd` runs for a complex input; the stage sum closes within 2.6% at 1T,
    so that trailing work is small at these sizes but it is not attributed.
- **Measurement traps found on the way, so the next session does not lose the time.**
  - `mkl_set_num_threads` segfaults in this MKL 2026.1 install before any computational call
    (reproduced under gdb, `mkl_set_num_threads_` in `libmkl_intel_lp64`); the budget has to be
    set through `omp_set_num_threads` and read back with `mkl_get_max_threads`. The campaign's
    MKL arm already carries the same note.
  - `LAPACKE_dsytrd` measured *after* a GEMM-heavy faer cell in the same process reads 4.4x
    slower (0.238 vs 0.054 ms at n = 64) and stably so; measured first in its own process it is
    0.054 ms, which is what closes MKL's `?syevd` total. Every cell in table B therefore runs in
    its own process. The mechanism is unexplained.
  - A stage probe handed a plain `Vec`-backed buffer reads 4.0-4.7x high (stage 1 at n = 128:
    1.62 vs 0.34 ms) and its sum does not close with the pipeline. It is not the stride type (the
    same buffer viewed with dynamic strides is fast) and not the 64-byte alignment of the base
    address (an offset Vec reads the same). The library's own allocation closes the sum, so table
    B is Mat-backed. Unexplained, and left as a lead: if it is not the allocation but something
    about a caller-owned `Vec`, every in-place kernel in the library has the same cliff.
- Host: AMD Ryzen AI 9 HX 470 (Zen5 mobile APU), 24 logical / 12 physical, SMT on, CPU boost
  off, governor `performance`, kernel 7.0.0-38-generic, rustc 1.99.0. Measured CPUs `0` (1T) and
  `0-3,12-15` (8T, the 16 MiB L3 domain and its SMT siblings); the observing side pinned to
  `4-11,16-23` (the other L3 domain) for the whole run. No `RAYON_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OMP_*` or `*_BLAS*` variable set; the probe sets the rayon/threadteam team width per cell and
  asserts MKL's read-back equals it. This is the `zen5-cpu` development host, the same machine the
  published pages are recorded on, not the `zen2-64-smt-off` measurement host.

### If we do implement: the options and what each buys

1. **Lower the D&C entry threshold when the team is wider than one** (make
   `SelfAdjointEvdParams::recursion_threshold` a function of `par.degree()`). Measured effect:
   13-32% off the call at 8T for n = 64..128, 8% at f64 n = 256, 3% at c64 n = 256, 1% at
   n = 512, and -5% to +9% at 1T.
   Those numbers belong to `recursion_threshold = 16` **only**; relaxing the fallback from
   `n <= qr_fallback_threshold` to `n <` is a different edit, was not run, and no number is claimed
   for it. Risk: the D&C merge's accuracy path differs from QR's; the fork's own
   `lapack_self_adjoint_evd_*` and spectrum-accuracy tests already run both thresholds (the
   QR/D&C choice is a parameter there), so the regression surface is those fixtures, not new
   code. Cost: one gate, no new algorithm.
2. **Parallelise the QR leaf** (deflated blocks as tasks inside `qr_algorithm`). This changes the
   arithmetic order rather than the schedule. It is not the only one that
   moves 1T: the threshold knob already does (table C, f64 n = 128 at 1T, 2.318 → 2.105 ms), it
   just stops at D&C's merge cost. What this option adds is the use of n = 64/96 with a threshold
   without paying that merge. Risk: the eigenvector chase is a sequential recurrence over
   rotations; block-parallel variants change the arithmetic order, so it needs its own accuracy
   and residual tests. Unmeasured.
3. **Leave it.** The band is 1.77-1.94x behind MKL at 8T (f64) and 1.29-1.38x at 1T; faer already
   wins f64 n = 512 at 1T (0.84x) but loses it at 8T (1.22x), and wins complex at 1T (0.76-0.83
   over n = 64..128, 0.63 at n = 512). If the target is the campaign's win rate rather than this
   band, option 1 is the cheapest 13-32% and option 3 is defensible.

## Evidence

All times in ms, median of the samples the row names. The driver is a standalone probe outside
both repositories (the fork carries no benchmark harness): it links this checkout by path and
`libmkl_rt` in one process, sizes every workspace with the library's own `*_scratch`, primes
each cell for 250 ms and takes 5 samples, and restores every input outside the clock.

### A. `eigh` faer (owning `self_adjoint_eigen`) vs MKL `?syevd`, ms

Pooled median of the 15 samples (3 sessions x 5 reps) of each cell, prime 250 ms per cell,
CPU boost off; `docs/worklogs/evidence/2026-11-10-eigh-raw-samples.csv` filtered to the cell
(reproduction A).

| dtype | order | n | faer 1T | MKL 1T | faer/MKL | faer 8T | MKL 8T | faer/MKL | faer 1T->8T | MKL 1T->8T |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | c | 64 | 0.508 | 0.394 | 1.29 | 0.567 | 0.336 | 1.69 | 0.90x | 1.17x |
| f64 | c | 96 | 1.203 | 0.898 | 1.34 | 1.273 | 0.700 | 1.82 | 0.94x | 1.28x |
| f64 | c | 128 | 2.316 | 1.841 | 1.26 | 2.387 | 1.263 | 1.89 | 0.97x | 1.46x |
| f64 | c | 256 | 9.930 | 9.891 | 1.00 | 6.661 | 5.336 | 1.25 | 1.49x | 1.85x |
| f64 | c | 512 | 49.123 | 60.196 | 0.82 | 25.076 | 21.396 | 1.17 | 1.96x | 2.81x |
| f64 | f | 64 | 0.506 | 0.387 | 1.31 | 0.567 | 0.320 | 1.77 | 0.89x | 1.21x |
| f64 | f | 96 | 1.203 | 0.875 | 1.38 | 1.273 | 0.687 | 1.85 | 0.95x | 1.27x |
| f64 | f | 128 | 2.321 | 1.796 | 1.29 | 2.397 | 1.233 | 1.94 | 0.97x | 1.46x |
| f64 | f | 256 | 9.998 | 9.755 | 1.02 | 6.776 | 5.141 | 1.32 | 1.48x | 1.90x |
| f64 | f | 512 | 49.413 | 59.158 | 0.84 | 25.301 | 20.773 | 1.22 | 1.95x | 2.85x |
| c64 | c | 64 | 0.639 | 0.801 | 0.80 | 0.655 | 0.707 | 0.93 | 0.98x | 1.13x |
| c64 | c | 96 | 1.660 | 2.022 | 0.82 | 1.503 | 1.400 | 1.07 | 1.10x | 1.44x |
| c64 | c | 128 | 3.302 | 4.376 | 0.75 | 2.742 | 2.308 | 1.19 | 1.20x | 1.90x |
| c64 | c | 256 | 17.520 | 26.212 | 0.67 | 9.208 | 10.752 | 0.86 | 1.90x | 2.44x |
| c64 | c | 512 | 105.199 | 169.258 | 0.62 | 41.305 | 51.806 | 0.80 | 2.55x | 3.27x |
| c64 | f | 64 | 0.640 | 0.791 | 0.81 | 0.657 | 0.687 | 0.96 | 0.98x | 1.15x |
| c64 | f | 96 | 1.658 | 1.997 | 0.83 | 1.499 | 1.391 | 1.08 | 1.11x | 1.44x |
| c64 | f | 128 | 3.302 | 4.324 | 0.76 | 2.758 | 2.261 | 1.22 | 1.20x | 1.91x |
| c64 | f | 256 | 17.580 | 25.926 | 0.68 | 9.291 | 10.541 | 0.88 | 1.89x | 2.46x |
| c64 | f | 512 | 105.580 | 167.972 | 0.63 | 41.803 | 51.011 | 0.82 | 2.53x | 3.29x |


### B. stage split, ms (one cell per process, order f)

faer stages are `tridiag::tridiag_in_place`, `tridiagonal_self_adjoint_evd` (with vectors),
`apply_block_householder_sequence_on_the_left_in_place_with_conj`. MKL stages are `?sytrd`/
`?hetrd` and `?stedc` (`compz='I'`) on the *same* tridiagonal faer's stage 1 produced.
`faer self_adjoint_evd` is the in-place route, `MKL ?syevd` is table A's column.
`MKL remainder` is `?syevd` minus `?sytrd` minus `?stedc`: **not** attributed to a stage.
Reproduction B.

**f64**

| stage | par | n=64 | n=96 | n=128 | n=256 | n=512 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| faer stage 1 (tridiagonalize) | 1T | 0.080 | 0.182 | 0.335 | 1.855 | 11.566 |
| faer stage 1 (tridiagonalize) | 8T | 0.079 | 0.254 | 0.423 | 1.592 | 6.475 |
| MKL ?sytrd | 1T | 0.053 | 0.154 | 0.338 | 2.623 | 21.317 |
| MKL ?sytrd | 8T | 0.054 | 0.216 | 0.439 | 1.901 | 9.367 |
| faer stage 2 (tridiag EVD + vectors) | 1T | 0.387 | 0.915 | 1.772 | 6.652 | 27.090 |
| faer stage 2 (tridiag EVD + vectors) | 8T | 0.386 | 0.915 | 1.771 | 5.196 | 15.919 |
| MKL ?stedc | 1T | 0.249 | 0.525 | 0.915 | 3.672 | 15.544 |
| MKL ?stedc | 8T | 0.205 | 0.404 | 0.610 | 2.425 | 6.982 |
| faer stage 3 (reflector apply) | 1T | 0.036 | 0.096 | 0.199 | 1.335 | 9.779 |
| faer stage 3 (reflector apply) | 8T | 0.037 | 0.066 | 0.109 | 0.419 | 2.796 |
| sum of faer's three stages | 1T | 0.503 | 1.193 | 2.305 | 9.842 | 48.435 |
| sum of faer's three stages | 8T | 0.502 | 1.234 | 2.303 | 7.208 | 25.190 |
| faer `self_adjoint_evd` | 1T | 0.508 | 1.200 | 2.318 | 9.884 | 49.282 |
| faer `self_adjoint_evd` | 8T | 0.565 | 1.263 | 2.391 | 6.649 | 25.814 |
| MKL `?syevd` | 1T | 0.387 | 0.875 | 1.796 | 9.755 | 59.158 |
| MKL `?syevd` | 8T | 0.320 | 0.687 | 1.233 | 5.141 | 20.773 |
| MKL remainder (`?syevd` - `?sytrd` - `?stedc`) | 1T | 0.084 | 0.195 | 0.544 | 3.460 | 22.296 |
| MKL remainder (`?syevd` - `?sytrd` - `?stedc`) | 8T | 0.062 | 0.067 | 0.183 | 0.816 | 4.424 |
| faer residual (`self_adjoint_evd` - 3 stages) | 1T | 0.005 | 0.007 | 0.013 | 0.042 | 0.848 |
| faer residual (`self_adjoint_evd` - 3 stages) | 8T | 0.062 | 0.029 | 0.089 | -0.558 | 0.624 |

**c64**

| stage | par | n=64 | n=96 | n=128 | n=256 | n=512 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| faer stage 1 (tridiagonalize) | 1T | 0.128 | 0.344 | 0.728 | 5.022 | 36.626 |
| faer stage 1 (tridiagonalize) | 8T | 0.175 | 0.359 | 0.610 | 2.715 | 13.782 |
| MKL ?sytrd | 1T | 0.247 | 0.732 | 1.597 | 11.548 | 88.560 |
| MKL ?sytrd | 8T | 0.316 | 0.639 | 1.112 | 5.632 | 29.996 |
| faer stage 2 (tridiag EVD + vectors) | 1T | 0.360 | 0.938 | 1.754 | 6.580 | 26.831 |
| faer stage 2 (tridiag EVD + vectors) | 8T | 0.362 | 0.938 | 1.759 | 4.625 | 16.038 |
| MKL ?stedc | 1T | 0.251 | 0.532 | 0.920 | 3.713 | 15.787 |
| MKL ?stedc | 8T | 0.207 | 0.409 | 0.693 | 2.270 | 6.246 |
| faer stage 3 (reflector apply) | 1T | 0.142 | 0.354 | 0.774 | 5.507 | 39.426 |
| faer stage 3 (reflector apply) | 8T | 0.087 | 0.151 | 0.283 | 1.487 | 10.181 |
| sum of faer's three stages | 1T | 0.630 | 1.636 | 3.257 | 17.109 | 102.883 |
| sum of faer's three stages | 8T | 0.624 | 1.448 | 2.651 | 8.827 | 40.002 |
| faer `self_adjoint_evd` | 1T | 0.639 | 1.653 | 3.287 | 17.570 | 105.428 |
| faer `self_adjoint_evd` | 8T | 0.654 | 1.494 | 2.780 | 9.222 | 41.147 |
| MKL `?syevd` | 1T | 0.791 | 1.997 | 4.324 | 25.926 | 167.972 |
| MKL `?syevd` | 8T | 0.687 | 1.391 | 2.261 | 10.541 | 51.011 |
| MKL remainder (`?syevd` - `?sytrd` - `?stedc`) | 1T | 0.292 | 0.733 | 1.807 | 10.665 | 63.625 |
| MKL remainder (`?syevd` - `?sytrd` - `?stedc`) | 8T | 0.163 | 0.344 | 0.457 | 2.639 | 14.769 |
| faer residual (`self_adjoint_evd` - 3 stages) | 1T | 0.009 | 0.017 | 0.030 | 0.462 | 2.544 |
| faer residual (`self_adjoint_evd` - 3 stages) | 8T | 0.030 | 0.046 | 0.128 | 0.396 | 1.146 |

### C. the `recursion_threshold` knob: `self_adjoint_evd` in place, ms

`default` is 128 (what `SelfAdjointEvdParams::auto` gives), `16` is the knob
`recursion_threshold = 16`, `QR forced` is `usize::MAX`. One cell per process, 5 samples.
Reproduction C.

| dtype | n | 1T default | 1T thr=16 | 1T QR forced | 8T default | 8T thr=16 | 8T QR forced | thr=16 vs default 1T | 8T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 64 | 0.508 | 0.514 | 0.506 | 0.565 | 0.472 | 0.564 | -1% | +16% |
| f64 | 96 | 1.200 | 1.183 | 1.202 | 1.263 | 1.043 | 1.263 | +1% | +17% |
| f64 | 128 | 2.318 | 2.105 | 2.327 | 2.391 | 1.627 | 2.386 | +9% | +32% |
| f64 | 256 | 9.884 | 9.450 | 12.350 | 6.649 | 6.088 | 11.762 | +4% | +8% |
| f64 | 512 | 49.282 | 48.716 | 76.401 | 25.814 | 25.430 | 67.711 | +1% | +1% |
| c64 | 64 | 0.639 | 0.668 | 0.641 | 0.654 | 0.568 | 0.652 | -5% | +13% |
| c64 | 96 | 1.653 | 1.630 | 1.649 | 1.494 | 1.173 | 1.494 | +1% | +21% |
| c64 | 128 | 3.287 | 3.161 | 3.288 | 2.780 | 2.049 | 2.738 | +4% | +26% |
| c64 | 256 | 17.570 | 17.220 | 19.764 | 9.222 | 8.909 | 13.954 | +2% | +3% |
| c64 | 512 | 105.428 | 104.920 | 131.838 | 41.147 | 40.811 | 85.127 | +0% | +1% |

The CSV also holds the 20 `faer_tridiag_evd_t16` cell rows (this table's thr=16 arm); they are the source of that
column and are cited rather than dropped.

## Reproduction

From the repository root. `A` prints the rows of table A and the samples behind them, `B` the rows of table B and
the two remainders, `C` the rows of table C, and `layout` the order bounds quoted above.

```sh
python3 - <<'PY'
import csv, statistics as st
rows = list(csv.DictReader(open('docs/worklogs/evidence/2026-11-10-eigh-raw-samples.csv')))
def med(v, d, o, n, t):
    return st.median([float(x) for r in rows
        if (r['variant'], r['dtype'], r['order'], r['n'], r['threads']) == (v, d, o, str(n), str(t))
        for x in r['samples_ms'].split(';')])
N = (64, 96, 128, 256, 512)
print('A')                                    # table A and its samples
for d in ('f64', 'c64'):
    for o in 'cf':
        for n in N:
            f1, k1, f8, k8 = (med(v, d, o, n, t) for t in (1, 8) for v in ('faer_eigen', 'mkl'))
            print(d, o, n, f'{f1:.3f} {k1:.3f} {f1/k1:.2f} {f8:.3f} {k8:.3f} {f8/k8:.2f} {f1/f8:.2f}x {k1/k8:.2f}x')
print('B')                                    # table B, and the two remainders
for d in ('f64', 'c64'):
    for t in (1, 8):
        for v in ('faer_tridiag', 'mkl_tridiag', 'faer_tridiag_evd', 'mkl_stage2', 'faer_apply', 'faer_evd', 'mkl'):
            print(d, t, v, ' '.join(f'{med(v, d, "f", n, t):.3f}' for n in N))
        s3 = [sum(med(v, d, 'f', n, t) for v in ('faer_tridiag', 'faer_tridiag_evd', 'faer_apply')) for n in N]
        print(d, t, 'sum3', ' '.join(f'{x:.3f}' for x in s3))
        print(d, t, 'MKL remainder', ' '.join(f'{med("mkl", d, "f", n, t) - med("mkl_tridiag", d, "f", n, t) - med("mkl_stage2", d, "f", n, t):.3f}' for n in N))
        print(d, t, 'faer residual', ' '.join(f'{med("faer_evd", d, "f", n, t) - x:.3f}' for n, x in zip(N, s3)))
print('C')                                    # table C
for d in ('f64', 'c64'):
    for n in N:
        print(d, n, ' '.join(f'{med(v, d, "f", n, t):.3f}' for t in (1, 8)
              for v in ('faer_evd', 'faer_evd_t16', 'faer_evd_qr')))
print('layout')                               # the order bounds in the text
for v in ('faer_eigen', 'mkl'):
    for t in (1, 8):
        w = max(((abs(med(v, d, 'c', n, t) / med(v, d, 'f', n, t) - 1) * 100, d, n)
                 for d in ('f64', 'c64') for n in N))
        print(v, t, f'max {w[0]:.2f}% at {w[1]} n={w[2]}')
PY
```
