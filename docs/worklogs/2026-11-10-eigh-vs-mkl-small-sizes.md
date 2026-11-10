# `eigh` at n = 64..128 against MKL: which stage carries the loss

Issue: [tensor4all/faer#354](https://github.com/tensor4all/faer/issues/354).

Measured revision: `1ac38c6d7bab168c4478c9e2379a5bd9e69a2272` (fork `main`, clean tree), built
`--no-default-features --features std,rayon,threadteam` (the campaign's feature set: AVX2
kernels, `x86-v4` off). MKL 2026.1 (`libmkl_rt.so.3`, Intel OpenMP), driver links the fork by
path and MKL in one process.

## Decisions

- **The loss is one stage, not the whole call.** `self_adjoint_evd` splits into the
  tridiagonalization (`tridiag::tridiag_in_place`), the tridiagonal eigenproblem with vectors
  (`tridiagonal_self_adjoint_evd`) and the block-Householder apply onto `U`. At n = 64..128 the
  middle one is **76% of the f64 call** (53-57% for c64, where the complex apply is itself
  large) and is the only stage slower than MKL's counterpart: f64 n = 128 at 1T is 1.772 ms
  against MKL's `?stedc` 0.915 ms (**1.94x**). Stage 1 is at parity with MKL's `?sytrd` there
  (0.335 vs 0.338 ms), and stage 3 is 2.7x *faster* than the reflector apply left inside MKL's
  `?syevd` (0.199 ms against 0.540 ms, the latter derived as `?syevd` minus `?sytrd` minus
  `?stedc`). The 1T gap of the whole call (2.318 vs 1.793, 1.29x) is therefore *smaller* than
  stage 2's own gap: stage 3 pays part of it back.
- **Below n = 128 the tridiagonal eigenproblem has no parallel route at all**, which is why faer
  gains nothing from 8 threads in that band while MKL does. `qr_algorithm` takes no `Par`; the
  entry gate is `n < params.recursion_threshold` (default 128), and at exactly n = 128
  `divide_and_conquer` is entered and immediately falls back to QR because its first test is
  `n <= qr_fallback_threshold` with the same 128. The fork's own gate trace (`cost::trace`,
  built with `par-trace`) shows no `tridiag_evd` decision being asked at all at n = 128, so the
  D&C fork is not declined - it is never reached. MKL's `?syevd` runs divide and conquer at
  these sizes (its `?stedc` scales 1.22-1.50x from 1T to 8T at n = 64..128).
- **The measured consequence of that threshold, not a hypothesis:** `recursion_threshold = 16`
  (D&C at every size, the only knob, no code change) gives, at 8T, f64 n = 64 0.472 ms (default
  0.565), n = 96 1.043 (1.263), n = 128 1.627 (2.391) - **16-32% off the call** - and c64 0.568
  (0.654), 1.173 (1.494), 2.049 (2.780) - **13-26%**. faer/MKL at 8T moves from 1.77/1.86/1.94 to
  1.48/1.52/1.32 for f64, and from 0.95/1.08/1.22 to 0.83/0.84/0.91 for c64: the complex cells
  flip from a small loss to a win. At 1T the knob is neutral at n = 64 and 96 (+1% / -1%) and
  9-10% better at n = 128 (2.105 vs 2.318). So the default is not wrong at 1T; it is wrong at 8T,
  where D&C's merge is cheaper than a serial QR of the same tridiagonal.
- **Hypothesis (2) (thread start-up) is real but an order of magnitude too small to explain
  2x.** f64 stage 1 at 8T is 0.71x (n = 96) and 0.79x (n = 128) of its own 1T time: the parallel
  gate opens a region that does not pay, costing 20-29% of a stage that is 15% of the call.
  MKL's `?sytrd` loses the same way (0.71x, 0.77x), so this is not a fork-specific defect.
- **Hypothesis (3) (row-major vs column-major) is refuted at this size.** faer's c and f cells
  differ by at most 1.0% (f64 n = 64: 0.508 vs 0.505; c64 n = 512: 105.20 vs 105.58), and the
  mechanism is in the source:
  `self_adjoint_evd` copies the lower triangle into a contiguous temporary
  (`trid.copy_from_triangular_lower(A)`) before any kernel sees it, so the caller's order never
  reaches a kernel. MKL's row-major entry point is 1.4-2.7% slower than its column-major one.
- **Not implemented.** The knob is worth 13-32% at 8T, but after it faer is still 1.3-1.5x behind
  MKL on the f64 cells at 8T and 1.2x at 1T, so a threshold change alone does not reach parity
  and the remaining question is a design one (a parallel QR leaf, a threaded D&C merge, or MKL's
  two-stage reduction), not a tuning one. The options, their risk and their expected effect are
  below for the maintainer's decision.

## Verification conclusions and constraints

- What the tables establish: (A) the faer/MKL ratio per dtype, order, size and thread budget;
  (B) the stage split, with MKL's own `?sytrd`/`?stedc` measured on the *same* tridiagonal, and
  the sum of faer's three stages closing with `self_adjoint_evd` within 2.6% at 1T and 4.6% at
  8T, except f64 n = 64 at 8T (-11%) and f64 n = 256 at 8T (+8.4%) - the residual is the
  lower-triangle copy, the team dispatch and each region's fill/drain, which no single stage row
  contains (a stage probe that does not close would not be a decomposition); (C) the threshold
  knob; (D) the raw samples of the cells under test.
- **Hypothesis (1) is half right, and the half that is wrong matters.** MKL does use divide and
  conquer where faer uses QR, and that is the mechanism. But "add a small-size D&C branch" is
  already there - what is missing is that the branch is unreachable at n <= 128 (the `n <=
  qr_fallback_threshold` fallback) and that its leaves are serial QR. A change that only lowers
  the threshold gets the 13-32% and stops.
- **What remains unverified.**
  - Whether a parallel QR leaf (deflated blocks as tasks) beats lowering the threshold. The
    threshold knob is measured; a threaded QR is not, and it is the only route that also fixes
    the 1T gap (where D&C is only 10% better at n = 128 and nothing at n = 64/96).
  - How much of MKL's `?stedc` advantage is its merge implementation rather than its parallelism.
    At 1T MKL is 1.55-1.94x faster than faer's QR at n = 64..128, so part of the gap survives
    any parallelisation of faer's current algorithm.
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
- The tree measured is the tree this record travels in: `cargo test --lib` on it gives 523
  passed, 0 failed, 138 ignored (661 library tests) and 1 doc test passed. The change is a record
  only, so no measured path moves.
- Host: AMD Ryzen AI 9 HX 470 (Zen5 mobile APU), 24 logical / 12 physical, SMT on, CPU boost
  off, governor `performance`, kernel 7.0.0-38-generic, rustc 1.99.0. Measured CPUs `0` (1T) and
  `0-3,12-15` (8T, the 16 MiB L3 domain and its SMT siblings); the observing side pinned to
  `4-11,16-23` (the other L3 domain) for the whole run. No `RAYON_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OMP_*` or `*_BLAS*` variable set; the probe sets the rayon/threadteam team width per cell and
  asserts MKL's read-back equals it. This is the `zen5-cpu` development host, the same machine the
  published pages are recorded on, not the `zen2-64-smt-off` measurement host.

### If we do implement: the options and what each buys

1. **Lower the D&C entry threshold when the team is wider than one** (make
   `SelfAdjointEvdParams::recursion_threshold` a function of `par.degree()`, or change the
   `n <= qr_fallback_threshold` fallback to `<`). Measured effect: 13-32% off the call at 8T for
   n = 64..128, 0-10% at 1T, no change at n >= 256. Risk: the D&C merge's accuracy path is
   different from QR's; the fork's own `lapack_self_adjoint_evd_*` and spectrum-accuracy tests
   already cover both thresholds (the QR/D&C choice is a parameter there), so the regression
   surface is the accuracy fixtures, not new code. Cost: one gate, no new algorithm.
2. **Parallelise the QR leaf** (deflated blocks as tasks inside `qr_algorithm`). This is the only
   option that can move the 1T gap too, and it is what would let n = 64/96 use a threshold
   without paying D&C's merge. Risk: the eigenvector chase is a sequential recurrence over
   rotations; block-parallel variants change the arithmetic order, so it needs its own accuracy
   and residual tests. Unmeasured.
3. **Leave it.** The band is 1.8-1.9x behind MKL at 8T and 1.3x at 1T; faer already wins at 512
   at 1T (0.84x) but loses it at 8T (1.21x), and wins complex at 1T (0.62-0.83x). If the target is
   the campaign's win rate rather than this band, option 1 is the cheapest 13-32% and option 3 is
   defensible.

## Evidence

All times in ms, median of the samples the row names. The driver is a standalone probe outside
both repositories (the fork carries no benchmark harness): it links this checkout by path and
`libmkl_rt` in one process, sizes every workspace with the library's own `*_scratch`, primes
each cell for 250 ms and takes 5 samples, and restores every input outside the clock.

### A. `eigh` faer (owning `self_adjoint_eigen`) vs MKL `?syevd`

Median of 15 samples (3 sessions x 5 reps), prime 250 ms per cell, CPU boost off.

| dtype | order | n | faer 1T | MKL 1T | faer/MKL | faer 8T | MKL 8T | faer/MKL | faer 1T->8T | MKL 1T->8T |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | c | 64 | 0.508 | 0.394 | 1.29 | 0.567 | 0.336 | 1.68 | 0.90x | 1.17x |
| f64 | c | 96 | 1.202 | 0.898 | 1.34 | 1.272 | 0.728 | 1.75 | 0.95x | 1.23x |
| f64 | c | 128 | 2.314 | 1.841 | 1.26 | 2.388 | 1.263 | 1.89 | 0.97x | 1.46x |
| f64 | c | 256 | 9.930 | 9.888 | 1.00 | 6.664 | 5.339 | 1.25 | 1.49x | 1.85x |
| f64 | c | 512 | 49.120 | 60.196 | 0.82 | 25.075 | 21.476 | 1.17 | 1.96x | 2.80x |
| f64 | f | 64 | 0.505 | 0.387 | 1.31 | 0.564 | 0.318 | 1.77 | 0.90x | 1.22x |
| f64 | f | 96 | 1.203 | 0.881 | 1.37 | 1.275 | 0.687 | 1.86 | 0.94x | 1.28x |
| f64 | f | 128 | 2.321 | 1.793 | 1.29 | 2.397 | 1.233 | 1.94 | 0.97x | 1.45x |
| f64 | f | 256 | 9.998 | 9.755 | 1.02 | 6.761 | 5.141 | 1.31 | 1.48x | 1.90x |
| f64 | f | 512 | 49.405 | 59.105 | 0.84 | 25.257 | 20.834 | 1.21 | 1.96x | 2.84x |
| c64 | c | 64 | 0.639 | 0.800 | 0.80 | 0.655 | 0.702 | 0.93 | 0.98x | 1.14x |
| c64 | c | 96 | 1.660 | 2.021 | 0.82 | 1.503 | 1.402 | 1.07 | 1.10x | 1.44x |
| c64 | c | 128 | 3.301 | 4.376 | 0.75 | 2.744 | 2.308 | 1.19 | 1.20x | 1.90x |
| c64 | c | 256 | 17.505 | 26.212 | 0.67 | 9.188 | 10.723 | 0.86 | 1.91x | 2.44x |
| c64 | c | 512 | 105.199 | 169.258 | 0.62 | 41.286 | 51.746 | 0.80 | 2.55x | 3.27x |
| c64 | f | 64 | 0.640 | 0.790 | 0.81 | 0.656 | 0.688 | 0.95 | 0.98x | 1.15x |
| c64 | f | 96 | 1.658 | 1.996 | 0.83 | 1.499 | 1.391 | 1.08 | 1.11x | 1.44x |
| c64 | f | 128 | 3.303 | 4.324 | 0.76 | 2.756 | 2.261 | 1.22 | 1.20x | 1.91x |
| c64 | f | 256 | 17.580 | 25.926 | 0.68 | 9.291 | 10.541 | 0.88 | 1.89x | 2.46x |
| c64 | f | 512 | 105.580 | 167.972 | 0.63 | 41.803 | 51.011 | 0.82 | 2.53x | 3.29x |

### B. stage split, ms (each cell its own process, order f)

faer stages are `tridiag::tridiag_in_place`, `tridiagonal_self_adjoint_evd` (with vectors),
`apply_block_householder_sequence_on_the_left_in_place_with_conj`. MKL stages are `?sytrd`/
`?hetrd` and `?stedc` (`compz='I'`) on the *same* tridiagonal faer's stage 1 produced.
`faer total` is `self_adjoint_evd` in place; `MKL total` is `?syevd` from table A.

**f64**

| stage | 1T/8T | n=64 | n=96 | n=128 | n=256 | n=512 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| faer stage 1 (tridiagonalize) | 1T | 0.080 | 0.182 | 0.335 | 1.855 | 11.566 |
| faer stage 1 (tridiagonalize) | 8T | 0.079 | 0.254 | 0.423 | 1.592 | 6.475 |
| faer stage 1 (tridiagonalize) | 1T/8T | 1.01x | 0.71x | 0.79x | 1.17x | 1.79x |
| MKL ?sytrd | 1T | 0.053 | 0.154 | 0.338 | 2.623 | 21.317 |
| MKL ?sytrd | 8T | 0.054 | 0.216 | 0.439 | 1.901 | 9.367 |
| MKL ?sytrd | 1T/8T | 0.99x | 0.71x | 0.77x | 1.38x | 2.28x |
| faer stage 2 (tridiag EVD + vectors) | 1T | 0.387 | 0.915 | 1.772 | 6.652 | 27.090 |
| faer stage 2 (tridiag EVD + vectors) | 8T | 0.386 | 0.915 | 1.771 | 5.196 | 15.919 |
| faer stage 2 (tridiag EVD + vectors) | 1T/8T | 1.00x | 1.00x | 1.00x | 1.28x | 1.70x |
| MKL ?stedc | 1T | 0.249 | 0.525 | 0.915 | 3.672 | 15.544 |
| MKL ?stedc | 8T | 0.205 | 0.404 | 0.610 | 2.425 | 6.982 |
| MKL ?stedc | 1T/8T | 1.22x | 1.30x | 1.50x | 1.51x | 2.23x |
| faer stage 3 (reflector apply) | 1T | 0.036 | 0.096 | 0.199 | 1.335 | 9.779 |
| faer stage 3 (reflector apply) | 8T | 0.037 | 0.066 | 0.109 | 0.419 | 2.796 |
| faer stage 3 (reflector apply) | 1T/8T | 0.98x | 1.47x | 1.83x | 3.18x | 3.50x |
| faer total (3 stages) | 1T | 0.503 | 1.193 | 2.305 | 9.842 | 48.435 |
| faer total (3 stages) | 8T | 0.502 | 1.234 | 2.303 | 7.208 | 25.190 |
| faer `self_adjoint_evd` | 1T | 0.508 | 1.200 | 2.318 | 9.884 | 49.282 |
| faer `self_adjoint_evd` | 8T | 0.565 | 1.263 | 2.391 | 6.649 | 25.814 |
| MKL `?syevd` | 1T | 0.387 | 0.881 | 1.793 | 9.755 | 59.105 |
| MKL `?syevd` | 8T | 0.318 | 0.687 | 1.233 | 5.141 | 20.834 |

**c64**

| stage | 1T/8T | n=64 | n=96 | n=128 | n=256 | n=512 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| faer stage 1 (tridiagonalize) | 1T | 0.128 | 0.344 | 0.728 | 5.022 | 36.626 |
| faer stage 1 (tridiagonalize) | 8T | 0.175 | 0.359 | 0.610 | 2.715 | 13.782 |
| faer stage 1 (tridiagonalize) | 1T/8T | 0.73x | 0.96x | 1.19x | 1.85x | 2.66x |
| MKL ?sytrd | 1T | 0.247 | 0.732 | 1.597 | 11.548 | 88.560 |
| MKL ?sytrd | 8T | 0.316 | 0.639 | 1.112 | 5.632 | 29.996 |
| MKL ?sytrd | 1T/8T | 0.78x | 1.15x | 1.44x | 2.05x | 2.95x |
| faer stage 2 (tridiag EVD + vectors) | 1T | 0.360 | 0.938 | 1.754 | 6.580 | 26.831 |
| faer stage 2 (tridiag EVD + vectors) | 8T | 0.362 | 0.938 | 1.759 | 4.625 | 16.038 |
| faer stage 2 (tridiag EVD + vectors) | 1T/8T | 0.99x | 1.00x | 1.00x | 1.42x | 1.67x |
| MKL ?stedc | 1T | 0.251 | 0.532 | 0.920 | 3.713 | 15.787 |
| MKL ?stedc | 8T | 0.207 | 0.409 | 0.693 | 2.270 | 6.246 |
| MKL ?stedc | 1T/8T | 1.21x | 1.30x | 1.33x | 1.64x | 2.53x |
| faer stage 3 (reflector apply) | 1T | 0.142 | 0.354 | 0.774 | 5.507 | 39.426 |
| faer stage 3 (reflector apply) | 8T | 0.087 | 0.151 | 0.283 | 1.487 | 10.181 |
| faer stage 3 (reflector apply) | 1T/8T | 1.63x | 2.35x | 2.74x | 3.70x | 3.87x |
| faer total (3 stages) | 1T | 0.630 | 1.636 | 3.257 | 17.109 | 102.883 |
| faer total (3 stages) | 8T | 0.624 | 1.448 | 2.651 | 8.827 | 40.002 |
| faer `self_adjoint_evd` | 1T | 0.639 | 1.653 | 3.287 | 17.570 | 105.428 |
| faer `self_adjoint_evd` | 8T | 0.654 | 1.494 | 2.780 | 9.222 | 41.147 |
| MKL `?syevd` | 1T | 0.790 | 1.996 | 4.324 | 25.926 | 167.972 |
| MKL `?syevd` | 8T | 0.688 | 1.391 | 2.261 | 10.541 | 51.011 |

### C. the `recursion_threshold` knob: `self_adjoint_evd` in place, ms

`default` is 128 (the value `SelfAdjointEvdParams::auto` gives), `16` forces divide and
conquer at every size, `usize::MAX` forces QR iteration at every size.

| dtype | n | 1T default | 1T thr=16 | 1T QR forced | 8T default | 8T thr=16 | 8T QR forced | thr=16 vs default (8T) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 64 | 0.508 | 0.514 | 0.506 | 0.565 | 0.472 | 0.564 | +16% |
| f64 | 96 | 1.200 | 1.183 | 1.202 | 1.263 | 1.043 | 1.263 | +17% |
| f64 | 128 | 2.318 | 2.105 | 2.327 | 2.391 | 1.627 | 2.386 | +32% |
| f64 | 256 | 9.884 | 9.450 | 12.350 | 6.649 | 6.088 | 11.762 | +8% |
| f64 | 512 | 49.282 | 48.716 | 76.401 | 25.814 | 25.430 | 67.711 | +1% |
| c64 | 64 | 0.639 | 0.668 | 0.641 | 0.654 | 0.568 | 0.652 | +13% |
| c64 | 96 | 1.653 | 1.630 | 1.649 | 1.494 | 1.173 | 1.494 | +21% |
| c64 | 128 | 3.287 | 3.161 | 3.288 | 2.780 | 2.049 | 2.738 | +26% |
| c64 | 256 | 17.570 | 17.220 | 19.764 | 9.222 | 8.909 | 13.954 | +3% |
| c64 | 512 | 105.428 | 104.920 | 131.838 | 41.147 | 40.811 | 85.127 | +1% |

### D. raw samples of the cells under test (order f), ms

Every sample of the 3 sessions (15 per cell).

| cell | samples | median |
| --- | --- | ---: |
| faer_eigen f64 64 1T | 0.5054, 0.5038, 0.5052, 0.5059, 0.5057, 0.5106, 0.5059, 0.5132, 0.5147, 0.5110, 0.5075, 0.5069, 0.5039, 0.5035, 0.5028 | 0.5059 |
| mkl f64 64 1T | 0.3864, 0.3895, 0.3883, 0.3855, 0.3891, 0.3856, 0.3879, 0.3846, 0.3841, 0.3891, 0.3868, 0.3884, 0.3866, 0.3897, 0.3855 | 0.3868 |
| faer_eigen f64 96 1T | 1.2039, 1.2097, 1.2033, 1.2004, 1.2010, 1.2022, 1.2051, 1.2015, 1.1983, 1.2010, 1.2063, 1.2047, 1.2072, 1.2036, 1.2017 | 1.2033 |
| mkl f64 96 1T | 0.8926, 0.8749, 0.8843, 0.8902, 0.8737, 0.8755, 0.8720, 0.8727, 0.8719, 0.8716, 0.8729, 0.8810, 0.8836, 0.8825, 0.8747 | 0.8749 |
| faer_eigen f64 128 1T | 2.3185, 2.3205, 2.3288, 2.3269, 2.3237, 2.3209, 2.3242, 2.3220, 2.3178, 2.3200, 2.3254, 2.3262, 2.3171, 2.3156, 2.3185 | 2.3209 |
| mkl f64 128 1T | 1.7897, 1.8107, 1.7830, 1.7964, 1.7915, 1.8309, 1.8016, 1.7923, 1.8186, 1.8299, 1.7963, 1.7935, 1.7899, 1.7835, 1.7966 | 1.7963 |
| faer_eigen f64 64 8T | 0.5710, 0.5750, 0.5678, 0.5673, 0.5668, 0.5680, 0.5623, 0.5625, 0.5623, 0.5681, 0.5629, 0.5636, 0.5638, 0.5780, 0.5690 | 0.5673 |
| mkl f64 64 8T | 0.3165, 0.3163, 0.3279, 0.3161, 0.3202, 0.3192, 0.3222, 0.3307, 0.3291, 0.3232, 0.3304, 0.3343, 0.3157, 0.3177, 0.3174 | 0.3202 |
| faer_eigen f64 96 8T | 1.2813, 1.2772, 1.2733, 1.2733, 1.2770, 1.2674, 1.2730, 1.2691, 1.2752, 1.2708, 1.2669, 1.3060, 1.2756, 1.2699, 1.2746 | 1.2733 |
| mkl f64 96 8T | 0.7148, 0.6897, 0.6864, 0.6810, 0.6867, 0.6836, 0.7223, 0.7124, 0.7192, 0.6874, 0.6776, 0.6795, 0.6812, 0.6784, 0.7164 | 0.6867 |
| faer_eigen f64 128 8T | 2.4167, 2.4143, 2.4358, 2.4096, 2.4055, 2.3650, 2.3673, 2.3776, 2.3679, 2.3682, 2.3984, 2.3970, 2.3917, 2.3923, 2.3983 | 2.3970 |
| mkl f64 128 8T | 1.2179, 1.2690, 1.2231, 1.2199, 1.2177, 1.2727, 1.2928, 1.2779, 1.2251, 1.2358, 1.2338, 1.2327, 1.2325, 1.2442, 1.2284 | 1.2327 |
| faer_eigen c64 64 1T | 0.6374, 0.6417, 0.6422, 0.6374, 0.6418, 0.6394, 0.6411, 0.6371, 0.6400, 0.6402, 0.6405, 0.6357, 0.6413, 0.6413, 0.6373 | 0.6402 |
| mkl c64 64 1T | 0.8155, 0.7907, 0.7902, 0.7902, 0.7903, 0.7919, 0.7888, 0.7864, 0.7926, 0.7880, 0.7836, 0.7908, 0.7926, 0.7908, 0.7905 | 0.7905 |
| faer_eigen c64 96 1T | 1.6736, 1.6700, 1.6597, 1.6659, 1.6589, 1.6566, 1.6576, 1.6580, 1.6553, 1.6600, 1.6582, 1.6549, 1.6570, 1.6584, 1.6528 | 1.6582 |
| mkl c64 96 1T | 1.9998, 1.9970, 1.9992, 1.9980, 2.0078, 1.9944, 1.9966, 1.9950, 1.9982, 1.9963, 1.9888, 2.0037, 1.9873, 1.9876, 1.9869 | 1.9966 |
| faer_eigen c64 128 1T | 3.3079, 3.3035, 3.3251, 3.3748, 3.2974, 3.2987, 3.2986, 3.3073, 3.3026, 3.3031, 3.3007, 3.3019, 3.2984, 3.2997, 3.3002 | 3.3019 |
| mkl c64 128 1T | 4.3243, 4.3311, 4.3203, 4.3188, 4.3278, 4.3603, 4.3314, 4.3359, 4.3326, 4.3509, 4.2865, 4.3076, 4.3175, 4.3064, 4.3078 | 4.3243 |
| faer_eigen c64 64 8T | 0.6619, 0.6613, 0.6578, 0.6606, 0.6618, 0.6566, 0.6517, 0.6563, 0.6579, 0.6522, 0.6555, 0.6566, 0.6499, 0.6557, 0.6586 | 0.6566 |
| mkl c64 64 8T | 0.6795, 0.6792, 0.6813, 0.6824, 0.6899, 0.6964, 0.6867, 0.6882, 0.6893, 0.6847, 0.6880, 0.6815, 0.6982, 0.6878, 0.6853 | 0.6867 |
| faer_eigen c64 96 8T | 1.5035, 1.4993, 1.5017, 1.5047, 1.5030, 1.4992, 1.5037, 1.5018, 1.4980, 1.4990, 1.4904, 1.4889, 1.4900, 1.4889, 1.4903 | 1.4992 |
| mkl c64 96 8T | 1.3754, 1.3757, 1.4008, 1.3569, 1.3746, 1.3918, 1.3927, 1.3640, 1.3729, 1.3907, 1.3907, 1.3917, 1.3689, 1.3942, 1.4031 | 1.3907 |
| faer_eigen c64 128 8T | 2.7542, 2.7586, 2.7565, 2.7560, 2.7556, 2.7509, 2.7590, 2.7535, 2.7506, 2.7588, 2.7611, 2.7579, 2.7628, 2.7576, 2.7682 | 2.7576 |
| mkl c64 128 8T | 2.2880, 2.2246, 2.2260, 2.2403, 2.2192, 2.2578, 2.2920, 2.2958, 2.2607, 2.2614, 2.2502, 2.2687, 2.3295, 2.2757, 2.3009 | 2.2614 |

Between-session spread (max/min of the three session medians): median 0.3% (faer 1T),
0.9% (faer 8T); one cell (`mkl c64 64 8T`) read 0.702 / 0.702 / 1.996 ms in the three
sessions and was re-run three times at 0.699-0.706 ms, so the 1.996 is a discarded outlier,
not the cell.
