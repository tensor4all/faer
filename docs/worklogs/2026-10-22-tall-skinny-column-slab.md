# Tall-skinny matmul: split the destination's columns over the budget

## Decisions

**The defect.** `matmul_imp` hands `private_gemm_x86::gemm` the caller's thread count once the
product clears its flop floor, and that kernel threads whatever it is handed. Above one thread it
leaves the fused serial path (`millikernel_colmajor`, taken when `n_threads <= 1 && !rowmajor && m <
l2 && n < l2`) and runs its chunked kernel, which hands out one microkernel tile at a time through a
shared job queue. When the destination is a **single row panel** (`M` no larger than the microkernel's
`mr`) and the output is narrow, that queue costs more than the tiles it hands out.

Measured on `zen5-cpu` at the campaign pin `50c2b98` with `f64`, column-major, `m = 16`, `k = 4096`
(`probes/small-gemm`, best of 11, 1T on CPU 4, 8T on 4-11, observer on 0-3,12-15):

| n | 60 | 120 | **240** | 480 | 960 | 1920 |
| --- | --- | --- | --- | --- | --- | --- |
| 1T (µs) | 252 | 502 | 1019 | 2116 | 4171 | 8524 |
| 8T (µs) | 158 | 162 | **1097** | 1237 | 1550 | 2393 |

The serial cost is linear in `n` (n = 30 costs 128 µs), so eight column slabs of a 240-column
destination would each cost about 128 µs and run concurrently. The campaign's own cell
(`gemm/16x4096x240`, the Householder reduction leg `tmp = V^H C` of `tlinalg-rs#46`) reads 1.136 ms
against OpenBLAS's 0.223 and MKL's 0.347, and faer is *faster* than both at one thread — the deficit
is the parallel decomposition, not the kernel.

**The change.** In `matmul_imp`, before the native kernel call: when the caller asked for more than
one thread and the destination is that class of shape, split its columns into `d` slabs, run
`matmul_imp` on each with `Par::Seq` — so each slab takes the fused serial path — and return.
`d = min(budget, N / 16)`, taken only when `d >= 2`.

The split uses the primitive the rank-update path already uses,
`spindle::for_each(d, dst.par_col_partition_mut(d).zip(rhs.par_col_partition(d)), ..)`, so it is one
region, not `d` of them. Each slab is multiplied by the public
`matmul_with_conj(dst_slab, beta, lhs, conj_lhs, rhs_slab, conj_rhs, alpha, Par::Seq)`, which
re-enters the same function with a budget of one and therefore ends on the kernel's serial path; that
keeps the slab types dynamic and needs no `Shape` juggling. The block is inside the `rayon` branch,
because `spindle` is an optional `rayon` dependency and `Par::Rayon` - and so a budget above one -
only exists with that feature.

A budget of one is necessary but not sufficient for the fused kernel: it also wants `!rowmajor &&
m < l2 && n < l2` after its own normalisation, and `faer` cannot see `l2`. A slab that misses those
preconditions falls to the kernel's *serial* chunked path - still a serial slab, so the change
degrades to the one-thread time rather than to the threaded queue. The measured class (16 rows,
30-column slabs, column-major) meets them; the row-major cells of the same shape are the ones where
this is an assumption rather than a reading, and they stay in the target list so the measurement
tests it.

**The gate.** `budget > 1 && M <= 32 && d >= 2`, where `budget` is the number of threads the native
kernel is already handed - the same expression the call below uses today, hoisted into one `let` so
the split and the kernel cannot disagree, and so the existing flop floor still decides whether a
budget exists at all.

`32` and `16` are empirical, and the record says so: the kernel's row tile is `mr = 24` for AVX-512
`f64` and `12` for AVX2, so 32 rows is one or two row tiles on this host, and `faer` cannot read
`mr`. 16 columns per slab is the narrowest slab that is worth opening the region for. The declared
grid has a single above-floor shape with `M <= 32` in each order and dtype (`16x4096x240`), so the
boundary cannot be guarded by a declared cell; a diagnostic sweep over `M` in {16, 24, 32, 33, 48}
is predeclared for it and its result decides whether the cap moves.

**Rejected.**

- Handing the kernel one thread and leaving the destination whole: 1019 µs instead of 1136 at 8T, a
  1.10x where OpenBLAS is 4.6x ahead.
- Threading the fused kernel itself (`t4a-private-gemm-x86`): a second repository, and a larger
  change than the defect needs.
- Splitting by rows instead: the row dimension has nothing to give here. `M = 16 / 32 / 64` at 8T
  against 1T read 1141 / 1036, 2082 / 2022 and 4248 / 4035 µs, so the budget buys nothing at 8T on any
  of them.

## Verification conclusions and constraints

Predeclared before the run, per the shared Performance-Gated Experiment Protocol.

- Baseline `../faer-benchmark/extern/t4a-faer` at `50c2b98` - the pin the published page was recorded
  at, and the revision this branch starts from, with `faer/Cargo.toml` byte-identical to the pinned
  checkout so the native kernel dependency cannot move under the comparison. Candidate this branch.
- Case list: everything the change can reach. The split fires on any product with `M <= 32`, `N >=
  32` and a budget, and `matmul` is the innermost product of the factorization families, so
  `householder_apply`, `householder_factor`, `qr_factor`, `qr`, `svd_values`, `svd_thin`, `lu` and
  `full_piv_lu` are *affected*, not controls - a blocked QR's `tmp = V^H C` has `M` equal to the basis
  width. Measured: `gemm-small`, `gemm-mid`, `gemm-tall`, `gemm-wide` in full; `linalg-mid` and
  `linalg-tall` in full; both dtypes, both orders, 4T and 8T, target cells at 1T. `cholesky`, `eigh`,
  `solve` and `triangular_solve` ride along in the same arm as the collateral control.
- Statistic the campaign's `candidate_compare.py` reports (rotation-averaged ratio of process
  medians); the frozen `--reps 5 --prime-ms 250` policy, every guard-rejected attempt discarded and
  retained rather than retried selectively, `--aa` at a multiple of the engine count.
- Target: `gemm/16x4096x240` at 8T at least 3x faster than the baseline, in both orders and both
  dtypes.
- Gate: no measured cell slower than 1.05x, and no affected family outside its recorded spread.
- `gemm/16x4096x240` is the campaign's only above-floor `M <= 32` shape, so the gate's boundaries are
  guarded by a predeclared diagnostic instead: `probes/small-gemm` over `M` in {16, 24, 32, 33, 48},
  `n = 240`, `k = 4096`, 1T on CPU 4 and 8T on 4-11, best of 11, observer on 0-3,12-15. It is a
  diagnostic and adopts nothing.

## Test

`matmul::tests::thread_budget_does_not_change_the_product` gains cases the branch actually takes:
`(16, 512, 240)` - above the flop floor, `d = 8`, even slabs; `(16, 512, 231)` - uneven slabs;
`(24, 512, 288)` - `M` just under the cap; a `(16, 512, 240)` call with `Accum::Add` into a
pre-filled destination; the same in `c64` with `Conj::Yes`; and one with a reversed destination. A
unit test on `column_slab_count` pins the boundaries (`M = 32` splits, `M = 33` does not; `N = 31`
does not, `N = 32` gives two slabs; a budget of one never splits). The lib is also built without the
`rayon` feature, which is what keeps the new block's `cfg` honest.

## Diagnostic measurements (not the acceptance arm)

`probes/small-gemm`, f64 column-major unless it says c64, best of 11, 1T on CPU 4, 4T on 4-7, 8T on
4-11, the observing process on 0-3,12-15. The baseline is `extern/t4a-faer` at `50c2b98`, the
candidate is this branch, and both were built from the same driver.

Target, `m = 16, n = 240, k = 4096` (µs, baseline -> candidate):

| threads | baseline | candidate | |
| --- | --- | --- | --- |
| 1 | 1019 | 1018 | unchanged: a budget of one never splits |
| 4 | 666 | 264 | 2.5x |
| 8 | 1092 | 137 | 8.0x |
| 8, c64 | 2373 | 544 | 4.4x |

At 8 threads the candidate is 137 µs against OpenBLAS's 223 and MKL's 347 on the same shape.

Where the row cap comes from, at 8 threads, `n = 240, k = 4096`, split taken for every `M` (the cap
lifted to 4096 for the sweep), against leaving the destination whole:

| shape | whole (baseline) | split | |
| --- | --- | --- | --- |
| `16x240x4096` | 1092 | 135 | 8.1x faster |
| `64x240x4096` | 3687 | 540 | 6.8x faster |
| `128x240x4096` | 2933 | 1147 | 2.6x faster |
| `128^3` | 48.7 | 27.4 | 1.8x faster |
| `256^3` | 173 | 146 | 1.2x faster |
| `512^3` | 1157 | 1831 | **1.6x slower** |
| `4096x64x64` | 164 | 479 | **2.9x slower** |

So the defect is not "one row tile": `M` up to 128 is worth splitting, and past that the budget is
better spent on rows the kernel can already decompose, because each slab makes every thread re-read
its share of the left operand. `COLUMN_SLAB_MAX_ROWS = 128` is that measured edge; `256^3` sits
outside it because it gains 1.2x at best.
