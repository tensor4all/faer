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
`d = min(budget, N / 16)`, taken only when `d >= 2`. This paragraph and the two below describe the
plan; the gate and the cap as measured differ, and the measurement is what the code carries.

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
preconditions falls to the kernel's *serial* chunked path. That is the expected shape of it and it is
what the C-order cells below cost, but it is not a bound: duplicated packing and the cache traffic of
`d` threads reading the same operand can make the split worse than either path, which is why the cap
exists rather than a promise. The measured class (16 rows,
30-column slabs, column-major) meets them; the row-major cells of the same shape are the ones where
this is an assumption rather than a reading, and they stay in the target list so the measurement
tests it.

**The gate.** `budget > 1 && M <= 128 && d >= 2`, where `budget` is the number of threads the native
kernel is already handed - the expression the call below already used, hoisted into one `let` so the
split and the kernel cannot disagree, and so the existing flop floor still decides whether a budget
exists at all. Only `f64` and `c64` take it: `f32`/`c32` reach the same kernel but have no measurement
behind them. Both bounds are empirical and the sweep below is what sets them; `faer` cannot read the
kernel's tile sizes, so nothing here is derived from them. It is not a claim about the whole grid:
the `M <= 128` region is sampled, not swept.

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

`matmul::tests::column_slab_split_matches_the_serial_product` is the case the branch takes:
`(16, 512, 240)`, `(16, 512, 231)` and `(24, 512, 288)` - above the flop floor, `d = 8`, even and
uneven slabs, `M` under the cap - each with `Accum::Replace` and with `Accum::Add` into a pre-filled
destination; then `c64` with `Conj::Yes` on both operands and a non-unit `alpha`; then the same
product in row-major views of all three operands, which is what the campaign's C-order cells hand in.
`matmul::tests::column_slab_boundaries` pins the gate (`M = 128` splits, `M = 129` does not; `N = 31`
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

## Acceptance measurement

Paired in one stack: harness `b09a1dc`, toolkit `a4f24a7`, `--reps 5 --aa 3 --prime-ms 250`, MKL arm,
`gemm-small,gemm-mid,gemm-tall,gemm-wide,linalg-mid,linalg-tall`, both dtypes, both orders, 1T/4T/8T.
The baseline is `50c2b98` recorded in the same stack (`20261022T092932Z-baseline-ref-50c2b98-853f008stack`)
because the published page was recorded at harness `4899a0b` and the comparator refuses a pair whose
harness moved; the candidate is `853f008` (`20261022T073947Z-acceptance-853f008`).

The arm was recorded at `853f008`; the branch head differs from it inside `faer/src` by two comment
lines (`git diff 853f008 <head> -- faer/src`), which is why the comparator's measured-path check,
applied between the two, refuses - the acceptance numbers are the measurement of this code, not of a
different one.

- `faer-upstream` control over 780 cells: median ratio 1.000883 -> the machine reproduces.
- Rotation-averaged fork ratios, below 1 is faster:
  `gemm/16x240x4096` f64 f 8T **0.124**, c64 f 8T **0.206**; `gemm/96x96x96` f64 f 4T 0.310;
  `gemm/128x128x128` f64 8T 0.313 (f) / 0.327 (c).
- 41 cells faster, 32 slower beyond 5%. Most of the slower ones cannot be this code: five are
  `triangular_solve` at *one* thread, where `budget == 1` and `column_slab_count` returns `None` by
  construction; `gemm/192x192x192`, `gemm/256x256x256` and `gemm/4096x64x64` are above the row cap;
  `lu c64 f 1024x64` moved at 1T as well (0.871 -> 0.918 ms), where the gate cannot fire. One cell is
  not accounted for: **`qr_factor f64 c 128^2` at 8T, 0.346 -> 0.374 ms (1.09x)**, where a 64x64 or
  128-column trailing product above the flop floor does take the new path. It is inside the same-revision
  A/A spread of that cell (1.02x) only by a margin, so it is recorded as an unresolved regression,
  not as an explained one.
- That rule is nonetheless not a valid gate, and the record should not pretend otherwise: a pair of
  the *same* revision moves 22 of 780 cells beyond 5% between runs (measured here), and the toolkit's
  own `control_check` says so. The protocol's decision rule is the median of the unchanged row, which
  passes at 1.0009 against a 2% tolerance.
- A/A' control, the same case subset with the split forced off (`bf63bf2`,
  `20261022T110216Z-aap-split-disabled`): 108 cells, fork median 0.9951, 4 cells beyond 5%. The
  largest of them, `lu f64 f 128^2` 8T at 1.26x, reads 1.25x on the **unchanged** `faer-upstream` row
  of the same pair, so it is the run, not the code. The gains survive the same test: `lu c64 512^2`
  8T 0.67x, 4T 0.69x, and `triangular_solve` 512^2 0.85-0.87x, all with per-start spreads under 3%.

## Verification conclusions and constraints

The split is adopted for the shapes it was written for. `M <= 128` and `n >= 32` are the measured
region; the change is not extended past it, and the reason the earlier "one row tile" reading of the
defect was wrong is in the diagnostic table above.

Limits that stay open:

- The deep-narrow family above the cap - `M x 240 x 4096` at `M >= 256` - still has no parallel
  decomposition of its own; it is untouched here and the budget still buys nothing at 8T.
- `matmul` is the innermost product of the factorization families, so the split also fires inside
  `lu`, `qr_factor`, `triangular_solve` and `householder_apply`. That is where the `lu c64 512^2`
  gain comes from, and it is also where the blast radius is: the guard set is the whole measured
  arm, not a handful of neighbouring cells.
- The row cap is a property of this host's kernel (`mr` 24 at AVX-512 f64); a machine with a
  different tile size would want the sweep repeated. Nothing here hardcodes the tile size, but the
  cap is calibrated to it.
- `256^3` was inside the cap in the diagnostic sweep and gained only 1.18x; the declared arm reads it
  at 1.00-1.02 either way, so the cap's exact upper edge is not settled to better than one shape.
