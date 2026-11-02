# Pfaffian #192: giving the eight-thread team enough work at n = 256

Follow-up to [#192](https://github.com/tensor4all/faer/issues/192), based on
`e765e0a` (#181 merged). The issue's diagnosis was that the n = 256 factorizations
run out of *supply*: an instrumented trace showed nonempty ranks of 5, 4 and 3 in
the three widest trailing updates, so most of an eight-thread team idles, and #181
had already ruled out dispatch cost with an equivalent-workload GEMM control.

## What the measured phase split said

Temporary clocks (std only, removed before submission) at n = 256, width 32,
`f64`, eight threads, total 0.534 ms:

| phase | serial | 8 threads | note |
|---|---:|---:|---|
| panel factorization | 273 us | 273 us | serial by construction (pivot order) |
| panel gather into the shared compact panel | 23 us | 23 us | serial |
| trailing tiled update | 518 us | 232 us | 2.2x on 8 threads |

Inside the serial panel: materializing the pending column 93 us, the ordered
rank-two recurrence 96 us, the pivot search 37 us, normalization 38 us, for
`f64`; the `c64` numbers are 344 / 347 / 61 / 40 us of 1.31 ms. Two things follow:
the trailing update was at 27% parallel efficiency, and the *serial* panel was
half of the eight-thread wall. So the fix had to do both.

## The change

- **Compact panel.** The panel now lives in pivot order (`perm[start + i]` is row
  `i`), so a step touches only live rows, the pivot search and the rank-two
  recurrence are contiguous, and the trailing rows of that panel are *already*
  the shared compact panel the tiled update reads. The second gather and one
  scratch matrix are gone, and the panel arithmetic dropped by the live-row
  fraction (56% at n = 256) because eliminated rows were previously recomputed at
  full length. The panel is gathered per column, lazily, so a pivot exchange that
  brings in a column from outside the panel is re-read rather than moved; the
  finished columns are relabelled by a row exchange that keeps compact row `i`
  equal to physical row `perm[start + i]`.
- **Tile-level split of the trailing update.** Each `PANEL x PANEL` tile of the
  lower triangle is one job; a rank owns a contiguous tile range and one private
  scratch slot. The old split cut row blocks only, and one row block of the
  224-row trailing block is a quarter of the triangle, so the widest rank could
  never be helped: 28 tiles, one rank with 7 of them.
- **Width cap.** The split is capped by the tile count and by the tile scratch
  (one slot per `PANEL`-wide strip of `n`), not by the row-block count, which is
  what left the shallow panels at width 2 to 5.
- **One reciprocal per step** for the native scalars, as LAPACK's unblocked
  `getf2` does. The extended-precision scalars keep exactly rounded division
  (their accuracy bound is pinned at 1e-32 by the public test), and a reciprocal
  that would itself overflow falls back to division, so the reachable exponent
  range and the subnormal/extreme-scale behaviour of the public test are
  unchanged.
- **Panel width 32 -> 24.** The width trades serial panel work (grows with it)
  against the elementwise gather/store of the trailing update (shrinks with it).
  Same session, n = 256, eight threads, `f64`/`c64` speedup over PfaPack:
  16: 3.72 / 5.27, 20: 3.79 / 4.99, 24: 3.91 / 4.99, 28: 3.59 / 4.47,
  32: 3.77 / 4.37. 24 is the measured real optimum and within noise of the
  complex one, so one width serves both.

Seq and Team agree bit for bit on every fixture (the harness now compares the
result's bit patterns rather than Python's `==`, which folds `+0.0` into `-0.0`).
Against the previous release the value moves by 1.4e-14 to 2.1e-13 relative on the
harness fixtures, and by 8.5e-13 on the `n384/f64` fixture. That is **not** a
rounding-only equivalence claim: the panel boundary change moves updates from the
ordered panel recurrence into the two-GEMM trailing product, whose summation order
differs, so a tie between near-equal pivot candidates can resolve differently at a
different width. An adversarial matrix (unit skew pairs plus two entries that
differ by one ulp) can therefore return a different last-digit result, or even a
zero pivot, at width 24 where width 32 found a nonzero one. The claim here is the
measured one: the harness fixtures and the 192 correctness comparisons agree with
the reference inside 5e-10, Seq and Team agree exactly, and no cell changes which
answer it returns.

## Measured (evo, Ryzen AI 9 HX 470, AVX2 lane, boost off, cores 4-11)

Two harnesses produced the numbers below, and they are not interchangeable.

The `n = 256` table and the `n = 384` row come from the author's alternating driver
(`~/work/pfaffian-192/measure.py`): PfaPack 1T, a probe built from `e765e0a`
(#181, the merged predecessor) and a probe built from this commit run in one
schedule, rotated per session, every cell idle-guard clean through the campaign's
`pinned.sh`, observer on 0-3,12-15 (outside the measured L3 domain), load1
recorded per cell.

The `n = 512` and `n = 1024` rows and the `384`/`512` `f64` Seq guard come from
the repository harness `scripts/pfaffian-acceptance/run.py`, which this commit
also updates: its baseline arm is now `e765e0a` (it was `71b25ffb`, whose
dependency graph still used `gemm` 0.19 and could not be built from the same lock
as this tree), it measures a `cand` arm at the candidate's default width in
addition to the `b16`/`b32` width scan, so its schedule is 192 guarded processes
rather than 126, and it does not measure `n = 384` at all.

n = 256, three sessions, median of session medians:

| dtype | PfaPack ms | base Team8 ms | base x | candidate Team8 ms | candidate x |
|---|---:|---:|---:|---:|---:|
| f64 | 1.2492 | 0.5373 | 2.325 | 0.2995 | **4.170** |
| c64 | 3.3240 | 1.3010 | 2.555 | 0.6221 | **5.344** |

Acceptance cells n = 384/512/1024, two sessions, both dtypes and both paths:

| n | dtype | PfaPack ms | base Team8 x | candidate Team8 x | base Seq/F | candidate Seq/F |
|---|---:|---:|---:|---:|---:|---:|
| 384 | f64 | 2.9353 | 2.377 | 3.930 | 0.898 | 0.935 |
| 384 | c64 | 10.2076 | 3.486 | 6.643 | 0.593 | 0.557 |
| 512 | f64 | 9.9195 | 4.298 | 6.229 | 0.588 | 0.641 |
| 512 | c64 | 26.4445 | 4.922 | 8.266 | 0.506 | 0.496 |
| 1024 | f64 | 63.0778 | 5.368 | 6.815 | 0.736 | 0.848 |
| 1024 | c64 | 222.8630 | 7.517 | 10.762 | 0.508 | 0.546 |

No cell regresses on the eight-thread side; every one improves. The serial side
does pay for the narrower panel at large `n`: Seq/Fortran for `f64` moves from
0.736 to 0.848 at n = 1024 (0.588 -> 0.641 at 512, 0.898 -> 0.935 at 384), which
is 4 to 15% more serial time inside the unchanged 1.05 gate. That is the
trade-off #192 asked to measure: the same width that buys supply at n = 256 pays
33% more elementwise trailing traffic, and Seq has nothing else to hide it behind.

## Candidates that lost

- **Single-triangle work storage** (one store per unordered pair, mirror read
  with the sign): f64 0.337 ms vs 0.330, c64 0.853 ms vs 0.762 at n = 256. The
  conditional negation per element costs more than the store it saves,
  especially for `c64` where it does not become a branchless select.
- **One fused GEMM per tile** (`[X|Y] [Y|-X]^T` with twice the inner dimension,
  built once per panel): f64 0.3456 vs 0.3254 ms, c64 0.6961 vs 0.6706. The
  per-call overhead of two half-width GEMMs is not where the trailing update
  spends its time, so the extra operand build and the longer inner loop only add.
- **Width 16 or 32 instead of 24**, and **caching `absmax` in the pivot search**:
  no repeatable effect.

## Tests

`cargo test -p t4a-faer --lib` is 590 passed / 138 ignored (the new tile-split
test included). The 12 public Pfaffian tests pass, including the extended
precision and extreme-scale cases the reciprocal normalization had to be scoped
around. `scripts/pfaffian-acceptance/test_no_perf.py` passes with the updated
arm count.

## Not measured

No `perf` counter, no DRAM bandwidth (the machine's `perf_event_paranoid=4`), no
other host, no GPU path, no architecture outside `x86_64-avx2`. The worklog's
byte model still describes matrix-interface traffic only. No PfaPack or mvmc
source was consulted; the algorithm remains the published Wimmer and Satyarth
equations cited in the module.
