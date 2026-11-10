# The tiled triangular copy (#282): the #290 acceptance run

#282 was merged on one slide cell, `1024` f64 8T row-major `cholesky`, taken with boost on; #290 asked for the
run `docs/adoption-protocol.md` describes and for that slide cell to be labelled as what it is. This is that
run and what it says. No code changes here: the loss it found below the cache boundary is the one #323
measured and #330 gated on (tiles only when the two buffers reach 4 MiB), and that gate is already on `main`.

## What the protocol asks, quoted

- `docs/adoption-protocol.md:15-19` — the frozen contract: `--prime-ms 250 --reps 5 --aa 3 --statistic median`,
  **CPU boost off**, the observing side pinned off the measured L3 domains, "an idle guard that watches every
  CPU sharing the measured cache", `PINNED_RETRIES=8`.
- `:30` — a shared helper reaches "every family that copies or owns a matrix".
- `:32-34` — "measure the affected families at every declared shape, dtype and order at 4T and 8T, the target
  cells at 1T, and two or three cells of untouched families as a collateral control".
- `:55-62` — before a merge: `cargo test` plus the change's own test; `fbench verify` clean for the affected
  populations and budgets, where "a mismatch or an undeclared skip fails acceptance"; "The target cells
  improve beyond the between-start spread the baseline recorded, and no measured guard cell regresses beyond
  it"; the change "carries its own test and its own record".
- `docs/review-rules.md:93-95` — the record states what was verified and what remains unverified "without
  inflating the evidence".

## The run

evo2 (`EVO-X2`), Zen 5, one measured L3 domain. 54 timed processes: 3 process sets × 3 passes × 1T/4T/8T ×
`before`/`after`, every cell measured by all three passes on both binaries (2148 cells over the three row
identities). Window 2026-11-07 16:17 to 2026-11-08 04:21 +09:00, started after the `faer#279` sweep released
the machine; 1T on CPU 8, 4T on 8-11, 8T on 8-15, driver on 0-7/16-23. `pinned.sh` with `PINNED_RETRIES=8`
admitted all 54 on the first attempt.

| item | this run |
| --- | --- |
| `before` | fork `06d8e84c` (`main` just before #282), clean |
| `after` | fork `2b87010c` (#282 merged), clean; the measured paths differ only in `faer/src/linalg/mod.rs` and `faer/src/mat/matmut.rs` |
| harness / vendor / upstream row | faer-benchmark `94fcae3`, oneMKL 2026.2, faer `0539947f` (`faer-v0.24.4`) |
| contract | `--reps 5 --prime-ms 250 --statistic median`, `--rotate-rows 0,1,2` (the protocol's `--aa 3`), one lock |
| shapes | A: n = 32, 64, 96, 128, 256, 512 · B: n = 255, 257, 511, 513, 1024, 1536 |
| dtypes | A: f64, c64, f32, c32 · B: f64, c64 (the suites' own split) |
| orders | both, every shape |
| families | target `cholesky`, `eigh` (both ops); collateral `lu`, `qr_factor`, `gemm` at 1024, f64/c64, both orders, 1T/4T/8T |
| controls in every process | `faer-upstream` (code identical in both binaries) and `lapack-mkl` |
| verify | `fbench verify` 18 of 18 clean (`verify/`); `cargo test -p t4a-faer --lib` at `2b87010c`, 685 passed / 0 failed (`gates/`) |
| cost | 12 h 04 min (verification 2 h 37 min, timing 9 h 27 min), against the protocol's "about an hour" |

**One clause is not met: boost off.** On this host `/sys/devices/system/cpu/cpufreq/boost` reads `1` and is
not writable by this user; the kernel refuses the write ("`store_boost: Cannot disable BOOST!`", boot log
2026-11-04), while every per-policy `boost` reads `0` and the governor is `performance`. The run therefore
**is not an adoption-protocol acceptance**; it is a control-complete re-measurement under an unverifiable
boost state, and the protocol's `:16` clause is the reason. Read-only evidence, for what it is worth: the
effective clock of CPUs 8-15, sampled from the observer side every 5 s (`freq.tsv`, 8672 samples per CPU),
had a loaded median of 3004 MHz and a maximum of 3036 MHz against a 3000 MHz nominal — consistent with boost
being effectively off, and not the protocol's switch.

## Results

All ratios are `after`/`before` of the fork row, `< 1` faster; per cell the median of the three process
medians per binary. The between-start spread is max/min of those three, for the `before` binary.

**Controls** (rows whose code is identical in both binaries): `faer-upstream` 912 cells, median **0.9991**
(1T 0.9987, 4T 0.9989, 8T 0.9999); `lapack-mkl` 324 cells, **1.0003**. The machine reproduced.

**Collateral** (separate processes, same CPU set): `lu` 24 cells 1.009, `qr_factor` 12 cells 0.999, `gemm` 12
cells 0.999.

**Target path**, row-major `cholesky` owning (`Llt::new`), median over 1T/4T/8T:

| dtype | 32 | 64 | 96 | 128 | 255 | 256 | 257 | 511 | 512 | 513 | 1024 | 1536 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 | 0.991 | 1.139 | 1.192 | 1.045 | 1.148 | 0.863 | 1.230 | 1.172 | **0.476** | 1.177 | **0.442** | **0.508** |
| c64 | 0.949 | 1.054 | 1.082 | 1.003 | 1.082 | 0.865 | 1.054 | 1.098 | **0.710** | 1.115 | **0.713** | 0.802 |
| f32 | 1.012 | 1.241 | 1.288 | 1.270 | — | 1.018 | — | — | 0.750 | — | — | — |
| c32 | 1.005 | 1.038 | 1.060 | 1.027 | — | 0.961 | — | — | 0.620 | — | — | — |

The table's cells are thread aggregates (median over 1T/4T/8T per dtype and n); the individual cells at the
winning sizes are not all wins. Over the 24 cells at 512/1024/1536 the per-cell ratio runs **0.40-1.04**, and
exactly one of them loses: c64 1536 1T, 75.472 → 78.416 ms, ratio **1.039**, which is inside that cell's own
before spread of 1.131. With the thread medians aggregated the range is 0.44-0.80.

Per budget, f64 `before` → `after`: 512 0.59/0.48/0.44 (1.78 → 1.06 ms at 1T, 1.28 → 0.57 at 8T), 1024
0.64/0.44/0.40 (9.85 → 6.32 at 1T, 5.44 → 2.17 at 8T), 1536 0.72/0.51/0.46 (13.56 → 6.18 ms at 8T). c64 is the
same shape of result with a smaller win: 1024 0.88/0.71/0.66.

Row-major `eigh` moves little, because the copy is a small part of the call: 192 cells at n ≥ 64 (both ops
reach the copy), median 1.003, 0.95-0.96 at 512/1024/1536 for f64 and 0.94-1.03 overall.

Against the protocol's criterion (`:59-60`): of the 288 row-major target cells at n ≥ 64 (owning `cholesky`,
both `eigh` ops), **129 are slower than the before binary's own between-start spread**, 76 faster, 83 inside
it. Split by family: `cholesky` 51 slower / 29 faster / 16 inside (median 1.046), `eigh` 78 / 47 / 67 (median
1.003). The direction is not consistent across the target set, and one guard cell family regresses beyond the
spread (below). **The run does not support the unconditional form of #282.** What it does support is the working-set condition: the tiles win above the cache pair and lose below
it, with `256x256` mixed below the boundary (0.86-1.02: f64, c64 and c32 win, f32 loses) and `513x513` the
ragged cell that keeps them above it; that is the boundary #323 measured and #330 implements.

**Guards.** Column-major `cholesky` owning, which never leaves the elementwise branch: median 1.010 (f64),
0.996 (c64), 1.016 (f32), 0.998 (c32). n = 32, below the tiles' element floor: 1.005. One family moves
outside its spread and is not explained by the copy: **`cholesky` `in_place`, row-major c32, n = 32/64/96/128:
+25 %, +28 %, +14 %, +4 %** in every budget with a before spread of 1.001-1.005. The census (`census/`) shows
that route never calls `copy_from_triangular_lower`, and n = 32 is below the tile floor, so the copy is
excluded as the cause. What remains is inside the fork crate's own binary, and nothing in this archive
separates code generation from layout: the cause is untraced.

**Resolution.** The `faer-upstream` row is the drift control; its per-n p90 |drift| is 7.1 % at n = 32, 19.2 %
at 64, 7.1 % at 96, 5.1 % at 128, 4.9 % at 255, 2.2 % at 511, 1.9 % at 1024 and 2.7 % at 1536. Below n = 256
this harness cannot resolve a few percent, so the +4 % to +6 % at 96 and 128 is not established upward; the
losses at 255/257/511/513 are above that floor.

## #282's own cell, and the descriptions to correct

- The slide cell reproduces: 1024 f64 8T row-major `cholesky` 5.439 → 2.169 ms (ratio 0.399), with
  `faer-upstream` in the same processes at 5.844 → 5.889 ms (1.008). The number is real, and it is the most
  favourable cell of the grid. **It is supporting evidence, not the acceptance measurement.**
- "both copies" (#282's description): only `MatMut::copy_from_triangular_lower` takes the tiles.
  `copy_from`, the full-matrix copy, still runs elementwise, as `PATCHSET.md`'s row says.
- "8-member LU" (#282's description): the 8 is `crate::team::with_team(8, ...)` around the `Llt::new` call
  only; the `PartialPivLu` pair runs `Par::Seq` in both orders, so no LU comparison runs on 8 members. The
  width is an argument of the closure that runs the `Llt` call, not a property of the test.

- An aggregate that does not reproduce: #290's issue comment, and through it `PATCHSET.md`'s row, quotes the
  win at 512/1024/1536 as -13.9 %, -18.2 % and -16.4 %. The thread-aggregated medians of the same CSVs
  (median over 1T/4T/8T) are 0.476, 0.442 and 0.508 for f64 and 0.710, 0.713 and 0.802 for c64, with per-cell
  ratios of 0.40-1.04, and no grouping tried here reproduces the three quoted numbers; the ordering the
  threshold was pinned to (511 loses, 512 wins) holds in every reduction. The record keeps the per-cell
  numbers, which `cells.csv` rebuilds.

## What remains unverified

- The boost clause (`:16`): the switch is not writable on this host, so no run on it can claim it. Whoever
  needs that clause met needs a host where the write succeeds.
- n = 96 and 128 (below the drift floor), and the c32 `in_place` guard regression, whose cause is not traced.
- One host. That the working-set size is what drives the crossover is a working hypothesis, not something
  this grid establishes: a timing grid on one host does not separate cache capacity from everything else that
  changes with n there, and nothing here measures a second microarchitecture. (`TILE_COPY_CACHE_BYTES` is a
  byte count by construction, so the threshold reads no cache size at run time — which keeps it host
  independent, and is not evidence for why the crossover sits where it does.)
- The two edges #330 names are the two edges of this table: `513x513` (+5 % to +26 %) is above the byte
  threshold and keeps the tiles, `256x256` (0.86-1.02) is below it — mixed rather than a win the tiles give up:
  f64 0.863, c64 0.865 and c32 0.961 win (thread medians), while the three f32 cells are **1.015, 1.018 and
  1.029, losses**.

Evidence: [`evidence/2026-11-07-tiled-triangular-copy-290.tar.gz`](evidence/2026-11-07-tiled-triangular-copy-290.tar.gz)
— 54 process CSVs, the idle-guard verdicts, the 18 verify logs, `driver.log`, `power-*.txt`, `freq.tsv`, the
census, the test gate, and `cells.csv` with the reduction every table above is read from
(`python3 -I analyze-290.py` rebuilds both).
