# The tiled copy at a power-of-two column stride (#352)

## Decisions

- `copy_lower_across_layouts` tiled on the working set alone since `#330`, and `Llt::new` on a row-major
  f64 matrix was the one losing cell: at n = 256 the elementwise walk takes 13.6% more time than the
  tiles (1T) and 15.7% more (8T) -- the tiles are 12.0% / 13.6% faster in time -- while at n = 240,
  244, 248, 252, 260, 264, 272 the walk wins by 3.5-7.3% (1T) and 9.6-11.1% (8T).
- The n that loses is the only one of those eight whose **destination column stride** is a power of two:
  2048 bytes. 248/252/260/264 are 1984-2112 bytes; 256 is 2048. The walk writes the destination a
  column at a time, and a power-of-two stride lands every column on the same cache sets as the one
  before it.
- Chosen: a second gate in `tile_copy_pays`, `TILE_COPY_POWER_OF_TWO_STRIDE_BYTES = 2048`, on the
  destination's column stride **in bytes** (so it follows `size_of::<T>()` exactly as the working-set
  gate does; nothing branches on the scalar). Below the byte threshold, the copy tiles when the stride
  is a power of two at or above 2048 bytes.
- What the sweep cannot separate: those inputs are square, so the 2048-byte stride is at once the
  row-major source's row stride and the destination's column stride. The clause is written on the
  destination stride -- the one the elementwise walk steps by a column at a time -- but a sweep that
  holds one stride fixed while moving the other (padded views) was not run, so that attribution is
  the mechanism the code reads, not a measurement that rules the source stride out.
- Not chosen: a shape test (square only) or an f64-only clause. The stride is what the measurement
  separates, and the same 2048 bytes of stride is a `256 x 256` f64, a `128 x 128` c64 or a
  `512 x 512` f32 destination.
- Not done: 1024 bytes (`128 x 128` f64) is not in the clause. The sweep's only power of two is 2048,
  and 1024 bytes was not measured.

## Verification conclusions and constraints

Sweep that established the mechanism (`#352` scan, evo/zen5, `Llt::new`, f64, order c, 5 interleaved
process sets, `--reps 5 --prime-ms 250 --statistic median`, 1T = cpu4, 8T = cpu4-11 in one L3 domain;
head `32f907b3` against base `620b96a0`):

| n | column stride | 1T head/base | 8T head/base |
|---|---|---|---|
| 240 | 1920 B | 0.928 | 0.900 |
| 244 | 1952 B | 0.927 | 0.894 |
| 248 | 1984 B | 0.941 | 0.904 |
| 252 | 2016 B | 0.929 | 0.892 |
| **256** | **2048 B** | **1.136** | **1.157** |
| 260 | 2080 B | 0.960 | 0.889 |
| 264 | 2112 B | 0.965 | 0.893 |
| 272 | 2176 B | 0.962 | 0.893 |

The dispatch-identical controls in the same run show the systematic bias of that pair of binaries:
`in_place` (same code in both) 1.006-1.020, order f (never reaches this gate) 1.019-1.060. Subtracting
it, the 256 regression is about +12% (1T) / +14% (8T).

Acceptance of the clause (`5ba37e58`, same protocol and host, `cholesky` f64/c32/c64, order c, three
arms: base `620b96a0` = tiles everywhere here, `32f907b3` = the walk the clause is for, and this
commit):

| dtype | n | 1T 352/base | 1T 352/330 | 8T 352/base | 8T 352/330 |
|---|---|---|---|---|---|
| f64 | 128 | 0.959 | 0.945 | 0.954 | 0.944 |
| f64 | 256 | **1.003** | **0.908** | **1.000** | **0.877** |
| f64 | 512 | 0.989 | 0.996 | 1.000 | 0.994 |
| c32 | 128 | 0.975 | 1.003 | 0.968 | 1.002 |
| c32 | 256 | 1.000 | 0.984 | 1.001 | 0.968 |
| c32 | 512 | 1.001 | 1.000 | 0.996 | 0.998 |
| c64 | 128 | 0.998 | 0.993 | **1.003** | **1.004** |
| c64 | 256 | 1.001 | 0.959 | 0.993 | 0.931 |
| c64 | 512 | 0.999 | 0.995 | 0.995 | 0.985 |

Worst cell against the base: **+0.32%** (8T c64 128), as the run's own aggregation over the five
interleaved process sets gives it (the per-process observations are in the raw CSVs, not in the
repository). Worst effect of the clause over the arm without it: **+0.37%** (same cell); the clause's win at 256 is 9.2-12.3% (f64), 4.1-6.9% (c64) and 1.6-3.2% (c32).

Guard cells -- the QR/LQ dispatch cells whose copied factor has a power-of-two destination
stride, f64, order c, five sets. The copied factor is not the input matrix: `copy_r` on a
`2048 x 256` input copies a `256 x 256` factor (1 MiB over both buffers), and `copy_l` on a
`256 x 2048` input does the same; `copy_r` on `64 x 2048` and `copy_l` on `2048 x 64` copy 2 MiB.
**All four are below `TILE_COPY_CACHE_BYTES`, so the clause flips every one of them** from the walk
to the tiles -- they are the cells the clause changes, not controls. Measured against the base:
`qr_factor 2048x256` 1T 0.990, 8T 1.001; `qr_factor 64x2048` 1T 1.006, 8T 0.997; `lq 2048x64` 1T
1.010, 8T **1.025**; `lq 256x2048` 1T 0.998, 8T 0.996. The worst of them is +2.5%, inside the 3% the
issue set. The #330 head was not measured on these four cells, so this table bounds the clause
against the base and says nothing about its size relative to #330.

Not measured, and therefore not claimed: f32; n = 1024 (tiled by the byte gate in both arms); tall and
wide factors at other powers-of-two strides (`1024 x 64` etc.); whether 1024 bytes of stride aliases
too. The clause is on the byte count, so those cells move without a measurement of their own; the two
gate tests pin which of them tile.

## Files

- `faer/src/linalg/mod.rs`: `TILE_COPY_POWER_OF_TWO_STRIDE_BYTES`, `tile_copy_pays` (new
  `col_stride` parameter), the call site, and the gate tests.
- `faer/src/linalg/qr/no_pivoting/factor.rs`, `faer/src/linalg/lq/no_pivoting/factor.rs`: the
  dispatch tests that reach this gate through a transposed copy (their `256 x 256` and
  `2048 x 64` / `64 x 2048` cells now tile).
- Raw runs, host and gate log: `~/work/faer330-evo/logs/{run352.log,raw352/}` and
  `logs/{run352g.log,raw352g/}` on the measuring host.
