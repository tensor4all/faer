# perf-faer-001: QR of a row-major matrix is up to 31 times slower than copying it to column major first

**What happens:** the column-pivoted `qr_in_place` takes 28 to 31 times as long on a row-major matrix as copying it into a column-major buffer, factoring that and copying the factor back; the unpivoted one takes 1.3 to 1.6 times as long. The copied route gives the column-major factorization, permutation included, bit for bit.
**Which input:** any tall or square `f64` matrix stored row major (a column stride of 1). The program uses 256 x 256 to 1024 x 1024 and the tall 1024 x 64, 1024 x 256 and 4096 x 256.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-11-06.

## Reproduce

```sh
cargo run --release
```

The program factors the same matrix three ways, with `Par::Seq`, for both `qr::no_pivoting::factor::qr_in_place` (recommended block size) and `qr::col_pivoting::factor::qr_in_place`:
the row-major storage as given; the adapted route (copy into a column-major `Mat`, factor, copy back), with both copies inside the timed region; and a column-major `Mat` with no copy, for scale.
It asserts that the adapted factor, block coefficients and (column pivoting) permutation equal the column-major ones bit for bit, and exits with status 1 when a row-major call takes at least 1.25 times the adapted route.

**Host: Apple M5 Max** (6 performance and 12 efficiency cores), macOS 26.7.2, rustc 1.96.0.
This is not the Ryzen host the rest of the fork's campaign uses, and these numbers are not comparable with that campaign's.
macOS has no thread affinity, so the process asks for the performance cores (`QOS_CLASS_USER_INTERACTIVE`) and is not pinned.
The host was shared with other work during the run (load average 5 to 6 on 18 cores).

| QR | m x n | row major, ms | adapted (copy, factor, copy back), ms | column major, ms | row major / adapted | adapted = column major, bit for bit |
|---|---|---:|---:|---:|---:|---|
| no pivoting | 256 x 256 | 1.16 | 0.79 | 0.70 | **1.46** | yes |
| no pivoting | 512 x 512 | 6.76 | 4.92 | 4.34 | **1.37** | yes |
| no pivoting | 1024 x 1024 | 45.26 | 34.76 | 32.27 | **1.30** | yes |
| no pivoting | 1024 x 256 | 6.22 | 3.97 | 3.41 | **1.57** | yes |
| no pivoting | 4096 x 256 | 29.88 | 18.68 | 15.55 | **1.60** | yes |
| column pivoting | 256 x 256 | 34.27 | 1.19 | 1.09 | **28.84** | yes |
| column pivoting | 512 x 512 | 281.72 | 8.96 | 8.46 | **31.43** | yes |
| column pivoting | 1024 x 64 | 13.97 | 0.50 | 0.41 | **28.01** | yes |
| column pivoting | 4096 x 256 | 870.50 | 27.83 | 24.25 | **31.27** | yes |

The direct row-major factorization is a different rounding of the same QR: its `R` differs from the column-major one by at most 1.3e-13 (4096 x 256) in this run, which the program prints and does not assert.

Each ratio is the median of 7 samples per route, the three routes alternating sample by sample, each sample at least 100 ms of calls (one call where a call is longer).
The input is reloaded before every call, outside the timed region; one untimed call per route primes the buffers.

## Cause

Neither QR has a path for a row-major matrix.

* **Column pivoting** (`src/linalg/qr/col_pivoting/factor.rs`). After each reflector the pivot search needs the remaining column norms. With a unit row stride and SIMD it updates them by a delayed rank-one correction (`delayed_update`, lines 170-172). Otherwise it applies the rank-one update to the whole trailing matrix through `matmul` and recomputes every remaining column norm with `norm_l2` (lines 173-192), which on a row-major matrix walks each column one element per cache line, at every step.
* **No pivoting** (`src/linalg/qr/no_pivoting/factor.rs:258`): with the `perf-warn` feature it logs that QR prefers column major (lines 270-281), and the unblocked panel core is column oriented: `make_householder_in_place` reads and scales the pivot column (line 37), and the rank-one update walks each column of the block with `inner_prod` and `z!(tail, v)` (lines 65-80). The blocked updates outside the panel go through `matmul`, which handles either layout, so the loss is the panel's.

## How the fork fixed it

The fork's `qr_in_place` (both flavours) copies a matrix whose columns are not contiguous into a column-major buffer, factors it there and copies the factor back, as this program's adapted route does: commits `044f947`, `ae92684` (no pivoting) and `10ee1c9` (column pivoting); ledger rows "Row-major QR layout adaptation" and "Row-major col-pivoting QR".
The arithmetic is the column-major factorization's, so the result is the column-major factor bit for bit.
The fork's copy is tiled 16 x 16; this program's copy is `faer`'s own `copy_from`.
tensor4all/faer#23 later limited the copy to matrices of at least 64 x 64 elements, where it pays.

## Affected versions

Confirmed by running this program: `faer` 0.24.4 on an Apple M5 Max.
The fork measured the same defect on x86_64 before fixing it (`docs/worklogs/2026-10-15-qr-row-major-layout.md`, `docs/worklogs/2026-10-15-row-major-adaptation-lu-colpiv.md`); those numbers are not repeated here.
Not fixed upstream in 0.24.4, the latest release.

## References

* Fork fix: commits `044f947`, `ae92684`, `10ee1c9`; records `docs/worklogs/2026-10-15-qr-row-major-layout.md` and `docs/worklogs/2026-10-15-row-major-adaptation-lu-colpiv.md`.
* Copy floor: tensor4all/faer#23.
