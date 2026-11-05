# perf-faer-002: the unit lower triangular solve is up to 2.6 times slower at a power-of-two stride

**What happens:** `solve_unit_lower_triangular_in_place` takes up to 2.6 times as long when the right hand side's column stride is a power of two as when the same values are copied to a padded buffer, solved there and copied back, and the two give the same bits.
**Which input:** a `b x b` unit lower triangle and a `b x b` right hand side, `f64`, with a column stride of 512, 1024 or 2048 elements (4, 8 or 16 KiB). This is the stride of the `U` panel solve inside a partial pivoting LU of a power-of-two matrix.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-11-06.

## Reproduce

```sh
cargo run --release
```

The program solves the same system three ways with `Par::Seq`: in place at the power-of-two stride; in place at that stride plus 8 elements; and staged, which copies the right hand side out of the power-of-two storage into a padded buffer, solves there and copies back, both copies inside the timed region.
It asserts that the three solutions are bit-identical, and exits with status 1 when the in-place solve is at least 1.5 times the staged one for some case.

**Host: Apple M5 Max** (6 performance and 12 efficiency cores), macOS 26.7.2, rustc 1.96.0.
This is not the Ryzen host the rest of the fork's campaign uses, and these numbers are not comparable with that campaign's.
macOS has no thread affinity, so the process asks for the performance cores (`QOS_CLASS_USER_INTERACTIVE`) and is not pinned.
The host was shared with other work during the run (load average 5 to 6 on 18 cores).

| b | stride, elements (bytes) | in place, us | stride + 8, us | staged, us | in place / staged | results bit-identical |
|---:|---|---:|---:|---:|---:|---|
| 64 | 512 (4 KiB) | 14.3 | 7.1 | 8.1 | **1.77** | yes |
| 64 | 1024 (8 KiB) | 17.6 | 7.0 | 8.2 | **2.16** | yes |
| 64 | 2048 (16 KiB) | 22.4 | 7.1 | 8.5 | **2.63** | yes |
| 128 | 512 (4 KiB) | 80.2 | 48.9 | 54.4 | **1.47** | yes |
| 128 | 1024 (8 KiB) | 94.5 | 49.4 | 55.8 | **1.69** | yes |
| 128 | 2048 (16 KiB) | 115.0 | 49.5 | 55.7 | **2.06** | yes |
| 256 | 512 (4 KiB) | 481.9 | 347.5 | 371.4 | **1.30** | yes |
| 256 | 1024 (8 KiB) | 531.9 | 347.1 | 369.1 | **1.44** | yes |
| 256 | 2048 (16 KiB) | 598.5 | 342.2 | 365.4 | **1.64** | yes |

Each time is the median of 11 samples per arm, the three arms alternating sample by sample, each sample the mean of at least 20 ms of calls.
The right hand side is reloaded before every call, outside the timed region, so every arm starts from the same warm data; one untimed call per arm primes it.

## Cause

`solve_unit_lower_triangular_in_place_imp` (`src/linalg/triangular_solve.rs:420`) splits a right hand side of more than 64 columns in halves while the triangle has at most 128 rows (lines 430-452), and below that recurses down to a base case of at most 4 rows (`recursion_threshold`, line 213; base case at line 454).
The base case and the short products beside it walk the rows of a chunk of up to 64 columns.
When the column stride is a power of two of at least 4 KiB, one row of every column of the chunk maps to the same few L1 sets, so a row walk keeps evicting the lines it just loaded.
Neither the triangle's stride nor the arithmetic matters: with only the right hand side padded the solve is as fast as fully padded.

## How the fork fixed it

The fork stages such a node: a node of 16 to 128 rows whose columns would put at least 24 lines into one L1 set copies its right hand side into a per-thread buffer at a stride that does not alias, runs the unchanged recursion there and copies back (tensor4all/faer#276, commits `e114eebc`, `17f3d73b`).
Same operations in the same order, so the result is bit-identical, which is what this program asserts for its simpler whole-matrix staging.
The fork measured 2.5 to 4.6 times at a power-of-two stride on a Zen 5 host before the change (`docs/worklogs/2026-11-05-trsm-aliased-rhs.md`); those numbers are not repeated here.

## Affected versions

Confirmed by running this program: `faer` 0.24.4 on an Apple M5 Max.
Not fixed upstream in 0.24.4, the latest release.

## References

* Fork fix: tensor4all/faer#276; record `docs/worklogs/2026-11-05-trsm-aliased-rhs.md`; ledger row "Staged right hand side in the unit-lower solve at an aliasing stride".
