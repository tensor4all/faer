# perf-gemm-001: `gemm` does not use the matrix unit of an Apple M4 or M5

**What happens:** on an Apple chip with SME, a single-threaded `gemm::gemm` product runs on the NEON kernels, 4.7 to 8 times slower than the same product on SME, with bit-identical results.
**Which input:** any `f64` or `f32` product from n = 512 up on a host with `FEAT_SME` (Apple M4 and later). The program uses square products of n = 256 to 2048.
**Which version:** `gemm` 0.19.0, the latest release on crates.io, with or without its `experimental-apple-amx` feature.
**Added to this list:** 2026-11-06.

## Reproduce

```sh
cargo run --release
cargo run --release --features upstream-amx   # upstream's own opt-in Apple path
```

The program times `C = A B` (column major, `Parallelism::None`) with `gemm` 0.19.0 and with `t4a-gemm` 0.20.1, which is the same crate plus Apple SME kernels, in one process.
It exits with status 1 when upstream is at least 2 times slower than SME at every size from n = 512.

**Host: Apple M5 Max** (6 performance and 12 efficiency cores), macOS 26.7.2, rustc 1.96.0, `FEAT_SME = 1`, `FEAT_SME_F64F64 = 1`.
This is not the Ryzen host the rest of the fork's campaign uses, and these numbers are not comparable with that campaign's.
macOS has no thread affinity, so the process asks for the performance cores (`QOS_CLASS_USER_INTERACTIVE`) and is not pinned.
The host was shared with other work during the run (load average 5 to 6 on 18 cores).

| type | n | `gemm` 0.19.0, ms | GFLOP/s | SME, ms | GFLOP/s | upstream / SME | results bit-identical |
|---|---:|---:|---:|---:|---:|---:|---|
| f64 | 256 | 0.586 | 57.3 | 0.179 | 187.2 | **3.27** | yes |
| f64 | 512 | 4.709 | 57.0 | 1.002 | 268.0 | **4.70** | yes |
| f64 | 1024 | 38.584 | 55.7 | 6.643 | 323.3 | **5.81** | yes |
| f64 | 2048 | 306.202 | 56.1 | 43.241 | 397.3 | **7.08** | yes |
| f32 | 256 | 0.301 | 111.4 | 0.085 | 394.4 | **3.54** | yes |
| f32 | 512 | 2.372 | 113.2 | 0.445 | 603.4 | **5.33** | yes |
| f32 | 1024 | 19.202 | 111.8 | 3.020 | 711.1 | **6.36** | yes |
| f32 | 2048 | 156.223 | 110.0 | 19.734 | 870.6 | **7.92** | yes |

With `--features upstream-amx` the upstream column is the same within the run-to-run spread (f64 n = 1024: 35.6 ms, 60.3 GFLOP/s; ratio 5.98), and the results are still bit-identical to SME's.

Each ratio is the median of 11 samples per implementation, the two implementations alternating sample by sample, each sample at least 50 ms of back-to-back calls after two untimed priming calls.
The whole `gemm` call is timed, including its dispatch and packing.
Before timing, four entries of each result are checked against a sum formed in the program, and the two results are compared bit for bit.
The program prints each implementation's max/min sample ratio; it was at most 1.6 in this run.

## Cause

`gemm` 0.19.0 chooses its aarch64 kernel in `gemm-common-0.19.0/src/gemm.rs:1026-1035`: NEON, unless the `experimental-apple-amx` feature is on and `HasAmx::get()` is true.
`has_amx_impl` (`gemm-common-0.19.0/src/cache.rs:52-70`) is true only when the CPU brand string is `Apple M1`, `Apple M2` or `Apple M3`.
On an M4 or M5 the AMX path is therefore never taken, and there is no SME kernel at all, so every product runs on the NEON microkernel.
That is about 57 GFLOP/s (f64) and 112 GFLOP/s (f32) on one M5 Max performance core, whatever the size.

## How it reaches faer

`faer` 0.24.4's `matmul` calls `gemm::gemm` on every target other than x86_64 with AVX2 and FMA (`src/linalg/matmul/mod.rs:1416`), which includes every Apple chip.
Every blocked factorization that reaches `matmul` through its trailing update gets the same kernel.
This program does not measure that path through faer.

## How the fork fixed it

The fork depends on `t4a-gemm` 0.20.1 (tensor4all/faer#189 `6e7cfcc6`, #190 `573c6181`), a fork of `gemm` that adds SME kernels behind a runtime probe (`FEAT_SME`, the matching element-type feature, a 64-byte vector length, macOS 26 or later) and a per-element-type size gate.
A host that fails the probe keeps the NEON kernel.
tensor4all/faer#189 also records that faer's QR and LU at 6 threads got slower with the first SME kernel, because their trailing updates are narrow-K products; the size gate in 0.20.1 is the fork's answer to that, and this program does not measure it.

## Affected versions

Confirmed by running this program: `gemm` 0.19.0 on an Apple M5 Max.
Not checked on an M4.
Not fixed upstream: 0.19.0 is the latest release.

## References

* Fork fix: tensor4all/faer#189, tensor4all/faer#190; the kernels are in [tensor4all/t4a-gemm](https://github.com/tensor4all/t4a-gemm).
