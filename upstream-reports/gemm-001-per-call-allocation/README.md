# gemm-001: every matrix product allocates on the heap

**What happens:** each call of `gemm::gemm` allocates on the heap, even when the same product was just computed.
**Which input:** any product. Size and element type do not matter.
**Which version:** `gemm` 0.19.0 and `gemm-common` 0.19.0, the latest releases on crates.io.
**Added to this list:** 2026-10-30.

## Reproduce

```sh
cargo run --release
```

The program counts heap allocations with a counting `#[global_allocator]`.
It computes the same f64 square product twice, at sizes 8 to 256, and prints the allocations of the second call.

| n | allocations, x86_64 with AVX-512 |
|---:|---:|
| 8 | 1 |
| 16 | 1 |
| 32 | 1 |
| 64 | 1 |
| 128 | 1 |
| 256 | 1 |

The issue that found this, tensor4all/faer#58, counted 2 allocations per product on aarch64 (Apple Silicon).
That was through faer, which reaches `gemm` from its decompositions.

## Cause

`gemm-common` 0.19.0 builds its work buffers per call:

* the packed operands: `dyn_stack::MemBuffer::new(rhs_req.and(lhs_req))` (`src/gemm.rs:437`), on the paths that pack;
* the "was this left block packed" flags: `alloc::vec![false; mc / MR]` (`src/gemm.rs:490`, and `:654` for the threaded path).

Only the L2 slab is reused (`src/gemm.rs:59`, a `thread_local!` buffer).
`gemm` has no entry point that takes a caller's `MemStack`, so a caller cannot supply these buffers.

## How it reaches faer

faer's `matmul` calls `gemm::gemm` when it cannot use `private-gemm-x86`.
That is every target other than x86_64, and x86_64 without AVX2 and FMA.
Upstream faer 0.24.4 has the same dispatch (`src/linalg/matmul/mod.rs:1358` and `:1416`).
So on those targets a faer decomposition allocates on every call, even when the caller passes a `MemStack` sized by the matching `*_scratch` request.
Two examples: `self_adjoint_evd` with eigenvectors, through `upgrade_householder_factor`, and `bidiagonal_svd` from n = 128, through the divide and conquer merge.

On x86_64 with AVX2 and FMA, faer calls `private-gemm-x86` instead, and these decompositions do not allocate.
The fork's test `faer/tests/evd_svd_vectors_no_alloc.rs` checks this.

## Affected versions

Confirmed by running this program: `gemm` 0.19.0 on x86_64.
Confirmed through faer on aarch64: tensor4all/faer#58.
Not fixed upstream.

## References

* Found in: tensor4all/faer#58.
* Fork plan: tensor4all/faer#56. The fork intends to replace the `gemm` fallback with a product dispatched on the caller's team, which can also take the caller's scratch.
* Not fixed in the fork yet.
