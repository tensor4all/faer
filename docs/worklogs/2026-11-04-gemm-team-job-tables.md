# The x86 kernel's job tables come from the team caller's workspace

faer#223: `gemm_team` allocated the `private_gemm_x86` kernel's three parallel job tables on
every call (and per depth panel). `t4a-private-gemm-x86` 0.1.28 offers `gemm_with_scratch`, which
takes them from caller memory; this is the faer side that supplies them.

## Decisions

- **The workspace is a per-thread reusable buffer in `utils::team`**, grown on demand to
  `MilliScratch::required(m, n)` (`m + n` bytes) and lent to `gemm_with_scratch` for the duration
  of one call. faer's `AGENTS.md` ("Working Memory In Parallel Code") asks for exactly this shape:
  a buffer the owning thread allocates once and reuses, and the region gate
  (`tests/no_step_alloc.rs`, faer#111) warms up before it counts.
- **Rejected: a public workspace** (`matmul_scratch` plus a `matmul` entry taking a `MemStack`).
  The plain matmul route has no workspace today, and every internal caller (`spicy_matmul`, QR,
  LU, Cholesky) and the FFI would have to thread one. That is a public contract change for a
  follow-up to a kernel change.
- **Rejected: a slot on the team token.** `threadteam::TeamExec` has none, it would be a
  cross-repository change, and it would put mutable state behind a token that is exclusive
  authority.
- **Not the cache the gemm-x86 side rejected.** Only the bytes survive the call: no team, fork or
  closure is stored, and the borrow ends with `gemm_team`. The gemm-x86 side has no pool of its
  own, so a table cache there would have had to carry `Par::Team`'s lifetime; here the buffer
  belongs to the thread that already holds the token.
- **A nested product on one thread falls back.** `job_tables` takes the buffer with
  `try_borrow_mut`; if it is already lent out, `gemm_team` calls `gemm_with`, whose kernel
  allocates the tables. No panic, no wrong result, and the fallback is not on any current path
  (nothing in the kernel's `Fork::run` re-enters a team product).

## Verification conclusions and constraints

- Allocations per `gemm_team` call, this host, 4 participants, one process per number:
  **3** with faer on gemm-x86 0.1.27 (three tables, per depth panel), **2** with faer on 0.1.28
  still calling `gemm_with` (the kernel's fallback), **0** with this change, warm, and 0 again for
  a larger shape once the workspace has grown and for a shape below the grown size.
  `tests/no_step_alloc.rs` asserts this and fails on the pre-change code with exactly 2.
- **Products are bit-identical** to `origin/main` for 4 dtypes x 4 shapes
  (128x96x64, 384x320x448, 96x64x32, 301x129x33) at 4 participants, so the requested 1e-13
  relative bound holds for f64 and c64 with zero difference, and f32/c32 are bit-identical too, so
  it holds for them as well; the in-tree tolerance check uses 2e-5 for the single-precision types,
  because that is their epsilon.
- **No performance claim.** The change removes allocations, not work. Spot check, same host, cpus
  0-7, 4 participants, one run each, median us: 128^3 32.3 -> 31.4, 256^3 230.3 -> 229.8,
  512^3 1631 -> 1596. Not a campaign pair and not an adoption-protocol measurement.
- Tests: `utils::team` 35 pass, including the two added here (a grow -> shrink shape sequence on
  one held team for all four dtypes; the nested-borrow fallback) and the existing `Par::Team`
  nesting/lifetime, dtype and triangular/indexed sweeps. `tests/no_step_alloc` passes.
- aarch64: `cargo check -p t4a-faer --lib --target aarch64-unknown-linux-gnu` passes. The whole
  change is inside `cfg(all(target_arch = "x86_64", feature = "std"))`, the crate's gemm-x86
  dependency is x86_64-only, and `gemm_tiled` / `Par` are untouched.
- Constraints: `t4a-private-gemm-x86` 0.1.28 was not on crates.io when this branch was written
  (0.1.27 was the newest published), so the local verification patched the dependency to the
  merged commit with `--config 'patch.crates-io...'`, which is not part of the change. 0.1.28 was
  published on 2026-11-03 before the merge; `origin/main` pins it from the registry, and the
  merge commit's `Cargo.lock` carries that entry.
- Upstream: the team route and `gemm_with_scratch` are fork-only. Nothing was prepared for or sent
  upstream.
