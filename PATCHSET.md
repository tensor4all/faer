# PATCHSET

The ledger of the experiments this fork runs. It records the diff against
upstream `faer`.

## Status legend

- **TODO** — not started. The issue column names the driving issue, or `—`.
- **EXPERIMENT** — being implemented on a branch. Breaking things is fine.
- **KEEP** — tensor4all keeps using it. Whether it goes upstream is a separate
  decision.
- **DROPPED** — tried and thrown away. The reason stays here.

## Fork infrastructure (not candidates for upstream)

| Change | What |
| --- | --- |
| Crate names | `faer` → `t4a-faer`, `faer-traits` → `t4a-faer-traits`. `[lib] name` stays `faer` / `faer_traits`, so downstream `use faer::...` is unchanged. |
| Workspace | `faer-ffi` is excluded (the C API and its cbindgen build are unused by tensor4all). |
| Publication | crates.io, in the order `t4a-faer-traits` then `t4a-faer`. The tag is `v<faer version>`. A version that is already on crates.io is skipped, so a release that bumps only one crate still succeeds. |
| LICENSE | A `LICENSE` copy lives in each crate directory so the published archive ships the MIT text (`license-file` next to `license` makes Cargo warn). |

## Experiments

| Experiment | What | Status | Issues | Notes |
| --- | --- | --- | --- | --- |
| Unpivoted QR column skip | Householder QR drops a column whose part orthogonal to the previous ones is below `16·(m−k)·ε`. The tall/wide `svd` path goes through it too. LAPACK has no such skip. | TODO | tlinalg-rs#28, tensor4all-rs#836 | Likely an upstream bug. Any upstream issue or PR is opened only after the fork maintainer confirms. |
| Typed uninitialized overwrite destination | A `MatUninitMut<T>`-style destination that never reads the prior output, accepted by `matmul_with_conj`. | TODO | strided-rs#198 | strided-rs currently returns a Faer-only `Unsupported`. |
| Native SVD / eigh direct output | At nbatch=1, stop writing U and the eigenvectors into an intermediate `Mat` and copying them out; write into the caller's output region instead. | TODO | tlinalg-rs#25 | Preserve output shape, initialization, and failure cleanup. |
| Borrowed reflector columns | Stop copying reflectors out of the compact QR buffer into `basis` scratch; borrow them through a column split. | TODO | tlinalg-rs#17 | Numerics, the tau convention, and the restored diagonal beta stay identical. |
| Thread policy | A path that maps the caller's rayon pool / work model (lanes × item threads) onto `faer::Par` without relying on ambient global state. | TODO | tenferro-rs#2000, strided-rs#6 | Nested parallelism and the thread budget are the subject. |
| Scratch / plan reuse | Hold workspaces, scratch, and plans across calls. | TODO | — | Decide whether to file an issue when work starts. |
| Strided / batched views | Accept general strides and batches directly, without packing. | TODO | — | — |
| c64 planar | Accept a split real/imaginary complex layout directly. | TODO | — | — |

## Ground rules

- One experiment per branch. Keep the diff small enough to rebase onto an
  upstream release tag.
- No unrelated formatting churn or dependency bumps; every extra diff is a
  rebase cost.
- Every experiment carries its own test, run locally by whoever wrote it. The PR
  gate (build) is not a substitute for recording the result.
- Anything intended for upstream is cleaned up and reviewed separately before it
  is proposed.
