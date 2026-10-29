# Matrix product into uninitialized storage (#136)

## Goal

strided-rs#198 holds output buffers as `MaybeUninit<T>` and wants a product that overwrites
them without first filling them, which the upstream request sarah-quinones/faer-rs#269 asks
for. This fork adds `matmul_with_conj_uninit` taking a `MatUninitMut` and returning the
initialized `MatMut`. The view type and the first tests come from shinaoka's upstream PR #270.
Nothing was posted upstream.

## Decisions

- **Kernel routes.** Overwriting is not enough; the destination must never be read and never
  be borrowed as `&T`/`&mut T` before it is written.
  - `t4a-private-gemm-x86` with `Accum::Replace` meets both. It writes every element in its
    first depth block (later blocks add to what that block wrote) and loads the destination
    only when accumulating. It is the route for the native types on x86_64 with AVX2 or
    AVX-512, directly or through `gemm_team` on a team.
  - The `gemm` crate with `read_dst = false` was the first plan for the other targets. It was
    dropped after the independent review traced two exceptions in `gemm-common` 0.19: its GEMV
    builds a `&mut [T]` over the destination before writing, and its experimental Apple AMX
    kernel copies the destination into a staging tile before it checks `alpha`.
  - `nano-gemm` scales the destination by `beta`, so it reads it.
  - On those targets the native types are therefore zero-filled through raw writes and then
    multiplied by the ordinary path. This is correct everywhere, and gains nothing there.
- **Other scalars.** One dot product per element, written with `ptr::write`. A team splits
  the columns.
- **Panics.** The `MatMut` is formed only after the whole write. A panic leaves the caller's
  storage `MaybeUninit`, possibly partly written, and never exposed as initialized.

## Limitations

- Miri cannot run the SIMD kernel. Under Miri the native types take the zero-fill route, so
  Miri checks the generic and fill paths, and the NaN-prefill tests check that the x86 kernel
  does not carry old contents into the product. Neither proves the absence of a discarded read
  inside the kernel; that rests on reading its epilogues.
- The timing is one shape on a shared host (in `PATCHSET.md`), not a quiet-host measurement.
