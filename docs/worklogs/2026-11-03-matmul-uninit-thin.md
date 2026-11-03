# Overwrite-only matmul inherits the replace entry's one-row route

## Decisions

- Route a one-row (`m == 1`) destination in `matmul_uninit_imp` through `fill_zero` +
  `matmul_with_conj(Accum::Replace)`. That is the path the non-x86 native fallback already takes, so
  the fresh entry reuses `matmul_imp`'s `matvec_*` branch instead of selecting a second kernel. The
  destination's prior contents are still never read, and the O(n) initialization is not measurable.
- Rejected: an uninit-safe `matvec` implementation. Reusing `matvec_colmajor`/`matvec_rowmajor`
  requires initialized storage, because forming `MatMut` over `MaybeUninit` elements is not allowed;
  a separate uninit matvec would duplicate the kernel and its accumulation order.
- Rejected: covering `n == 1` in the same change. It has the same mismatch (on broccoli, base fresh
  is 0.28-0.77 of replace and the bits differ), but the fix makes `c64 141x282x1` 4.1x faster while
  `f64 141x282x1` becomes 1.28x slower than replace at 4T (10.5 -> 13.9 us). That needs its own
  decision and is recorded in tensor4all/faer#200.

## Verification conclusions and constraints

- Cause: `matmul_imp` has `dst.ncols() == 1` / `dst.nrows() == 1` branches to
  `matvec_colmajor`/`matvec_rowmajor`; `matmul_uninit_imp` had no such branch, and its x86 native
  block ran `private_gemm_x86::gemm`. Other targets already fall back to `matmul_with_conj` and were
  unaffected.
- Paired probe (replace vs fresh, same buffers, one process, per-iteration interleave, median of 200,
  three alternating process rounds), broccoli EPYC 7702P, measured CPU 30 at 1T, 28-31 at 4T (one
  L3 domain) and 28-35 at 8T (two 4-core L3 domains, the topology limit), observer 12-15, load
  8.96-12.86 on 64 cores. At every thread count, the `m = 1` cells moved from fresh/replace
  0.45-0.70 with 1-50 of 282 bits equal to 1.12-2.03 with all bits equal; `m = 2/8/16` and
  `141x282x141` are unchanged (ratios within 3%, all bits equal in both builds).
- Apple M5 Max (aarch64), `Par::Seq`: unchanged within noise, all bits equal, because that target
  already took the same path.
- `cargo test -p t4a-faer --lib`: 589 passed, 138 ignored. The focused
  `test_matmul_with_conj_uninit_thin_matches_replace_bits` covers f64 and c64 at `1x141x282` and
  `1x141x141` and fails on x86 AVX2 before the change.
- Not measured: the campaign comparator (this is a paired isolated driver, not a campaign run), a
  many-core scaling sweep beyond 8 threads, and M5 under the `threadteam` route for this cell.
- Upstream: Codeberg `main` at `0ecd846d` has no `matmul_with_conj_uninit` entry at all, so the
  defect cannot occur there; the entry is fork-only. Nothing was prepared for or sent upstream.
