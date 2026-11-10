# LQ on a column-major matrix: the panel is staged, the trailing update stays in place (tensor4all/faer#314, slice 3, step A)

## Decisions

- `lq::no_pivoting::factor::lq_in_place` factors the transposed view through the blocked QR
  kernel (`qr_in_place_packed`), and on that view every panel is a block of **rows** of `A`,
  `A.col_stride()` apart. Nothing in the panel recursion is layout-aware: the reflector is made on
  a strided column, the single-reflector SIMD apply runs only on a column-major target, and the
  small block applies inside the panel are products of strided slivers. Before this change a
  column-major LQ was 5.6x the same-flop column-major QR at `512 x 512` (one thread) and 11-18x at
  eight threads.
- **The panel is staged.** Each block that starts on the diagonal is copied into a column-major
  buffer of `(m - row) x block_size` (on the transposed view; `block_size` rows of `A`, each
  contiguous), factored there by the unchanged panel kernel (`factor_panel`: the recursion and the
  completed block factor), and copied back (`factor_panel_staged`). This is `gelqf`'s panel of
  `nb` rows with the rows made contiguous. Only the panel is copied, twice per block, about
  `m * n - k^2 / 2` elements over the whole factorization; no `m x n` buffer.
- **The trailing update stays a pair of matrix products on `A`, in place, but takes its basis from
  the staged buffer and keeps its temporary row-major**
  (`householder::apply_block_householder_transpose_on_the_left_in_place_with_conj_row_major_tmp`,
  a flag on the shared generic apply; every other caller keeps the column-major temporary).
  Staging the panel alone left the trailing apply 1.5x the column-major one at power-of-two
  leading dimensions: the reduction `V^H C` runs down the strided direction of `C`, the x86 GEMM
  does not pack that operand (`t4a-private-gemm-x86` packs its right-hand side only for a tall
  destination), and at a 1024- or 4096-element leading dimension every step lands in one L1 set.
  With the basis contiguous and the temporary row-major the reduction reads `C` along its rows:
  13.4 -> 6.4 ms against 5.9 for the column-major problem (`4032 x 960` view, 64 reflectors, one
  thread). At a 1000-element leading dimension all three are within 5%, so this is the
  power-of-two case, which is every benchmark shape.
- **Two panel buffers, so the look-ahead keeps working.** The look-ahead factors the next panel on
  a few participants while the others apply the current block to the far columns; the far apply
  reads the current basis from one buffer while the next panel is staged in the other, and the two
  swap. The next block's columns are copied into its buffer, and copied back after the join, on the
  whole team: inside the panel group the copies ran 2.5x slower than alone because they shared the
  memory system with the far apply (`1024 x 4096`, eight threads: 39.1 -> 30.3 ms).
- **Same packed form.** The reflectors, `L`, the coefficient blocks, the rank, the `infinity`
  markers and the zeros of the rank-deficient path are where they were, with the same values up to
  rounding, so `unitary`, `reconstruct`, `copy_l` and `LqParams`/`LqInfo` are unchanged. It is not
  the old route bit for bit (the panel's SIMD paths now run), and not the column-major QR of an
  explicit `A^T` bit for bit either (largest difference `5e-14` at `512 x 512`). A block that
  starts below the diagonal (after a rank loss) and the later chunks of a rank-deficient block are
  factored on `A`'s strides as before: exact rank loss is rare and that path is known to be right.
- **Implemented in the shared blocked driver, not as a second `gelqf` kernel.** The LQ of `A` is
  the QR of `A^T` step for step, so a separate driver in LQ coordinates would repeat the look-ahead,
  the rank-deficient chunking and the factor merge. The staging is a flag on the top level of
  `qr_in_place_blocked`, passed by `lq_in_place` only; QR's own entry passes `false`, so no QR
  result changes in this step. Step B of the slice decides where row-major QR takes it.
- **Scratch.** `lq_in_place_scratch` now reserves the two buffers (`2 * n * min(block, m, n)`),
  the panel's own applies and the transposed temporary. With a smaller stack the driver does not
  stage and the panels run on `A`'s strides, as before.
- **Measured and not taken here:** a tile copy without the `Option` staging array in
  `linalg::copy_across_layouts_block` makes the panel copies 1.6-2x faster (`1024 x 4096`, eight
  threads: 30.3 -> 27.9 ms end to end). That helper is shared by every row-major adaptation, so it
  belongs in its own change with that blast radius measured. Running the copies sequentially
  inside the panel group instead was worse (copy-in 10.5 -> 13.4 ms).

## Verification conclusions and constraints

- Timing, in-process and paired: the old route (the unstaged kernel on the transposed view, a
  hidden entry added for the probe only and not committed) and the new one alternate in one
  process, 11 rounds after 2 warm-up rounds, `f64`, column-major input, faer's own
  `recommended_block_size`, a held `threadteam` team for 8T; median ms. Code `3cdfaf34`, AMD Ryzen
  AI MAX+ 395 (Zen 5, 32 logical CPUs), CPU 4 for 1T and 4-11 for 8T, boost not switchable (no
  sudo); a foreign process (`kache-engine`, 11-16% of one CPU, on CPU 18) ran throughout. The
  last column is the same run's column-major QR of the explicit `A^T` (same flops), the cost the
  LQ would have with no layout penalty at all.

  | LQ `m x n` | t | before | after | after / before | column-major QR of `A^T` | after / QR |
  |---|---|---|---|---|---|---|
  | `512 x 512` | 1 | 14.72 | 3.74 | 0.254 | 3.24 | 1.153 |
  | `1024 x 1024` | 1 | 67.53 | 23.25 | 0.344 | 20.96 | 1.109 |
  | `4096 x 1024` | 1 | 219.58 | 128.45 | 0.585 | 117.29 | 1.095 |
  | `1024 x 4096` | 1 | 531.87 | 131.83 | 0.248 | 112.40 | 1.173 |
  | `512 x 512` | 8 | 17.00 | 1.74 | 0.103 | 1.30 | 1.342 |
  | `1024 x 1024` | 8 | 52.58 | 6.49 | 0.123 | 4.57 | 1.419 |
  | `4096 x 1024` | 8 | 90.50 | 22.17 | 0.245 | 18.37 | 1.207 |
  | `1024 x 4096` | 8 | 414.77 | 30.14 | 0.073 | 23.73 | 1.270 |

  The old route's `512 x 512` 8T cell moved between 13.0 and 17.0 ms across sessions; the others
  repeated within a few per cent. This is a probe, not the campaign's protocol, and no `fbench`
  cell was run.
- What remains, from a per-stage timing pass (temporary timers, removed): at one thread the
  trailing apply is 1.08-1.09x the column-major one and the two panel copies are 9% of the
  `1024 x 4096` time; at eight threads the far apply of the look-ahead is 1.25-1.3x the
  column-major one (the column split hands each task a block of rows of `A` that touches every
  column) and the copies are 6 of 30 ms at `1024 x 4096`. The faster tile copy above is the
  measured lead for the second; the first is not investigated.
- Not timed: `f32`, `c32`, `c64`, shapes off powers of two, 2-7 threads. `c64` is covered for
  correctness only.
- Inherited and not fixed here: column-major `c64` QR on a two-thread team differs from the
  sequential one by up to `3.4e-7` (Frobenius) on the `faer/tests/common` fixtures at
  `129 x 257` and `257 x 257` on `main` (`c7470bd`), and both reconstruct the matrix to `7e-16`.
  The fixtures are ill-conditioned (`1e9`-`1e11`), so this is conditioning, not a wrong result;
  the LQ comparison on them had held only because the strided route summed in the same order on
  one and two threads.
- Tests: `lq::no_pivoting::factor::tests::staged_panels_write_the_unstaged_packed_form` (staged
  against `qr_in_place_packed(.., false)` on the transposed view, f64 and c64, 48 cases with zero
  rows at the start, inside a panel and from a block boundary on, a padded leading dimension,
  sequential and on a 4-thread team). Dropping the copy-back of one panel column makes it fail at
  `7 x 9`. The existing look-ahead tests (`test_lookahead_matches_sequential`,
  `test_lookahead_rank_loss`) now run the staged path, and the LAPACK LQT01/02/03 family and
  `faer/tests/lq.rs` / `lq_no_alloc.rs` cover the rest. `cargo test -p t4a-faer`: 1033 passed,
  162 ignored at `3cdfaf34`, against 1032 and 162 at `c7470bd` (the new test).
- `faer/tests/lq.rs::lq_parallel_matches_sequential` failed at first (c64 `257 x 129`: `3.0e-7`
  against a bound of `6.5e-11`). It compares `Q` from a two-thread team with the sequential one
  entry by entry at `200 n eps`, on `sample_mat` fixtures whose condition numbers are
  `1e9`-`1e11`, where `Q` moves by about `cond * eps` under any change of summation order. The
  packed forms of the two runs agree to `1e-15`, and the staged team run reconstructs `A` as
  well as the sequential one. Its fixture now carries `16` on the diagonal (condition number at
  most 21); it passes at `3cdfaf34` and at `c7470bd`.

## Upstream

- `lq_in_place` is the fork's own entry point (tensor4all/faer#130); the strided panel it suffered
  from is the inherited QR kernel run on a transposed view, which upstream does not do. No
  upstream revision was measured for this record.
