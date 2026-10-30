# Equal-area split of the Cholesky trailing update (faer#160 item 2, first instance of faer#168 item 1)

Revision measured: `perf/160-chol-syrk` `d696bd1` (probe `d44a15d`) against its own `fork` arm, which
is `main` `7ff08ed`'s split. No upstream revision was measured for this change.

## Decisions

- **Cause.** At 2048/16T the trailing update `A11 -= A10 A10ᴴ` was 77 % of `cholesky_in_place` on
  broccoli (faer#160). The stage-1 trace pointed at a 2-way join at `triangular.rs:622`, but that line
  is the gate of the generic recursion. On x86_64 a native scalar hands the product to the
  private-gemm kernel on a fork of the team (`gemm_with`, `DstKind::Lower`), and the kernel splits it
  into tiles. That costs a pack section and a compute section per cache block, and the tiles above the
  diagonal leave participants without work.
- **Equal-area column ranges** (OpenBLAS `driver/level3/syrk_thread.c`). The triangle is cut into
  `width` column ranges of equal area, with boundaries on whole microkernel columns (8). Each
  participant makes one sequential kernel call on its trapezoid (rows `c0..m`, columns `c0..c1`), and
  all of them run in one `team::run` section. The width is the kernel's (`kernel_width`).
- **Working memory.** Each participant packs into the kernel's thread-local buffers, which the owning
  thread first-touches and reuses across calls. That is per-thread working memory, which AGENTS.md
  allows; nothing is allocated per call. Every participant packs its own `A10` rows, as OpenBLAS's
  threads do. The duplicated packing is at most `2p/m` of the arithmetic.
- **A shared part (faer#168).** The split is `utils::partition::lower_area_split(m, n, i, parts,
  align)`, with no Cholesky-specific input. An upper triangle uses the transpose, and a strict triangle
  shifts the rows. The trsm/trmm products, the LU (#123) update and the symmetric reductions can take
  their static parts from it.
- **Probe switch.** `cost::trace::LOWER_SPLIT` (feature `par-trace`) restores the kernel's split, so
  one binary measures both arms.

## Verification conclusions and constraints

- **Tests.**
  - `utils::team::tests::lower_product_by_area_matches_sequential` checks every entry against the
    sequential product. It covers both accumulation modes, the strict triangle, a row-major
    destination, sizes that are not multiples of the rounding, and c64, and it asserts that the area
    path ran. Shifting a part's offset makes it fail.
  - `utils::partition::tests::lower_parts_cover_the_triangle_with_equal_areas` checks the split.
  - The existing team triangular and factorization tests pass, as does the full `--lib` suite.
- **Decision rule declared before the run.** Area must beat fork beyond the spread at 2048/4096 for
  4-16T, and the whole call must not be slower beyond the spread anywhere.
  - Met: all six target cells have `max(area) < min(fork)`. The update ratios are 0.71-0.86, and the
    whole call at 4096 is 0.61-0.66 of `main`.
  - No cell regresses with separated ranges.
- **Residual: 1024/64T.** The whole call is +7.4 % at the median (3.35 → 3.60 ms). The ranges
  overlap, but 6 of the 7 paired repetitions are slower. Two observations bound it:
  - The standalone outer update moves by only 0.03 ms there, 0.75 → 0.78 ms, with overlapping
    ranges.
  - The rest of the call (`chol − update`, which includes the 64-wide updates inside the 128 diagonal
    blocks) grows by 0.22 ms.

  It is unattributed. One hypothesis is that at 64 parts on the small trailing blocks, rounding the
  boundaries to 8 columns leaves many parts empty, and the section still wakes every participant. A
  cap of `parts <= area / (align · m)` would bound that. It is not implemented, because it was not
  measured. This change ships the measured code.
- **Against MKL.** `dpotrf` is still ahead at 4096 (`area/MKL` 1.22-1.54 at 4-64T). At 2048/8-16T the
  two are about level (1.02-1.03). The update itself is at or below MKL `dsyrk` from 16T at 4096. What
  is left is outside the outer update, and this run cannot attribute it (see faer#160's tables).
- **Unverified.**
  - evo 1-8T non-regression is deferred to after the weekly benchmark.
  - One host (Zen 2) and f64 only.
  - The C driver does not check `dpotrf` INFO, and no run checks the factorization's residual. The
    tests check correctness.
