# Row-major input to `Llt::new`: the triangular copy

## Decisions
- The campaign's 1024 f64 8T `cholesky` cell was 3.3x slower row-major than column-major (5.5 against 1.7 ms) while
  MKL moved 7%. The factorization is the same call on the same column-major matrix in both orders. The difference
  is `copy_from_triangular_lower`. Its zip has no common contiguous direction for a row-major source and a
  column-major destination, so it reads one element per cache line. Gap at 1T and at 8T, fork and upstream faer:
  3.7-3.8 ms. Copy alone: 4.16 against 0.30 ms.
- Chosen: the tile `to_owned` already uses, applied band by band to the lower triangle. A pure copy, so the factors
  cannot change; the gate test compares the factor of one matrix through both orders bit for bit.
- Not chosen: removing the copy. `Llt` returns an owned column-major factor, so a row-major input is transposed
  once. Reading the row-major buffer as its column-major transpose would read A's upper triangle, which
  `Llt::new(A, Side::Lower)` promises not to.

## Verification conclusions and constraints
- Copy at 1024 f64, one thread: 4.16 -> 0.65 ms row-major; 0.30 ms column-major, unchanged.
- Campaign cell (fbench, pinned, three processes per arm, MKL in process): row-major 5.445 -> 2.236 ms, column-major
  1.674 -> 1.649. LU is not on this path and did not move.
- Not done: `copy_from` (full matrix, 8.6 ms row-major) and a parallel copy for `Llt::new`, which has a `par`.
  The remaining row-major gap, about 0.4-0.6 ms, is the tiled copy itself plus its `zeros` pass.
- Harness observation, not about this change: the `lu` cell's row-major time depends on what ran earlier in the
  process (4.7 to 8.0 ms) unless glibc's mmap and trim thresholds are pinned. Measured cells here were checked both
  ways.
