# Pfaffian: compact shared panels and swap-free tiled stores

## Decisions

- Keep physical slots fixed and pivot only the logical index map. Panel vectors
  use physical rows; after each completed panel, gather the surviving factors
  once into one shared compact panel. No matrix or factor permutation pass
  remains. Upper entries are negative transposes, never conjugates.
- Split fixed logical row blocks on the caller's team. Each block reuses its own
  scratch tile, invokes sequential faer GEMM for `X Y^T - Y X^T`, and scatters
  each unordered pair and its opposite sign together. Tile boundaries and GEMM
  summation order do not depend on the team width. Seq uses the same route.
- The initial panel reads authoritative lower input directly; subsequent panels
  read only initialized surviving pairs. Keep the complete finite-input scan,
  but remove the standalone dense input copy and never read eliminated slots.
  Working storage adds O(n b), not a second O(n²) matrix or a panel per worker.
- `b` counts elimination columns, hence b/2 rank-two pairs. Both 16 and 32 have
  correctness evidence. Keep the existing 32 **provisionally**, not as a tuned
  winner: only evo/zen5-cpu can choose the value and decide adoption.

## Byte model and provenance

Main already used delayed updates with 32-column panels. The change is coefficient
reuse through the owning GEMM implementation and compact active rows, not the
invention of blocking. The old ordered SIMD recurrence remains for dependent
panel-column materialization, not the trailing matrix.

For f64, a useful trailing pair has 2b flops and 24 matrix-interface bytes (one
read plus two skew stores): 12/b bytes/flop, .75 at b=16 and .375 at b=32. This
excludes coefficients, tile scratch, GEMM packing, validation, panel work and
edge-tile arithmetic, and is **not measured DRAM traffic**. Extra diagonal-tile
flops are not included in the conventional n³/3 useful-flop numerator.

The algorithm remains the independent implementation of Wimmer and Satyarth et
al. cited in the module. GEMM is faer's existing implementation. No vendor or
PfaPack source was consulted; the numerical comparisons use previously recorded
public PfaPack outputs, with a direct 2x2 oracle.

## Verification conclusions and constraints

- The committed implementation at `8bad72f` passes the Pfaffian integration
  tests (12 passed, 0 failed) and the frozen numerical grid (48 passed, 0 failed)
  on broccoli. Maximum grid-relative error is 1.3350018239792163e-13, and all
  four execution modes agree exactly. CPUs 60–63 were used for correctness;
  Team8 was oversubscribed, not an eight-core performance experiment. No
  correctness fix was needed. Raw per-case outputs, source/binary identities
  and gate logs are retained under
  `/home/shinaoka/work/pfaffian-ref/176-swap-fusion/pr181-check.pDkDAf/`.
- Both b=16 and b=32 pass the frozen 48-case grid: n=2/64/128/256/512/1024,
  f64/c64, Seq/Team1/Team4/Team8. Maximum relative errors against the public
  reference are 3.480e-13 and 1.335e-13 respectively. On each build, all four
  execution modes agree bitwise. These are correctness-only broccoli runs.
- Both widths pass all 12 public contract tests, the complete-call zero-heap-
  allocation guard and the private tiled-update oracle. The oracle covers
  arbitrary physical permutations, first/later updates, partial tiles, four
  native types, fx128/cx128, NaN-poisoned owned work/ignored input and exact
  finite canaries for unowned/padding entries. At n258 f64 it asserts actual
  parallel trailing-row dispatch, not merely a requested team width.
- The final default library suite passes: **589 passed, 138 ignored**. The
  no-std linalg check, three Pfaffian doctests and focused nightly formatting
  also pass. Broccoli's actual runtime lane is V3/AVX2 (f64 four lanes, c64 two).
- Std-only tests have an inherited determinant-oracle failure: nine pass and
  one fails with exactly the same values on untouched #177 and this candidate.
  Clippy is also not green: its 34 dependency-free errors match the untouched
  base's messages and locations exactly; dependency linting stops earlier in
  faer-traits. Neither gate was relaxed or reported as passing.
- Completed independent post-reviews cover the sequential/API and trailing/
  ownership scopes. Their nonfinite-pivot classification finding reproduces on
  #177: a NaN behind a zero candidate could be misreported as ZeroPivot. Reject
  nonfinite live candidates; the 8x8 regression now returns NumericalFailure
  under Seq/Team3/Team8. The sentinel finding is covered by exact finite canaries.
  Both 48-case grids and the relevant gates were rerun after these corrections.
- Raw fixtures, source snapshots, per-case output and gate logs are retained at
  `/home/shinaoka/work/pfaffian-ref/176-swap-fusion/`; see its `REPORT.md`.
  Upstream control and formal performance adoption have not been measured.

## What remains unverified

No new wall times or measured GFLOP/s are claimed, and deleting the physical swap
routine is not yet a measured zero-cost profile. Metadata index swaps still
exist. evo is reserved for other work: width selection, phase profiling,
coefficient/cache traffic, n64/n1024 Seq guards and the unchanged Seq/Fortran<=1.05
and Team8/Fortran-speedup>=4 gates at n256..1024 remain pending. Draft PR
[#181](https://github.com/tensor4all/faer/pull/181) preserves the candidate; it
is not ready for performance adoption, merge or release, and #176 remains open.
This correctness follow-up does not revalidate the historical broader gates or
resolve the existing std-only/clippy failures. evo was not accessed. Historical
instrumented and uninstrumented runs must keep their source/build identities
separate; neither establishes this candidate's performance.
