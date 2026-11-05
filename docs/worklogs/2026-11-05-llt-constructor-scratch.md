# Caller-owned scratch for owning LLT construction

## Decisions

Colmat retains faer's owning Llt to reuse its factors. Its canonical constructor
allocates kernel scratch internally and offers no way to consume a caller's
workspace, although the low-level factor kernel already has a query/stack API.
The current main 2d4ea4e four-scalar probe observes two allocations per warmed
32-by-32 construction: the owning factor and temporary scratch. Scratch is
128/256/256/512 bytes for f32/f64/c32/c64. Input/setup and probe workspace are
outside the measured constructor. No latency claim is made.

Add Llt::new_scratch and require MemStack in the canonical Llt::new, keeping the
existing numerical implementation and result/error semantics. Its output still
owns independently allocated factor storage; only kernel temporaries use caller
scratch. Preflight the query before allocating the output. Mat::llt remains the
existing automatic-workspace convenience, and the generalized self-adjoint
reduction uses that same convenience. No alternative compatibility constructor
or downstream duplicate factor representation is introduced.

Other owning dense constructors and Colmat's integration remain outstanding.
The public contract checks four native scalars, actual complex phases, both
selected triangles and C/F layouts, conjugated views, zero/singleton shapes,
undersized workspace, and nonpositive pivots. Inner numerical loops and parallel
splits are unchanged. No vendor implementation is consulted.

## Verification conclusions and constraints

The complete crate run passes 915 tests, with 162 pre-existing ignored,
including 84 doctests. The expanded public cases also pass with and without
threadteam: two tests covering both sides/layouts at n=0/1/3/65/257 (two
128-column panels and a tail), plus complex phases/conjugation/pivot errors.
Eight opt-in LAPACK-inspired Cholesky tests pass. The real nightly no_std
executable constructs and checks an owning factor using caller scratch.

The identical independent release probe against the candidate reports one
allocation, the owning factor, for every scalar and all three warmed
repetitions; the parent's two-allocation result and all residuals are unchanged.
Generic field implementations and unmeasured shapes are not covered by that
allocation observation. The later public-test expansion and no_std formatting
leave the production expressions of the complete crate run unchanged.
Hosted CI explicitly covers sequential-only and team configurations and is
pending. Query scratch excludes owning factor
storage and automatic-workspace convenience calls still allocate that scratch.

Acceptance scope is Cholesky and its generalized self-adjoint EVD consumer,
all declared layouts/dtypes at 1/4/8 threads, with LU/QR/solve controls. The
canonical campaign, fbench verification, paired candidate comparison and hosted
checks are pending. The available Ryzen 9 PRO 8945HS exposes one L3 domain over
all 16 logical CPUs, so it cannot satisfy off-domain observer pinning. No
hardware/profile/pinning waiver, timing/scaling claim, adoption or registry
publication is implied. Keep the change draft until acceptance is complete.

The executed allocation reproducer and canonical API proposal are preserved in
[issue #268](https://github.com/tensor4all/faer/issues/268).
