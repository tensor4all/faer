# Caller workspace for scaled Schur back substitution

## Decisions

- Address tensor4all/faer#270 at the private range-scaling owner. The issue's
  predeclared public workload (n=32, four native scalars, ordinary/large/small,
  preallocated output/query buffer, Seq, three warmed repetitions) establishes
  the need: the public call has one hidden matrix allocation at extreme scales.
  This is a declared-workspace contract change, not a time-share hypothesis.
- Replace the optional owning copy with a fully initialized stack matrix.
  Preserve thresholds, two-factor powers of two, copied Schur band, arithmetic
  order, conjugation and the original public T used by final normalization.
  Do not pre-scale the whole public call or change numerical loops.
- Thread the private stack requirement through saved-Schur, full EVD left/right,
  and partial EVD consumers. Persistent output V/X stays live while the optional
  scaling copy is borrowed; sequential left/right phases reuse the same region.
- The existing parallel partition and team ownership stay unchanged. One shared
  normalized input is formed before the solves; this adds no per-thread copy,
  task cap, threshold, cache or allocator pool. Use the existing temporary-matrix
  layout, preserving this ownership change's layout against its parent rather
  than coupling a leading-dimension optimization into it.

## Verification conclusions and constraints

- Parent and fork baseline is `2d4ea4eab8d8f724269508e16b4aaafaf5c67ae7`.
  No Codeberg update, vendor implementation, LAPACK code or new algorithm is used.
- Functional acceptance covers four scalars, C/F and conjugated views, empty and
  singleton matrices, real pairs, ordinary/threshold/extreme/subnormal scales,
  insufficient scratch without output mutation, independent residuals and norms,
  full EVD left/right/both and partial EVD consumers, plus allocation probes.
  These are requirements; candidate outcomes are not established yet.
- Performance acceptance is predeclared for saved-Schur vector targets at n=32
  large/small scale (four scalars, C/F, 1T/4T/8T), all declared full/partial EVD
  populations at 4T/8T and target 1T, with LU/QR/SVD collateral controls under the
  adoption protocol's fixed priming/reps/AA/statistic/noise/affinity policy. The
  matching nonsymmetric/partial EVD harness populations must first exist; the
  current harness has `eigh`, not these operations. Baseline/candidate clean
  revisions and the same harness identity are recorded before any campaign.
- No campaign, fbench verification or paired comparison has run. The available
  single-L3 host cannot supply the required off-domain observer pin. No protocol
  requirement is waived; a functional candidate remains draft and unadopted
  until the declared acceptance and required hosted checks are complete.

## Executed functional evidence

- `cargo check -p t4a-faer`: pass.
- `cargo test -p t4a-faer`: 28 completed test binaries/doc-test groups,
  915 passed and 162 pre-existing ignored tests; no failures. This run preceded
  the final expansion of the allocation integration test, which was rerun
  separately below. Production source was unchanged between these runs.
- `cargo test -p t4a-faer --test schur_scaling_no_alloc`: pass after expansion.
  All four scalar types, n=0/1/3/17/32, C/F input and output, three scales and
  three warmed repetitions: 360 allocation observations, each zero, with
  dimensionless residual and unit-norm checks. Four additional one-byte-short
  workspace probes panic before mutating the output sentinel. These probes use
  `Par::Seq` and identity Q; they do not establish held-team coverage.
- The original standalone release reproducer passes all 36 candidate probes
  with zero allocations, versus one hidden allocation at large/small scale in
  the recorded baseline. This is allocation evidence, not timing evidence.
- Private range-copy tests pass for four scalars, both retained-band widths,
  threshold boundaries, extreme/subnormal scale and empty/zero input, with
  poisoned scratch to check complete logical initialization.

Remaining functional work includes held-team allocation coverage, targeted full
and partial EVD consumer coverage, and the opt-in numerical validation families.
Performance acceptance remains outstanding as described above.

## Held-team follow-up defect

Executed the same standalone four-scalar n=257 C/F, identity-Q, held-team2
probe at the unchanged baseline and this candidate. Both complete all numerical
residual and norm checks. Ordinary F allocation counts are 0/48/72/126 for
f32/f64/c32/c64 (0/91776/130560/395136 requested bytes). Ordinary C counts are
zero. Extreme scales require one additional matrix at the baseline; this
candidate removes that matrix and leaves the same 0/48/72/126 allocations in
both layouts. All three repetitions agree per cell. Exact allocating kernel
has not yet been isolated; these are whole-call observations.

Reported separately with executable source and unchanged-baseline results as
[tensor4all/faer#271](https://github.com/tensor4all/faer/issues/271). Preserve
its stronger zero-allocation assertion as an explicitly ignored opt-in
regression probe, rather than accepting observed allocations as a new budget.
The default Seq guard remains strict zero-allocation. The held-team guard was
executed and failed at f64 as expected; do not cite it as passing. Standalone
baseline/candidate probes retain numerical assertions and collect all counts.

`cargo +nightly check -p t4a-faer --no-default-features --features linalg`
also passed (existing warnings only).

## Consumer validation and final functional gate

- Opt-in `cargo test -p t4a-faer --lib lapack_nonsymmetric -- --ignored`:
  all eight tests pass (four scalars, sequential cases and held-team parallel
  cases). Existing checks cover full EVD values-only/left/right/both, multiple
  layouts including conjugation, saved-Schur extraction and range fixtures.
- Added four public partial-EVD tests: n=96, min/max Krylov dimensions 32/64,
  three eigenvectors, deterministic triangular operator and scales 1, 2^30,
  2^-30. All 12 cases pass using exactly the public query. Verify the three
  expected largest eigenvalues and dimensionless relative residuals; complex
  cases have nonzero imaginary components. The tolerance passed to partial
  EVD is absolute and scales with the operator, as the implementation's
  `residual <= tol` requires. Partial EVD does not promise unit normalization,
  so the oracle requires a finite nonzero norm and divides residual by it.
  Initial tests incorrectly assumed normalization and an unscaled tolerance;
  these assumptions were corrected from the documented/source contract,
  without changing the residual threshold or production code.
- Final default crate suite: 29 completed groups, 919 passed, 163 ignored
  (162 inherited plus the explicitly retained #271 allocation probe).
  The complex fixture was subsequently strengthened to nonzero imaginary
  components and its focused four tests rerun: all pass.

Functional evidence does not satisfy the performance adoption gate. Keep the
PR draft and the downstream dependency on the unchanged baseline. In particular
#271 is an executed inherited failure, not a passing acceptance cell.
