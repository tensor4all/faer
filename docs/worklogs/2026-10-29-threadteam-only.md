# Threadteam as the only parallel route (tensor4all/faer#85)

Issue: tensor4all/faer#85 (also closes #59, who holds spindle's worker group).
Branch: `refactor/85-threadteam-only`, base `origin/main` `0a4713a`.

## Why

The fork carried two complete parallel routes. Every consumer (tenferro-rs, tlinalg-rs,
tcontract-rs) enables `std,rayon,threadteam` and passes `Par::Team` or `Par::Seq`; nothing
constructed `Par::Rayon` any more, but the library still shipped it: `Par::Rayon`/`ParShape::Rayon`,
the Rayon arms of every dispatch entry point, a Rayon cost model with its own hand-kept copy of the
host tables, `spindle` as a dependency, and 109 `#[cfg(feature = "rayon")]` sites that made the
second route look load-bearing. Maintaining two routes means every gate, every kernel arm and every
test had to be read twice. The maintainer's decision (2026-10-28) was to delete the second route and
keep `threadteam`, and the caller-side answer to #59 is `faer::team::with_team`.

## What the branch removes

- `Par::Rayon`, `Par::rayon`, `ParShape::Rayon`, `ParShape::rayon`, `Par::__Lifetime`.
- The Rayon arms of `thread::join_raw`/`for_each`/`for_each_with_par`/`split`, of
  `ParShape::dispatch_cost`/`min_dispatch_cost`, of `par_pays_time`/`par_can_pay_time`, and of the
  GEMM route. `parallelism_degree` folds into `Par::degree`, `is_team` goes.
- The Rayon cost model in `utils::cost`: `outside_cost_of`, `in_pool_cost_of`, `width_model`,
  `rayon_cost`, `min_rayon_cost`, `caller_in_pool`, the outside/in-pool anchor tables, and the
  tests that asserted them. faer's own serial `Rates` table and its `HostClass` key stay (the rates
  are faer's, not the dispatch model's); the rule `PAYS`/`pays_at` is now re-exported from
  `threadteam` (one implementation, no mirror to keep in sync).
- The `spindle` dependency, from `Cargo.toml` and from the `rayon` feature. No production code
  opened a group; the remaining users were three test bodies and the two example harnesses, which
  now use a team. The lock still names spindle through `t4a-private-gemm-x86`, which is a separate
  matter.
- `faer::team::with_team(width, |par| ..)` is added (it owns a `threadteam::Pool` for the closure,
  a width of one hands out `Par::Seq`), and it is reachable as `faer::team::with_team`. This is
  what `faer-benchmark`'s `fbench` and `tensor4all-meta`'s threshold probe use instead of
  `Par::rayon(n)`, so #59 (who holds the worker group) is answered by the caller owning a team.

## The feature graph

The route is a feature, on by default: `threadteam` compiles the team (and pulls `rayon`, the
substrate the team iterates through); `rayon` alone keeps the `gemm` dependency's Rayon arm and the
public `par_*` iterators. Both names stay because the consumers pass `features = ["std", "rayon",
"threadteam"]`. With `threadteam` off there is no `Par::Team` and no team module: `Par<'team>`
carries `Seq` and the uninhabited `__Lifetime` variant that owns the lifetime (the shape the crate
used while the route was experimental), and nothing pulls Rayon, i.e. std, into the build. That is
what keeps `faer-no-std-test` linkable and the inherited `nostd-nightly` job green: a build without
the route has `Par::Seq` as its only token. The matches that would otherwise be non-exhaustive get
`#[cfg(not(feature = "threadteam"))]` arms, and the code that names Rayon types keeps its
`#[cfg(feature = "rayon")]` gate.

## Tests and examples

81 lib-test sites and 15 example sites constructed the removed constructor. The conversion is
`rayon::ThreadPoolBuilder` + `pool.install(|| f(.., Par::rayon(n), ..))` →
`faer::team::with_team(n, |par| f(.., par, ..))`; a `[Par::Seq, Par::rayon(n)]` pair becomes the
sequential run followed by the same closure inside `with_team`; `Par::rayon(0)` in a test becomes a
team of four (the fork's build/test budget). Tests that only asserted the removed route went with
it: `rayon_uses_the_outside_cost`, `outside_rows_follow_the_classes`, `min_rayon_cost_is_the_cheapest_width`,
`in_pool_rows_follow_the_classes`, `a_held_spindle_group_is_a_worker_of_the_pool`,
`derived_floors_outside_the_pool_are_larger_than_on_a_team`. The tests that stated a contract
survive, converted: the recursive-leaf count and the sized-join shares (`utils::thread::tests`), the
"the region reaches the team and not the caller alone" barrier (spindle's thread-index test became a
peak-concurrency barrier), the threshold semantics and the degree-one rejection in `utils::cost`, and
the LU region rules against the team table. An explicit width in the base is kept as it was
(`with_team(8, ..)` where `Par::rayon(8)` stood, `12` where it stood); only the ambient
`Par::rayon(0)` becomes four, the fork's build/test budget. Two new tests carry the oracles the
deleted comparisons used to carry for the complex scalars: `bidiag::tests::
bidiag_on_a_team_reconstructs` (blocks 4/3 and 8/8, five shapes, f64 and c64) and
`hessenberg::tests::test_hessenberg_cplx_team_reconstructs_blocked` (96x96, blocked path) reassemble
`A` from the reflectors the team route produced.

`examples/bench.rs` (upstream's own benchmark example) built `Par::rayon(0)` per function and held a
spindle group in `main`. It now resolves a width (`RAYON_NUM_THREADS` or the machine), reports it in
the row name, hands the vendor its thread count, and runs the body of each benchmark inside
`faer::team::with_team(width, ..)` for the faer arm only — a vendor call is not run beside a
reserved faer team. The spindle spin/pause tuning and `bench_sparse.rs`'s wrapper around a
sequential benchmark are deleted.

## Verification

- `cargo test -p t4a-faer --lib` (all of it).
- `cargo check`/`build` for `--no-default-features`, `--no-default-features --features std`, default,
  `--features std,rayon,threadteam`, and `cargo +nightly-2025-01-08 run --profile nostd` in
  `faer-no-std-test` (the inherited nostd job).
- `rg 'Par::rayon|ParShape::Rayon|spindle'` in `faer/src` and `faer/examples` finds no route.

## Acceptance measurement

The claim is mechanism, not speed: deleting the second route must not move the team route. So both
revisions are measured *on the team route* (`origin/main` carries `Par::Team` too), paired, with
the same driver.

Driver: `faer-benchmark/probes/team-nonregress` (out-of-tree diagnostic, not a measured arm: the
campaign harness `fbench` still builds its rows from `Par::rayon`, so it cannot measure this
branch until its own change lands). It links one faer checkout by path, owns a `threadteam::Team`
for the whole cell and lends the token to each repetition (no acquisition inside a timed region,
which is what a caller holding a budget does on both revisions), and covers `gemm`, `qr_factor`,
`qr`, `lu`, `cholesky`, `svd_values`, `svd_thin`, `eigh` at 128/256/512, `f64` and `c64`, at 1T,
4T and 8T. Protocol: boost off (checked, `/sys/devices/system/cpu/cpufreq/boost` = 0), 250 ms
priming, 5 timed repetitions, median per cell, three process sets per cell combined by the median
of their medians; the measured process pinned to `0` (1T), `0-3` (4T) or `4-11` (8T) — one logical
CPU per physical core, one L3 domain per cell — and the observer (this agent and its shell) moved
off that domain for the duration.

| budget | cells | median candidate/baseline | min | max | cells outside +-5% |
| --- | --- | --- | --- | --- | --- |
| 1T | 48 | 1.0001 | 0.9423 | 1.0586 | 3 |
| 4T | 48 | 0.9989 | 0.9633 | 1.0654 | 1 |
| 8T | 48 | 0.9978 | 0.9665 | 1.0648 | 1 |

The cells outside +-5%:

- `cholesky f64 128` at 1T/4T/8T: 1.059 / 1.065 / 1.065, the one *slower* cell. It is a 54 us
  call, so the delta is about 3 us, consistent across the three budgets, and the baseline's own
  three process sets span 0.99-1.07 of its median there at 1T and 8T (0.991-1.006 at 4T). A
  constant few-microsecond difference on the smallest cell is what a codegen change in the gate
  path looks like, but this measurement cannot separate that from a real regression: recorded as
  the one regression to watch.
- `svd_values c64 128/256` at 1T: 0.942 / 0.948 (and 0.963-0.978 at 4T/8T) - the candidate is
  *faster* there, with a tight baseline spread (0.997-1.001); not explained by the route removal,
  recorded as what was measured.

Base revision: `5e45c57` (the merge of #52 that this branch is rebased onto).

OpenBLAS arm, alongside (the same shapes, host and protocol, from the campaign's existing
`lapack-openblas` arm binary, 3 process sets): the fork row is **0.63x** the vendor at 4T,
**0.59x** at 8T (median of the 48 cell ratios per budget) and **0.77x** at 1T. The vendor is ahead
on `eigh` `f64` at 128 only (1.2-1.25x); on `qr` the fork is 0.13-0.30x of it. Full table in
`work/runs/faer-85/vendor/`.

Limits of this evidence: it is an out-of-tree diagnostic driver with the campaign's *timing*
protocol, not a `record_run.py` campaign run (no manifest, no comparator, no pin movement), so
these numbers are indicative and are not a page. The vendor row is a cross-harness comparison.


## What remains unverified

1. **`threadteam` needs std** (`Pool` spawns OS threads and its `parallel` feature pulls Rayon). A
   build that wants the route therefore has std, and a build without std has no route:
   `faer-no-std-test` links and runs with `Par::Seq`, but a no_std application cannot have parallel
   faer. That is the limit of the graph, not something missing.
2. **Complex factorizations on the team route.** At a forced floor (`ParThreshold::Fixed(1)`,
   block 8, 96x96) the team route produces a different bidiagonalization (1.38 max element
   difference) and Hessenberg reduction (9.26) than the sequential one, for `c64` only; `f64`
   agrees exactly. Measured at `origin/main` `0a4713a` as well (same numbers with `Par::Team`), so
   it is the fork's team route, not this PR. Both are *valid* factorizations: the reflectors from
   the team route reassemble `A` in both cases (`bidiag::tests::bidiag_on_a_team_reconstructs`,
   `hessenberg::tests::test_hessenberg_cplx_team_reconstructs_blocked`). The element-by-element
   comparison between the two routes therefore stays for the real scalars only; for the complex
   ones both routes still run and must produce finite factors, and the reconstruction is the
   oracle. Whether two complex routes agree up to the phase freedom of a Householder reflector is
   not analysed; nothing downstream depends on the representation, only on the reconstruction and
   on the spectra, which the SVD/EVD suites check.
3. **`faer-ffi`** (`faer-ffi/src/lib.rs`) still converts `ParTag::Rayon` to `Par::rayon` and is
   not built by this workspace; it needs its own execution-ownership design (a C caller cannot
   own a scoped team). Not compiled or verified here.
4. **The campaign harness and the threshold probe** (`faer-benchmark`'s `fbench`, `tensor4all-meta`
   `docs/design/thresholds-data/src/thr-faer.rs`) still build `Par::rayon`; they cannot build
   against this branch until their own changes land. That is the stated follow-up order.
5. The width of the tests that used `Par::rayon(0)` ("all threads") is now four, the fork's build
   and test budget; no test was verified to behave the same at eight.
