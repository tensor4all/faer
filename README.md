# t4a-faer

`t4a-faer` provides dense and sparse linear algebra in Rust for scientific and HPC applications.
`tenferro` and other tensor4all libraries use it. It is not API compatible with upstream faer.

The caller owns the `threadteam` team and passes `Par::Team` for parallel work, or `Par::Seq` for sequential work.
faer does not use an implicit global thread pool. See the [migration guide](docs/par-team-migration.md).

[Benchmarks](https://tensor4all.org/faer-benchmark/): published measurements and their conditions live here.
**Status:** Published versions follow semver; APIs can change in incompatible releases.

See the [migration guides](docs/migration/README.md) for Eigen, BLAS, LAPACK, Julia, NumPy/SciPy,
nalgebra, ndarray, and upstream faer users. The [LAPACK routine table](docs/migration/lapack-names.md)
maps `?getrf`, `?geqp3`, `?gesvd`, `?syevd`, `?potrf` and the rest to the call that replaces each.
Coding agents in a downstream project can be given
[`skills/faer-usage/SKILL.md`](skills/faer-usage/SKILL.md).

# Why trust it

Each item below is one click from a file in this repository. No number here is a projection; where a
number is a snapshot of one run, the file and the run are named.

- **Differential validation against LAPACK.** Sixteen families are ported from LAPACK 3.12.0's
  `lapack-netlib/TESTING` (via the OpenBLAS snapshot recorded in
  [`docs/lapack-validation-sources.json`](docs/lapack-validation-sources.json)): SVD, Hessenberg,
  Schur, nonsymmetric EVD, QZ and Schur reordering, balancing, tridiagonal and self-adjoint EVD, LU,
  Cholesky, indefinite LDL, QR, LQ, triangular solves and least squares
  ([`docs/lapack-validation.md`](docs/lapack-validation.md)). The recorded combined run selects 325
  tests: 71,856 principal calls and 50,952 dependent calls. The suite runs nightly and by hand
  ([`.github/workflows/lapack-validation.yml`](.github/workflows/lapack-validation.yml)); a run is
  green only when nothing fails outside [`scripts/lapack-known-failures.json`](scripts/lapack-known-failures.json),
  which is **empty**, so nothing is currently excluded from the green.
- **Allocation contracts are tested, not asserted.** A counting allocator proves zero heap
  allocations in a steady-state call: `faer/tests/svd_values_no_alloc.rs` (values-only SVD, four
  scalar types, 1×1 to 700×700), `faer/tests/no_step_alloc.rs`, `faer/tests/lq_no_alloc.rs`,
  `faer/tests/evd_svd_vectors_no_alloc.rs` and
  `faer/tests/generalized_schur_reorder_no_alloc.rs`. The reason a 0.20.1 transitive dependency
  broke one of them is written next to the version pin in `faer/Cargo.toml`.
- **Defects found in the published upstream crates are reproducible.** [`upstream-reports/`](upstream-reports/)
  holds 40 defects, each a standalone Cargo project that depends only on the published crate, prints
  expected and actual values and exits non-zero; 38 are against `faer` 0.24.4, two against `gemm` and
  `private-gemm-x86` ([`upstream-reports/README.md`](upstream-reports/README.md)). Each entry names
  the release it was confirmed against and whether a fix has reached upstream.
- **CI gates the contract, not only the build.** [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
  builds the crates with and without default features, runs the selected no-`rayon` test targets,
  checks that the migration guides are byte-identical to the examples CI runs
  (`scripts/check-migration-examples.py`), builds and runs those examples, runs the doctests, and
  checks the package contents. `.github/workflows/code-quality.yml` runs `cargo clippy -D warnings` as
  a **blocking** gate for `t4a-faer-traits`, and the compiler is pinned in
  [`rust-toolchain.toml`](rust-toolchain.toml) (1.96.0) so a local run lints with the same toolchain
  as CI. `t4a-faer` itself has 2,340 clippy warnings on that toolchain and its clippy step is
  therefore advisory — the count is recorded in the workflow file rather than hidden.
- **Performance is measured on named hardware at pinned revisions, and published with its conditions.**
  [tensor4all.org/faer-benchmark](https://tensor4all.org/faer-benchmark/) carries the acceptance
  protocol and the host profiles. Pages recorded on the development host are marked *provisional* and
  will be superseded by the dedicated measurement host's dataset (faer-benchmark,
  `docs/provisional-and-clean-datasets.md`). Read them as conditions-bound numbers, not as a promise
  of parity with any vendor library.


This is a tensor4all fork of [faer](https://codeberg.org/sarah-quinones/faer).
The [GitHub repository](https://github.com/sarah-quinones/faer-rs) is a mirror of upstream.
The crate body stays `MIT`, keeping the upstream copyright and attribution, and code ported from
other projects keeps its own `COPYING` terms (see [`NOTICE`](NOTICE)).

# Community

Questions and discussion about this fork go to the tensor4all Matrix space:
[#tensor4all:matrix.org](https://matrix.to/#/#tensor4all:matrix.org). Bug reports and feature
requests go to the [GitHub issues](https://github.com/tensor4all/faer/issues).

The upstream community is the [zulip server](https://faer.zulipchat.com), for questions about upstream
faer itself. The fork's own channel is the Matrix space above.

<p align="center">
  <img src="https://faer.veganb.tw/faer-logo-color.png" alt="faer logo"/ width="25%">
</p>

# faer

[![documentation](https://docs.rs/t4a-faer/badge.svg)](https://docs.rs/t4a-faer)
[![crate](https://img.shields.io/crates/v/t4a-faer.svg)](https://crates.io/crates/t4a-faer)

`faer` is a rust crate that implements low level linear algebra routines and a high level wrapper
for ease of use, in pure rust, with a focus on portability, correctness and performance. This fork
keeps upstream's API where it can and changes it where the tensor4all workload needs another shape;
[`PATCHSET.md`](PATCHSET.md) lists every difference.

Upstream documentation, for the same crate with an API that differs in places: the
[upstream website](https://faer.veganb.tw),
[docs.rs/faer](https://docs.rs/faer/latest/faer), and the upstream
[zulip server](https://faer.zulipchat.com).

# parallelism

There is no process wide parallelism setting. Every operation that can run in parallel takes a
`Par`: `Par::Seq` runs it on the calling thread, and `Par::Team` runs its regions on the members of a
`threadteam` team that the caller owns. Owning the team is the primary route: a caller with a thread
budget builds the pool and the team once and lends the team to each call.

```rust
// once: 8 participants, the calling thread and 7 reserved workers
let pool = threadteam::Pool::builder().threads(8).build()?;
let team = pool.into_team(8).ok().expect("an idle pool");

// per call: lend the team to faer for the duration of the closure
for (a, h) in work {
    let mut exec = team.exec(8);
    exec.team(|t| qr_in_place(a.as_mut(), h.as_mut(), Par::Team(t), stack, params))
        .ok()
        .expect("a team token for an idle team");
}
```

A caller that already holds a token lends a sub-team to a phase with `par.reborrow()`; a width of one
runs the phase on the calling thread.

`faer::team::with_team` is the one-off route. It builds a pool, reserves its members, runs one closure,
and releases them afterwards, so it suits a single call or a script that runs a single operation:

```rust
faer::team::with_team(8, |par| {
    // every faer call in here with `par` runs its regions on 8 participants: the calling thread
    // and 7 reserved workers
    qr_in_place(a.as_mut(), h.as_mut(), par, stack, params);
});
```

Each `with_team` call acquires a pool and team for its closure, and threadteam can reuse cached
workers across calls rather than spawning and joining them each time. A program with a sequence of
calls pays that acquisition per call, which is why it should own the `team` as above. The
[migration guide](docs/migration-guide.md#explicit-parallelism) gives the measured cost of each route.

See the *parallelism* section of the crate documentation for the full picture.

# minimum supported rust version

1.89.0, declared as `rust-version` in `faer/Cargo.toml`. Rust 1.89 is the first release where the
AVX-512 target features are stable, which is what lets the `x86-v4` kernels build without the nightly
compiler. They stay opt in until that path is settled.

# contributing

Upstream's contribution guide applies to this fork as well: start with the issues labelled "good
first issue", and ask on the [upstream zulip server](https://faer.zulipchat.com) or in the issue
itself. Fork specific reports belong in this repository's issues, and a defect that also reproduces on
the published upstream crates belongs in
[`upstream-reports/`](upstream-reports/README.md). Undefined behaviour is checked weekly by Miri on the
`t4a-faer-traits` crate only; [`docs/miri.md`](docs/miri.md) says how to run it and why the main
crate is not in that lane yet (test selection, not a capability limit).

# benchmarks

The fork's own results are the published campaign pages, and the upstream benchmark page is at
[faer.veganb.tw/benchmarks](https://faer.veganb.tw/benchmarks/). Performance lessons that carry over
between changes are in [`PERFORMANCE_TIPS.md`](PERFORMANCE_TIPS.md).
