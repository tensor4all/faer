<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tensor4all/tensor4all-meta/main/branding/colmat/svg/colmat-wordmark-dark.svg">
  <img alt="colmat" src="https://raw.githubusercontent.com/tensor4all/tensor4all-meta/main/branding/colmat/svg/colmat-wordmark.svg" height="64">
</picture>

[![documentation](https://docs.rs/colmat/badge.svg)](https://docs.rs/colmat)
[![crate](https://img.shields.io/crates/v/colmat.svg)](https://crates.io/crates/colmat)

`colmat` is a pure-Rust dense and sparse linear algebra library for scientific and HPC work. It is
the linear algebra backend of [tenferro](https://github.com/tensor4all/tenferro-rs) and the other
[tensor4all](https://github.com/tensor4all) libraries, and it changes upstream faer's API where
their workloads need a different shape: parallel work runs on a `threadteam` team that the caller owns
and passes as `Par::Team` (or `Par::Seq` for the calling thread), with no implicit global thread
pool. Its decompositions are validated against LAPACK's own test suites. Published versions follow
semver, and APIs can change in incompatible releases.

**Renamed from `t4a-faer`.** The crate was published as `t4a-faer` up to 0.33.0; 0.34.0 is the
same library under the name `colmat` (`use colmat::...`). See the
[rename note](CHANGELOG.md#breaking-the-crate-is-renamed-to-colmat). The repository moved from
`tensor4all/faer` to `tensor4all/colmat-rs`; old links redirect, and issue numbers are unchanged.

## Links

- Crate and API docs: [crates.io/crates/colmat](https://crates.io/crates/colmat), [docs.rs/colmat](https://docs.rs/colmat)
- Benchmarks, with their measurement conditions: [tensor4all.org/faer-benchmark](https://tensor4all.org/faer-benchmark/)
- Migration guides: [from upstream faer, nalgebra and ndarray](docs/migration-guide.md), and
  [from Eigen, BLAS/LAPACK, Julia and NumPy/SciPy](docs/migration/README.md)
- Issues: [github.com/tensor4all/colmat-rs/issues](https://github.com/tensor4all/colmat-rs/issues)
  (read the [review rules](docs/review-rules.md) first)
- Community: [#tensor4all:matrix.org](https://matrix.to/#/#tensor4all:matrix.org)

## Origin and acknowledgment

colmat is derived from [faer](https://codeberg.org/sarah-quinones/faer) by Sarah Quiñones
(GitHub mirror: [sarah-quinones/faer-rs](https://github.com/sarah-quinones/faer-rs)). Its design,
kernels and most of its code are faer's, and we thank Sarah Quiñones and the upstream faer
contributors for them. [`PATCHSET.md`](PATCHSET.md) records how colmat differs from upstream, and
[`upstream-reports/`](upstream-reports/README.md) holds reproducers for defects found in the published
upstream crates. colmat is not upstream faer, and questions about it belong here rather than with
the upstream project.

colmat is released under the MIT license ([`LICENSE`](LICENSE)), which carries the upstream copyright,
Copyright (c) 2026 sarah quiñones el kazdadi. Changes made in this repository are Copyright (c)
tensor4all contributors under the same license. Code ported from other projects keeps its own terms;
[`NOTICE`](NOTICE) lists the attributions and the `COPYING.*` files that ship with the crate.
