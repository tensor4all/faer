> **The linear algebra foundation of the tensor4all stack.**
>
> `t4a-faer` is a tensor4all fork of [faer](https://codeberg.org/sarah-quinones/faer). The crates are
> published on [crates.io](https://crates.io/crates/t4a-faer) and `tenferro` and the other tensor4all
> libraries depend on them. The crate body stays `MIT`, keeping the upstream copyright and
> attribution, and code ported from other projects keeps its own `COPYING` terms (see
> [`NOTICE`](NOTICE)).
>
> ### What differs from upstream
>
> - **Parallelism runs on `threadteam`.** Every parallel region runs on a `Par::Team` token that the
>   caller owns, and `faer::team::with_team(width, |par| ..)` is the helper for a scoped call.
>   `Par::Rayon` and the spindle route are gone.
> - **Correctness fixes** that upstream has not taken yet, each with its own worklog.
> - **A nightly LAPACK numerical validation**, whose known failures are listed with the issue that
>   tracks them.
> - **MSRV 1.89**, and the AVX-512 kernels (`x86-v4`) are an opt-in feature.
>
> Every change is recorded in [`PATCHSET.md`](PATCHSET.md), one row per experiment, and defects that
> also reproduce on the published upstream crates are written up in
> [`upstream-reports/`](upstream-reports/). Published versions follow semver.

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
`threadteam` team that the caller owns. A caller with a thread budget scopes its work once:

```rust
faer::team::with_team(8, |par| {
    // every faer call in here with `par` runs its regions on the eight reserved members
    qr_in_place(a.as_mut(), h.as_mut(), par, stack, params);
});
```

A caller that already owns a team lends a sub-team to each call with `par.reborrow()`; a width of one
is `Par::Seq`. `with_team` reserves a team for the duration of the closure, so a caller that keeps a
budget across many calls should own a team itself. See the *parallelism* section of the crate
documentation for the full picture.

# minimum supported rust version

1.89.0, declared as `rust-version` in `faer/Cargo.toml`. Rust 1.89 is the first release where the
AVX-512 target features are stable, which is what lets the `x86-v4` kernels build without the nightly
compiler. They stay opt in until that path is settled.

# contributing

Upstream's contribution guide applies to this fork as well: start with the issues labelled "good
first issue", and ask on the [upstream zulip server](https://faer.zulipchat.com) or in the issue
itself. Fork specific reports belong in this repository's issues, and a defect that also reproduces on
the published upstream crates belongs in
[`upstream-reports/`](upstream-reports/README.md).

# benchmarks

The fork's own results are the published campaign pages, and the upstream benchmark page is at
[faer.veganb.tw/benchmarks](https://faer.veganb.tw/benchmarks/). Performance lessons that carry over
between changes are in [`PERFORMANCE_TIPS.md`](PERFORMANCE_TIPS.md).
