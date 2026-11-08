# Miri

Miri runs weekly (`.github/workflows/miri.yml`) over `cargo miri test -p t4a-faer-traits --lib` and
files an issue labelled `miri-failure` when it fails; run it locally with
`cargo +nightly miri test -p t4a-faer-traits --lib` (add `rustup component add --toolchain nightly miri
rust-src` once). The main `t4a-faer` crate is out of scope: its SIMD kernels come from
`pulp` 0.22.2, whose x86_64 helpers are inline assembly (`pulp-0.22.2/src/x86.rs:263` and `:281`, the
AVX2 masked u32 load/store), and Miri implements no inline assembly: a kernel that reaches one aborts
with `unsupported operation: inline assembly is not supported`. A whole-lib run is out of budget
anyway — the `matmul` and `lblt` cases on their own did not finish here within 40 and 15 minutes.
Covering the main crate therefore needs an asm-free code path, and `pulp` 0.22.2 has no feature that
turns those kernels off (`std`, `x86-v3`, `x86-v4`, `nightly`, `macro` only), so a feature of our own
would have to select a scalar-only build before a Miri lane can include it. Three float tests are
`#[cfg_attr(miri, ignore)]` because they assert the last bit of a subnormal or near-maximum result,
which Miri's floating point does not reproduce; they are a precision check, not a UB report.
