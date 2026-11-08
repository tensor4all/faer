# Miri

Miri runs weekly (`.github/workflows/miri.yml`) over `cargo miri test -p t4a-faer-traits --lib` and
files an issue labelled `miri-failure` when it fails; run it locally with
`cargo +nightly miri test -p t4a-faer-traits --lib` (add `rustup component add --toolchain nightly miri
rust-src` once).

The main `t4a-faer` crate is not in the lane yet, and the reason is budget and test selection rather
than a capability limit. On an aarch64 host (2026-11-08, `miri 0.1.0 (daf2e5e18b)`, nightly) the crate
builds under Miri in about 20 s and its kernel tests do run:

- `cargo +nightly miri test -p t4a-faer --lib linalg::mat_ops::test::test_matrix_add -- --exact`
  — 1 passed, 0.84 s;
- the same command for `linalg::mat_ops::test::test_matmul_col_row` — 1 passed, 0.84 s.

What keeps the crate out is the shape of its suite: a whole `--lib` run costs far more per test than
the traits crate, and parts of it are slow even natively (`linalg::matmul::tests::test_matmul` carries
`#[ignore = "takes too long"]` in `faer/src/linalg/matmul/mod.rs:2128`), so including the main crate
means selecting its tests, not running all of them. We have not timed a complete main-crate `--lib`
run under Miri, so no budget number is claimed here.

On an x86_64 host there is a second, real obstacle: `pulp` 0.22.2 implements its masked u32 load/store
with `core::arch::asm!` (`pulp-0.22.2/src/x86.rs:263` and `:281`), and Miri executes no inline
assembly. `pulp` 0.22.2 also has an `Arch::Scalar` arm that `Arch::new()` selects when it detects
neither AVX2 nor AVX-512 (`pulp-0.22.2/src/x86.rs:313-334`). Whether an x86_64 Miri run reaches those
asm blocks at all is therefore **not established**: it can only be answered by running the main crate
under Miri on x86_64, and this lane never does. So this page makes no claim that an asm-free build path
is required before the main crate can join.

Three float tests are `#[cfg_attr(miri, ignore)]` because they assert the last bit of a subnormal or
near-maximum result, which Miri's floating point does not reproduce; they are a precision check, not a
UB report.
