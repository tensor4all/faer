# Adding a new linear-algebra routine: checklist

This page is the short, executable version of the repository's testing rules for a new
decomposition, solver or kernel. It points at the layer that owns each guarantee; the rules
themselves stay in [AGENTS.md](../AGENTS.md), [review-rules.md](review-rules.md) and
[adoption-protocol.md](adoption-protocol.md).

A new routine is done when every box below is either ticked or explicitly waived with a reason in
the PR and the worklog. The fork's CI is a **build** gate, not the full suite: the numerical
evidence has to be produced locally and recorded.

## 1. Decide the contract before writing code

Answer these in the PR description first; they decide which tests exist.

| Question | Usually |
| --- | --- |
| Scalar types | `f32`, `f64` for real-only algorithms; also `c32`, `c64` where the operation is defined. Justify any narrowing. |
| Storage and layout | Which triangle is read, what strides are accepted, row-major or column-major only, aliasing rules. |
| Scratch | `*_scratch` query must be exact; a caller-sized `MemStack` must not be overrun and must not be padded "just in case". |
| Parallel route | `Par::Seq` and `Par::Team` must both run, and the results must agree within the family's tolerance. Bitwise equality is a fixed-thread-count promise only, never across thread counts ([reproducibility.md](reproducibility.md)). |
| Failure | What `Err` / `NonConvergence` leaves in the outputs, and whether inputs are allowed to be left unspecified. |
| Allocation | Nothing on the heap per call in a steady-state call (see [tensor4all/faer#111](https://github.com/tensor4all/faer/issues/111)); per-thread reusable working memory is allowed and is documented as such. |
| Provenance | Is any arithmetic, test oracle or fixture adapted from LAPACK, OpenBLAS, Eigen, or another project? Record project, revision, file, routine and the material changes at the port site. |

## 2. The layers a new routine touches

| Layer | Where it lives | What it proves |
| --- | --- | --- |
| Unit | `#[cfg(test)] mod tests` in the owning module (child module, so private items are reachable) | Arithmetic, boundaries, layout and scratch contracts |
| Public contract | `faer/tests/<name>.rs` with `mod common;` | Behaviour through the public API, output buffers and rejection paths |
| Numerical validation | `<module>::lapack::<dtype>::lapack_<family>_*` (opt-in, `#[ignore]`) | Agreement with LAPACK's own checkers on a bounded case set |
| Non-functional | `faer/tests/*no_alloc.rs`, `faer-no-std-test`, CI | Zero per-call allocation, no_std build, MSRV, formatting |
| Performance | outside this repository (`faer-benchmark`), see [adoption-protocol.md](adoption-protocol.md) | Speed; never a substitute for the numerical layers |

Do not add a `benchmarks/` directory here; it is forbidden. Performance evidence goes to the
campaign, and the PR cites the run.

## 3. Unit tests

- Put them in a child test module of the owning module, following the surrounding faer pattern.
  Do not widen production visibility for a test, and do not move existing tests.
- Cover, per scalar type:
  - the exact `0`, `1` and `2` cases, including empty scratch and empty output buffers;
  - a deterministic sample at the sizes in the `SIZES` list of
    `faer/tests/common/mod.rs`, with `sample(i, j)`-style
    pseudo-random entries (no `rand` feature dependency);
  - repeated/clustered spectra, a Jordan-like defective case, and a graded matrix where the
    algorithm's scaling matters;
  - subnormal, `min_positive`, `max_positive` and exact powers of two around the range limits,
    when the routine rescales or reciprocates;
  - every layout the public entry accepts, views with non-unit strides, and a reused output buffer
    that previously held a different result;
  - `Par::Seq` versus `Par::Team` (at least two threads) on the same input, compared within tolerance.
- Assert **properties**, not shapes: reconstruction, residual, orthogonality/unitarity, ordering,
  invariants, and a known exact solution. A finite-only assertion is not a numerical test.
- Tolerances must be precision-, dimension- and scale-aware. The shared helpers in
  `faer/tests/common/mod.rs` (`close`, `unitary`, `schur_form`, `tolerance`) are the pattern; a new
  copy of them is a sign the test belongs beside the existing ones.
- Every experiment carries **one focused test** that fails before the change and passes after. Run
  it against the parent revision and paste the before/after outcome in the worklog.

## 4. Public contract test

Add `faer/tests/<routine>.rs` when the routine is public and has a contract worth pinning:
output modes, buffer reuse, error paths, views, and the dispatch boundary. Use `mod common;` and the
existing helpers. Keep it lightweight; a full combination sweep belongs in the opt-in validation.

## 5. Scratch and allocation

- Add or extend an allocation test when the routine allocates. The pattern is
  `faer/tests/no_step_alloc.rs` (a counting global allocator around a warmed-up call) or
  `faer/tests/svd_values_no_alloc.rs` (per-thread counting for a `Par::Seq` call; a process-wide
  flag once counted libtest's own watchdog output).
- The scratch query must be exact: build the buffer from `*_scratch`, never a hand-picked size, and
  add a test that runs at the exact query size for every output mode.
- If the routine takes `par`, add the region to the zero-allocation test only if it is reachable in
  a steady-state call.

## 6. LAPACK numerical validation (when LAPACK has the routine)

This is the layer that catches most numerical defects. Registration has four non-obvious
requirements; miss any and the case silently stops being counted.

1. **Name and module.** The test name must contain `::lapack::` and end with one of
   `_cases`, `_parallel`, `_subnormal`, `_range` (the runner only counts those as case groups), and
   its leaf must start with `lapack_<family>_`. Example:
   `linalg::lu::lapack::f64::lapack_lu_cases`.
2. **Family registration.** Add the family to `FAMILIES` and `LIMITS` in
   `scripts/run-lapack-validation.py`. `FAMILIES` is execution order; `svd_large` stays last.
   `classify()` sorts families by length, so `schur` and `schur_reorder` can coexist.
3. **Record line.** The group must print
   `LAPACK_CASES family=<family> dtype=<...> planned=<n> attempted=<n> call_failures=<n> diagnostics=<n>`
   and assert `attempted == planned`. An incomplete loop is a failure, not a skip.
4. **Offline check.** Add or extend a case in `scripts/test-lapack-validation.py` for the new
   classification, and run it: `python3 scripts/test-lapack-validation.py`.

Then:

- Port the checker semantics, not just the inputs. Keep LAPACK's threshold (20/30/50/60) and the
  source's dimension and safe-minimum factors. **Do not loosen an absolute tolerance so the test
  passes** ([review-rules.md](review-rules.md)). If a dimension-dependent threshold is genuinely
  needed, make it a maintainer decision and record it (the `20 * max(1, sqrt(n / 16))` decision in
  [#112](https://github.com/tensor4all/faer/issues/112) is the precedent).
- Exercise the four native types, views/reused buffers, and sequential plus dedicated 2-/4-thread
  groups.
- Update the inventory table in [lapack-validation.md](lapack-validation.md): filter suffix, port
  status, LAPACK sources and helpers, threshold, owning test module, and what is deliberately
  omitted.
- Add every inspected source file with its SHA and line count to
  [lapack-validation-sources.json](lapack-validation-sources.json). The pinned distribution is
  LAPACK 3.12.0 via OpenBLAS `791b02a1a00c996b66a894f6df93421c53b62d80`; the reference checkout is
  `extern/OpenBLAS` (a submodule).
- Keep the BSD notice and `faer/COPYING.LAPACK.BSD`; adapted test files carry the copyright header
  and an SPDX identifier.

Run the family locally, then the whole suite:

```sh
scripts/run-lapack-validation.sh                 # all families, one build
scripts/run-lapack-validation.sh --limit-scale 3 # slower hosts (the aarch64 lane)
```

A failure that is real but not fixed yet goes into `scripts/lapack-known-failures.json` **with the
issue that tracks it**. An entry that starts passing fails the run, so the fix removes the entry in
the same change. Never add an entry without an issue.

## 7. Record the change

- `PATCHSET.md` — one row: what the experiment does, status (`TODO` / `EXPERIMENT` / `KEEP` /
  `DROPPED` / `HOLD`), the driving issue, the evidence class, and links.
- `docs/worklogs/YYYY-MM-DD-<slug>.md` — decisions, rejected alternatives with the measurement that
  dropped them, verification conclusions, and what remains unverified. No command logs, no
  chronology ([worklogs/README.md](worklogs/README.md)).
- `upstream-reports/<slug>/` — only when the change fixes a defect that also exists upstream; the
  fork's policy is to record and defer, not to open anything upstream.
- For a public API addition, check the migration guides under `docs/migration/` and `llms.txt`.

## 8. Verification commands

```sh
cargo test -p t4a-faer --lib                         # unit tests
cargo test -p t4a-faer --test <integration-name>     # new integration test
cargo test -p t4a-faer                               # lib + integration + doc tests
cargo build -p t4a-faer --no-default-features --features std
( cd faer-no-std-test && cargo run --profile nostd )
cargo test --doc --workspace -j8 -- --test-threads=4
python3 scripts/check-migration-examples.py          # if a migration example changed
cargo package -p t4a-faer --no-verify --list         # packaging still works
python3 scripts/test-lapack-validation.py            # runner selection logic
scripts/run-lapack-validation.sh                     # new/changed numerical family
```

CI runs the build, the traits, the doc tests, the migration examples and the packaging check, plus
the inherited upstream jobs (MSRV 1.89, no_std, nightly coverage, clippy/fmt as non-blocking). The
LAPACK validation runs nightly on x86_64 and weekly on aarch64. None of that replaces a local run of
the focused test.

## 9. Review gates for the PR

- [ ] One focused regression test, failing on the parent and passing on the change.
- [ ] Every accepted scalar type and layout is exercised.
- [ ] Scratch query is exact and a zero-allocation test covers the call if it allocates.
- [ ] `Par::Seq` and `Par::Team` agree within tolerance (bitwise only at a fixed thread count, [reproducibility.md](reproducibility.md)).
- [ ] LAPACK family registered, record line emitted, inventory and source list updated.
- [ ] No tolerance was loosened to make a test pass; any new threshold has a recorded decision.
- [ ] Provenance and licence notices preserved at the port site.
- [ ] `PATCHSET.md` row and worklog exist; performance claims cite the campaign, not this repo.
- [ ] `cargo fmt`/`clippy` introduce no new warnings in touched files.

## 10. What this checklist deliberately does not cover

- Packed and banded storage, `?RFS`/`?CON` condition and error estimates, MRRR and range selection,
  and the generalized nonsymmetric eigenproblem are outside the current LAPACK adaptation; adding
  them is a validation-program change, not a one-line port.
- The LAPACK case generator is a bounded adaptation, not `LATMS`/`LATMR`'s random sequence; a
  fixture that needs a new distribution belongs in a small ordinary test unless it is calibrated
  against a checker.
- Extended-precision scalar types are not in the native validation set.
- Performance acceptance is a separate campaign with its own frozen protocol; a green numerical
  suite says nothing about speed.
