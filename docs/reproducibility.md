# Reproducibility: bitwise results are promised at a fixed thread count

Owner decision, 2026-11-08 ([tensor4all/faer#327](https://github.com/tensor4all/faer/issues/327)).

## What is promised

On the **same machine and the same build**, at a **fixed thread count**, a call returns bitwise
identical results across fresh threads, fresh processes and repeated runs. That is the whole promise.
"Fixed thread count" means the same `Par` width in every run being compared: `Par::Seq`, or a team of
one stated size.

Anything that compares results recorded earlier — a cache that spans runs, a benchmark table, a
regression digest — fixes the thread count first and records which count its numbers belong to.

## What is not promised

Bitwise identical results when the thread count changes. Earlier work on this fork treated that as a
goal; it is **withdrawn** as a requirement, because keeping the work partition independent of the
number of participants constrains how the work may be split, and that constraint is paid on every
call to buy a property no caller needs. No API in this repository claims it.

## The mechanism, so a reader can predict where it shows up

Parallel work is sized by a **cost model**, not by the requested degree.
`faer/src/utils/cost.rs:613-619` states the rule: the width is the widest one in `2..=degree` at
which the region's estimated serial time pays for the dispatch, and 1 (sequential) when none does.
The estimate is in flops, so it is dtype-dependent. `faer/src/utils/cost.rs:711-713` returns 1 for a
`Par` of degree below 2, so a one-thread team runs the region on the calling thread. The matmul team
path shows both halves at once: `faer/src/linalg/matmul/internal/mod.rs:286-297` computes that width
with `crate::utils::cost::par_width` and, when it is narrower than the degree, reruns the call on a
sub-team of that width.

Two consequences a reader can use:

- A requested degree of 2 and a requested degree of 8 reach the **same width** on a problem that only
  pays for 2 participants, so those two runs use the same partition and produce the same bits.
- The cost is priced in flops, so complex arithmetic crosses the threshold at a different problem size
  than real. `PATCHSET.md:135` prices the native gemm thread budget in flops for exactly this reason:
  complex keeps its two-times gain at `64^3`.

So "the result differs across thread counts" means "the serial path and the parallel path differ", not
"the summation order follows the thread count".

## Measured, not promised

Observed on an AMD EPYC 7713P (Zen 3, AVX2+FMA), as recorded in
[tensor4all/faer#327](https://github.com/tensor4all/faer/issues/327). This is a record of what the
mechanism does, not a contract:

| case | 1 vs 2 threads | 2 vs 8 threads |
| --- | --- | --- |
| f64 `mat_mul` | identical | identical |
| Complex64 `mat_mul`, `contract_pair` at `16³`, `32³`, `48³` | identical | identical |
| Complex64 `mat_mul`, `contract_pair` at `64³` and larger | **differ** | identical |

The complex row is the mechanism above: the one-thread case takes the serial path, two or more take
the cost-model-capped parallel path, and complex crosses the threshold below `64³` where real does
not.

## What this asks of a contribution

- Do not add a test that asserts bitwise identical results across thread counts.
- Do add or keep a test that asserts bitwise identical results **at a fixed thread count**, with the
  count pinned through the API (`Par::Seq`, or a team of a stated width) rather than through an
  environment variable.
- Comparisons between routes that use a tolerance (`Par::Seq` against `Par::Team`, as
  [new-routine-checklist.md](new-routine-checklist.md) asks) are unaffected: they are not bitwise
  assertions.
- When writing a performance claim, pin the thread count for **both** sides of the comparison.
