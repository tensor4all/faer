# Reproducibility: bitwise results are promised at a fixed thread count

Owner decision, 2026-11-08 ([tensor4all/colmat-rs#327](https://github.com/tensor4all/colmat-rs/issues/327)).

## What is promised

On the **same machine and the same build**, at a **fixed thread count**, a call returns bitwise
identical results across fresh threads, fresh processes and repeated runs. That is the whole promise.
"Fixed thread count" means the same `Par` width in every run being compared: `Par::Seq`, or a team of
one stated size.

Anything that compares results recorded earlier — a cache that spans runs, a benchmark table, a
regression digest — fixes the thread count first and records which count its numbers belong to.

No test in this repository compares a second process, or repeats a run after a restart. What the tree
does check pins the width: `faer/tests/lu_rank_truncation.rs:183-218` records a digest at `Par::Seq`
and on a team of four and compares them against stored goldens (on `aarch64` only). The promise
therefore describes what the code is expected to deliver; the across-process half is not enforced by a
check.

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
path shows both halves at once: the native-kernel gate is
`faer/src/linalg/matmul/internal/mod.rs:252-278` (`crate::utils::team::kernel::kernel_width`), and the
generic path at `:286-297` applies the same widest-paying-width rule with `crate::utils::cost::par_width`
and, when that width is narrower than the degree, reruns the call on a sub-team of that width.

Two consequences a reader can use:

- Within one path, a requested degree of 2 and a requested degree of 8 reach the **same width** on a
  problem that only pays for 2 participants, so those two runs use the same partition and produce the
  same bits. The width is a property of the size *and the path*: the kernel fork pays two sections
  (`faer/src/utils/team.rs:515` prices `2`), the generic gate one
  (`faer/src/linalg/matmul/internal/mod.rs:289-293` passes `1`), so at a size between the two floors
  the native arm can stay sequential while the generic arm already runs at width 2 or more.


- The cost is priced in flops, so complex arithmetic crosses the threshold at a different problem size
  than real. `PATCHSET.md:135` prices the native gemm thread budget in flops for exactly this reason:
  complex keeps its two-times gain at `64^3`.

For the products measured below, "the result differs across thread counts" therefore means "the serial
path and the parallel path differ", not "the summation order follows the thread count". That reading is
specific to the cost-model gate and does not carry over to a routine that partitions by the count
itself: `faer/src/linalg/evd/tridiag.rs:386-409` splits the trailing columns as `idx / nthreads` and
adds the per-column partial vectors at `:457-459`, so its own boundaries are `[0, 2, 8]` at two threads
against `[0, 1, 2, 4, 8]` at four (with `n2 = 8`), and two parallel widths can group the same sum
differently.

## Measured, not promised

Observed on an AMD EPYC 7713P (Zen 3, AVX2+FMA), as recorded in
[tensor4all/colmat-rs#327](https://github.com/tensor4all/colmat-rs/issues/327). This is a record of what the
mechanism does, not a contract:

The numbers below are the ones recorded in that issue; it carries the host description, the sizes and
the thread counts, but no stored digests, so a replay has to rerun those sizes on the same host class.

| case | 1 vs 2 threads | 2 vs 8 threads |
| --- | --- | --- |
| f64 `mat_mul` | identical | identical |
| Complex64 `mat_mul`, `contract_pair` at `16³`, `32³`, `48³` | identical | identical |
| Complex64 `mat_mul`, `contract_pair` at `64³` and larger | **differ** | identical |

The complex row correlates with the mechanism above; it does not prove which gate ran. A one-thread
call takes the serial path, and two or more threads reach the widest width that pays. Complex costs
about 4.5x the estimated serial time per MAC of real (`0.165` against `0.037` ns/MAC), so it crosses
a given floor at a smaller size. Bit equality alone does not show the gate stayed shut: at the
prices this build uses, a
width-2 kernel fork costs `PAYS x 2 x 1423 ns = 8.5 us` and the generic gate `PAYS x 1 x 1423 ns =
4.3 us`, while an `f64` `64³` product already estimates `262144 x 0.037 ns = 9.7 us` of serial work,
above the first floor. So `f64` may run parallel at `64³` and still agree bitwise.

Why the `Complex64` rows differ is a hypothesis rather than a recorded fact: the two paths
accumulate in a different order, and the issue records the outcome, not a trace of which
partition ran. Treating it as the likely cause is enough to predict where to look; it is not
evidence from #327.

## What this asks of a contribution

- Do not add a test that asserts bitwise identical results across thread counts.
- Do add or keep a test that asserts bitwise identical results **at a fixed thread count**, with the
  count pinned through the API (`Par::Seq`, or a team of a stated width) rather than through an
  environment variable.
- Comparisons between routes that use a tolerance (`Par::Seq` against `Par::Team`, as
  [new-routine-checklist.md](new-routine-checklist.md) asks) are unaffected: they are not bitwise
  assertions.
- When writing a performance claim, pin the thread count for **both** sides of the comparison.
