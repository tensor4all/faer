# Caller-owned Householder factor-upgrade workspace

The compact-WY deep reduction currently allocates a contiguous `threads*n*n`
partial array on each warmed team call (faer #93, comment 6648218975; also #168
item 5). The four-scalar standalone MWE measures one allocation with width 2 and
none with width 1. Its independent Gram checks pass. This change supplies that
same working set through the canonical upgrade API and a matching scratch query;
it does not change the partition, cost gates, reduction order, or arithmetic.

The query must preserve the dispatch costs embedded in ParShape and model the
actual recursive join budgets. Concurrent diagonal children get disjoint stack
regions; scratch from sequential phases is reused. Generic initialized partials
retain the previous live-T and drop contract. No zero-allocation or performance
claim is established until the final tests and required acceptance evidence run.

The API change reaches unpivoted/pivoted QR, Hessenberg/tridiagonal reductions,
and bidiagonalization (hence EVD/SVD). Block widths must be inputs to the
tridiagonalization/bidiagonalization scratch queries so they can describe the
actual factor outputs, rather than over-reserving for every possible width.
QR look-ahead must account for simultaneous panel and trailing-update scratch.

Predeclared acceptance scope: qr_factor/qr and pivoted QR; Hessenberg/general and
self-adjoint EVD; bidiagonalization/SVD, all four scalars and declared layouts at
1T/4T/8T, with untouched LU/solve controls. Canonical library tests, a focused
four-scalar exact-query/Gram regression, no_step_alloc and no_std checks are
required. This note does not waive campaign, hosted, or hardware requirements.
No registry publication or automatic adoption is authorized by this record.

Remaining evidence and decisions will be recorded against the tested candidate.

## Initial functional evidence

The source candidate builds with std/linalg/threadteam. The complete library run
passes 680 tests (162 pre-existing ignored), the four-scalar public exact-query/
layout test passes, and no_step_alloc passes with added deep upgrades at all
four scalar types and its 4/8/70-width token budgets. The independent release MWE
reports zero allocation at widths 1/2 for every scalar, with the same parallel
section counts and Gram errors as the pinned parent. This evidence precedes the
separate #265 correction and is not the final adoption evidence.

The allocation work exposed another valid public branch: multiple target blocks
split at block_count/2 columns instead of that count times block_size. The pinned
main 9-column/block-2/previous-1 standalone MWE exits 101 with three wrong
coefficients. It is reported separately as tensor4all/faer#265 and will be fixed
in its own commit, with the matching scratch partition corrected too.

Campaign acceptance and hosted checks remain unverified. The change is an
experiment, not adopted. Final functional evidence is recorded below.

## Multiple-block correctness (#265)

The recursion split is a column offset, so `(block_count/2)*block_size` replaces
`block_count/2` in execution and query. A new four-scalar public test fails on
43089e0 at the first missing within-block coefficient (4,5), then passes after
the two offsets change. Cases include odd block counts, partial final blocks,
previous widths 1/8, and 2/8-participant trees with simultaneous deep children.
Cross-block sentinel entries and original diagonal denominators are checked.
This numerical correction is a separate commit from the storage/API change.

## Final functional evidence and remaining acceptance

The post-#265 source passes the complete crate test run: 916 passed, 162
pre-existing ignored, including 84 doctests. The public upgrade contract has
three passing tests: four-scalar C/F layouts, multiple-block boundaries, and
short workspace rejected before any factor update. The process-wide allocation
regression passes. An independent release MWE reports zero allocations/bytes
for f32/f64/c32/c64 at widths 1 and 2 across three repetitions, with two actual
parallel sections at width 2 and the parent's Gram errors unchanged. Setup
and scratch acquisition are outside the counted step; this is not a claim
about arbitrary scalar implementations or allocation of the owning outputs.
A real nightly no_std executable also exercises the provided-stack upgrade
and independently checks its coefficient. The additional opt-in numerical
validation passes 77 tests: QR 8, LQ 8,
Hessenberg 8, nonsymmetric EVD 8, self-adjoint EVD 12, and SVD 33. The
SVD filter includes an existing opt-in bidiagonal regression in addition to
the four-scalar LAPACK-inspired families. These are correctness checks, not
benchmark campaign verification.

No timing, speedup, scaling or adoption claim follows from the allocation
counter. The development host is an AMD Ryzen 9 PRO 8945HS with all 16 logical
CPUs in one L3 sharing domain, so the observing process cannot be pinned off
the measured L3 domain as the canonical protocol requires. Neither existing
harness hardware profile describes this host. The declared campaign, fbench
verification, paired candidate comparison and hosted CI remain pending; no
pinning or hardware requirement has been waived. Keep this change as a draft
until those acceptance requirements are satisfied. Colmat has not adopted the
unmerged API change.

The reviewable experiment is [PR #267](https://github.com/tensor4all/faer/pull/267),
kept draft pending acceptance.
The hosted build explicitly registers the new public regression in sequential
and threadteam configurations, plus the process-wide allocation test. Those
new hosted runs remain pending. The final cosmetic source reflow preserves
the expressions used by the completed local numerical tests.

## Coverage integration

Main's #266 (`2d4ea4e`) removes a duplicate whole QZ call from the #253
child test while retaining its exact AED-branch workspace check; the public
QZ scratch regression still executes the decomposition. It is merged into
this branch without any production-expression change. The retained child
regression passes at the integrated tree. The completed Householder,
allocation, no_std and full numerical results above remain applicable to the
same production implementation. Hosted checks must still pass at the new head.
The prior coverage run received a signal after 318 passes; four running tests
were terminated and 510 were not run, so it established no complete success.
No numerical assertion failure appears in that log and its cancellation cause
has not been established.
