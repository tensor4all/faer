# Eigenvector back substitution clamps relative to the eigenvalue (tensor4all/faer#112)

## Decisions
- **Comparison with LAPACK.** The test's residual is GET22's: `|A X - X W|_1 / (|A|_1 |X|_1 ulp)`, with no
  dimension factor (`dget22.f` 3.12.0). CHKHS holds it to 20 at n <= 16 (`nep.in`). The opt-in family runs it
  at n = 63, 64, 65 and 129. For a reference, the same ratio was computed for OpenBLAS 0.3.34's `dgeev`
  (through numpy) on the same matrices:

  | fixture | faer before | faer after | LAPACK `dgeev` |
  | --- | --- | --- | --- |
  | kind 3 (Jordan block), n = 63 / 64 / 65 | 62.5 / 63.5 / 64.5 | < 20 | 0.50 at every n |
  | kind 5 (clustered triangle), n = 63 to 65 | 63.5 to 64.5 | < 20 | not computed |
  | kinds 2, 4, 6, 7, n = 63 | 20 to 28 | 20 to 29 (unchanged) | 17.5 to 28.5 |
  | kind 4, n = 129 | 37 | 36 to 38 (unchanged) | 34.4 |

- **The defect.** The growth with n on kinds 3 and 5 is faer's. The back substitution replaced a vanishing
  shift `t_ii - lambda` by `eps * sum |t_ij|`, a floor that grows with `n`. The eigenvector then carries a
  residual of that size: on the Jordan block the entries sum to `2n - 1`, which gives a ratio of
  `(2n - 1) / 2`.
  - LAPACK's `?TREVC` uses `max(ulp * (|wr| + |wi|), safmin * n / ulp)` on a matrix that `?GEEV` has already
    scaled into range.
  - faer does not scale, so its floor is now `eps * max(|lambda|, max |t_ij|)`, at least the smallest
    positive subnormal. Both terms are `|re| + |im|`, multiplied by `eps` before the sum so that the floor
    does not overflow near the top of the range; the f32 range-limit test caught the unscaled form.
  - The subnormal Jordan 2x2 test confirms that a floor of `safmin * n / ulp` without LAPACK's prescaling
    would be wrong: it is far above the eigenvalue there.
  - A clamp this small is range safe only on entries near one. The post-implementation review traced two
    counterexamples, and both reproduced:
    - a Jordan chain scaled to 2^972 overflowed in the products with the back-substituted entries (about
      `1 / eps`) before the solvers' rescaling ran;
    - a complex one scaled to 2^-972 had a clamp whose reciprocal overflows.

    A Schur form whose largest entry is outside `[2^-20, 2^20]` is therefore first scaled by a power of two,
    applied as two factors so that neither overflows, into `[1, 2)`, which is `?GEEV`'s prescaling.
    `jordan_chains_at_both_ends_of_the_range` covers both counterexamples and fails without it.
- **The threshold.** Kinds 2, 4, 6 and 7 at n = 63 to 129 stay above 20. On the fixtures that were compared,
  LAPACK's own result is in the same range (17.5 to 34.4). That does not show a faer-specific defect there,
  but it covers the compared fixtures only, not every side and precision.
  - The threshold is unchanged. Test tolerances are the maintainer's to change (R-testing-51).
  - Candidates for that decision: `20 * max(1, sqrt(n / 16))` is 40 at 63 and 57 at 129. It passes LAPACK's
    and faer's current values, and it would still have caught the clamp defect (62.5 at 63). A linear
    `20 * n / 16` would not have caught it.

## Verification conclusions and constraints
- **Regression test:** `lapack_nonsymmetric_evd_clustered_residuals_do_not_grow_with_n` covers kinds 3 and 5 at n = 63
  and 65, both eigenvector sides, residual < 20, every dtype. It runs with the regular tests and gives 62.5
  without the change.
- **Opt-in family:** `lapack_nonsymmetric_evd_cases` diagnostics drop from 306 / 282 / 220 / 216 to
  192 / 168 / 112 / 116 (f64 / f32 / c32 / c64). The parallel group (kind 4 at n = 129) is unchanged at 24.
- **Existing tests:** `linalg::evd` unit tests pass, including the range-limit, subnormal-clamp and
  Jordan-overflow regressions.
- **Provenance:** only LAPACK's formulas and parameter files were read. No source was copied.
