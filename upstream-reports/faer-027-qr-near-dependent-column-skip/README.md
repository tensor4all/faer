# faer-027: unpivoted QR skips a nearly dependent column of a tall matrix

**What happens:** `Mat::qr` of a tall full rank matrix drops a column whose independent part is small, so `Q R` differs from `A` by thousands of `eps` and `R` gets a zero on its diagonal.
**Which input:** a tall matrix with a column whose distance from the span of the earlier columns is below about `16 m eps` times its norm. The program below uses `m x 3` matrices with nearly equal columns.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

Column 0 is all ones, column 1 is ones plus `1e-10` in row 0, and column 2 is ones plus `1e-6` in row 1.
The matrix has full rank 3.
A backward stable QR gives `|A - Q R|_F / |A|_F` of a few `eps` at every `m`.

| m | expected | `faer` 0.24.4 | `R[2,2]` in `faer` 0.24.4 |
|---|---|---|---|
| 100 | a few eps | 1.6e-16 (1 eps) | 9.949e-7 |
| 1000 | a few eps | **2.6e-12 (11619 eps)** | **0** |
| 8192 | a few eps | **9.0e-13 (4055 eps)** | **0** |

At `m = 100` the column is kept.
At `m = 1000` and `m = 8192` the column with the `1e-10` entry is skipped.
`R` then has only two nonzero rows, and the `1e-10` part of `A` is lost.

## Cause

`qr_in_place_unblocked` builds a Householder reflector for each column.
It computes `threshold = eps * 16 (m - row) * norm`, where `norm` is the norm of the whole column (`src/linalg/qr/no_pivoting/factor.rs:52-55`).
It applies the reflector and advances `row` only when `info.norm > threshold` (line 64), where `info.norm` is the part of the column below the current row.
Otherwise the column is skipped and its reflector is not applied to the remaining columns.
For column 1 here, `info.norm` is about `1e-10` and `norm` is about `sqrt(m)`.
So the threshold is about `16 m eps sqrt(m)`: 3.5e-12 at `m = 100`, 1.1e-10 at `m = 1000` and 2.6e-9 at `m = 8192`.
From `m = 1000` on it exceeds `1e-10`, and column 1 is skipped.
The threshold grows with `m`, so the skipped part can be far above rounding level.
The blocked QR recurses into this function, and the tall path of `svd` goes through it too.
LAPACK's `dgeqrf` applies a reflector for every column.

## How the fork fixed it

The fork removes the threshold.
The reflector is applied and `row` advances for every column with `info.norm > 0`, as in LAPACK.
Run against the fork, the program prints 1.6e-16, 2.1e-15 and 4.6e-15, and `R[2,2]` is about `1e-6` at every `m`.

Fork regression test: `linalg::qr::no_pivoting::factor::tests::test_near_dependent_columns_are_not_skipped` (this matrix at m = 100, 1000, 8192, error at most 1e-13), and `test_near_dependent_rows_are_not_skipped` for its transpose.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#1 (commits `2e6ea52` and `d16929b`).
* Where it was found: tensor4all/tlinalg-rs#28 and tensor4all/tensor4all-rs#836.
