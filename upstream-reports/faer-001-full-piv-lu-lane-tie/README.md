# faer-001: full pivoting LU picks a different pivot depending on the SIMD width

**What happens:** `full_piv_lu` chooses a different pivot for the same matrix when the SIMD vector holds a different number of elements.
**Which input:** a matrix where two or more entries tie for the largest magnitude (common with small integer entries).
**Which version:** `faer` 0.24.4, seen on an AVX2 machine by comparing `f64` (4 elements per vector) with `f32` (8 elements per vector).

## Reproduce

```sh
cargo run
```

The program factors one 5 x 5 matrix of small integers in `f64` and in `f32`.
Both types store every entry exactly.
Two entries tie for the largest magnitude: 2 at (row 1, col 0) and 2 at (row 4, col 1).
It prints the first pivot chosen by each type.
It also prints the pivot that faer's own scalar search would choose.

| | first pivot (row, col) |
|---|---|
| expected (faer's scalar search, and the same answer for every type) | (1, 0) |
| `faer` 0.24.4, `f64` | **(4, 1)** |
| `faer` 0.24.4, `f32` | (1, 0) |

On an AVX-512 machine, `f64` also uses 8 elements per vector.
So the same `f64` matrix can get different pivots, and different factors, on an AVX2 machine and an AVX-512 machine.

## Cause

The SIMD pivot search keeps one running maximum per vector lane (a lane is one element slot of the SIMD vector).
Each row of a column goes to a lane that depends on the lane count.
At the end, `reduce_2d` in `src/linalg/lu/full_pivoting/factor.rs:64` finds the largest value and takes the first lane that holds it (`simd.first_true_mask(is_best)`).
With 4 lanes, row 4 sits in lane 0 and row 1 in lane 1, so row 4 wins.
With 8 lanes, row 1 sits in lane 1 and row 4 in lane 4, so row 1 wins.
The scalar search `best_in_matrix_fallback` in the same file scans column by column and keeps the first strict maximum, so it picks the smallest column and then the smallest row.
The SIMD search and the scalar search therefore disagree on ties, and the SIMD result changes with the vector width.

## How the fork fixed it

The fork breaks the tie in `reduce_2d` the way the scalar search does.
Among the lanes that hold the maximum, it takes the one with the smallest column, and then the smallest row.
The pivot sequence then no longer depends on the vector width.
With that change this program prints (1, 0) for both types.

Fork regression test: `pivot_search_breaks_ties_by_column_then_row` in `faer/src/linalg/lu/full_pivoting/factor.rs`.
It checks that the SIMD search and the scalar search pick the same pivot on tied matrices of several shapes.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.

## References

* Fork fix: tensor4all/faer#105 (commit `efaa5de`).
* Where it was found: tensor4all/faer#102. With AVX-512 enabled through `pulp/x86-v4`, faer's own test `test_gh238` (a singular 28 x 28 matrix of 0, 1 and 2) chose a different elimination order than with AVX2, and the fork observed the solve returning `NaN`. That AVX-512 observation was made on the fork, not rerun on 0.24.4.
