// The same 5 x 5 matrix, factored with full pivoting in f64 and in f32.
// Every entry is a small integer, so both types hold the matrix exactly.
// Two entries tie for the largest magnitude: 2 at (row 1, col 0) and at (row 4, col 1).
// We check which of the two becomes the first pivot.
use faer::prelude::*;

fn entry(i: usize, j: usize) -> f64 {
    if (i, j) == (1, 0) || (i, j) == (4, 1) {
        2.0
    } else if i == j {
        1.0
    } else {
        0.0
    }
}

// First pivot (row, col) chosen by `full_piv_lu`.
fn first_pivot<E: faer::traits::RealField>(a: Mat<E>) -> (usize, usize) {
    let lu = a.full_piv_lu();
    (lu.P().arrays().0[0], lu.Q().arrays().0[0])
}

fn main() {
    let n = 5;

    // Reference: scan column by column and keep the first strict maximum.
    // This is what faer's own non SIMD search (`best_in_matrix_fallback`) does.
    let mut best = (0, 0);
    for j in 0..n {
        for i in 0..n {
            if entry(i, j).abs() > entry(best.0, best.1).abs() {
                best = (i, j);
            }
        }
    }

    let f64_pivot = first_pivot(Mat::<f64>::from_fn(n, n, entry));
    let f32_pivot = first_pivot(Mat::<f32>::from_fn(n, n, |i, j| entry(i, j) as f32));

    println!("reference (scalar scan): {best:?}");
    println!("full_piv_lu, f64:        {f64_pivot:?}");
    println!("full_piv_lu, f32:        {f32_pivot:?}");
    if f64_pivot != f32_pivot {
        println!("DIFFERENT: the pivot depends on the element type");
        std::process::exit(1);
    }
}
