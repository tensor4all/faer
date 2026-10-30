//! The real nonsymmetric EVD of `[[2, 1], [-1, 0]]`, whose eigenvalue 1 is a double, defective
//! eigenvalue, returns an eigenvector with `|A v - v| = sqrt(2)`. LAPACK `dgeev` returns two nearly
//! parallel eigenvectors with residuals near rounding level.

use faer::{c64, Mat, Scale};

fn main() {
    let a = Mat::<f64>::from_fn(2, 2, |i, j| [[2.0, 1.0], [-1.0, 0.0]][i][j]);
    let ac = Mat::<c64>::from_fn(2, 2, |i, j| c64::new(a[(i, j)], 0.0));
    let evd = a.eigen().expect("eigen converges");
    let (s, u) = (evd.S().column_vector(), evd.U());
    let mut worst = 0.0f64;
    for k in 0..2 {
        let v = u.col(k);
        let residual = (&ac * v - v * Scale(s[k])).norm_l2();
        println!("eigenvalue {:?}: |v| = {:.3}, |A v - lambda v| = {residual:e}", s[k], v.norm_l2());
        worst = worst.max(residual);
    }
    let expected = 1e-14;
    println!("expected: every residual below {expected:e}");
    if worst > expected {
        println!("FAIL: the largest residual is {worst:e}");
        std::process::exit(1);
    }
    println!("OK");
}
