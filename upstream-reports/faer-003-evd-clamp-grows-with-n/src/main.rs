//! faer-003: the eigenvector residual of `Mat::eigen` grows with n on a Jordan block.
use faer::{c64, Mat, MatRef};

/// Largest column sum of absolute values (the matrix 1-norm).
fn norm1(m: MatRef<'_, c64>) -> f64 {
    (0..m.ncols())
        .map(|j| (0..m.nrows()).map(|i| m[(i, j)].norm()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn main() {
    // LAPACK's GET22 residual, |A U - U S|_1 / (|A|_1 |U|_1 eps); CHKHS accepts it below 20.
    let threshold = 20.0;
    let mut failed = false;
    println!("{:>4} | {:>10} | {:>10}", "n", "expected", "actual");
    for n in [16usize, 63, 65] {
        // a lower Jordan block: ones on the diagonal and on the subdiagonal
        let a = Mat::<f64>::from_fn(n, n, |i, j| if i == j || i == j + 1 { 1.0 } else { 0.0 });
        let eig = a.eigen().unwrap();
        let (u, s) = (eig.U(), eig.S());
        let ac = Mat::<c64>::from_fn(n, n, |i, j| c64::new(a[(i, j)], 0.0));
        let residual = &ac * u - u * s;
        let ratio = norm1(residual.as_ref()) / norm1(ac.as_ref()) / norm1(u) / f64::EPSILON;
        println!("{n:>4} | {:>10} | {ratio:>10.2}", format!("< {threshold}"));
        failed |= ratio >= threshold;
    }
    if failed {
        println!("the residual ratio grows with n: (2n - 1) / 2 on this matrix");
        std::process::exit(1);
    }
}
