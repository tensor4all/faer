//! faer-029: `Mat::eigen` loses every digit of the eigenvectors when the entries are tiny.
use faer::{c64, Mat};

/// LAPACK's GET22 residual `|A U - U S|_1 / (|A|_1 |U|_1 eps)`, in `f64`, on the dimensionless
/// matrix `a / scale` (exact for a power of two)
fn get22(a: &Mat<c64>, u: &Mat<c64>, s: &[c64], eps: f64) -> f64 {
    let n = a.nrows();
    let norm1 = |m: &Mat<c64>| {
        (0..n).map(|j| (0..n).map(|i| m[(i, j)].norm()).sum::<f64>()).fold(0.0, f64::max)
    };
    let us = Mat::<c64>::from_fn(n, n, |i, j| u[(i, j)] * s[j]);
    let residual = a * u - us;
    norm1(&residual) / norm1(a) / norm1(u) / eps
}

fn main() {
    let threshold = 20.0;
    let n = 64;
    // the Grcar matrix: ones on the diagonal and the three superdiagonals, minus one below
    let grcar = |i: usize, j: usize| match j as isize - i as isize {
        -1 => -1.0,
        0..=3 => 1.0,
        _ => 0.0,
    };
    let mut failed = false;
    println!("{:>5} | {:>6} | {:>10} | {:>12}", "type", "scale", "expected", "actual");
    for (ty, e) in [("f64", 0), ("f64", -960), ("f64", -1000), ("f32", 0), ("f32", -100), ("f32", -110)] {
        let scale = 2.0f64.powi(e);
        let dimless = Mat::<c64>::from_fn(n, n, |i, j| c64::new(grcar(i, j), 0.0));
        let (u, s, eps) = if ty == "f64" {
            let a = Mat::<f64>::from_fn(n, n, |i, j| grcar(i, j) * scale);
            let eig = a.eigen().unwrap();
            let s: Vec<c64> = (0..n).map(|i| eig.S()[i] / scale).collect();
            (eig.U().to_owned(), s, f64::EPSILON)
        } else {
            let a = Mat::<f32>::from_fn(n, n, |i, j| (grcar(i, j) * scale) as f32);
            let eig = a.eigen().unwrap();
            let s: Vec<c64> = (0..n).map(|i| c64::new(eig.S()[i].re as f64, eig.S()[i].im as f64) / scale).collect();
            let u = Mat::<c64>::from_fn(n, n, |i, j| c64::new(eig.U()[(i, j)].re as f64, eig.U()[(i, j)].im as f64));
            (u, s, f32::EPSILON as f64)
        };
        let ratio = get22(&dimless, &u, &s, eps);
        println!("{ty:>5} | 2^{e:<4} | {:>10} | {ratio:>12.4e}", format!("< {threshold}"));
        failed |= !(ratio < threshold);
    }
    if failed {
        println!("the eigenvectors of a matrix with tiny entries have lost every digit");
        std::process::exit(1);
    }
}
