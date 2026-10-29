// Eigenvectors of A = c * [[1, 1], [0, 1/2]] with `Mat::eigen`, at c = 1 and c = 2^1023.
// All entries and both eigenvalues (c and c/2) are finite.
// For every c, the eigenvector for c/2 is proportional to [-2, 1].
// We check that pair on the scaled down matrix A / c, which is exact:
// relative residual |(A / c - 1/2) x| / |x| = |x0 / 2 + x1| / |x|.
use faer::mat;

fn main() {
    let mut bad = false;
    for c in [1.0f64, 2f64.powi(1023)] {
        let a = mat![[c, c], [0.0, c / 2.0]];
        let e = a.eigen().unwrap();
        let (s, u) = (e.S(), e.U());
        let j = if (s[0].re - c / 2.0).abs() < (s[1].re - c / 2.0).abs() { 0 } else { 1 };
        let (x0, x1) = (u[(0, j)], u[(1, j)]);
        let r = (x0 * 0.5 + x1).norm() / (x0.norm().hypot(x1.norm()));
        println!("c = {c:e}: eigenvalue {:e}, vector [{}, {}], relative residual {r:.2e}", s[j].re, x0.re, x1.re);
        bad |= !(r < 1e-12);
    }
    println!("expected: vector proportional to [-2, 1], relative residual near 1e-16");
    if bad {
        println!("WRONG: the returned vector is not an eigenvector");
        std::process::exit(1);
    }
}
