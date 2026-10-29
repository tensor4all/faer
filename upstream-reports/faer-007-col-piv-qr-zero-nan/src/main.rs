// Column pivoted QR of a 3 x 2 zero matrix.
// A zero matrix has the exact factorization Q = I, R = 0, so every entry of R
// and of Q R must be 0. Any NaN means the factorization went wrong.
use faer::linalg::solvers::DenseSolveCore;
use faer::Mat;

fn main() {
    let a = Mat::<f64>::zeros(3, 2);
    let qr = a.col_piv_qr();
    let r00 = qr.R()[(0, 0)];
    let back = qr.reconstruct();
    println!("R[0,0]: expected 0, got {r00}");
    println!("(Q R)[0,0]: expected 0, got {}", back[(0, 0)]);
    let bad = (0..3).any(|i| (0..2).any(|j| !back[(i, j)].is_finite()));
    if bad || !r00.is_finite() {
        println!("WRONG: column pivoted QR of a zero matrix returns NaN");
        std::process::exit(1);
    }
}
