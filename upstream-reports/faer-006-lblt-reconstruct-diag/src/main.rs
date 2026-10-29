// Factor A = diag(2, 3) with the Bunch-Kaufman LBL^T factorization and rebuild A from the factors.
// The rebuilt matrix must equal A.
use faer::linalg::solvers::DenseSolveCore;
use faer::{mat, Side};

fn main() {
    let a = mat![[2.0f64, 0.0], [0.0, 3.0]];
    let r = a.lblt(Side::Lower).reconstruct();
    println!("expected diag = [2, 3]");
    println!("actual   diag = [{}, {}]", r[(0, 0)], r[(1, 1)]);
    if (&r - &a).norm_max() > 1e-12 {
        println!("WRONG: reconstruct does not return A");
        std::process::exit(1);
    }
}
