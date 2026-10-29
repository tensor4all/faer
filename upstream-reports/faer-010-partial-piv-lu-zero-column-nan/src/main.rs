// Partial pivoted LU of A = [[0, 1], [0, 2]], whose first column is zero.
// P = I, L = I, U = A is a valid factorization, so P^T L U must give back A.
use faer::linalg::solvers::DenseSolveCore;
use faer::mat;

fn main() {
    let a = mat![[0.0f64, 1.0], [0.0, 2.0]];
    let lu = a.partial_piv_lu();
    let back = lu.reconstruct();
    println!("L[1,0]: expected 0, got {}", lu.L()[(1, 0)]);
    println!("reconstruction: expected {a:?}");
    println!("                got      {back:?}");
    if (0..2).any(|i| (0..2).any(|j| back[(i, j)] != a[(i, j)])) {
        println!("WRONG: the factors do not reconstruct A");
        std::process::exit(1);
    }
}
