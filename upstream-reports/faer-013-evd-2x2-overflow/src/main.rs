// Self adjoint eigenvalues of the diagonal matrix 1e308 * I (2 x 2).
// Both exact eigenvalues are the finite value 1e308.
use faer::{mat, Side};

fn main() {
    let a = mat![[1e308f64, 0.0], [0.0, 1e308]];
    let s = a.self_adjoint_eigenvalues(Side::Lower).unwrap();
    println!("expected eigenvalues = [1e308, 1e308]");
    println!("actual   eigenvalues = {s:?}");
    if s.iter().any(|&x| x != 1e308) {
        println!("WRONG: a finite spectrum overflows to inf");
        std::process::exit(1);
    }
}
