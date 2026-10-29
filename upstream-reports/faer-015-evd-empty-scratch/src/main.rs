// Self adjoint eigendecomposition of a valid 0 x 0 matrix.
// Expected: Ok with empty eigenvalues and a 0 x 0 eigenvector matrix, no panic.
use faer::{Mat, Side};

fn main() {
    let a = Mat::<f64>::zeros(0, 0);
    let r = std::panic::catch_unwind(|| a.self_adjoint_eigen(Side::Lower).map(|e| e.U().shape()));
    println!("expected: Ok((0, 0))");
    match r {
        Ok(v) => println!("actual:   {v:?}"),
        Err(_) => {
            println!("WRONG: self_adjoint_eigen panics on an empty matrix");
            std::process::exit(1);
        }
    }
}
