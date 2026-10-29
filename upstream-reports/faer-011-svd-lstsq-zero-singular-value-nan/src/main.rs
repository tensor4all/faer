// Least squares solve with the SVD of a rank deficient 3 x 2 matrix.
// A = [[1, 0], [0, 0], [0, 0]], b = [2, 0, 0]. Then x = [2, 0] solves A x = b
// exactly, and it is the minimum norm solution. The answer must be finite.
use faer::linalg::solvers::SolveLstsq;
use faer::mat;

fn main() {
    let a = mat![[1.0f64, 0.0], [0.0, 0.0], [0.0, 0.0]];
    let b = mat![[2.0f64], [0.0], [0.0]];
    let svd = a.svd().unwrap();
    let x = svd.solve_lstsq(b.as_ref());
    let reference = svd.pseudoinverse() * &b;
    println!("pseudoinverse * b: {reference:?}");
    println!("solve_lstsq:       {x:?}");
    if !(x[(0, 0)] == 2.0 && x[(1, 0)].is_finite()) {
        println!("WRONG: solve_lstsq does not return a finite least squares solution");
        std::process::exit(1);
    }
}
