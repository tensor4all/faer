// Generalized eigenproblem A u = lambda B u with A = [[2, 1], [0, 3]] and B = I.
// The right eigenvector buffer is passed in filled with 2.0 instead of zeros.
// Each returned column must satisfy A u beta - B u alpha = 0.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::{evd::ComputeEigenvectors, gevd};
use faer::{diag::Diag, mat, Mat, Par};

fn main() {
    let original = mat![[2.0f64, 1.0], [0.0, 3.0]];
    let mut a = original.clone();
    let mut b = Mat::<f64>::identity(2, 2);
    let (mut alpha, mut imag, mut beta) = (Diag::<f64>::zeros(2), Diag::<f64>::zeros(2), Diag::<f64>::zeros(2));
    let mut right = Mat::<f64>::full(2, 2, 2.0);
    let mut mem = MemBuffer::new(gevd::gevd_scratch::<f64>(
        2, ComputeEigenvectors::No, ComputeEigenvectors::Yes, Par::Seq, Default::default()));
    gevd::gevd_real(a.as_mut(), b.as_mut(), alpha.as_mut(), imag.as_mut(), beta.as_mut(),
        None, Some(right.as_mut()), Par::Seq, MemStack::new(&mut mem), Default::default()).unwrap();
    let residual = ((&original * &right) * beta.as_ref() - &right * alpha.as_ref()).norm_max();
    println!("eigenvalues alpha/beta = [{}, {}]", alpha[0] / beta[0], alpha[1] / beta[1]);
    println!("first right eigenvector = [{}, {}] (expected proportional to [1, 0])", right[(0, 0)], right[(1, 0)]);
    println!("residual |A U beta - U alpha|_max: expected below 1e-12, actual {residual:e}");
    if residual > 1e-12 {
        println!("WRONG: the eigenvectors depend on the old buffer contents");
        std::process::exit(1);
    }
}
