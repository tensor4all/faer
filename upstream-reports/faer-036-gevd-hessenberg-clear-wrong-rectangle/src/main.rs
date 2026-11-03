// Generalized Schur reduction of a regular 40 x 40 pencil, asking for the eigenvalues only and
// for the left eigenvectors only. Both requests take the branch of the generalized Hessenberg
// reduction that clears the stored rotations with a rectangle covering the already reduced
// columns, so the QZ iteration never deflates and `beta` stays zero.
//
// A = B*C with B = X X^T + I symmetric positive definite and C the tridiagonal 1-D Laplacian:
// the pencil is regular, so every eigenvalue alpha/beta is finite and beta != 0.
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd;
use faer::{Mat, Par};

fn pencil(n: usize) -> (Mat<f64>, Mat<f64>) {
    let x = Mat::<f64>::from_fn(n, n, |i, j| (1 + i + j * n) as f64 / (n * n) as f64);
    let b = &x * x.transpose() + Mat::<f64>::identity(n, n);
    let c = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            2.0
        } else if i.abs_diff(j) == 1 {
            -1.0
        } else {
            0.0
        }
    });
    (&b * &c, b)
}

fn beta_zeros(n: usize, left: bool, right: bool) -> usize {
    let (a, b) = pencil(n);
    let mut a = a;
    let mut b = b;
    let (mut alpha, mut imag, mut beta) =
        (Diag::<f64>::zeros(n), Diag::<f64>::zeros(n), Diag::<f64>::zeros(n));
    let mut u_left = Mat::<f64>::zeros(n, n);
    let mut u_right = Mat::<f64>::zeros(n, n);
    let pair = |yes: bool| {
        if yes {
            ComputeEigenvectors::Yes
        } else {
            ComputeEigenvectors::No
        }
    };
    let mut mem = MemBuffer::new(gevd::gevd_scratch::<f64>(
        n,
        pair(left),
        pair(right),
        Par::Seq,
        Default::default(),
    ));
    gevd::gevd_real(
        a.as_mut(),
        b.as_mut(),
        alpha.as_mut(),
        imag.as_mut(),
        beta.as_mut(),
        if left { Some(u_left.as_mut()) } else { None },
        if right { Some(u_right.as_mut()) } else { None },
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .unwrap();
    (0..n).filter(|j| beta[*j] == 0.0).count()
}

fn main() {
    let n = 40;
    let modes = [
        ("eigenvalues only", false, false),
        ("left only", true, false),
        ("right only", false, true),
        ("both", true, true),
    ];
    let mut bad = false;
    for (name, left, right) in modes {
        let zeros = beta_zeros(n, left, right);
        println!("{name:17}: beta == 0 for {zeros} of {n} eigenvalues");
        if zeros != 0 {
            bad = true;
        }
    }
    println!();
    println!("the pencil is regular, so every eigenvalue alpha/beta is finite and beta != 0;");
    println!("a nonzero count is the defect (the reduced A/B of the block after the first were");
    println!("cleared before the QZ iteration).");
    if bad {
        println!("WRONG: the spectrum depends on which eigenvectors are requested");
        std::process::exit(1);
    }
}
