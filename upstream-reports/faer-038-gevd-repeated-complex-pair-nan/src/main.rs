// Eigenvectors of the real pencil (A, I) with A = diag(R, R, 3), R = [[2, 1], [-1, 2]], and a
// coupling A[1, 3] between the two copies of R. The spectrum is 2 +- i twice and 3.
//
// The eigenvector substitution for one copy of the pair meets the other copy shifted by the same
// eigenvalue, an exactly singular 2 x 2 system, and divides by its zero pivot. The right vectors
// of the second pair and the left vectors of the first pair come out NaN, with and without the
// coupling.
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd;
use faer::{Mat, Par};

fn pencil(coupling: f64) -> (Mat<f64>, Mat<f64>) {
    let mut a = Mat::<f64>::zeros(5, 5);
    for o in [0usize, 2] {
        a[(o, o)] = 2.0;
        a[(o, o + 1)] = 1.0;
        a[(o + 1, o)] = -1.0;
        a[(o + 1, o + 1)] = 2.0;
    }
    a[(4, 4)] = 3.0;
    a[(1, 3)] = coupling;
    (a, Mat::<f64>::identity(5, 5))
}

/// The columns of the left and right eigenvector matrices that hold a non-finite entry.
fn non_finite_columns(coupling: f64) -> (Vec<usize>, Vec<usize>) {
    let n = 5;
    let (mut a, mut b) = pencil(coupling);
    let (mut re, mut im, mut beta) =
        (Diag::<f64>::zeros(n), Diag::<f64>::zeros(n), Diag::<f64>::zeros(n));
    let mut u_left = Mat::<f64>::zeros(n, n);
    let mut u_right = Mat::<f64>::zeros(n, n);
    // 0.24.4's `gevd_scratch` leaves no room for the complex 2 x 2 solve's pivoting LU (a
    // separate defect: `gevd_real` panics in `lu_in_place` with exactly that buffer), so the
    // buffer gets extra room to reach the solve itself
    let mut mem = MemBuffer::new(
        gevd::gevd_scratch::<f64>(
            n,
            ComputeEigenvectors::Yes,
            ComputeEigenvectors::Yes,
            Par::Seq,
            Default::default(),
        )
        .and(StackReq::new::<f64>(1024)),
    );
    gevd::gevd_real(
        a.as_mut(),
        b.as_mut(),
        re.as_mut(),
        im.as_mut(),
        beta.as_mut(),
        Some(u_left.as_mut()),
        Some(u_right.as_mut()),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .unwrap();
    let bad = |u: &Mat<f64>| {
        (0..n)
            .filter(|&j| (0..n).any(|i| !u[(i, j)].is_finite()))
            .collect::<Vec<_>>()
    };
    (bad(&u_left), bad(&u_right))
}

fn main() {
    let mut defect = false;

    // the reported call, through the public API
    let (a, b) = pencil(0.75);
    let eig = a.generalized_eigen(&b).unwrap();
    let u = eig.U();
    let finite = (0..5)
        .all(|j| (0..5).all(|i| u[(i, j)].re.is_finite() && u[(i, j)].im.is_finite()));
    println!("generalized_eigen, coupling 0.75: U finite = {finite} (expected true)");
    defect |= !finite;

    for coupling in [0.75, 0.0] {
        let (left, right) = non_finite_columns(coupling);
        println!(
            "gevd_real, coupling {coupling:4}: non-finite left columns {left:?}, right columns {right:?} (expected [] and [])"
        );
        defect |= !left.is_empty() || !right.is_empty();
    }

    if defect {
        std::process::exit(1);
    }
}
