// Eigenvalues of two symmetric tridiagonal matrices with finite, representable spectra.
// 1. diag = [-1e308, 1e308], offdiag = [0]: already diagonal, eigenvalues -1e308 and 1e308.
// 2. diag = [2s, 2s, 2s], offdiag = [-s, -s], s = 1e-310: eigenvalues s (2 - sqrt 2, 2, 2 + sqrt 2).
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd;
use faer::{diag::{DiagMut, DiagRef}, Mat, Par};

fn tridiag(d: &[f64], e: &[f64]) -> (Result<(), evd::EvdError>, Vec<f64>) {
    let n = d.len();
    let (mut w, mut z) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
    // 0.24.4 has no scratch query for the tridiagonal entry point; the dense one covers it.
    let mut buf = MemBuffer::new(evd::self_adjoint_evd_scratch::<f64>(n, evd::ComputeEigenvectors::Yes, Par::Seq, Default::default()));
    let r = evd::tridiagonal_self_adjoint_evd(DiagRef::from_slice(d), DiagRef::from_slice(e),
        DiagMut::from_slice_mut(&mut w), Some(z.as_mut()), Par::Seq, MemStack::new(&mut buf), Default::default());
    (r, w)
}

fn main() {
    let (r1, w1) = tridiag(&[-1e308, 1e308], &[0.0]);
    println!("case 1: expected Ok, [-1e308, 1e308]; actual {r1:?}, {w1:?}");
    let s = 1e-310;
    let (r2, w2) = tridiag(&[2. * s; 3], &[-s, -s]);
    let x = [s * (2.0 - 2f64.sqrt()), 2.0 * s, s * (2.0 + 2f64.sqrt())];
    println!("case 2: expected Ok, {x:?}; actual {r2:?}, {w2:?}");
    let ok1 = r1.is_ok() && w1 == [-1e308, 1e308];
    let ok2 = r2.is_ok() && w2.iter().zip(x).all(|(a, b)| (a - b).abs() <= 1e-3 * s);
    if !(ok1 && ok2) {
        println!("WRONG: a finite spectrum near the range limits is lost");
        std::process::exit(1);
    }
}
