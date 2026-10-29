// Symmetric 127 x 127 matrix A = Q diag(s) Q^T with known eigenvalues s_i = 1 - 2 i / 126.
// Q is a product of adjacent plane rotations with c = 0.6, s = 0.8.
// 1. `self_adjoint_eigenvalues` (values only): uses the tridiagonal QR algorithm.
// 2. Control: `self_adjoint_evd` on the same A with eigenvectors and recursion threshold 8,
//    which uses divide and conquer.
// We print max_i |lambda_i - s_i| / eps. LAPACK's CHKST test accepts a ratio below 60.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self, ComputeEigenvectors, SelfAdjointEvdParams};
use faer::{Mat, Par, Side, Spec};

macro_rules! ratios {
    ($t:ty) => {{
        let n = 127;
        let s: Vec<$t> = (0..n).map(|i| 1.0 - 2.0 * i as $t / (n - 1) as $t).collect();
        let mut q = Mat::<$t>::identity(n, n);
        for i in 1..n {
            for j in 0..n {
                let (x, y) = (q[(i - 1, j)], q[(i, j)]);
                q[(i - 1, j)] = 0.6 * x - 0.8 * y;
                q[(i, j)] = 0.8 * x + 0.6 * y;
            }
        }
        // Plain triple loop so the input does not depend on the matrix multiply kernel.
        let a = Mat::<$t>::from_fn(n, n, |i, j| (0..n).fold(0.0, |acc, k| acc + q[(i, k)] * s[k] * q[(j, k)]));
        let ratio = |mut w: Vec<$t>| {
            w.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let mut known = s.clone();
            known.sort_by(|x, y| x.partial_cmp(y).unwrap());
            (w.iter().zip(&known).fold(0.0 as $t, |m, (x, y)| m.max((x - y).abs())) / <$t>::EPSILON) as f64
        };
        let qr = ratio(a.self_adjoint_eigenvalues(Side::Lower).unwrap());
        let mut p: Spec<SelfAdjointEvdParams, $t> = Default::default();
        p.recursion_threshold = 8;
        let (mut w, mut u) = (Mat::<$t>::zeros(n, 1), Mat::<$t>::zeros(n, n));
        let mut mem = MemBuffer::new(evd::self_adjoint_evd_scratch::<$t>(n, ComputeEigenvectors::Yes, Par::Seq, p));
        evd::self_adjoint_evd(a.as_ref(), w.as_mut().col_mut(0).as_diagonal_mut(), Some(u.as_mut()),
            Par::Seq, MemStack::new(&mut mem), p).unwrap();
        (qr, ratio((0..n).map(|i| w[(i, 0)]).collect()))
    }};
}

fn main() {
    let (q32, d32) = ratios!(f32);
    let (q64, d64) = ratios!(f64);
    println!("max |lambda - s| / eps, expected below 60");
    println!("f32: values only (QR) {q32:.1}, divide and conquer {d32:.1}");
    println!("f64: values only (QR) {q64:.1}, divide and conquer {d64:.1}");
    if q32 >= 60.0 || q64 >= 60.0 {
        println!("WRONG: the QR eigenvalues are less accurate than the LAPACK test bound");
        std::process::exit(1);
    }
}
