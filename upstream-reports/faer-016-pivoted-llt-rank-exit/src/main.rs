// Pivoted Cholesky of the positive definite matrix A = ones(3, 3) + eps * I.
// The factorization stops at numerical rank 1. Rebuilding L L^T from the returned
// factor should give back A up to a truncation error of about eps, not an O(1) error.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt_pivoting::{factor, reconstruct};
use faer::{Mat, Par};

fn main() {
    let a = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { 1.0 + f64::EPSILON } else { 1.0 });
    let mut l = a.clone();
    let (mut p, mut pinv) = ([0usize; 3], [0usize; 3]);
    let mut mem = MemBuffer::new(factor::cholesky_in_place_scratch::<usize, f64>(3, Par::Seq, Default::default()));
    let (info, perm) = factor::cholesky_in_place(l.as_mut(), &mut p, &mut pinv, Par::Seq,
        MemStack::new(&mut mem), Default::default()).unwrap();
    let mut out = Mat::<f64>::zeros(3, 3);
    let mut mem = MemBuffer::new(reconstruct::reconstruct_scratch::<usize, f64>(3, Par::Seq));
    reconstruct::reconstruct(out.as_mut(), l.as_ref(), perm, Par::Seq, MemStack::new(&mut mem));
    println!("numerical rank = {}", info.rank);
    println!("expected A[2,2] = {}, reconstructed A[2,2] = {}", a[(2, 2)], out[(2, 2)]);
    if (out[(2, 2)] - a[(2, 2)]).abs() > 1e-12 {
        println!("WRONG: the factor still holds input entries past the rank");
        std::process::exit(1);
    }
}
