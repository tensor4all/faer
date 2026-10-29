// Full SVD of a tall 4 x 2 matrix, written into a U buffer that holds ones.
// The full U must be orthogonal (U^T U = I) whatever the buffer held before.
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
use faer::{Mat, Par};

fn main() {
    let a = Mat::<f64>::identity(4, 2);
    let mut u = Mat::<f64>::full(4, 4, 1.0); // a reused buffer with old contents
    let mut v = Mat::<f64>::zeros(2, 2);
    let mut s = Diag::<f64>::zeros(2);
    let full = ComputeSvdVectors::Full;
    let mut mem = MemBuffer::new(svd_scratch::<f64>(4, 2, full, full, Par::Seq, Default::default()));
    svd(a.as_ref(), s.as_mut(), Some(u.as_mut()), Some(v.as_mut()), Par::Seq, MemStack::new(&mut mem), Default::default()).unwrap();

    let back = u.as_ref().get(.., ..2) * s.as_ref() * v.transpose();
    let ortho = (u.transpose() * &u - Mat::<f64>::identity(4, 4)).norm_max();
    println!("max |A - U S V^T|: {:e}", (&back - &a).norm_max());
    println!("max |U^T U - I|: expected about 1e-16, got {ortho:e}");
    println!("U = {u:?}");
    if ortho > 1e-12 {
        println!("WRONG: the full U is not orthogonal");
        std::process::exit(1);
    }
}
