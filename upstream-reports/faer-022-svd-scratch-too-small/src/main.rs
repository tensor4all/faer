// Values and V only SVD of an 8 x 8 rank one f32 matrix, with divide and conquer
// forced at size 4 (recursion_threshold = 4).
// The workspace comes from svd_scratch with exactly the same arguments,
// so svd must run without panicking for lack of memory.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors, SvdParams};
use faer::{Col, Mat, Par, Spec};

fn main() {
    let a = Mat::<f32>::from_fn(8, 8, |i, j| if i == 0 && j == 0 { 1.0 } else { 0.0 });
    let mut s = Col::<f32>::zeros(8);
    let mut v = Mat::<f32>::zeros(8, 8);
    let mut params = Spec::<SvdParams, f32>::default();
    params.recursion_threshold = 4;
    let req = svd_scratch::<f32>(8, 8, ComputeSvdVectors::No, ComputeSvdVectors::Thin, Par::Seq, params);
    println!("svd_scratch asks for {} bytes", req.size_bytes());
    let mut mem = MemBuffer::new(req);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        svd(a.as_ref(), s.as_mut().as_diagonal_mut(), None, Some(v.as_mut()), Par::Seq, MemStack::new(&mut mem), params)
    }));
    match r {
        Ok(res) => println!("svd returned {res:?}, s[0] = {}", s[0]),
        Err(_) => {
            println!("WRONG: svd panicked inside the workspace that svd_scratch sized");
            std::process::exit(1);
        }
    }
}
