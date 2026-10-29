// Eigendecomposition of an empty (0 x 0) complex tridiagonal matrix, eigenvectors requested.
// Expected: Ok(()) with nothing written, as for the real type and for `self_adjoint_evd`.
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::evd;
use faer::{c64, diag::{DiagMut, DiagRef}, Mat, Par};

fn main() {
    let r = std::panic::catch_unwind(|| {
        let (d, e, mut w) = (Vec::<c64>::new(), Vec::<c64>::new(), Vec::<c64>::new());
        let mut u = Mat::<c64>::zeros(0, 0);
        // Sized by hand: 0.24.4 has no scratch query for this entry point, and the dense
        // query `self_adjoint_evd_scratch` itself overflows on n = 0 in a debug build.
        let mut buf = MemBuffer::new(StackReq::new::<c64>(1024));
        evd::tridiagonal_self_adjoint_evd(DiagRef::from_slice(&d), DiagRef::from_slice(&e),
            DiagMut::from_slice_mut(&mut w), Some(u.as_mut()), Par::Seq, MemStack::new(&mut buf), Default::default())
    });
    println!("expected: Ok(())");
    match r {
        Ok(v) => println!("actual:   {v:?}"),
        Err(_) => {
            println!("actual:   panic");
            println!("WRONG: the empty complex case panics");
            std::process::exit(1);
        }
    }
}
