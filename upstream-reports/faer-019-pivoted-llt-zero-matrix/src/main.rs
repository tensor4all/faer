// Pivoted Cholesky of the zero matrix (n = 1 and n = 3).
// The zero matrix is positive semidefinite with rank 0, so the factorization
// should succeed with rank 0, as LAPACK dpstrf does.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt_pivoting::factor;
use faer::{Mat, Par};

fn main() {
    let mut bad = false;
    for n in [1usize, 3] {
        let mut l = Mat::<f64>::zeros(n, n);
        let (mut p, mut pinv) = (vec![0usize; n], vec![0usize; n]);
        let mut mem = MemBuffer::new(factor::cholesky_in_place_scratch::<usize, f64>(n, Par::Seq, Default::default()));
        let r = factor::cholesky_in_place(l.as_mut(), &mut p, &mut pinv, Par::Seq,
            MemStack::new(&mut mem), Default::default());
        let actual = match &r { Ok((info, _)) => format!("Ok, rank {}", info.rank), Err(e) => format!("{e:?}") };
        println!("n = {n}: expected Ok, rank 0; actual {actual}");
        bad |= !matches!(&r, Ok((info, _)) if info.rank == 0);
    }
    if bad {
        println!("WRONG: the zero matrix does not give rank 0");
        std::process::exit(1);
    }
}
