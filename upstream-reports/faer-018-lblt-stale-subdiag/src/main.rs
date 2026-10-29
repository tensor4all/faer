// Bunch-Kaufman LBL^T factorization of A = diag(1, 2, ..., n) with the low level API.
// The caller's `subdiag` buffer is prefilled with 7.0 (a reused buffer).
// A diagonal A has only 1 x 1 pivots, so every returned subdiagonal entry must be 0.
// Then the factors solve A x = A [1, ..., 1]^T; x must be all ones.
// n = 100 is above the default block size, so the blocked path runs too.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::lblt::{factor, solve};
use faer::{diag::Diag, Conj, Mat, Par};

fn main() {
    std::panic::set_hook(Box::new(|_| {})); // a panic is reported in the table below
    let mut bad = false;
    for n in [1usize, 3, 100] {
        let a = Mat::<f64>::from_fn(n, n, |i, j| if i == j { 1.0 + i as f64 } else { 0.0 });
        let mut lb = a.clone();
        let mut subdiag = Diag::<f64>::full(n, 7.0);
        let (mut p, mut pinv) = (vec![0usize; n], vec![0usize; n]);
        let mut mem = MemBuffer::new(factor::cholesky_in_place_scratch::<usize, f64>(n, Par::Seq, Default::default()));
        let (_, perm) = factor::cholesky_in_place(lb.as_mut(), subdiag.as_mut(), &mut p, &mut pinv,
            Par::Seq, MemStack::new(&mut mem), Default::default());
        let stale = (0..n).filter(|&i| subdiag[i] != 0.0).count();
        let mut x = Mat::<f64>::from_fn(n, 1, |i, _| a[(i, i)]);
        let mut mem = MemBuffer::new(solve::solve_in_place_scratch::<usize, f64>(n, 1, Par::Seq));
        let solved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve::solve_in_place_with_conj(lb.as_ref(), lb.diagonal(), subdiag.as_ref(), Conj::No, perm,
                x.as_mut(), Par::Seq, MemStack::new(&mut mem));
            (0..n).map(|i| (x[(i, 0)] - 1.0).abs()).fold(0.0, f64::max)
        }));
        let err = match solved { Ok(e) => format!("{e:.1e}"), Err(_) => "panic".into() };
        println!("n = {n}: nonzero subdiag entries: expected 0, actual {stale}; solve max|x - 1|: expected near 1e-16, actual {err}");
        bad |= stale > 0 || !matches!(solved, Ok(e) if e < 1e-12);
    }
    if bad {
        println!("WRONG: subdiag keeps values from the caller's buffer");
        std::process::exit(1);
    }
}
