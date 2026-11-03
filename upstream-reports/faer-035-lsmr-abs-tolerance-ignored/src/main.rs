// `LsmrParams::abs_tolerance` is never read: the convergence test uses
// `norm_ref * rel_tolerance` only. An initial normal residual below
// `abs_tolerance`, with `rel_tolerance = 0` and `max_iters = 0`, is therefore
// reported as `NoConvergence` instead of converging at the initial guess.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::lsmr::{lsmr, lsmr_scratch, LsmrParams};
use faer::{Mat, Par};

fn main() {
    // A = I, b = [1, 1], x0 = 0: the initial normal residual |A^H (b - A x0)| is sqrt(2).
    let a = Mat::<f64>::identity(2, 2);
    let b = Mat::<f64>::from_fn(2, 1, |_, _| 1.0);
    let mut x = Mat::<f64>::zeros(2, 1);
    let atol = 10.0;
    let initial_residual = b.norm_l2();
    let mut mem = MemBuffer::new(lsmr_scratch(&a, &a, 1, Par::Seq));
    let result = lsmr(
        x.as_mut(),
        &a,
        &a,
        b.as_ref(),
        LsmrParams {
            abs_tolerance: atol,
            rel_tolerance: 0.0,
            max_iters: 0,
            ..Default::default()
        },
        |_| {},
        Par::Seq,
        MemStack::new(&mut mem),
    );
    println!("initial normal residual = {initial_residual:e}, abs_tolerance = {atol:e}");
    println!("expected: Ok, converged at the initial guess (residual below abs_tolerance)");
    println!("actual  : {result:?}");
    if result.is_err() {
        println!("WRONG: abs_tolerance was ignored");
        std::process::exit(1);
    }
}
