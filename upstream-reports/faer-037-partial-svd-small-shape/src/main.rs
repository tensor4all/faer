// The public `partial_svd` driver clips `max_dim` to `min(m, n)` and then hands
// it to `partial_svd_imp`, which asserts `max_dim < min(m, n)`. Every nonempty
// matrix with `min(m, n) <= 2 * MIN_DIM` (64) therefore panics under the default
// parameters, and so does any explicit `max_dim >= min(m, n)`.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::eigen::{partial_svd, partial_svd_scratch, PartialEigenParams};
use faer::{Col, Mat, Par};

fn main() {
    // the issue's MWE: a 4x4 identity, default parameters, one singular triplet
    let a = Mat::<f64>::identity(4, 4);
    let mut u = Mat::<f64>::zeros(4, 1);
    let mut v = Mat::<f64>::zeros(4, 1);
    let mut s = [0.0];
    let v0 = Col::<f64>::from_fn(4, |i| (i + 1) as f64);
    let params = PartialEigenParams::default();
    let mut mem = MemBuffer::new(partial_svd_scratch(&a, 1, Par::Seq, params));
    println!("4x4 identity, default parameters, one singular triplet");
    println!("expected: n_converged = 1 and s[0] = 1");
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        partial_svd(
            u.as_mut(),
            v.as_mut(),
            &mut s,
            &a,
            v0.as_ref(),
            1e-10,
            Par::Seq,
            MemStack::new(&mut mem),
            params,
        )
    }));
    match run {
        Ok(info) => {
            println!("actual  : n_converged = {}, s[0] = {:?}", info.n_converged_eigen, s[0]);
            if info.n_converged_eigen != 1 || (s[0] - 1.0).abs() > 1e-8 {
                println!("WRONG: the requested triplet was not returned");
                std::process::exit(1);
            }
        }
        Err(_) => {
            println!("actual  : panic in partial_svd_imp (see the message above)");
            println!("WRONG: a public call with default parameters panicked");
            std::process::exit(1);
        }
    }
}
