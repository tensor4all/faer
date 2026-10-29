// Ask `partial_svd` for the 10 largest singular triplets of a random 200 x 120 matrix.
// Then check each pair it reports as converged: A^T A v must equal s^2 v.
// A converged pair should leave a residual near rounding level (about 1e-11 here).
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::eigen::{partial_svd, partial_svd_scratch, PartialEigenParams};
use faer::prelude::*;

// Small deterministic normal generator, so the input is the same on every machine.
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

fn main() {
    let (m, n, k) = (200, 120, 10);
    let mut rng = Rng(0x9E3779B97F4A7C15 ^ (107 * 7919 + 200));
    let a = Mat::<f64>::from_fn(m, n, |_, _| rng.normal());
    let mut v0 = Col::<f64>::from_fn(n, |_| rng.normal());
    v0 /= v0.norm_l2();

    // A small search space (12 to 24 vectors) so the algorithm restarts several times.
    let params = PartialEigenParams { min_dim: 12, max_dim: 24, ..Default::default() };
    let (mut u, mut v, mut s) = (Mat::<f64>::zeros(m, k), Mat::<f64>::zeros(n, k), vec![0.0; k]);
    let mut mem = MemBuffer::new(partial_svd_scratch::<f64>(&a, k, Par::Seq, params));
    let tol = f64::EPSILON * 128.0;
    let info = partial_svd(u.as_mut(), v.as_mut(), &mut s, &a, v0.as_ref(), tol, Par::Seq, MemStack::new(&mut mem), params);

    println!("pairs reported converged: {} of {k}", info.n_converged_eigen);
    let mut worst = 0.0f64;
    for j in 0..info.n_converged_eigen {
        let r = (a.transpose() * (&a * v.col(j)) - Scale(s[j] * s[j]) * v.col(j)).norm_l2();
        println!("pair {j}: s = {:.6}, residual |A^T A v - s^2 v| = {r:.1e}", s[j]);
        worst = worst.max(r);
    }
    if worst > 1e-8 {
        println!("WRONG: a pair reported as converged is far from an exact singular pair");
        std::process::exit(1);
    }
}
