// SVD of a 128 x 128 f32 lower bidiagonal matrix whose entries are eps^u,
// with u uniform in [-2, 2] (LAPACK's xCHKBD test type 16). Default parameters.
// Checks the ratio |I - V^T V|_1 / (n eps) used by LAPACK's ORT01, which must stay
// below 50. U and the reconstruction are checked the same way for comparison.
use faer::Mat;

// Small deterministic generator, so the input is the same on every machine.
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

// |I - Q^T Q|_1 / (n eps), accumulated in f64.
fn ortho_ratio(q: faer::MatRef<'_, f32>) -> f64 {
    let n = q.ncols();
    let q = Mat::<f64>::from_fn(q.nrows(), n, |i, j| q[(i, j)] as f64);
    let g = q.transpose() * &q - Mat::<f64>::identity(n, n);
    let one_norm = (0..n).map(|j| (0..n).map(|i| g[(i, j)].abs()).sum::<f64>()).fold(0.0, f64::max);
    one_norm / (n as f64 * f32::EPSILON as f64)
}

fn main() {
    let n = 128;
    let mut rng = Rng(0x9E3779B97F4A7C15 ^ 32);
    let a = Mat::<f32>::from_fn(n, n, |i, j| {
        if i == j || i == j + 1 { f32::EPSILON.powf((rng.uniform() * 4.0 - 2.0) as f32) } else { 0.0 }
    });
    let svd = a.svd().unwrap();
    let (rv, ru) = (ortho_ratio(svd.V()), ortho_ratio(svd.U()));
    let back = svd.U() * svd.S() * svd.V().transpose();
    let rec = (&a - &back).norm_l2() / (a.norm_l2() * n as f32 * f32::EPSILON);
    println!("|A - U S V^T|_F / (|A|_F n eps): {rec:.3e} (limit 50)");
    println!("U orthogonality ratio: {ru:.3e} (limit 50)");
    println!("V orthogonality ratio: {rv:.3e} (limit 50)");
    if rv > 50.0 {
        println!("WRONG: V is not orthogonal");
        std::process::exit(1);
    }
}
