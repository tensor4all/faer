// Unpivoted QR of a tall m x 3 matrix with nearly dependent columns:
// column 0 is all ones, column 1 is ones plus 1e-10 in row 0,
// column 2 is ones plus 1e-6 in row 1. The matrix has full rank 3.
// A backward stable QR gives |A - Q R|_F / |A|_F of a few eps (about 1e-16)
// at every m. The program prints that error and the error divided by eps.
use faer::linalg::solvers::DenseSolveCore;
use faer::Mat;

fn main() {
    let mut worst = 0.0f64;
    for m in [100, 1000, 8192] {
        let a = Mat::<f64>::from_fn(m, 3, |i, j| match j {
            0 => 1.0,
            1 => 1.0 + if i == 0 { 1e-10 } else { 0.0 },
            _ => 1.0 + if i == 1 { 1e-6 } else { 0.0 },
        });
        let qr = a.qr();
        let err = (&a - qr.reconstruct()).norm_l2() / a.norm_l2();
        println!("m = {m:5}: |A - Q R|_F / |A|_F = {err:.1e} ({:.0} eps), R[2,2] = {:.3e}", err / f64::EPSILON, qr.R()[(2, 2)]);
        worst = worst.max(err);
    }
    if worst > 1e-13 {
        println!("WRONG: QR does not reproduce A to rounding level");
        std::process::exit(1);
    }
}
