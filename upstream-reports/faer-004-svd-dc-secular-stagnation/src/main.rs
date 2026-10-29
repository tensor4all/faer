// Thin SVD of a 128 x 128 upper bidiagonal matrix (diagonal d, superdiagonal e):
// d_k = 1 and e_k = 1 for k < 64, d_64 = 1e-13, every other entry 0.
// 128 is the size where the SVD switches to divide and conquer.
// U S V^T must reproduce A to rounding level (about 1e-15), and s[64] is 1.2403e-14
// (LAPACK gesdd and faer's own QR algorithm agree on this value).
use faer::Mat;

fn main() {
    let n = 128;
    let a = Mat::<f64>::from_fn(n, n, |i, j| {
        if (i == j && i < 64) || (j == i + 1 && i < 64) {
            1.0
        } else if i == 64 && j == 64 {
            1e-13
        } else {
            0.0
        }
    });
    let svd = a.thin_svd().unwrap();
    let s = svd.S().column_vector();
    let err = (&a - svd.U() * svd.S() * svd.V().transpose()).norm_l2() / a.norm_l2();
    println!("|A - U S V^T|_F / |A|_F: expected about 1e-15, got {err:.1e}");
    println!("s[63]: expected 4.8327e-2, got {:.4e}", s[63]);
    println!("s[64]: expected 1.2403e-14, got {:.4e}", s[64]);
    if err > 1e-12 {
        println!("WRONG: the SVD factors do not reproduce A");
        std::process::exit(1);
    }
}
