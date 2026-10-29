// Compute the Euclidean norm of the row [s, s/2] of a column-major 2 x 2 matrix.
// The row is strided in memory. Its exact norm is sqrt(1.25) * s, a normal finite f64.
fn main() {
    let mut bad = false;
    for s in [1.0f64, 1e160, 1e-300] {
        let a = faer::mat![[s, s * 0.5], [0.0, s]];
        let strided = a.row(0).norm_l2();
        let contiguous = a.row(0).to_owned().norm_l2();
        let expected = s * 1.25f64.sqrt();
        println!("s = {s:e}: expected {expected:e}, strided {strided:e}, contiguous {contiguous:e}");
        if !((strided - expected).abs() <= 1e-14 * expected) {
            bad = true;
        }
    }
    if bad {
        println!("WRONG: the strided norm overflows or underflows");
        std::process::exit(1);
    }
}
