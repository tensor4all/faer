// `norm_max` drops NaN entries: the SIMD kernel accumulates with `simd.max`,
// the pairwise split combines halves with `fmax`, and the scalar fallback uses
// `fmax` as well. Neither the vector max nor IEEE `maxNum` is NaN-propagating,
// so a matrix that contains a NaN reports a finite number (0 when every entry
// is NaN), while `norm_l1` and `norm_l2` of the same input report NaN.
use faer::{c64, mat, Col, Mat};

fn main() {
    let mut failures = 0usize;

    let a = mat![[1.0f64, f64::NAN]];
    let v = Col::<f64>::from_fn(2, |i| [1.0, f64::NAN][i]);
    let z = Col::<c64>::from_fn(2, |i| [c64::new(1.0, 0.0), c64::new(f64::NAN, 0.0)][i]);
    let zi = Col::<c64>::from_fn(2, |i| [c64::new(1.0, 0.0), c64::new(0.0, f64::NAN)][i]);
    let m = Mat::<f64>::from_fn(2, 3, |i, j| if (i, j) == (1, 1) { f64::NAN } else { 1.0 });
    let only = Col::<f64>::from_fn(1, |_| f64::NAN);
    let big = Col::<f64>::from_fn(100, |i| if i == 57 { f64::NAN } else { i as f64 });
    // 129 elements is the first length whose column reaches the pairwise split
    // above LINEAR_IMPL_THRESHOLD (128).
    let split = Col::<f64>::from_fn(129, |i| if i == 128 { f64::NAN } else { i as f64 });

    let mut check = |label: &str, actual: f64| {
        println!("{label:<34} expected NaN        actual {actual}");
        if !actual.is_nan() {
            failures += 1;
        }
    };
    check("1x2 row-major, one NaN", a.norm_max());
    check("2-vector, one NaN", v.norm_max());
    check("2-vector c64, NaN real part", z.norm_max());
    check("2-vector c64, NaN imag part", zi.norm_max());
    check("2x3 column-major, one NaN", m.norm_max());
    check("2x3, row 1 strided", m.row(1).norm_max());
    check("all-NaN vector", only.norm_max());
    check("100-vector, NaN at 57", big.norm_max());
    check("129-vector, NaN at 128", split.norm_max());
    check("transpose of the 2x3", m.transpose().norm_max());

    println!();
    println!("reference: norm_l1 = {} (expected NaN)", v.norm_l1());
    println!("reference: norm_l2 = {} (expected NaN)", v.norm_l2());
    if !v.norm_l1().is_nan() || !v.norm_l2().is_nan() {
        failures += 1;
    }

    if failures > 0 {
        println!("\nWRONG: {failures} norm_max result(s) lost a NaN entry");
        std::process::exit(1);
    }
}
