// Lower triangular solves L x = b where L is diagonal with a subnormal pivot d and b = d.
// Every exact solution entry is 1.
// 1. f64, 1 x 1, d = 1e-310.
// 2. f64, 2 x 2, d = 2^-1074 (the smallest subnormal) on both diagonal entries.
// 3. c64, 1 x 1, d = i * 2^-1074.
use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::{c64, Mat, Par};

fn main() {
    let mut bad = false;
    for (n, d) in [(1, 1e-310f64), (2, f64::from_bits(1))] {
        let a = Mat::<f64>::from_fn(n, n, |i, j| if i == j { d } else { 0.0 });
        let mut b = Mat::<f64>::full(n, 1, d);
        solve_lower_triangular_in_place(a.as_ref(), b.as_mut(), Par::Seq);
        let x: Vec<f64> = (0..n).map(|i| b[(i, 0)]).collect();
        println!("f64 n = {n}, d = {d:e}: expected x = {:?}, actual x = {x:?}", vec![1.0; n]);
        bad |= x.iter().any(|&v| v != 1.0);
    }
    let d = c64::new(0.0, f64::from_bits(1));
    let a = Mat::<c64>::full(1, 1, d);
    let mut b = Mat::<c64>::full(1, 1, d);
    solve_lower_triangular_in_place(a.as_ref(), b.as_mut(), Par::Seq);
    println!("c64 n = 1, d = i * 2^-1074: expected x = 1+0i, actual x = {}", b[(0, 0)]);
    bad |= b[(0, 0)] != c64::new(1.0, 0.0);
    if bad {
        println!("WRONG: the solve multiplies by an overflowing reciprocal");
        std::process::exit(1);
    }
}
