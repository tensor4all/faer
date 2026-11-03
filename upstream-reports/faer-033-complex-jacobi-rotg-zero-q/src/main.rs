// `JacobiRotation::rotg(p, 0)` returns the rotation J = I and r = p, because
// J maps the column [p, 0]^T to [r, 0]^T = [p, 0]^T.
// The complex branch returns r = 1 instead, which equals p only when p = 1.
use faer::linalg::jacobi::JacobiRotation;
use faer::prelude::*;

fn main() {
    let p = c64::new(3.0, 2.0);
    let zero = c64::new(0.0, 0.0);
    let (j, r) = JacobiRotation::rotg(p, zero);
    // J = [[c, -conj(s)], [s, c]], so J [p, 0]^T = [c p, s p]^T.
    let first = j.c * p;
    let second = j.s * p;
    println!("rotg(p = {p}, q = 0) = (J, r) with");
    println!("  c = {}, s = {}", j.c, j.s);
    println!("  r = {r}, expected r = p = {p}");
    println!("  J [p, 0]^T = [{first}, {second}], expected [{p}, 0]");
    if r != p || first != p || second != zero {
        println!("WRONG: the returned r is not the entry the rotation carries");
        std::process::exit(1);
    }
}
