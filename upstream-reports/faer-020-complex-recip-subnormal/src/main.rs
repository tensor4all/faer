// Complex reciprocal of z = i * m / 2, where m = f64::MIN_POSITIVE (so z is subnormal).
// The exact reciprocal is -i * 2 / m = -i * 2^1023, a finite f64 value.
use faer::c64;
use faer::traits::math_utils::recip;

fn main() {
    let m = f64::MIN_POSITIVE;
    let z = c64::new(0.0, m / 2.0);
    let r = recip(&z);
    let expected = c64::new(0.0, -2.0 / m);
    println!("expected recip = {expected:e}");
    println!("actual   recip = {r:e}");
    if r != expected {
        println!("WRONG: the reciprocal of a subnormal value is not finite");
        std::process::exit(1);
    }
}
