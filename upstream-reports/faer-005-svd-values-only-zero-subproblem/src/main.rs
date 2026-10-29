// A 256 x 256 matrix with a single nonzero entry, A[128, 128] = 1.
// Its singular values are 1, 0, 0, ... . Asking for the values only
// must give the same leading value as the full SVD.
use faer::Mat;

fn main() {
    let mut a = Mat::<f64>::zeros(256, 256);
    a[(128, 128)] = 1.0;
    let values = a.singular_values().unwrap();
    let full = a.svd().unwrap();
    println!("leading singular value: expected 1");
    println!("  full SVD (U and V):  {}", full.S()[0]);
    println!("  singular_values():   {}", values[0]);
    if values[0] != 1.0 {
        println!("WRONG: the values only SVD loses the nonzero singular value");
        std::process::exit(1);
    }
}
