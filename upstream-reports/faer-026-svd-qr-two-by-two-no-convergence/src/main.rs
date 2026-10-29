// Thin SVD of a 12 x 10 complex matrix, read from src/matrix.txt.
// Each data line is `<row> <col> Complex { re: <f64>, im: <f64> }`; the decimals
// round trip to the exact bits. LAPACK zgesdd decomposes this matrix, with
// singular values 3.94, 3.94, 0.4815, 0.4815, 0.1133, 0.1133, 0.0119, 0.0119, ~3e-17, ~8e-18.
// The SVD must return Ok.
use faer::{c64, Mat};

fn main() {
    let text = include_str!("matrix.txt");
    let mut a = Mat::<c64>::zeros(12, 10);
    for line in text.lines().skip(4) {
        let mut parts = line.splitn(3, ' ');
        let i: usize = parts.next().unwrap().parse().unwrap();
        let j: usize = parts.next().unwrap().parse().unwrap();
        let body = parts.next().unwrap().trim_start_matches("Complex { re: ").trim_end_matches(" }");
        let (re, im) = body.split_once(", im: ").unwrap();
        a[(i, j)] = c64::new(re.parse().unwrap(), im.parse().unwrap());
    }
    match a.thin_svd() {
        Ok(svd) => {
            let s = svd.S().column_vector();
            println!("Ok, singular values: {:?}", (0..10).map(|i| s[i].re).collect::<Vec<_>>());
        }
        Err(e) => {
            println!("expected Ok, got Err({e:?})");
            println!("WRONG: the SVD gives up on a matrix that LAPACK decomposes");
            std::process::exit(1);
        }
    }
}
