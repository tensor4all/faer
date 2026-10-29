// A 4 x 4 real matrix with two complex eigenvalue pairs, +-1e-5 i and +-2e-5 i,
// in two 2 x 2 rotation blocks that are coupled by entries of size 1e8:
//
//     [  0     1e-5  1e8   1e8  ]
//     [ -1e-5  0     1e8   1e8  ]
//     [  0     0     0     2e-5 ]
//     [  0     0    -2e-5  0    ]
//
// We check every eigenvector u with eigenvalue lambda through the residual ratio
// |A u - lambda u|_1 / (|A|_1 |u|_1 eps). LAPACK's dgeev gives 9.6e-14 here.
use faer::prelude::*;

fn main() {
    let a = Mat::<f64>::from_fn(4, 4, |i, j| match (i, j) {
        (0, 1) => 1e-5,
        (1, 0) => -1e-5,
        (2, 3) => 2e-5,
        (3, 2) => -2e-5,
        (0 | 1, 2 | 3) => 1e8,
        _ => 0.0,
    });
    let evd = a.eigen().unwrap();
    let (u, s) = (evd.U(), evd.S());

    let ac = Mat::<c64>::from_fn(4, 4, |i, j| c64::new(a[(i, j)], 0.0));
    let col_norm = |m: MatRef<'_, c64>, j: usize| (0..4).map(|i| m[(i, j)].norm()).sum::<f64>();
    let a_norm = (0..4).map(|j| col_norm(ac.as_ref(), j)).fold(0.0, f64::max);
    let r = &ac * u - u * s.as_ref();
    let mut worst = 0.0f64;
    for j in 0..4 {
        let ratio = col_norm(r.as_ref(), j) / (a_norm * col_norm(u, j) * f64::EPSILON);
        println!("eigenvalue {:+.1e}i: residual ratio {ratio:.2e}", s[j].im);
        worst = worst.max(ratio);
    }
    if worst > 20.0 {
        println!("WRONG: an eigenvector is far from exact (dgeev: 9.6e-14)");
        std::process::exit(1);
    }
}
