// `apply_on_the_right_2x2` computes M * J for J = [[c, -conj(s)], [s, c]].
// It delegates to the left helper with the adjoint, which gives M * conj(J).
// With a non-real `s`, even M = I reproduces conj(J) instead of J.
use faer::linalg::jacobi::JacobiRotation;
use faer::prelude::*;

type M = [[c64; 2]; 2];

// Arguments and result use the same row-major order as `apply_on_the_left_2x2`.
fn right(j: &JacobiRotation<c64>, m: M) -> M {
    let (m00, m01, m10, m11) = j.apply_on_the_right_2x2(m[0][0], m[0][1], m[1][0], m[1][1]);
    [[m00, m01], [m10, m11]]
}

// M * J, computed from the definition of J.
fn expected(j: &JacobiRotation<c64>, m: M) -> M {
    let s_conj = j.s.conj();
    [
        [m[0][0] * j.c + m[0][1] * j.s, -m[0][0] * s_conj + m[0][1] * j.c],
        [m[1][0] * j.c + m[1][1] * j.s, -m[1][0] * s_conj + m[1][1] * j.c],
    ]
}

fn diff(a: M, b: M) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..2 {
        for k in 0..2 {
            let d = a[i][k] - b[i][k];
            worst = worst.max(d.re.hypot(d.im));
        }
    }
    worst
}

fn main() {
    let j = JacobiRotation {
        c: c64::new(0.6, 0.0),
        s: c64::new(0.0, 0.8),
    };
    let cases = [
        ("M = I", [[c64::new(1.0, 0.0), c64::new(0.0, 0.0)], [c64::new(0.0, 0.0), c64::new(1.0, 0.0)]]),
        (
            "M = [[1, 2i], [3, 4]]",
            [[c64::new(1.0, 0.0), c64::new(0.0, 2.0)], [c64::new(3.0, 0.0), c64::new(4.0, 0.0)]],
        ),
    ];
    let mut worst = 0.0f64;
    for (name, m) in cases {
        let e = diff(right(&j, m), expected(&j, m));
        worst = worst.max(e);
        println!("{name}: max |M J - apply_on_the_right_2x2| = {e:e}");
    }
    println!("expected: rounding level, at most 1e-14");
    if worst > 1e-14 {
        println!("WRONG: the right application uses conj(J), not J");
        std::process::exit(1);
    }
}
