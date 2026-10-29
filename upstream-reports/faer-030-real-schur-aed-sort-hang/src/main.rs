// `Mat::eigen` of an f32 near-Jordan matrix, J_n + 1e-6 G (Jordan block with eigenvalue 1,
// G standard normal from StdRng seed 112), divided by its largest entry and scaled by 2^-100.
// All entries are finite, normal f32 values.
// The call should return in well under a second; the program gives each size 60 s.
use faer::Mat;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::sync::mpsc;
use std::time::{Duration, Instant};

fn input(n: usize) -> Mat<f32> {
    let rng = &mut StdRng::seed_from_u64(112);
    let a = Mat::<f32>::from_fn(n, n, |i, j| {
        let g: f32 = rng.sample(StandardNormal);
        let jordan = if i == j || i + 1 == j { 1.0 } else { 0.0 };
        jordan + 1e-6 * g
    });
    let max = a.norm_max();
    Mat::from_fn(n, n, |i, j| a[(i, j)] / max * 2f32.powi(-100))
}

fn main() {
    let mut bad = false;
    for n in [128, 257] {
        let a = input(n);
        let (send, recv) = mpsc::channel();
        let start = Instant::now();
        std::thread::spawn(move || {
            let ok = a.eigen().is_ok();
            let _ = send.send(ok);
        });
        match recv.recv_timeout(Duration::from_secs(60)) {
            Ok(ok) => println!("n = {n}: returned after {:.2} s (Ok: {ok})", start.elapsed().as_secs_f64()),
            Err(_) => {
                println!("n = {n}: no return after 60 s");
                bad = true;
            }
        }
    }
    println!("expected: both calls return");
    if bad {
        println!("WRONG: Mat::eigen does not return");
        std::process::exit(1);
    }
}
