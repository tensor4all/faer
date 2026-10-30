//! Probe for tensor4all/faer#160, Cholesky item (2) (not for merging). Times the trailing update
//! of the blocked Cholesky, `A11 -= A10 A10^T` for every step of block `--block` (the steps of
//! `cholesky_in_place` at `n`), as a product into the lower triangle under the kernel's own split
//! (`fork`) and under equal-area column ranges (`area`), and as the rectangular product of the
//! same shape (`rect`, twice the arithmetic); then the whole `cholesky_in_place` under each split.
//! The arms alternate within every repetition, after one untimed round.
//!
//! ```text
//! syrk_probe_160 --threads T --n N [--reps R] [--block B]
//! ```
//! Output: `arm,n,threads,rep,ms`.
#![allow(non_snake_case)]
use dyn_stack::{MemBuffer, MemStack};
use faer::dyn_stack;
use faer::linalg::cholesky::llt::factor as llt;
use faer::linalg::matmul::{matmul, triangular};
use faer::prelude::*;
use faer::utils::cost::trace;
use faer::Accum;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

fn arg(args: &[String], name: &str, default: usize) -> usize {
	args.iter().position(|a| a == name).map(|i| args[i + 1].parse().unwrap()).unwrap_or(default)
}

fn seeded(m: usize, n: usize, seed: u64) -> Mat<f64> {
	let mut s = 0x9e3779b97f4a7c15u64 ^ seed;
	Mat::from_fn(m, n, |_, _| {
		s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
	})
}

fn main() {
	let args: Vec<String> = std::env::args().collect();
	let threads = arg(&args, "--threads", 1);
	let n = arg(&args, "--n", 2048);
	let reps = arg(&args, "--reps", 5);
	let block = arg(&args, "--block", 128);
	let panel = seeded(n, block, 1);
	let mut c = seeded(n, n, 2);
	let g = seeded(n, n, 3);
	let spd = &g * g.transpose() + Mat::<f64>::identity(n, n) * (n as f64);
	let mut a = spd.clone();
	let mut mem = MemBuffer::new(llt::cholesky_in_place_scratch::<f64>(n, faer::ParShape::team(threads), Default::default()));
	println!("arm,n,threads,rep,ms");
	faer::utils::team::with_team(threads, |mut par| {
		let mut update = |mut par: Par, structure: triangular::BlockStructure| {
			let mut j = 0;
			while j + block < n {
				let m = n - j - block;
				let A10 = panel.as_ref().subrows(j + block, m);
				let A11 = c.as_mut().submatrix_mut(j + block, j + block, m, m);
				if structure == triangular::BlockStructure::Rectangular {
					matmul(A11, Accum::Add, A10, A10.transpose(), -1e-3, par.reborrow());
				} else {
					triangular::matmul(
						A11,
						structure,
						Accum::Add,
						A10,
						triangular::BlockStructure::Rectangular,
						A10.transpose(),
						triangular::BlockStructure::Rectangular,
						-1e-3,
						par.reborrow(),
					);
				}
				j += block;
			}
		};
		for rep in 0..=reps {
			for arm in ["fork", "area", "rect", "chol-fork", "chol-area"] {
				trace::LOWER_SPLIT.store(if arm.ends_with("fork") { 1 } else { 0 }, Relaxed);
				if arm.starts_with("chol") {
					a.copy_from(&spd);
				}
				let t0 = Instant::now();
				match arm {
					"fork" | "area" => update(par.reborrow(), triangular::BlockStructure::TriangularLower),
					"rect" => update(par.reborrow(), triangular::BlockStructure::Rectangular),
					_ => {
						llt::cholesky_in_place(a.as_mut(), Default::default(), par.reborrow(), MemStack::new(&mut mem), Default::default())
							.unwrap();
					},
				}
				let ms = t0.elapsed().as_secs_f64() * 1e3;
				if rep > 0 {
					println!("{arm},{n},{threads},{},{ms:.3}", rep - 1);
				}
			}
		}
	});
}
