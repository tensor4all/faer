//! Acceptance driver for tensor4all/faer#160 / #90 (not for merging): times the decompositions
//! that use the block Householder apply on a team of `--threads` participants, through their
//! public entry points, on a seeded f64 matrix restored outside the clock. It uses no probe
//! feature, so the same file builds against `origin/main` and against the fix branch.
//!
//! ```text
//! factor_scaling_160 --threads T --op qr|q|hess|tridiag|bidiag|apply --n N [--reps R]
//!                    [--k K --cols C]      (apply: basis N x K, target N x C)
//! ```
//! Output: `op,n,k,cols,threads,rep,ms`, one line per repetition after one untimed call.
#![allow(non_snake_case)]
use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::dyn_stack;
use faer::linalg::{evd, householder, qr::no_pivoting::factor as qrf, svd};
use faer::prelude::*;
use faer::Conj;
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
	args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
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
	let threads: usize = arg(&args, "--threads").map(|s| s.parse().unwrap()).unwrap_or(1);
	let n: usize = arg(&args, "--n").map(|s| s.parse().unwrap()).unwrap_or(1024);
	let reps: usize = arg(&args, "--reps").map(|s| s.parse().unwrap()).unwrap_or(3);
	let op = arg(&args, "--op").unwrap_or("qr".into());
	let k: usize = arg(&args, "--k").map(|s| s.parse().unwrap()).unwrap_or(0);
	let cols: usize = arg(&args, "--cols").map(|s| s.parse().unwrap()).unwrap_or(0);
	let shape = faer::ParShape::team(threads);
	// probe builds: the block apply's schedule (0 the rule, 5 owner-packed; faer#160)
	#[cfg(feature = "par-trace")]
	if let Ok(mode) = std::env::var("FAER_APPLY_MODE") {
		faer::utils::cost::trace::N_TASKS_MODE.store(mode.parse().unwrap(), std::sync::atomic::Ordering::Relaxed);
	}
	let bs = qrf::recommended_block_size::<f64>(n, n);

	let mut a0 = seeded(n, n, n as u64);
	if op == "tridiag" {
		let s = &a0 + a0.transpose();
		a0 = s;
	}
	let mut a = a0.clone();
	// the compact factors of a QR of `a0`, for `q` (forming Q) and `apply` (a basis and its factor)
	let (basis, factor) = {
		let (m, kk) = if op == "apply" { (n, k) } else { (n, n) };
		let bsz = if op == "apply" { k } else { bs };
		let mut v = seeded(m, kk, 7);
		let mut h = Mat::<f64>::zeros(bsz, kk);
		let mut mem = MemBuffer::new(qrf::qr_in_place_scratch::<f64>(m, kk, bsz, faer::ParShape::Seq, Default::default()));
		if op == "q" || op == "apply" {
			qrf::qr_in_place(v.as_mut(), h.as_mut(), Par::Seq, MemStack::new(&mut mem), Default::default());
		}
		(v, h)
	};

	let mut h1 = Mat::<f64>::zeros(bs, n);
	let mut h2 = Mat::<f64>::zeros(bs, n.saturating_sub(1));
	let mut target = seeded(n, cols.max(1), 11);
	let target0 = target.clone();
	let mut q = Mat::<f64>::zeros(n, n);
	let req = match op.as_str() {
		"qr" => qrf::qr_in_place_scratch::<f64>(n, n, bs, shape, Default::default()),
		"q" => householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(n, bs, n),
		"hess" => evd::hessenberg::hessenberg_in_place_scratch::<f64>(n, bs, shape, Default::default()),
		"tridiag" => evd::tridiag::tridiag_in_place_scratch::<f64>(n, shape, Default::default()),
		"bidiag" => svd::bidiag::bidiag_in_place_scratch::<f64>(n, n, shape, Default::default()),
		"apply" => householder::apply_block_householder_transpose_on_the_left_in_place_scratch::<f64>(n, k, cols),
		_ => panic!("unknown op {op}"),
	};
	let mut mem = MemBuffer::new(StackReq::all_of(&[req]));

	println!("op,n,k,cols,threads,rep,ms");
	faer::utils::team::with_team(threads, |mut par| {
		let call = |a: &mut Mat<f64>, q: &mut Mat<f64>, target: &mut Mat<f64>, h1: &mut Mat<f64>, h2: &mut Mat<f64>, mem: &mut MemBuffer, par: Par<'_>| {
			let stack = MemStack::new(mem);
			match op.as_str() {
				"qr" => {
					qrf::qr_in_place(a.as_mut(), h1.as_mut(), par, stack, Default::default());
				},
				"q" => {
					householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
						basis.as_ref(),
						factor.as_ref(),
						Conj::No,
						q.as_mut(),
						par,
						stack,
					);
				},
				"hess" => evd::hessenberg::hessenberg_in_place(a.as_mut(), h2.as_mut(), par, stack, Default::default()),
				"tridiag" => evd::tridiag::tridiag_in_place(a.as_mut(), h2.as_mut(), par, stack, Default::default()),
				"bidiag" => svd::bidiag::bidiag_in_place(a.as_mut(), h1.as_mut(), h2.as_mut(), par, stack, Default::default()),
				"apply" => householder::apply_block_householder_transpose_on_the_left_in_place_with_conj(
					basis.as_ref(),
					factor.as_ref(),
					Conj::Yes,
					target.as_mut(),
					par,
					stack,
				),
				_ => unreachable!(),
			}
		};
		let restore = |a: &mut Mat<f64>, q: &mut Mat<f64>, target: &mut Mat<f64>| {
			a.copy_from(&a0);
			q.fill(0.0);
			q.diagonal_mut().fill(1.0);
			target.copy_from(&target0);
		};
		for rep in -1..reps as isize {
			restore(&mut a, &mut q, &mut target);
			let t0 = Instant::now();
			call(&mut a, &mut q, &mut target, &mut h1, &mut h2, &mut mem, par.reborrow());
			let ms = t0.elapsed().as_secs_f64() * 1e3;
			if rep >= 0 {
				println!("{op},{n},{k},{cols},{threads},{rep},{ms:.3}");
			}
		}
	});
	assert!(a[(0, 0)].is_finite() && q[(0, 0)].is_finite() && target[(0, 0)].is_finite());
}
