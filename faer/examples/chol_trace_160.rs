//! Probe for tensor4all/faer#160 cholesky (not for merging). For each width cap in `--caps`
//! (`0`: none), times `--reps` calls of `cholesky_in_place` at `--threads T` after one untimed call
//! and splits each into the top-level stages (diagonal block, triangular solve, trailing update);
//! then records the gate decisions and the team's regions of one more call (untimed).
//! Output: `n,threads,cap,rep,total_ms,diag_ms,trsm_ms,update_ms`, then `# ...` trace lines.
#![allow(non_snake_case)]
use dyn_stack::{MemBuffer, MemStack};
use faer::dyn_stack;
use faer::linalg::cholesky::llt::factor as llt;
use faer::prelude::*;
use faer::utils::cost::trace;
use std::collections::BTreeMap;

fn main() {
	let args: Vec<String> = std::env::args().collect();
	let get = |k: &str, d: usize| args.iter().position(|a| a == k).map(|i| args[i + 1].parse().unwrap()).unwrap_or(d);
	let threads = get("--threads", 64);
	let reps = get("--reps", 5);
	let caps: Vec<usize> = args.iter().position(|a| a == "--caps").map(|i| args[i + 1].split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![0]);
	let n = get("--n", 1024);
	let b = Mat::<f64>::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.5);
	let a0 = &b * b.transpose() + Mat::<f64>::identity(n, n) * (n as f64);
	let mut a = a0.clone();
	let mut mem = MemBuffer::new(llt::cholesky_in_place_scratch::<f64>(n, faer::ParShape::team(threads), Default::default()));
	use std::sync::atomic::Ordering::Relaxed;
	println!("n,threads,cap,rep,total_ms,diag_ms,trsm_ms,update_ms");
	faer::utils::team::with_team(threads, |mut par| {
		for &cap in &caps {
			trace::WIDTH_CAP.store(if cap == 0 { usize::MAX } else { cap }, Relaxed);
			a.copy_from(&a0);
			llt::cholesky_in_place(a.as_mut(), Default::default(), par.reborrow(), MemStack::new(&mut mem), Default::default()).unwrap();
			for rep in 0..reps {
				a.copy_from(&a0);
				for c in &trace::CHOL_NS {
					c.store(0, Relaxed);
				}
				let t0 = std::time::Instant::now();
				llt::cholesky_in_place(a.as_mut(), Default::default(), par.reborrow(), MemStack::new(&mut mem), Default::default()).unwrap();
				let total = t0.elapsed().as_secs_f64() * 1e3;
				let ms = |i: usize| trace::CHOL_NS[i].load(Relaxed) as f64 * 1e-6;
				println!("{n},{threads},{cap},{rep},{total:.3},{:.3},{:.3},{:.3}", ms(0), ms(1), ms(2));
			}
		}
		trace::WIDTH_CAP.store(usize::MAX, Relaxed);
		a.copy_from(&a0);
		faer::utils::team::stats::enable(true);
		trace::enable(true);
		llt::cholesky_in_place(a.as_mut(), Default::default(), par.reborrow(), MemStack::new(&mut mem), Default::default()).unwrap();
		trace::enable(false);
		faer::utils::team::stats::enable(false);
	});
	let mut agg: BTreeMap<(String, usize, bool), (usize, f64)> = BTreeMap::new();
	for r in trace::take() {
		let e = agg.entry((r.site.to_string(), r.width, r.parallel)).or_default();
		e.0 += 1;
		e.1 += r.estimate_ns;
	}
	for ((site, width, par), (count, est)) in agg {
		println!("# site={site} width={width} parallel={par} count={count} est_ms={:.3}", est * 1e-6);
	}
	let s = faer::utils::team::stats::take();
	let nz = |v: &Vec<usize>| v.iter().enumerate().filter(|(_, c)| **c > 0).map(|(i, c)| format!("{i}:{c}")).collect::<Vec<_>>().join(" ");
	println!("# run {}\n# split {}\n# width {}\n# gemm below_floor {} one_tile {} tiled {} team_kernel {}", nz(&s.run), nz(&s.split), nz(&s.width), s.gemm_below_floor, s.gemm_one_tile, s.gemm_tiled, s.gemm_team_kernel);
}
