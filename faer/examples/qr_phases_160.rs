//! Probe for tensor4all/faer#160 (not for merging): times `qr_in_place` on a square f64 matrix and
//! splits it into the stages of the top level of the blocked QR, under each block-apply column-split
//! mode, and optionally dumps the gate decisions (`par-trace`).
//!
//! Usage (one thread count per process, so `taskset` can give each its own CPU set):
//!
//! ```text
//! qr_phases_160 --threads T [--n 4096] [--modes 0,3,4,2] [--blocks 0] [--reps 3] [--trace]
//! ```
//!
//! `--blocks 0` is the library's `recommended_block_size`; other values fix the block width (the
//! number of columns of every top-level block apply). Output: one CSV line per repetition,
//! `n,threads,mode,block,rep,total_ms,panel_ms,leaf_ms,factor_ms,apply_ms,panel_apply_ms`, then
//! with `--trace` one `# trace` line per (mode, block, gate site, width).
#![allow(non_snake_case)]
use dyn_stack::{MemBuffer, MemStack};
use faer::dyn_stack;
use faer::linalg::qr::no_pivoting::factor as qrf;
use faer::prelude::*;
use faer::utils::cost::trace;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
	args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}
fn list(s: &str) -> Vec<usize> {
	s.split(',').map(|x| x.trim().parse().unwrap()).collect()
}

fn main() {
	let args: Vec<String> = std::env::args().collect();
	let threads: usize = arg(&args, "--threads").map(|s| s.parse().unwrap()).unwrap_or(1);
	let n: usize = arg(&args, "--n").map(|s| s.parse().unwrap()).unwrap_or(4096);
	let modes = arg(&args, "--modes").map(|s| list(&s)).unwrap_or(vec![0]);
	let blocks = arg(&args, "--blocks").map(|s| list(&s)).unwrap_or(vec![0]);
	let reps: usize = arg(&args, "--reps").map(|s| s.parse().unwrap()).unwrap_or(3);
	let want_trace = args.iter().any(|a| a == "--trace");
	if args.iter().any(|a| a == "--lookahead") {
		trace::LOOKAHEAD.store(true, Ordering::Relaxed);
	}
	if let Some(w) = arg(&args, "--lookahead-width") {
		trace::LOOKAHEAD_PANEL_WIDTH.store(w.parse().unwrap(), Ordering::Relaxed);
	}
	if args.iter().any(|a| a == "--panel-seq") {
		trace::PANEL_SEQ.store(true, Ordering::Relaxed);
	}
	if let Some(c) = arg(&args, "--row-split-cols") {
		trace::ROW_SPLIT_COLS.store(c.parse().unwrap(), Ordering::Relaxed);
	}

	// seeded, well conditioned: a fixed LCG in [-0.5, 0.5)
	let mut state = 0x9e3779b97f4a7c15u64 ^ n as u64;
	let a0 = Mat::<f64>::from_fn(n, n, |_, _| {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
	});
	let mut a = a0.clone();

	println!("n,threads,mode,block,rep,total_ms,panel_ms,leaf_ms,factor_ms,apply_ms,panel_apply_ms,pa_w1,pa_w2,pa_w4,pa_w8,pa_w16,pa_w32,pa_w64,pa_w128,panel_factor_ms");
	faer::utils::team::with_team(threads, |mut par| {
		for &block in &blocks {
			let bs = if block == 0 { qrf::recommended_block_size::<f64>(n, n) } else { block.min(n) };
			let mut qc = Mat::<f64>::zeros(bs, n);
			let mut mem = MemBuffer::new(qrf::qr_in_place_scratch::<f64>(n, n, bs, par.shape(), Default::default()));
			let mut reference: Option<Mat<f64>> = None;
			for &mode in &modes {
				trace::N_TASKS_MODE.store(mode, Ordering::Relaxed);
				// one untimed call: page faults, team warm-up
				a.copy_from(&a0);
				qrf::qr_in_place(a.as_mut(), qc.as_mut(), par.reborrow(), MemStack::new(&mut mem), Default::default());
				// every mode computes the same factorization: R against the first mode's, to rounding
				match &reference {
					None => reference = Some(a.clone()),
					Some(r) => {
						let mut worst = 0.0f64;
						for j in 0..n {
							for i in 0..=j {
								worst = worst.max((a[(i, j)] - r[(i, j)]).abs() / r[(j, j)].abs().max(1e-300));
							}
						}
						println!("# check,n={n},threads={threads},mode={mode},block={bs},max_rel_r_diff={worst:.3e}");
						assert!(worst < 1e-8, "mode {mode}: R differs from the first mode's by {worst}");
					},
				}
				for rep in 0..reps {
					a.copy_from(&a0);
					trace::phase_reset();
					let t0 = Instant::now();
					qrf::qr_in_place(a.as_mut(), qc.as_mut(), par.reborrow(), MemStack::new(&mut mem), Default::default());
					let total = t0.elapsed().as_secs_f64() * 1e3;
					let ms = |i: usize| trace::PHASE_NS[i].load(Ordering::Relaxed) as f64 * 1e-6;
					let by: Vec<String> = trace::PANEL_BY_WIDTH_NS.iter().map(|c| format!("{:.3}", c.load(Ordering::Relaxed) as f64 * 1e-6)).collect();
					println!(
						"{n},{threads},{mode},{bs},{rep},{total:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
						ms(trace::PHASE_PANEL),
						ms(trace::PHASE_LEAF),
						ms(trace::PHASE_FACTOR),
						ms(trace::PHASE_APPLY),
						ms(trace::PHASE_PANEL_APPLY),
						by.join(","),
					);
				}
				if want_trace {
					// one more call with the decisions recorded (not timed: the log takes a lock)
					a.copy_from(&a0);
					trace::enable(true);
					qrf::qr_in_place(a.as_mut(), qc.as_mut(), par.reborrow(), MemStack::new(&mut mem), Default::default());
					trace::enable(false);
					// (site, width, parallel) -> (count, sum of dims[1] (basis columns), sum estimate ns)
					let mut agg: BTreeMap<(String, usize, bool), (usize, usize, f64)> = BTreeMap::new();
					for r in trace::take() {
						let e = agg.entry((r.site.to_string(), r.width, r.parallel)).or_default();
						e.0 += 1;
						e.1 += r.dims[1];
						e.2 += r.estimate_ns;
					}
					for ((site, width, parallel), (count, cols, est)) in agg {
						println!(
							"# trace,n={n},threads={threads},mode={mode},block={bs},site={site},width={width},parallel={parallel},count={count},mean_dim1={:.1},est_ms={:.3}",
							cols as f64 / count as f64,
							est * 1e-6,
						);
					}
				}
			}
		}
	});
	eprintln!("# lookahead_steps={}", trace::LOOKAHEAD_STEPS.load(Ordering::Relaxed));
	// keep the result observable
	assert!(a[(0, 0)].is_finite());
}
