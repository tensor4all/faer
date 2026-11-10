//! tensor4all/faer#314 step 1: row-major versus column-major for every dense
//! factorization family, one table. This is a measurement harness, not part of
//! the crate: the PR is docs-only and the worklog carries this source.
//!
//! Build and run (from the repository root; the file is not part of the crate and no
//! gate compiles it -- see the worklog for the exact commands):
//!   cp docs/worklogs/evidence/2026-11-11-row-major-parity/rm314.rs faer/examples/rm314.rs
//!   cargo build --release --example rm314 --features perf-warn
//!   taskset -c 0-7 target/release/examples/rm314 time  <time.csv> 1     # Par::Seq
//!   taskset -c 0-7 target/release/examples/rm314 time  <time.csv> 8     # with_team(8)
//!   taskset -c 0-7 target/release/examples/rm314 alloc <alloc.csv> 1
//!   taskset -c 0-7 target/release/examples/rm314 alloc <alloc.csv> 8
//!   rm faer/examples/rm314.rs
//!
//! The GEVD 2048 pairs in the evidence were measured by reducing `FAMILIES` to `["gevd"]` in
//! this file (a one-line edit) so that the other 138 cells were not re-run; a pass with the file
//! as shipped covers all 148 cells per thread count and takes about 20 minutes at one thread.
//!
//! Every matrix of a cell -- the input, the coefficients, the eigenvector and
//! singular-vector outputs -- is allocated in the cell's layout, so a
//! "row-major" cell is the call a row-major caller actually makes.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::lblt::factor as lblt;
use faer::linalg::cholesky::llt::factor as llt;
use faer::linalg::evd::{self, ComputeEigenvectors};
use faer::linalg::gevd;
use faer::linalg::lq::no_pivoting::factor as lq;
use faer::linalg::lu::full_pivoting::factor as flu;
use faer::linalg::lu::partial_pivoting::factor as plu;
use faer::linalg::qr::col_pivoting::factor as qrcp;
use faer::linalg::qr::no_pivoting::factor as qr;
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::diag::Diag;
use faer::prelude::{Reborrow, ReborrowMut, default};
use faer::{MatMut, Par};
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------- allocator
//
// While `COUNTING` is set, every allocation on any thread adds its size to
// `BYTES` / `COUNT` and updates `LARGEST`. The flag is process-wide because a
// `Par::Team` cell allocates on the team's members, not on this thread.

static COUNTING: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

fn record(size: usize) {
	if COUNTING.load(Ordering::Relaxed) {
		BYTES.fetch_add(size, Ordering::Relaxed);
		COUNT.fetch_add(1, Ordering::Relaxed);
		LARGEST.fetch_max(size, Ordering::Relaxed);
	}
}
struct Counting;
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
		record(layout.size());
		unsafe { System.alloc(layout) }
	}
	unsafe fn alloc_zeroed(&self, layout: AllocLayout) -> *mut u8 {
		record(layout.size());
		unsafe { System.alloc_zeroed(layout) }
	}
	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: AllocLayout,
		new_size: usize,
	) -> *mut u8 {
		record(new_size);
		unsafe { System.realloc(ptr, layout, new_size) }
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

// ------------------------------------------------------------------ matrices

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Layout {
	Col,
	Row,
}
impl Layout {
	fn name(self) -> &'static str {
		match self {
			Layout::Col => "col",
			Layout::Row => "row",
		}
	}
	/// `(row_stride, col_stride)` of a contiguous `m x n` matrix.
	fn strides(self, m: usize, n: usize) -> (isize, isize) {
		match self {
			Layout::Col => (1, m as isize),
			Layout::Row => (n as isize, 1),
		}
	}
}

/// A contiguous `m x n` f64 buffer, in the cell's layout.
struct Buf {
	data: Vec<f64>,
	m: usize,
	n: usize,
	layout: Layout,
}
impl Buf {
	fn new(m: usize, n: usize, layout: Layout) -> Self {
		Buf { data: vec![0.0; m * n], m, n, layout }
	}
	/// # Safety
	/// `self.data` covers the strides and the view does not outlive the call.
	unsafe fn mat_mut(&mut self) -> MatMut<'_, f64> {
		let (rs, cs) = self.layout.strides(self.m, self.n);
		unsafe {
			MatMut::from_raw_parts_mut(
				self.data.as_mut_ptr(),
				self.m,
				self.n,
				rs,
				cs,
			)
		}
	}
	fn fill(&mut self, f: fn(usize, usize) -> f64) {
		let (m, n) = (self.m, self.n);
		let (rs, cs) = self.layout.strides(m, n);
		for j in 0..n {
			for i in 0..m {
				self.data[i * rs as usize + j * cs as usize] = f(i, j);
			}
		}
	}
}

/// Deterministic pseudo-random value in `[-0.5, 0.5]`, keyed by coordinates so
/// both layouts receive the same matrix.
fn noise(i: usize, j: usize) -> f64 {
	let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
		^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
	x ^= x >> 33;
	x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
	x ^= x >> 29;
	(x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

fn general(i: usize, j: usize) -> f64 {
	noise(i, j)
}
/// Symmetric positive definite: diagonally dominant, `A(i, j) = K(min, max)`.
fn sym_pd(i: usize, j: usize) -> f64 {
	let k = 1.0 / (1.0 + (i as f64 - j as f64).abs());
	k + if i == j { 2000.0 } else { 0.0 }
}
/// The same kernel under a diagonal sign flip. A diagonal `D` gives `D K D`, a congruence, so
/// this is still positive definite: Bunch-Kaufman takes no 2x2 pivots on it and the worklog says
/// so -- the LBL^T cells measure the 1x1 path only.
fn sym_signed(i: usize, j: usize) -> f64 {
	let s = |i: usize| if i % 2 == 0 { 1.0 } else { -1.0 };
	s(i) * s(j) * (1.0 / (1.0 + (i as f64 - j as f64).abs()))
}
/// Symmetric positive definite, second kernel, for the GEVD's `B`.
fn sym_pd_b(i: usize, j: usize) -> f64 {
	let k = 1.0 / (1.0 + 0.5 * (i as f64 - j as f64).abs());
	3.0 * k + if i == j { 2000.0 } else { 0.0 }
}

// --------------------------------------------------------------------- cells

const WARMUP: usize = 1;
const REPS: usize = 5;

const FAMILIES: [&str; 12] = [
	"lu_partial",
	"lu_full",
	"qr",
	"qr_colpiv",
	"lq",
	"llt",
	"lblt",
	"evd_vals",
	"evd_vecs",
	"svd_vals",
	"svd_vecs",
	"gevd",
];

fn shapes(family: &str) -> Vec<(usize, usize)> {
	// a smoke run: `RM314_MAX=<n>` keeps the square cells up to n
	if let Ok(v) = std::env::var("RM314_MAX") {
		let max: usize = v.parse().expect("RM314_MAX is a number");
		return vec![32usize, 64, 128, 512, 2048]
			.into_iter()
			.filter(|&n| n <= max)
			.map(|n| (n, n))
			.collect();
	}
	let squares = [32usize, 64, 128, 512, 2048];
	let tall = [(4096usize, 256usize), (1024, 64)];
	match family {
		// square only: the decomposition is defined on an `n x n` matrix
		"llt" | "lblt" | "evd_vals" | "evd_vecs" => {
			squares.iter().map(|&n| (n, n)).collect()
		}
		// the most expensive family: 25 s per 2048 call column-major, 110-124 s row-major
		"gevd" => squares.iter().map(|&n| (n, n)).collect(),
		_ => {
			let mut s: Vec<(usize, usize)> =
				squares.iter().map(|&n| (n, n)).collect();
			s.extend_from_slice(&tall);
			s
		}
	}
}

fn shape_name((m, n): (usize, usize)) -> String {
	if m == n { format!("{m}") } else { format!("{m}x{n}") }
}

struct Row {
	family: &'static str,
	shape: String,
	layout: &'static str,
	threads: usize,
	samples: Vec<f64>,
	alloc: Option<(usize, usize, usize)>,
	matrix_bytes: usize,
}

/// Measures one cell: one allocation-counted call, then `REPS` timed calls
/// after a discarded warm-up, each on a freshly filled input.
fn cell(
	family: &'static str,
	(m, n): (usize, usize),
	layout: Layout,
	threads: usize,
	fill: fn(usize, usize) -> f64,
	reps: usize,
	measure_alloc: bool,
	mut prep: impl FnMut(),
	mut call: impl FnMut(&mut Buf),
	out: &mut Vec<Row>,
) {
	let mut buf = Buf::new(m, n, layout);
	buf.fill(fill);
	let alloc = if measure_alloc {
		// The first GEMM a thread runs allocates its thread-local workspace in
		// `private_gemm_x86` (128 MiB, sizes/host in the worklog); count the
		// second call so a one-time lazy initialisation is not read as a copy.
		prep();
		call(&mut buf);
		prep();
		buf.fill(fill);
		BYTES.store(0, Ordering::Relaxed);
		COUNT.store(0, Ordering::Relaxed);
		LARGEST.store(0, Ordering::Relaxed);
		COUNTING.store(true, Ordering::SeqCst);
		call(&mut buf);
		COUNTING.store(false, Ordering::SeqCst);
		Some((
			COUNT.load(Ordering::Relaxed),
			BYTES.load(Ordering::Relaxed),
			LARGEST.load(Ordering::Relaxed),
		))
	} else {
		None
	};
	let mut samples = Vec::with_capacity(reps);
	for r in 0..(if reps == 0 { 0 } else { WARMUP + reps }) {
		prep();
		buf.fill(fill);
		let t = Instant::now();
		call(&mut buf);
		let dt = t.elapsed().as_secs_f64() * 1e6;
		if r >= WARMUP {
			samples.push(dt);
		}
	}
	if reps > 0 {
		eprintln!(
			"[{}] {} {} {}T: {} us",
		family,
		shape_name((m, n)),
		layout.name(),
			threads,
			samples
				.iter()
				.map(|x| format!("{x:.1}"))
				.collect::<Vec<_>>()
				.join("/")
		);
	}
	out.push(Row {
		family,
		shape: shape_name((m, n)),
		layout: layout.name(),
		threads,
		samples,
		alloc,
		matrix_bytes: m * n * 8,
	});
}

fn main() {
	let args: Vec<String> = std::env::args().collect();
	let mode = args.get(1).cloned().unwrap_or_else(|| "time".into());
	let out_path = args
		.get(2)
		.cloned()
		.unwrap_or_else(|| format!("314-{mode}.csv"));
	let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
	let measure_alloc = mode == "alloc";
	let reps = if measure_alloc { 0 } else { REPS };

	let mut rows: Vec<Row> = Vec::new();
	let run = |rows: &mut Vec<Row>, par: Par<'_>| {
		let mut par = par;
		for family in FAMILIES {
			let shapes = shapes(family);
			for (m, n) in shapes {
				let k = m.min(n);
				for layout in [Layout::Col, Layout::Row] {
					match family {
						"lu_partial" => {
							let mut perm = vec![0usize; m];
							let mut pinv = vec![0usize; m];
							let shape = par.shape();
							let mut mem = MemBuffer::new(
								plu::lu_in_place_scratch::<usize, f64>(
									m, n, shape, default(),
								),
							);
							cell(
								family, (m, n), layout, threads, general,
								reps,
								measure_alloc,
								|| {},
								|buf| {
									let mut a = unsafe { buf.mat_mut() };
									plu::lu_in_place(
										a.rb_mut(),
										&mut perm,
										&mut pinv,
										par.reborrow(),
										MemStack::new(&mut mem),
										default(),
									);
								},
								rows,
							);
						}
						"lu_full" => {
							let mut p = vec![0usize; m];
							let mut pi = vec![0usize; m];
							let mut q = vec![0usize; n];
							let mut qi = vec![0usize; n];
							let shape = par.shape();
							let mut mem = MemBuffer::new(
								flu::lu_in_place_scratch::<usize, f64>(
									m, n, shape, default(),
								),
							);
							cell(
								family, (m, n), layout, threads, general,
								reps,
								measure_alloc,
								|| {},
								|buf| {
									let mut a = unsafe { buf.mat_mut() };
									flu::lu_in_place(
										a.rb_mut(),
										&mut p,
										&mut pi,
										&mut q,
										&mut qi,
										par.reborrow(),
										MemStack::new(&mut mem),
										default(),
									);
								},
								rows,
							);
						}
						"qr" => {
							let bs = qr::recommended_block_size::<f64>(m, n);
							let mut coeff = Buf::new(bs, k, layout);
							let shape = par.shape();
							let mut mem = MemBuffer::new(
								qr::qr_in_place_scratch::<f64>(
									m, n, bs, shape, default(),
								),
							);
							cell(
								family, (m, n), layout, threads, general,
								reps,
								measure_alloc,
								|| {},
								|buf| {
									let mut a = unsafe { buf.mat_mut() };
									let mut c = unsafe { coeff.mat_mut() };
									qr::qr_in_place(
										a.rb_mut(),
										c.rb_mut(),
										par.reborrow(),
										MemStack::new(&mut mem),
										default(),
									);
								},
								rows,
							);
						}
						"qr_colpiv" => {
							let bs = qr::recommended_block_size::<f64>(m, n);
							let mut coeff = Buf::new(bs, k, layout);
							let mut perm = vec![0usize; n];
							let mut pinv = vec![0usize; n];
							let shape = par.shape();
							let mut mem = MemBuffer::new(
								qrcp::qr_in_place_scratch::<usize, f64>(
									m, n, bs, shape, default(),
								),
							);
							cell(
								family, (m, n), layout, threads, general,
								reps,
								measure_alloc,
								|| {},
								|buf| {
									let mut a = unsafe { buf.mat_mut() };
									let mut c = unsafe { coeff.mat_mut() };
									qrcp::qr_in_place(
										a.rb_mut(),
										c.rb_mut(),
										&mut perm,
										&mut pinv,
										par.reborrow(),
										MemStack::new(&mut mem),
										default(),
									);
								},
								rows,
							);
						}
						"lq" => {
							let bs = qr::recommended_block_size::<f64>(m, n);
							let mut coeff = Buf::new(bs, k, layout);
							let shape = par.shape();
							let mut mem = MemBuffer::new(
								lq::lq_in_place_scratch::<f64>(
									m, n, bs, shape, default(),
								),
							);
							cell(
								family, (m, n), layout, threads, general,
								reps,
								measure_alloc,
								|| {},
								|buf| {
									let mut a = unsafe { buf.mat_mut() };
									let mut c = unsafe { coeff.mat_mut() };
									lq::lq_in_place(
										a.rb_mut(),
										c.rb_mut(),
										par.reborrow(),
										MemStack::new(&mut mem),
										default(),
									);
								},
								rows,
							);
						}
						"llt" => {
							let shape = par.shape();
							let mut mem = MemBuffer::new(
								llt::cholesky_in_place_scratch::<f64>(
									n, shape, default(),
								),
							);
							cell(
								family, (m, n), layout, threads, sym_pd,
								reps,
								measure_alloc,
								|| {},
								|buf| {
									let mut a = unsafe { buf.mat_mut() };
									llt::cholesky_in_place(
										a.rb_mut(),
										llt::LltRegularization {
											dynamic_regularization_delta: 0.0,
											dynamic_regularization_epsilon: 0.0,
										},
										par.reborrow(),
										MemStack::new(&mut mem),
										default(),
									)
									.unwrap();
								},
								rows,
							);
						}
						"lblt" => {
							let mut subdiag = Diag::<f64>::zeros(n);
							let mut perm = vec![0usize; n];
							let mut pinv = vec![0usize; n];
							let shape = par.shape();
							let mut mem = MemBuffer::new(
								lblt::cholesky_in_place_scratch::<usize, f64>(
									n, shape, default(),
								),
							);
							cell(
								family, (m, n), layout, threads, sym_signed,
								reps,
								measure_alloc,
								|| {},
								|buf| {
									let mut a = unsafe { buf.mat_mut() };
									lblt::cholesky_in_place(
										a.rb_mut(),
										subdiag.as_mut(),
										&mut perm,
										&mut pinv,
										par.reborrow(),
										MemStack::new(&mut mem),
										default(),
									);
								},
								rows,
							);
						}
						"evd_vals" | "evd_vecs" => {
							let vectors = family == "evd_vecs";
							let ce = if vectors {
								ComputeEigenvectors::Yes
							} else {
								ComputeEigenvectors::No
							};
							let mut s = Diag::<f64>::zeros(n);
							let mut u = Buf::new(n, n, layout);
							let shape = par.shape();
							let mut mem = MemBuffer::new(
								evd::self_adjoint_evd_scratch::<f64>(
									n, ce, shape, default(),
								),
							);
							cell(
								family, (m, n), layout, threads, sym_pd,
								reps,
								measure_alloc,
								|| {},
								|buf| {
									let a = unsafe { buf.mat_mut() };
									let mut uu =
										unsafe { u.mat_mut() };
									evd::self_adjoint_evd(
										a.rb(),
										s.as_mut(),
										vectors.then(|| uu.rb_mut()),
										par.reborrow(),
										MemStack::new(&mut mem),
										default(),
									)
									.unwrap();
								},
								rows,
							);
						}
						"gevd" => {
							let mut b = Buf::new(n, n, layout);
							b.fill(sym_pd_b);
							// the QZ overwrites `B`, so it is refilled outside
							// the timer; a raw pointer keeps the two closures
							// from holding the same borrow at once
							let b_ptr = &mut b as *mut Buf;
							let mut sre = Diag::<f64>::zeros(n);
							let mut sim = Diag::<f64>::zeros(n);
							let mut bet = Diag::<f64>::zeros(n);
							let mut u = Buf::new(n, n, layout);
							let ce = ComputeEigenvectors::Yes;
							let shape = par.shape();
							let mut mem = MemBuffer::new(
								gevd::gevd_scratch::<f64>(
									n,
									ComputeEigenvectors::No,
									ce,
									shape,
									default(),
								),
							);
							cell(
								family, (m, n), layout, threads, sym_pd,
								reps,
								measure_alloc,
								|| unsafe { (*b_ptr).fill(sym_pd_b) },
								|buf| {
									let mut a = unsafe { buf.mat_mut() };
									let mut bb = unsafe { (*b_ptr).mat_mut() };
									let mut uu = unsafe { u.mat_mut() };
									gevd::gevd_real(
										a.rb_mut(),
										bb.rb_mut(),
										sre.as_mut(),
										sim.as_mut(),
										bet.as_mut(),
										None,
										Some(uu.rb_mut()),
										par.reborrow(),
										MemStack::new(&mut mem),
										default(),
									)
									.unwrap();
								},
								rows,
							);
						}
						"svd_vals" | "svd_vecs" => {
							let vectors = family == "svd_vecs";
							let cu = if vectors {
								ComputeSvdVectors::Thin
							} else {
								ComputeSvdVectors::No
							};
							let mut s = Diag::<f64>::zeros(k);
							let mut u = Buf::new(m, if vectors { k } else { 1 }, layout);
							let mut v = Buf::new(n, if vectors { k } else { 1 }, layout);
							let shape = par.shape();
							let mut mem = MemBuffer::new(
								svd::svd_scratch::<f64>(
									m, n, cu, cu, shape, default(),
								),
							);
							cell(
								family, (m, n), layout, threads, general,
								reps,
								measure_alloc,
								|| {},
								|buf| {
									let a = unsafe { buf.mat_mut() };
									let mut uu = unsafe { u.mat_mut() };
									let mut vv = unsafe { v.mat_mut() };
									svd::svd(
										a.rb(),
										s.as_mut(),
										vectors.then(|| uu.rb_mut()),
										vectors.then(|| vv.rb_mut()),
										par.reborrow(),
										MemStack::new(&mut mem),
										default(),
									)
									.unwrap();
								},
								rows,
							);
						}
						other => unreachable!("{other}"),
					}
				}
			}
		}
	};

	if threads <= 1 {
		run(&mut rows, Par::Seq);
	} else {
		faer::team::with_team(threads, |par| run(&mut rows, par));
	}

	let mut f = std::io::BufWriter::new(
		std::fs::File::create(&out_path).expect("create the output file"),
	);
	writeln!(
		f,
		"family,shape,layout,threads,rep,micros,alloc_count,alloc_bytes,alloc_largest,matrix_bytes"
	)
	.unwrap();
	for r in &rows {
		if let Some((c, b, l)) = r.alloc {
			writeln!(
				f,
				"{},{},{},{},alloc,,{c},{b},{l},{}",
				r.family, r.shape, r.layout, r.threads, r.matrix_bytes
			)
			.unwrap();
		}
		for (i, t) in r.samples.iter().enumerate() {
			writeln!(
				f,
				"{},{},{},{},{},{t:.3},,,,",
				r.family, r.shape, r.layout, r.threads, i
			)
			.unwrap();
		}
	}
	f.flush().unwrap();
	eprintln!("wrote {out_path}: {} cells", rows.len());
}
