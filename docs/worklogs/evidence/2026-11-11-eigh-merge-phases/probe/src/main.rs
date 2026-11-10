//! tensor4all/faer#391 increment 1 probe: one cell per process.
//!
//! usage: rm391 <mode> <dtype> <n> <threads> <thr> [prime_ms] [measure_ms]
//!   mode    stage2 | leaf     (leaf = stage 2 with recursion_threshold = usize::MAX, i.e. QR alone)
//!   dtype   f64 | c64
//!   thr     default | <usize> (recursion_threshold)
//!
//! Prints CSV rows `key,level,phase,value` on stdout. Timing values are per call
//! (accumulated over every timed call of the window, divided by the call count).
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{
	tridiag, tridiagonal_self_adjoint_evd, tridiagonal_self_adjoint_evd_scratch,
	ComputeEigenvectors, SelfAdjointEvdParams,
};
use faer::traits::ComplexField;
use faer::diag::Diag;
use faer::{c64, Mat, Par, ParShape, Spec};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, l: Layout) -> *mut u8 {
		ALLOCS.fetch_add(1, Relaxed);
		#[cfg(feature = "instr")]
		faer::dc_probe::ALLOCS.fetch_add(1, Relaxed);
		System.alloc(l)
	}
	unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
		ALLOCS.fetch_add(1, Relaxed);
		#[cfg(feature = "instr")]
		faer::dc_probe::ALLOCS.fetch_add(1, Relaxed);
		System.alloc_zeroed(l)
	}
	unsafe fn realloc(&self, p: *mut u8, l: Layout, s: usize) -> *mut u8 {
		ALLOCS.fetch_add(1, Relaxed);
		#[cfg(feature = "instr")]
		faer::dc_probe::ALLOCS.fetch_add(1, Relaxed);
		System.realloc(p, l, s)
	}
	unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
		System.dealloc(p, l)
	}
}
#[global_allocator]
static GA: Counting = Counting;

trait Scalar: ComplexField + Copy {
	fn sample(st: &mut u64) -> Self;
	fn herm(self) -> Self;
	fn re_only(self) -> Self;
}
fn unif(st: &mut u64) -> f64 {
	*st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
	((*st >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}
impl Scalar for f64 {
	fn sample(st: &mut u64) -> Self {
		unif(st)
	}
	fn herm(self) -> Self {
		self
	}
	fn re_only(self) -> Self {
		self
	}
}
impl Scalar for c64 {
	fn sample(st: &mut u64) -> Self {
		c64::new(unif(st), unif(st))
	}
	fn herm(self) -> Self {
		self.conj()
	}
	fn re_only(self) -> Self {
		c64::new(self.re, 0.0)
	}
}

fn run<T: Scalar>(mode: &str, n: usize, threads: usize, thr: Option<usize>, prime: Duration, window: Duration) {
	// the tridiagonal faer's stage 1 makes from a seeded random Hermitian matrix
	let mut st = 0x391u64;
	let mut a = Mat::<T>::zeros(n, n);
	for j in 0..n {
		for i in j..n {
			let mut x = T::sample(&mut st);
			if i == j {
				x = x.re_only();
			}
			a[(i, j)] = x;
			a[(j, i)] = x.herm();
		}
	}
	let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n, n);
	let mut h = Mat::<T>::zeros(bs, n - 1);
	let mut mem = MemBuffer::new(tridiag::tridiag_in_place_scratch::<T>(n, ParShape::Seq, Default::default()));
	tridiag::tridiag_in_place(a.as_mut(), h.as_mut(), Par::Seq, MemStack::new(&mut mem), Default::default());
	let diag = { let mut d = Diag::<T>::zeros(n); for i in 0..n { d[i] = a[(i, i)]; } d };
	let sub = { let mut d = Diag::<T>::zeros(n - 1); for i in 0..n - 1 { d[i] = a[(i + 1, i)]; } d };

	let mut params: Spec<SelfAdjointEvdParams, T> = Default::default();
	match (mode, thr) {
		("leaf", _) => params.recursion_threshold = usize::MAX,
		(_, Some(t)) => params.recursion_threshold = t,
		_ => {},
	}
	let shape = if threads == 1 { ParShape::Seq } else { ParShape::team(threads) };
	let mut mem = MemBuffer::new(tridiagonal_self_adjoint_evd_scratch::<T>(n, ComputeEigenvectors::Yes, shape, params));
	let mut s = Diag::<T>::zeros(n);
	let mut u = Mat::<T>::zeros(n, n);

	let mut body = |par: Par| {
		let mut call = |par: Par| {
			tridiagonal_self_adjoint_evd(
				diag.as_ref(),
				sub.as_ref(),
				s.as_mut(),
				Some(u.as_mut()),
				par,
				MemStack::new(&mut mem),
				params,
			)
			.unwrap();
		};
		let mut par = par;
		let t0 = Instant::now();
		while t0.elapsed() < prime {
			call(par.reborrow());
		}
		#[cfg(feature = "instr")]
		faer::dc_probe::reset();
		#[cfg(feature = "instr")]
		faer::dc_probe::reset_counts();
		let a0 = ALLOCS.load(Relaxed);
		let mut times = Vec::with_capacity(1 << 22);
		let a1 = ALLOCS.load(Relaxed); // the Vec above is outside the count
		let t0 = Instant::now();
		while t0.elapsed() < window {
			let c = Instant::now();
			call(par.reborrow());
			times.push(c.elapsed().as_nanos() as u64);
		}
		let allocs = ALLOCS.load(Relaxed) - a1;
		let _ = a0;
		(times, allocs)
	};
	let (mut times, allocs) = if threads == 1 {
		body(Par::Seq)
	} else {
		faer::team::with_team(threads, |par| body(par))
	};
	// 1 << 22 slots: no push reallocates (and is counted) within a window of this length
	let calls = times.len() as f64;
	let mean = times.iter().sum::<u64>() as f64 / calls;
	times.sort_unstable();
	let median = times[times.len() / 2] as f64;
	println!("calls,,,{}", times.len());
	println!("call_mean_ns,,,{mean:.1}");
	println!("call_median_ns,,,{median:.1}");
	println!("allocs_per_call,,,{:.4}", allocs as f64 / calls);
	#[cfg(feature = "instr")]
	{
		use faer::dc_probe as P;
		for l in 0..P::MAX_LEVEL {
			for p in 0..P::N_PHASES {
				let c = P::CALLS[l][p].load(Relaxed);
				if c == 0 {
					continue;
				}
				let name = P::PHASES[p];
				println!("ns,{l},{name},{:.1}", P::NS[l][p].load(Relaxed) as f64 / calls);
				println!("nodes,{l},{name},{:.4}", c as f64 / calls);
				println!("allocs,{l},{name},{:.4}", P::ALLOC[l][p].load(Relaxed) as f64 / calls);
			}
			for (key, arr) in [
				("halves_par", &P::HALVES_PAR),
				("halves_seq", &P::HALVES_SEQ),
				("gemm_par", &P::GEMM_PAR),
				("gemm_seq", &P::GEMM_SEQ),
				("k_sum", &P::K_SUM),
			] {
				let v = arr[l].load(Relaxed);
				if v != 0 {
					println!("{key},{l},,{:.4}", v as f64 / calls);
				}
			}
		}
		#[cfg(feature = "count")]
		for l in 0..P::MAX_LEVEL {
			for (key, arr) in [
				("sec_roots", &P::SEC_ROOTS),
				("sec_evals", &P::SEC_EVALS),
				("sec_batch", &P::SEC_BATCH),
				("sec_terms", &P::SEC_TERMS),
				("sec_bisect_roots", &P::SEC_BISECT_ROOTS),
				("sec_bisect_evals", &P::SEC_BISECT_EVALS),
			] {
				let v = arr[l].load(Relaxed);
				if v != 0 {
					println!("{key},{l},,{:.4}", v as f64 / calls);
				}
			}
		}
		for (sz, c) in P::LEAF_SIZE.iter().enumerate() {
			let c = c.load(Relaxed);
			if c != 0 {
				println!("leaf_size,{sz},,{:.4}", c as f64 / calls);
			}
		}
	}
}

fn main() {
	let a: Vec<String> = std::env::args().collect();
	let mode = a[1].as_str();
	let dtype = a[2].as_str();
	let n: usize = a[3].parse().unwrap();
	let threads: usize = a[4].parse().unwrap();
	let thr = if a[5] == "default" { None } else { Some(a[5].parse().unwrap()) };
	let prime = Duration::from_millis(a.get(6).map_or(250, |x| x.parse().unwrap()));
	let window = Duration::from_millis(a.get(7).map_or(1000, |x| x.parse().unwrap()));
	match dtype {
		"f64" => run::<f64>(mode, n, threads, thr, prime, window),
		"c64" => run::<c64>(mode, n, threads, thr, prime, window),
		_ => panic!("dtype"),
	}
}
