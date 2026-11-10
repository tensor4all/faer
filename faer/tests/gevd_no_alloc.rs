//! tensor4all/faer#375: with the scratch sized by `gevd_scratch`, a warm call of
//! `gevd_real` / `gevd_cplx` allocates nothing on the heap, also when `A` and
//! `B` are prescaled (entries far below `sqrt(safmin) / eps` or above its
//! inverse) and `alpha`, `beta` are scaled back.
//!
//! The allocator counts the allocations of the calling thread while the call
//! runs (a thread-local flag, so tests running on other threads are not
//! counted); the inputs, the outputs and the scratch are built outside the
//! counted interval. `Par::Seq`, so the call does not leave the thread.
#![cfg(feature = "std")]
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{GevdParams, gevd_cplx, gevd_real, gevd_scratch};
use faer::{Mat, Par, ParShape, Spec, c64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;
thread_local! {
	static COUNTING: Cell<bool> = const { Cell::new(false) };
	static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
fn count() {
	// `try_with`: the allocator also runs while thread locals are torn down
	let _ = COUNTING.try_with(|c| {
		if c.get() {
			let _ = ALLOCATIONS.try_with(|a| a.set(a.get() + 1));
		}
	});
}
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		count();
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		count();
		unsafe { System.alloc_zeroed(layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		count();
		unsafe { System.realloc(ptr, layout, new_size) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations of the second of two calls of `run`: first-use allocations
/// belong to the warm-up.
fn warm(mut run: impl FnMut()) -> usize {
	run();
	ALLOCATIONS.with(|a| a.set(0));
	COUNTING.with(|c| c.set(true));
	run();
	COUNTING.with(|c| c.set(false));
	ALLOCATIONS.with(|a| a.get())
}

/// A fixed pseudo-random value in `[-1, 1)`.
fn sample(i: usize, j: usize, seed: u64) -> f64 {
	let mut s = (i as u64 + 1)
		.wrapping_mul(0x9e37_79b9_7f4a_7c15)
		.wrapping_add((j as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9))
		.wrapping_add(seed.wrapping_mul(0x94d0_49bb_1331_11eb));
	s ^= s >> 31;
	s = s.wrapping_mul(0xd6e8_feb8_6659_fd93);
	s ^= s >> 32;
	((s >> 11) as f64 / (1u64 << 52) as f64) - 1.0
}

/// `1` is in range; `1e-300` and `1e300` are prescaled
const SCALES: [f64; 3] = [1.0, 1e-300, 1e300];
/// below and above the QZ's default blocking threshold (75)
const SIZES: [usize; 2] = [20, 100];

fn real_allocations(n: usize, s: f64, ce: ComputeEigenvectors) -> usize {
	let a = Mat::<f64>::from_fn(n, n, |i, j| s * sample(i, j, 1));
	let b = Mat::<f64>::from_fn(n, n, |i, j| {
		s * (sample(i, j, 2) + if i == j { 2.0 } else { 0.0 })
	});
	let (mut am, mut bm) = (a.clone(), b.clone());
	let mut re = Diag::<f64>::zeros(n);
	let mut im = Diag::<f64>::zeros(n);
	let mut beta = Diag::<f64>::zeros(n);
	let mut ul = Mat::<f64>::zeros(n, n);
	let mut ur = Mat::<f64>::zeros(n, n);
	let params = Spec::<GevdParams, f64>::default();
	let mut mem =
		MemBuffer::new(gevd_scratch::<f64>(n, ce, ce, ParShape::Seq, params));
	let vectors = matches!(ce, ComputeEigenvectors::Yes);
	warm(|| {
		am.copy_from(&a);
		bm.copy_from(&b);
		gevd_real(
			am.as_mut(),
			bm.as_mut(),
			re.as_mut(),
			im.as_mut(),
			beta.as_mut(),
			vectors.then(|| ul.as_mut()),
			vectors.then(|| ur.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		)
		.unwrap();
	})
}

fn cplx_allocations(n: usize, s: f64, ce: ComputeEigenvectors) -> usize {
	let a = Mat::<c64>::from_fn(n, n, |i, j| {
		c64::new(s * sample(i, j, 3), s * sample(i, j, 4))
	});
	let b = Mat::<c64>::from_fn(n, n, |i, j| {
		c64::new(
			s * (sample(i, j, 5) + if i == j { 2.0 } else { 0.0 }),
			s * sample(i, j, 6),
		)
	});
	let (mut am, mut bm) = (a.clone(), b.clone());
	let mut alpha = Diag::<c64>::zeros(n);
	let mut beta = Diag::<c64>::zeros(n);
	let mut ul = Mat::<c64>::zeros(n, n);
	let mut ur = Mat::<c64>::zeros(n, n);
	let params = Spec::<GevdParams, c64>::default();
	let mut mem =
		MemBuffer::new(gevd_scratch::<c64>(n, ce, ce, ParShape::Seq, params));
	let vectors = matches!(ce, ComputeEigenvectors::Yes);
	warm(|| {
		am.copy_from(&a);
		bm.copy_from(&b);
		gevd_cplx(
			am.as_mut(),
			bm.as_mut(),
			alpha.as_mut(),
			beta.as_mut(),
			vectors.then(|| ul.as_mut()),
			vectors.then(|| ur.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		)
		.unwrap();
	})
}

#[test]
fn gevd_warm_call_does_not_allocate() {
	let mut failures = Vec::new();
	for n in SIZES {
		for s in SCALES {
			for ce in [ComputeEigenvectors::No, ComputeEigenvectors::Yes] {
				let r = real_allocations(n, s, ce);
				let c = cplx_allocations(n, s, ce);
				println!(
					"n = {n}, s = {s:e}, vectors = {ce:?}: gevd_real {r}, gevd_cplx {c} allocations"
				);
				if r != 0 || c != 0 {
					failures.push((n, s, ce, r, c));
				}
			}
		}
	}
	assert!(failures.is_empty(), "allocations: {failures:?}");
}
