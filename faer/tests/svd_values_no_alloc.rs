//! #156: the values-only SVD allocates nothing on the heap once the caller's `MemStack` is sized by
//! `svd_scratch`. The dqds route (0.26.0) built its four work arrays with `vec!` (two allocations at
//! 1 x 1, four above), and sorted with a stable sort, whose merge buffer goes to the heap once the
//! slice outgrows a stack buffer (one allocation for `f64` and `c64` at 700 x 700). The allocator counts every
//! allocation of the process while the call runs; the matrices and the scratch are built outside the
//! counted interval.
#![cfg(feature = "std")]
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{ComputeSvdVectors, svd, svd_scratch};
use faer::diag::Diag;
use faer::traits::ComplexField;
use faer::{Auto, Mat, Par, ParShape, c32, c64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counting;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.alloc(layout) }
	}
	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.alloc_zeroed(layout) }
	}
	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.realloc(ptr, layout, new_size) }
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations of one values-only `svd` call on an `m x n` matrix, after a warm-up call.
fn values_only_allocations<T: ComplexField>(a: &Mat<T>) -> usize {
	let (m, n) = a.shape();
	let params = <faer::linalg::svd::SvdParams as Auto<T>>::auto().into();
	let mut mem = MemBuffer::new(svd_scratch::<T>(m, n, ComputeSvdVectors::No, ComputeSvdVectors::No, ParShape::Seq, params));
	let mut s = Diag::<T>::zeros(Ord::min(m, n));
	let mut run = || svd(a.as_ref(), s.as_mut(), None, None, Par::Seq, MemStack::new(&mut mem), params).unwrap();
	run(); // warm-up: first-use allocations (lazy statics, feature detection) are not the call's
	ALLOCATIONS.store(0, Ordering::SeqCst);
	COUNTING.store(true, Ordering::SeqCst);
	run();
	COUNTING.store(false, Ordering::SeqCst);
	ALLOCATIONS.load(Ordering::SeqCst)
}

fn sample(m: usize, n: usize, seed: u64) -> Mat<f64> {
	let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
	Mat::from_fn(m, n, |_, _| {
		x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
	})
}

// one test function: the counter is process wide, so nothing else may run while it counts
#[test]
fn values_only_svd_allocates_nothing() {
	// 1 x 1 (tlinalg's case), small, wide, square past the recursion threshold, tall enough to take
	// the QR route, and large enough for the stable sort's buffer to leave the stack
	for (m, n) in [(1, 1), (2, 2), (17, 17), (40, 33), (33, 40), (128, 128), (300, 60), (700, 700)] {
		let re = sample(m, n, 156);
		let im = sample(m, n, 157);
		let f32s = Mat::<f32>::from_fn(m, n, |i, j| re[(i, j)] as f32);
		let c64s = Mat::<c64>::from_fn(m, n, |i, j| c64::new(re[(i, j)], im[(i, j)]));
		let c32s = Mat::<c32>::from_fn(m, n, |i, j| c32::new(re[(i, j)] as f32, im[(i, j)] as f32));
		let counts = [
			values_only_allocations(&re),
			values_only_allocations(&f32s),
			values_only_allocations(&c64s),
			values_only_allocations(&c32s),
		];
		assert_eq!(counts, [0; 4], "{m} x {n}: allocations (f64, f32, c64, c32)");
	}
}
