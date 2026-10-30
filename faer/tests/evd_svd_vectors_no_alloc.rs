//! #58: with the scratch sized by the matching `*_scratch` request, the
//! self-adjoint EVD with eigenvectors and the bidiagonal SVD with vectors
//! (divide and conquer from n = 128) allocate nothing on the heap on a warm
//! call. The allocator counts every allocation of the process while the call
//! runs; the inputs and the scratch are built outside the counted interval.
//!
//! It holds on x86_64 with AVX2 and FMA. Elsewhere, and on x86_64 without
//! them, `matmul` falls back to the `gemm` crate, which allocates its packing
//! buffers per call and takes no caller scratch
//! (upstream-reports/gemm-001-per-call-allocation). That path is #56's to
//! replace, so the test is ignored off x86_64; the weekly aarch64 lane of the
//! LAPACK validation runs it with `--ignored` and reports its counts. Drop the
//! `ignore` when #56 lands.
#![cfg(feature = "std")]
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{
	ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch,
};
use faer::linalg::svd::{bidiagonal_svd, bidiagonal_svd_scratch};
use faer::traits::ComplexField;
use faer::{Mat, Par, ParShape, c64};
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
	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
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

/// Allocations of the second of two calls of `run`: first-use allocations (lazy statics, feature
/// detection, thread-local workspaces that are reused) belong to the warm-up.
fn warm(mut run: impl FnMut()) -> usize {
	run();
	ALLOCATIONS.store(0, Ordering::SeqCst);
	COUNTING.store(true, Ordering::SeqCst);
	run();
	COUNTING.store(false, Ordering::SeqCst);
	ALLOCATIONS.load(Ordering::SeqCst)
}

fn evd_allocations<T: ComplexField>(a: &Mat<T>) -> usize {
	let n = a.nrows();
	let params = Default::default();
	let mut mem = MemBuffer::new(self_adjoint_evd_scratch::<T>(
		n,
		ComputeEigenvectors::Yes,
		ParShape::Seq,
		params,
	));
	let mut s = Diag::<T>::zeros(n);
	let mut u = Mat::<T>::zeros(n, n);
	warm(|| {
		self_adjoint_evd(
			a.as_ref(),
			s.as_mut(),
			Some(u.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		)
		.unwrap()
	})
}

fn bidiagonal_svd_allocations(n: usize) -> usize {
	let d: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
	let e: Vec<f64> = (0..n - 1).map(|i| 0.5 + (i % 3) as f64).collect();
	let mut dd = d.clone();
	let (mut u, mut v) = (Mat::<f64>::zeros(n, n), Mat::<f64>::zeros(n, n));
	let params = Default::default();
	let mut mem =
		MemBuffer::new(bidiagonal_svd_scratch::<f64>(n, ParShape::Seq, params));
	warm(|| {
		dd.copy_from_slice(&d);
		bidiagonal_svd(
			&mut dd,
			&e,
			true,
			u.as_mut(),
			v.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		)
		.unwrap();
	})
}

fn hermitian(n: usize) -> Mat<c64> {
	Mat::from_fn(n, n, |i, j| {
		let (p, q) = (i.min(j), i.max(j));
		let v = c64::new(
			((p * 7 + q * 3) % 11) as f64,
			if p == q { 0.0 } else { ((p + q) % 5) as f64 },
		);
		if i <= j { v } else { v.conj() }
	})
}

// one test function: the counter is process wide, so nothing else may run while it counts
#[test]
#[cfg_attr(
	not(target_arch = "x86_64"),
	ignore = "#58: the gemm crate fallback allocates per call until #56"
)]
fn evd_and_bidiagonal_svd_with_vectors_allocate_nothing() {
	#[cfg(target_arch = "x86_64")]
	if !(std::arch::is_x86_feature_detected!("avx2")
		&& std::arch::is_x86_feature_detected!("fma"))
	{
		// the product takes the `gemm` crate fallback on this CPU; see the module comment
		return;
	}
	let mut counts = vec![];
	for n in [2usize, 17, 127, 128, 129, 300] {
		let c = hermitian(n);
		let r = Mat::<f64>::from_fn(n, n, |i, j| c[(i, j)].re);
		counts.push((
			n,
			evd_allocations(&c),
			evd_allocations(&r),
			bidiagonal_svd_allocations(n),
		));
	}
	assert!(
		counts.iter().all(|&(_, a, b, c)| a + b + c == 0),
		"(n, evd c64, evd f64, bidiagonal svd f64): {counts:?}"
	);
}
