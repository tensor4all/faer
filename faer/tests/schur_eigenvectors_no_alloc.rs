//! tensor4all/faer#270: with the scratch sized by
//! `eigenvectors_from_schur_scratch`, a warm call
//! of `eigenvectors_from_schur_real` / `eigenvectors_from_schur_cplx` allocates
//! nothing on the heap, also when the Schur form's entries are far from one and
//! the back substitution runs on a copy scaled into range. That copy was an
//! owned `Mat` (one `n` by `n` allocation per call); it now comes from the
//! caller's scratch.
//!
//! The allocator counts every allocation of the process while the call runs;
//! the inputs, the output and the scratch are built outside the counted
//! interval. The size is 32, with an orthogonal `q`, as in the issue's
//! reproducer, measured at 0 allocations on aarch64 macOS and x86_64 Linux at
//! ordinary scale.
#![cfg(feature = "std")]
mod common;
use common::*;
use faer::MatMut;
use faer::linalg::evd::{
	eigenvectors_from_schur_cplx, eigenvectors_from_schur_real,
	eigenvectors_from_schur_scratch,
};
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

/// Allocations of the second of two calls of `run`: first-use allocations (lazy
/// statics, feature detection, thread-local workspaces that are reused) belong
/// to the warm-up.
fn warm(mut run: impl FnMut()) -> usize {
	run();
	ALLOCATIONS.store(0, Ordering::SeqCst);
	COUNTING.store(true, Ordering::SeqCst);
	run();
	COUNTING.store(false, Ordering::SeqCst);
	ALLOCATIONS.load(Ordering::SeqCst)
}

const N: usize = 32;
const CASES: [&str; 3] = ["ordinary", "large", "small"];

/// Allocations of a warm call on `scale * t`, for each scale, with an
/// orthogonal `q`.
fn allocations<T: ComplexField>(
	t: &Mat<T>,
	scales: [f64; 3],
	call: fn(MatRef<'_, T>, MatRef<'_, T>, MatMut<'_, T>, &mut MemStack),
) -> [usize; 3] {
	let q = Mat::<T>::from_fn(N, N, |i, j| from_f64(sample(i, j)))
		.qr(Par::Seq)
		.compute_Q(Par::Seq);
	let mut x = Mat::<T>::zeros(N, N);
	let mut mem = MemBuffer::new(eigenvectors_from_schur_scratch::<T>(N));
	scales.map(|scale| {
		let scale = from_f64::<T::Real>(scale);
		let t = Mat::<T>::from_fn(N, N, |i, j| t[(i, j)].mul_real(&scale));
		warm(|| {
			call(t.as_ref(), q.as_ref(), x.as_mut(), MemStack::new(&mut mem))
		})
	})
}

fn real_schur<T: RealField>() -> Mat<T> {
	// standardized 2 by 2 blocks at rows 1..3, 4..6, ...
	Mat::from_fn(N, N, |i, j| {
		let pair = |k: usize| k % 3 == 1 && k + 1 < N;
		if i == j {
			let k = if i > 0 && pair(i - 1) { i - 1 } else { i };
			from_f64((k + 1) as f64)
		} else if i + 1 == j && pair(i) {
			from_f64(0.7)
		} else if i == j + 1 && pair(j) {
			from_f64(-0.3)
		} else if i < j {
			from_f64(0.03 * sample(i, j))
		} else {
			zero()
		}
	})
}

fn complex_schur<T: ComplexField>(make: fn(f64, f64) -> T) -> Mat<T> {
	Mat::from_fn(N, N, |i, j| {
		if i == j {
			let phase = 0.37 * i as f64;
			make((i + 1) as f64 * phase.cos(), (i + 1) as f64 * phase.sin())
		} else if i < j {
			make(0.03 * sample(i, j), 0.02 * sample(j, i))
		} else {
			make(0.0, 0.0)
		}
	})
}

fn real_call<T: RealField>(
	t: MatRef<'_, T>,
	q: MatRef<'_, T>,
	x: MatMut<'_, T>,
	stack: &mut MemStack,
) {
	eigenvectors_from_schur_real(
		t,
		Some(q),
		x,
		Par::Seq,
		stack,
		Default::default(),
	)
	.unwrap();
}
fn complex_call<T: ComplexField>(
	t: MatRef<'_, T>,
	q: MatRef<'_, T>,
	x: MatMut<'_, T>,
	stack: &mut MemStack,
) {
	eigenvectors_from_schur_cplx(
		t,
		Some(q),
		x,
		Par::Seq,
		stack,
		Default::default(),
	)
	.unwrap();
}

#[test]
fn warm_schur_eigenvectors_allocate_nothing_at_any_scale() {
	let counts = [
		(
			"f32",
			allocations(
				&real_schur::<f32>(),
				[1.0, 1e30, 1e-30],
				real_call::<f32>,
			),
		),
		(
			"f64",
			allocations(
				&real_schur::<f64>(),
				[1.0, 1e300, 1e-300],
				real_call::<f64>,
			),
		),
		(
			"c32",
			allocations(
				&complex_schur::<c32>(|r, i| c32::new(r as f32, i as f32)),
				[1.0, 1e30, 1e-30],
				complex_call::<c32>,
			),
		),
		(
			"c64",
			allocations(
				&complex_schur::<c64>(c64::new),
				[1.0, 1e300, 1e-300],
				complex_call::<c64>,
			),
		),
	];
	let mut report = String::new();
	for (name, count) in &counts {
		for (case, count) in CASES.iter().zip(count) {
			report += &format!("{name} {case} allocations={count}\n");
		}
	}
	eprint!("{report}");
	assert!(
		counts
			.iter()
			.all(|(_, count)| count.iter().all(|&c| c == 0)),
		"{report}"
	);
}
