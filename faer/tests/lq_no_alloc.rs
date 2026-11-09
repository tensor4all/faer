//! tensor4all/faer#130: the packed LQ factorization and the factor extraction
//! (`copy_l`, `unitary`, `null`, `reconstruct`) allocate nothing on the heap once
//! the caller's `MemStack` is sized by the matching `*_scratch` query, on the
//! serial path.
//!
//! The transposed view that `lq_in_place` factors is never column-major, so a
//! per-call copy would be allocated by the QR layout adaptation on every wide
//! call; the LQ entry reaches the packed kernel directly instead. The allocator
//! counts every allocation **of the counting thread** while the call runs; the
//! matrices, the coefficients and the scratch are built outside the counted
//! interval.
//!
//! Not covered here, and inherited from the QR kernel: `upgrade_householder_factor`
//! allocates one `partials` buffer when the `V^H V` reduction is split over more
//! than one thread, and the matmul fallback on hosts without AVX2/FMA allocates
//! (PATCHSET rows #58/#56). Both apply to a `Par::Team` call, not to this one.
//!
//! A third path is host-dependent and does reach this one: where the `gemm` crate
//! dispatches to its aarch64 SME kernel, `PackBuffer::take` builds a fresh
//! `MemBuffer` on every call, so `unitary` allocates through
//! `apply_block_householder_on_the_left_in_place_generic` on such a host. Measured
//! on an aarch64 macOS host at `60 x 300`, the one shape far enough into that
//! dispatch: 13, 12, 17, 17 allocations for (f64, f32, c64, c32), the same numbers
//! at `620b96a` and at head. On `x86_64` the test passes, so this is the kernel the
//! build's `gemm` selects, not the LQ code.
#![cfg(feature = "std")]
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lq::no_pivoting::{factor, reconstruct, unitary};
use faer::traits::ComplexField;
use faer::{Auto, Mat, Par, ParShape, c32, c64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
// Per thread, not per process: the call under test is `Par::Seq`, so every
// allocation it makes is made on this thread, and a line the harness prints on
// another thread (its watchdog, a result line) is not the call's.
thread_local! {
	static COUNTING: Cell<bool> = const { Cell::new(false) };
}
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn counting() -> bool {
	COUNTING.with(Cell::get)
}
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if counting() {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if counting() {
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
		if counting() {
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

/// Allocations of one complete LQ pipeline on `a`, after a warm-up call.
fn lq_allocations<T: ComplexField>(a: &Mat<T>) -> usize {
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let params = <factor::LqParams as Auto<T>>::auto().into();
	let block_size =
		Ord::max(factor::recommended_block_size::<T>(m, n), 1);
	let mut factor_mem = MemBuffer::new(factor::lq_in_place_scratch::<T>(
		m,
		n,
		block_size,
		ParShape::Seq,
		params,
	));
	let mut reconstruct_mem = MemBuffer::new(reconstruct::reconstruct_scratch::<T>(
		m,
		n,
		block_size,
		ParShape::Seq,
	));
	let mut unitary_mem = MemBuffer::new(unitary::unitary_scratch::<T>(
		m,
		n,
		k,
		block_size,
		ParShape::Seq,
	));
	let mut full_mem = MemBuffer::new(unitary::unitary_scratch::<T>(
		m,
		n,
		n,
		block_size,
		ParShape::Seq,
	));
	let mut null_mem = MemBuffer::new(unitary::null_scratch::<T>(
		m,
		n,
		block_size,
		ParShape::Seq,
	));
	let mut packed = a.clone();
	let mut coeff = Mat::<T>::zeros(block_size, k);
	let mut l = Mat::<T>::zeros(m, k);
	let mut l_full = Mat::<T>::zeros(m, n);
	let mut q = Mat::<T>::zeros(k, n);
	let mut q_full = Mat::<T>::zeros(n, n);
	let mut null = Mat::<T>::zeros(n - k, n);
	let mut rebuilt = Mat::<T>::zeros(m, n);
	let mut run = || {
		factor::lq_in_place(
			packed.as_mut(),
			coeff.as_mut(),
			Par::Seq,
			MemStack::new(&mut factor_mem),
			params,
		);
		factor::copy_l(l.as_mut(), packed.as_ref(), true);
		factor::copy_l(l_full.as_mut(), packed.as_ref(), true);
		unitary::unitary(
			q.as_mut(),
			packed.as_ref(),
			coeff.as_ref(),
			true,
			Par::Seq,
			MemStack::new(&mut unitary_mem),
		);
		unitary::unitary(
			q_full.as_mut(),
			packed.as_ref(),
			coeff.as_ref(),
			true,
			Par::Seq,
			MemStack::new(&mut full_mem),
		);
		unitary::null(
			null.as_mut(),
			packed.as_ref(),
			coeff.as_ref(),
			Par::Seq,
			MemStack::new(&mut null_mem),
		);
		reconstruct::reconstruct(
			rebuilt.as_mut(),
			packed.as_ref(),
			coeff.as_ref(),
			Par::Seq,
			MemStack::new(&mut reconstruct_mem),
		);
	};
	run(); // warm-up: first-use allocations (lazy statics, feature detection) are not the call's
	ALLOCATIONS.store(0, Ordering::SeqCst);
	COUNTING.with(|c| c.set(true));
	run();
	COUNTING.with(|c| c.set(false));
	ALLOCATIONS.load(Ordering::SeqCst)
}

fn sample(m: usize, n: usize, seed: u64) -> Mat<f64> {
	let mut x = seed
		.wrapping_mul(6364136223846793005)
		.wrapping_add(1442695040888963407);
	Mat::from_fn(m, n, |_, _| {
		x = x
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
	})
}

/// Serializes the counted windows. The counter is process wide, so no other
/// test of this binary may allocate while one counts.
static GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Allocations of the four scalar types on an `m x n` matrix, asserted to be
/// none.
fn check_cell(m: usize, n: usize) {
	let _turn = GATE.lock().unwrap_or_else(|e| e.into_inner());
	let re = sample(m, n, 130);
	let im = sample(m, n, 131);
	let f32s = Mat::<f32>::from_fn(m, n, |i, j| re[(i, j)] as f32);
	let c64s =
		Mat::<c64>::from_fn(m, n, |i, j| c64::new(re[(i, j)], im[(i, j)]));
	let c32s = Mat::<c32>::from_fn(m, n, |i, j| {
		c32::new(re[(i, j)] as f32, im[(i, j)] as f32)
	});
	let counts = [
		lq_allocations(&re),
		lq_allocations(&f32s),
		lq_allocations(&c64s),
		lq_allocations(&c32s),
	];
	assert_eq!(counts, [0; 4], "{m} x {n}: allocations (f64, f32, c64, c32)");
}

// One test per shape, because the harness reports a test that runs for over 60
// seconds from inside the test process, and its watchdog thread allocates while
// the counting window is open.
macro_rules! cells {
	($($name:ident: ($m:expr, $n:expr);)*) => {
		$(
			#[test]
			fn $name() {
				check_cell($m, $n);
			}
		)*
	};
}
// 1 x 1, small, square past the recursion threshold, tall and wide enough for
// the blocked panel to have a trailing update, and a shape that is a multiple of
// 1 KiB per row (128 f64), which the transposed panel walks with a 1 KiB stride.
cells! {
	lq_allocates_nothing_1x1: (1, 1);
	lq_allocates_nothing_2x2: (2, 2);
	lq_allocates_nothing_17x17: (17, 17);
	lq_allocates_nothing_40x33: (40, 33);
	lq_allocates_nothing_33x40: (33, 40);
	lq_allocates_nothing_128x128: (128, 128);
	lq_allocates_nothing_300x60: (300, 60);
	lq_allocates_nothing_60x300: (60, 300);
}
