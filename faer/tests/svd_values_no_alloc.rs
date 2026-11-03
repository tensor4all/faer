//! #156: the values-only SVD allocates nothing on the heap once the caller's `MemStack` is sized by
//! `svd_scratch`. The dqds route (0.26.0) built its four work arrays with
//! `vec!` (two allocations at 1 x 1, four above), and sorted with a stable
//! sort, whose merge buffer goes to the heap once the slice outgrows a stack
//! buffer (one allocation for `f64` and `c64` at 700 x 700). The allocator
//! counts every allocation **of the counting thread** while the call runs; the
//! matrices and the scratch are built outside the counted interval.
#![cfg(feature = "std")]
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{ComputeSvdVectors, svd, svd_scratch};
use faer::traits::ComplexField;
use faer::{Auto, Mat, Par, ParShape, c32, c64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
// Per thread, not per process: the call under test is `Par::Seq`, so every
// allocation it makes is made on this thread, and a line the harness prints on
// another thread (its "has been running for over 60 seconds" watchdog, a result
// line) is not the call's. A process-wide flag counted the watchdog's own
// string: CI run 41407771163 failed at 700 x 700 with two `c32` allocations the
// warning had made, while the same test passed at 52 s when it stayed under the
// 60 s mark.
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

/// Allocations of one values-only `svd` call on an `m x n` matrix, after a
/// warm-up call.
fn values_only_allocations<T: ComplexField>(a: &Mat<T>) -> usize {
	let (m, n) = a.shape();
	let params = <faer::linalg::svd::SvdParams as Auto<T>>::auto().into();
	let mut mem = MemBuffer::new(svd_scratch::<T>(
		m,
		n,
		ComputeSvdVectors::No,
		ComputeSvdVectors::No,
		ParShape::Seq,
		params,
	));
	let mut s = Diag::<T>::zeros(Ord::min(m, n));
	let mut run = || {
		svd(
			a.as_ref(),
			s.as_mut(),
			None,
			None,
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		)
		.unwrap()
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
/// test of this binary may allocate while one counts: under `cargo test` the
/// tests of one binary run on threads of one process, so they take turns here,
/// and under nextest each test already has its own process.
static GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Allocations of the four scalar types on an `m x n` matrix, asserted to be
/// none.
fn check_cell(m: usize, n: usize) {
	let _turn = GATE.lock().unwrap_or_else(|e| e.into_inner());
	let re = sample(m, n, 156);
	let im = sample(m, n, 157);
	let f32s = Mat::<f32>::from_fn(m, n, |i, j| re[(i, j)] as f32);
	let c64s =
		Mat::<c64>::from_fn(m, n, |i, j| c64::new(re[(i, j)], im[(i, j)]));
	let c32s = Mat::<c32>::from_fn(m, n, |i, j| {
		c32::new(re[(i, j)] as f32, im[(i, j)] as f32)
	});
	let counts = [
		values_only_allocations(&re),
		values_only_allocations(&f32s),
		values_only_allocations(&c64s),
		values_only_allocations(&c32s),
	];
	assert_eq!(
		counts, [0; 4],
		"{m} x {n}: allocations (f64, f32, c64, c32)"
	);
}

// One test per shape, because the harness reports a test that runs for over 60
// seconds *from inside the test process*: the message is formatted and printed
// by libtest's watchdog thread, which allocates, and the counting window is
// open at that moment. A single 8-shape test crossed 60 s under coverage
// (60.119 s) and failed with two `c32` allocations that the warning had made
// (CI run 41407771163). The assertions are unchanged; only the shape loop is
// split.
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
// 1 x 1 (tlinalg's case), small, wide, square past the recursion threshold,
// tall enough to take the QR route, and large enough for the stable sort's
// buffer to leave the stack
cells! {
	values_only_svd_allocates_nothing_1x1: (1, 1);
	values_only_svd_allocates_nothing_2x2: (2, 2);
	values_only_svd_allocates_nothing_17x17: (17, 17);
	values_only_svd_allocates_nothing_40x33: (40, 33);
	values_only_svd_allocates_nothing_33x40: (33, 40);
	values_only_svd_allocates_nothing_128x128: (128, 128);
	values_only_svd_allocates_nothing_300x60: (300, 60);
	values_only_svd_allocates_nothing_700x700: (700, 700);
}
