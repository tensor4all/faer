//! tensor4all/faer#314: a small square row-major matrix that `qr_in_place`
//! used to copy into a column-major buffer is factored through the transpose
//! dual instead on a sequential call, and no `m * n` buffer is allocated. The
//! allocator records the largest single allocation the calling thread makes
//! while `qr_in_place` runs; the copy is one `m * n` buffer, so the dual stays
//! below that size.
//!
//! The control: the shapes and calls that keep the copy (a tall matrix, a
//! square one from 192 up, a team call, a stack smaller than
//! `qr_in_place_scratch`, a view whose strides are both not 1), where the same
//! counter must see it.
#![cfg(feature = "std")]
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::qr::no_pivoting::factor as qr;
use faer::linalg::temp_mat_scratch;
use faer::{MatMut, Par};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct LargestOnThisThread;
thread_local! {
	static COUNTING: Cell<bool> = const { Cell::new(false) };
	static LARGEST: Cell<usize> = const { Cell::new(0) };
}
fn record(size: usize) {
	// `try_with`: the allocator also runs while a thread's locals are torn down
	let _ = COUNTING.try_with(|on| {
		if on.get() {
			let _ = LARGEST.try_with(|l| l.set(Ord::max(l.get(), size)));
		}
	});
}
unsafe impl GlobalAlloc for LargestOnThisThread {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		record(layout.size());
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		record(layout.size());
		unsafe { System.alloc_zeroed(layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		record(new_size);
		unsafe { System.realloc(ptr, layout, new_size) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: LargestOnThisThread = LargestOnThisThread;

/// The largest single allocation, in bytes, the calling thread makes while an
/// `m x n` f64 matrix with element strides `(rs, cs)` (both positive) is
/// factored by `qr_in_place` with a stack of `stack_req`. The call runs twice
/// on fresh copies of the matrix and only the second is counted, so a buffer a
/// kernel keeps per thread across calls (the GEMM's packing buffer) is not
/// mistaken for a per-call one.
fn largest_allocation(
	m: usize,
	n: usize,
	(rs, cs): (usize, usize),
	stack_req: Option<StackReq>,
	mut par: Par<'_>,
) -> usize {
	let len = (m - 1) * rs + (n - 1) * cs + 1;
	let a: Vec<f64> = (0..len)
		.map(|k| {
			((k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64
				/ (1u64 << 53) as f64
				- 0.5
		})
		.collect();
	let bs = qr::recommended_block_size::<f64>(m, n);
	let k = Ord::min(m, n);
	let mut coeff = vec![0.0f64; bs * k];
	let req = stack_req.unwrap_or_else(|| {
		qr::qr_in_place_scratch::<f64>(m, n, bs, par.shape(), Default::default())
	});
	let mut mem = MemBuffer::new(req);
	let mut largest = 0;
	for pass in 0..2 {
		let mut a = a.clone();
		// SAFETY: `a` holds `(m - 1) * rs + (n - 1) * cs + 1` elements, and the
		// strides address distinct elements (`rs >= n * cs` or `cs >= m * rs` in
		// every caller)
		let view = unsafe {
			MatMut::from_raw_parts_mut(
				a.as_mut_ptr(),
				m,
				n,
				rs as isize,
				cs as isize,
			)
		};
		let coeff = MatMut::from_column_major_slice_mut(&mut coeff, bs, k);
		LARGEST.with(|l| l.set(0));
		COUNTING.with(|c| c.set(pass == 1));
		qr::qr_in_place(
			view,
			coeff,
			par.reborrow(),
			MemStack::new(&mut mem),
			Default::default(),
		);
		COUNTING.with(|c| c.set(false));
		largest = LARGEST.with(|l| l.get());
	}
	largest
}

#[test]
fn row_major_qr_does_not_copy_the_matrix() {
	for n in [64, 96, 128, 176] {
		let whole = n * n * size_of::<f64>();
		// contiguous rows, then a padded leading dimension
		for (rs, cs) in [(n, 1), (n + 3, 1)] {
			let largest = largest_allocation(n, n, (rs, cs), None, Par::Seq);
			assert!(
				largest < whole,
				"{n}x{n}, strides ({rs}, {cs}): an allocation of {largest} \
				 bytes, the whole matrix is {whole}"
			);
		}
	}
	// the control: these keep the copy, and the counter sees it
	let keeps_copy = |m: usize, n: usize, strides: (usize, usize), stack: Option<StackReq>, par: Par<'_>, why: &str| {
		let whole = m * n * size_of::<f64>();
		let largest = largest_allocation(m, n, strides, stack, par);
		assert!(
			largest >= whole,
			"{m}x{n} ({why}): largest allocation {largest} bytes, the copy is \
			 {whole}"
		);
	};
	keeps_copy(256, 64, (64, 1), None, Par::Seq, "tall");
	keeps_copy(256, 256, (256, 1), None, Par::Seq, "square from 192 up");
	let bs = qr::recommended_block_size::<f64>(128, 128);
	// the stack of the query before the dual, too small for its panels
	let small = temp_mat_scratch::<f64>(bs, 128);
	keeps_copy(128, 128, (128, 1), Some(small), Par::Seq, "small stack");
	keeps_copy(128, 128, (256, 2), None, Par::Seq, "neither stride 1");
	#[cfg(feature = "threadteam")]
	faer::team::with_team(4, |par| keeps_copy(128, 128, (128, 1), None, par, "team"));
}
