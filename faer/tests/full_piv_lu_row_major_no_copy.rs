//! tensor4all/faer#314: a small square row-major matrix is factored by the
//! full-pivoting LU in place, through the transpose dual, and not through a
//! column-major copy of the whole matrix. The allocator records the largest
//! single allocation the calling thread makes while `lu_in_place` runs; the
//! copy is one `m * n` buffer, so the dual path stays below that size.
//! The shapes that still take the copy are the control: there the same counter
//! must see it.
#![cfg(all(feature = "std", feature = "threadteam"))]
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::full_pivoting::factor as flu;
use faer::{Auto, MatMut, Par};
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
/// factored (permutations and scratch are allocated before).
fn largest_allocation_strided(
	m: usize,
	n: usize,
	(rs, cs): (usize, usize),
	mut par: Par<'_>,
) -> usize {
	let len = (m - 1) * rs + (n - 1) * cs + 1;
	let mut a: Vec<f64> = (0..len)
		.map(|k| {
			((k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64
				/ (1u64 << 53) as f64
				- 0.5
		})
		.collect();
	let (mut p, mut pi, mut q, mut qi) = (
		vec![0usize; m],
		vec![0usize; m],
		vec![0usize; n],
		vec![0usize; n],
	);
	let mut mem = MemBuffer::new(flu::lu_in_place_scratch::<usize, f64>(
		m,
		n,
		par.shape(),
		Default::default(),
	));
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
	LARGEST.with(|l| l.set(0));
	COUNTING.with(|c| c.set(true));
	flu::lu_in_place(
		view,
		&mut p,
		&mut pi,
		&mut q,
		&mut qi,
		par.reborrow(),
		MemStack::new(&mut mem),
		Default::default(),
	);
	COUNTING.with(|c| c.set(false));
	LARGEST.with(|l| l.get())
}
/// [`largest_allocation_strided`] for a contiguous row-major matrix.
fn largest_allocation(m: usize, n: usize, par: Par<'_>) -> usize {
	largest_allocation_strided(m, n, (n, 1), par)
}

#[test]
fn small_square_row_major_full_piv_lu_does_not_copy_the_matrix() {
	let run = |mut par: Par<'_>| {
		for n in [64, 96, 127] {
			let whole = n * n * size_of::<f64>();
			let largest = largest_allocation(n, n, par.reborrow());
			assert!(
				largest < whole,
				"{n}x{n}: an allocation of {largest} bytes, the whole matrix \
				 is {whole}"
			);
		}
		// contiguous rows with a padded leading dimension are still row-major
		let whole = 96 * 96 * size_of::<f64>();
		let largest =
			largest_allocation_strided(96, 96, (100, 1), par.reborrow());
		assert!(
			largest < whole,
			"96x96, strides (100, 1): an allocation of {largest} bytes, the \
			 whole matrix is {whole}"
		);
		// the control: these shapes keep the copy, and the counter sees it
		for (m, n) in [(128, 128), (256, 32)] {
			let whole = m * n * size_of::<f64>();
			let largest = largest_allocation(m, n, par.reborrow());
			assert!(
				largest >= whole,
				"{m}x{n}: largest allocation {largest} bytes, the copy is \
				 {whole}"
			);
		}
		// a generic-stride view (neither stride 1) is not row-major: it keeps
		// the copy (tensor4all/faer#342 review), where in place it would miss
		// the fused SIMD update
		for (rs, cs) in [(2, 192), (96 * 2, 2)] {
			let largest =
				largest_allocation_strided(96, 96, (rs, cs), par.reborrow());
			assert!(
				largest >= whole,
				"96x96, strides ({rs}, {cs}): largest allocation {largest} \
				 bytes, the copy is {whole}"
			);
		}
	};
	run(Par::Seq);
	faer::team::with_team(4, |par| run(par));
}

/// The pivot search keeps the first strict maximum of the matrix it factors:
/// the smallest column, then the smallest row. The transpose dual factors the
/// transpose, so a row-major input breaks the same tie by the smallest row
/// first (`lu_in_place`, "Pivot ties").
#[test]
fn full_piv_lu_tie_order_depends_on_the_layout() {
	let n = 64;
	let at = |i: usize, j: usize| {
		if (i, j) == (1, 2) || (i, j) == (0, 4) {
			2.0
		} else {
			0.0
		}
	};
	let first_pivot = |row_major: bool| {
		let mut a = vec![0.0f64; n * n];
		for i in 0..n {
			for j in 0..n {
				a[if row_major { i * n + j } else { i + j * n }] = at(i, j);
			}
		}
		let view = if row_major {
			MatMut::from_row_major_slice_mut(&mut a, n, n)
		} else {
			MatMut::from_column_major_slice_mut(&mut a, n, n)
		};
		let (mut p, mut pi, mut q, mut qi) = (
			vec![0usize; n],
			vec![0usize; n],
			vec![0usize; n],
			vec![0usize; n],
		);
		let mut mem = MemBuffer::new(flu::lu_in_place_scratch::<usize, f64>(
			n,
			n,
			Par::Seq.shape(),
			Default::default(),
		));
		let params = flu::FullPivLuParams {
			max_rank: Some(1),
			..<flu::FullPivLuParams as Auto<f64>>::auto()
		};
		let (info, _, _) = flu::lu_in_place(
			view,
			&mut p,
			&mut pi,
			&mut q,
			&mut qi,
			Par::Seq,
			MemStack::new(&mut mem),
			params.into(),
		);
		(p[0], q[0], info.transposition_count)
	};
	assert_eq!(first_pivot(false), (1, 2, 2));
	assert_eq!(first_pivot(true), (0, 4, 1));
}
