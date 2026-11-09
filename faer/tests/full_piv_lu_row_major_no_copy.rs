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
use faer::traits::ComplexField;
use faer::traits::math_utils::zero;
use faer::{Auto, MatMut, Par, c64, cx128, fx128};
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

/// The first pivot `lu_in_place` picks on a zero matrix with two entries of
/// magnitude 2, row-major or column-major, at `max_rank = Some(1)`: `(row,
/// column, transposition count)`.
fn first_pivot(
	m: usize,
	n: usize,
	ties: [(usize, usize); 2],
	row_major: bool,
) -> (usize, usize, usize) {
	let mut a = vec![0.0f64; m * n];
	for (i, j) in ties {
		a[if row_major { i * n + j } else { i + j * m }] = 2.0;
	}
	let view = if row_major {
		MatMut::from_row_major_slice_mut(&mut a, m, n)
	} else {
		MatMut::from_column_major_slice_mut(&mut a, m, n)
	};
	let (mut p, mut pi, mut q, mut qi) = (
		vec![0usize; m],
		vec![0usize; m],
		vec![0usize; n],
		vec![0usize; n],
	);
	let mut mem = MemBuffer::new(flu::lu_in_place_scratch::<usize, f64>(
		m,
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
}

/// `lu_in_place`, "Pivot ties": the matrix that is searched is the input when
/// `|row_stride| < |col_stride|` (or its column-major copy) and its transpose
/// otherwise; a tie goes to the smallest column, then the smallest row, of the
/// matrix searched.
#[test]
fn full_piv_lu_tie_order_depends_on_the_layout() {
	// two columns: the searched column wins
	let ties = [(1, 2), (0, 4)];
	assert_eq!(first_pivot(64, 64, ties, false), (1, 2, 2));
	assert_eq!(first_pivot(64, 64, ties, true), (0, 4, 1));
	// a wide row-major matrix is never copied, so it is searched transposed
	// too (tensor4all/faer#342 review)
	assert_eq!(first_pivot(8, 16, ties, false), (1, 2, 2));
	assert_eq!(first_pivot(8, 16, ties, true), (0, 4, 1));
	// a tall row-major matrix that takes the copy is searched column-major
	assert_eq!(first_pivot(256, 32, ties, true), (1, 2, 2));

	// one searched column, two rows: the smaller row wins, across the SIMD
	// accumulators (tensor4all/faer#342 review: (2, 8) was returned)
	assert_eq!(first_pivot(64, 64, [(2, 6), (2, 8)], true), (2, 6, 2));
	assert_eq!(first_pivot(64, 64, [(6, 2), (8, 2)], false), (6, 2, 2));
}

/// First pivot `(row, column, transpositions)` on a zero `m x n` complex matrix
/// with two entries, at `max_rank = Some(1)`.
fn first_pivot_complex<T: ComplexField>(
	m: usize,
	n: usize,
	ties: [(usize, usize, T); 2],
	row_major: bool,
) -> (usize, usize, usize) {
	let mut a = vec![zero::<T>(); m * n];
	for (i, j, v) in ties {
		a[if row_major { i * n + j } else { i + j * m }] = v;
	}
	let view = if row_major {
		MatMut::from_row_major_slice_mut(&mut a, m, n)
	} else {
		MatMut::from_column_major_slice_mut(&mut a, m, n)
	};
	let (mut p, mut pi, mut q, mut qi) = (
		vec![0usize; m],
		vec![0usize; m],
		vec![0usize; n],
		vec![0usize; n],
	);
	let mut mem = MemBuffer::new(flu::lu_in_place_scratch::<usize, T>(
		m,
		n,
		Par::Seq.shape(),
		Default::default(),
	));
	let params = flu::FullPivLuParams {
		max_rank: Some(1),
		..<flu::FullPivLuParams as Auto<T>>::auto()
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
}

/// `lu_in_place`, "Pivot ties" for a complex `T`: the index order is not part
/// of the contract, and which of the two documented scores the search uses is
/// visible. `(0, 1)` holds `3 + 4i` and `(2, 1)` holds `5`, so `|re| + |im|` is
/// larger at `(0, 1)` and the largest single component at `(2, 1)`. The scalar
/// search compares the sum, the SIMD one the components; both fixtures below
/// are asserted as a pair, so the test needs no knowledge of which backend the
/// build dispatches to (tensor4all/faer#342 review).
#[test]
fn full_piv_lu_complex_tie_follows_the_search_score() {
	let ties = [(0, 1, c64::new(3.0, 4.0)), (2, 1, c64::new(5.0, 0.0))];
	let swapped = [(0, 1, c64::new(5.0, 0.0)), (2, 1, c64::new(3.0, 4.0))];
	let first = first_pivot_complex(16, 16, ties, false);
	let second = first_pivot_complex(16, 16, swapped, false);
	// Component score keeps the larger component: (2, 1) then (0, 1). Sum score
	// keeps the larger |re| + |im|: (0, 1) then (2, 1). Index order would keep
	// (0, 1) in both, so either arm below is a score and the pair rules it out.
	assert!(
		(first, second) == ((2, 1, 2), (0, 1, 1))
			|| (first, second) == ((0, 1, 1), (2, 1, 2)),
		"first pivot {first:?}, swapped {second:?}"
	);
}

/// The component score belongs to the native `c32` and `c64` vectors only; a
/// generic `Complex<T>` such as `cx128` sums `|re| + |im|` on its SIMD path too
/// (tensor4all/faer#342 review r5).
#[test]
fn full_piv_lu_cx128_pivot_uses_the_sum_score() {
	let c = |re: f64, im: f64| cx128 {
		re: fx128::from_f64(re),
		im: fx128::from_f64(im),
	};
	let ties = [(0, 1, c(3.0, 4.0)), (2, 1, c(5.0, 0.0))];
	let swapped = [(0, 1, c(5.0, 0.0)), (2, 1, c(3.0, 4.0))];
	// the larger |re| + |im| wins in both fixtures, column-major (SIMD search
	// on the input) and row-major (SIMD search on the transpose)
	for row_major in [false, true] {
		assert_eq!(
			(
				first_pivot_complex(16, 16, ties, row_major),
				first_pivot_complex(16, 16, swapped, row_major),
			),
			((0, 1, 1), (2, 1, 2)),
			"row_major = {row_major}"
		);
	}
}
