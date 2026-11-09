//! #322: `factorize_symbolic_{lu,qr}_with_column_ordering` sized its `MemBuffer`
//! as `StackReq::or(colamd::order_scratch(..), ..)` even when the caller
//! supplies the ordering, although COLAMD only runs in the ordering-less
//! branch. The allocator records the largest single allocation of the counting
//! thread while the call runs (the guard is the one `svd_values_no_alloc.rs`
//! uses).
#![cfg(feature = "std")]
use faer::perm::Perm;
use faer::sparse::SymbolicSparseColMat;
use faer::sparse::linalg::{SupernodalThreshold, colamd, lu, qr};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// Both the flag and the running maximum are thread-local: the tests below run
// in parallel, and a process-wide counter let one test's window reset the
// value the other was about to read.
thread_local! {
	static COUNTING: Cell<bool> = const { Cell::new(false) };
	static LARGEST: Cell<usize> = const { Cell::new(0) };
}

struct Counting;
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		let _ = COUNTING.try_with(|counting| {
			if counting.get() {
				let _ = LARGEST.try_with(|largest| {
					largest.set(largest.get().max(layout.size()));
				});
			}
		});
		unsafe { System.alloc(layout) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

fn largest_during(f: impl FnOnce()) -> usize {
	LARGEST.with(|l| l.set(0));
	COUNTING.with(|c| c.set(true));
	f();
	COUNTING.with(|c| c.set(false));
	LARGEST.with(Cell::get)
}

/// Diagonal plus three off-diagonals per column, as in the issue's reproducer.
fn sample(n: usize) -> SymbolicSparseColMat<usize> {
	let mut cols: Vec<Vec<usize>> = (0..n)
		.map(|j| {
			let mut c =
				vec![j, (j * 7 + 3) % n, (j * 13 + 5) % n, (j * 31 + 11) % n];
			c.sort();
			c.dedup();
			c
		})
		.collect();
	let mut ptr = vec![0usize];
	let mut idx = Vec::new();
	for c in cols.drain(..) {
		idx.extend(c);
		ptr.push(idx.len());
	}
	SymbolicSparseColMat::new_checked(n, n, ptr, None, idx)
}

fn colamd_bytes(a: &SymbolicSparseColMat<usize>) -> usize {
	colamd::order_scratch::<usize>(a.nrows(), a.ncols(), a.compute_nnz())
		.size_bytes()
}

fn params() -> (lu::LuSymbolicParams<'static>, qr::QrSymbolicParams<'static>) {
	let mut lu_params = lu::LuSymbolicParams::default();
	lu_params.supernodal_flop_ratio_threshold =
		SupernodalThreshold::FORCE_SIMPLICIAL;
	let mut qr_params = qr::QrSymbolicParams::default();
	qr_params.supernodal_flop_ratio_threshold =
		SupernodalThreshold::FORCE_SIMPLICIAL;
	(lu_params, qr_params)
}

#[test]
fn explicit_column_ordering_does_not_size_the_colamd_workspace() {
	let n = 100;
	let a = sample(n);
	let identity = Perm::<usize>::new_checked(
		(0..n).collect::<Vec<_>>().into_boxed_slice(),
		(0..n).collect::<Vec<_>>().into_boxed_slice(),
		n,
	);
	let colamd = colamd_bytes(&a);
	let (lu_params, qr_params) = params();
	let lu_largest = largest_during(|| {
		lu::factorize_symbolic_lu_with_column_ordering(
			a.as_ref(),
			lu_params,
			Some(identity.as_ref()),
		)
		.unwrap();
	});
	assert!(
		lu_largest < colamd,
		"symbolic LU with an explicit ordering allocated {lu_largest} B, \
		 which reaches the {colamd} B COLAMD workspace"
	);
	let qr_largest = largest_during(|| {
		qr::factorize_symbolic_qr_with_column_ordering(
			a.as_ref(),
			qr_params,
			Some(identity.as_ref()),
		)
		.unwrap();
	});
	assert!(
		qr_largest < colamd,
		"symbolic QR with an explicit ordering allocated {qr_largest} B, \
		 which reaches the {colamd} B COLAMD workspace"
	);
}

#[test]
fn absent_column_ordering_sizes_the_colamd_workspace() {
	let n = 100;
	let a = sample(n);
	let colamd = colamd_bytes(&a);
	let (lu_params, qr_params) = params();
	let lu_largest = largest_during(|| {
		lu::factorize_symbolic_lu_with_column_ordering(
			a.as_ref(),
			lu_params,
			None,
		)
		.unwrap();
	});
	assert!(
		lu_largest >= colamd,
		"symbolic LU without an ordering allocated {lu_largest} B, less than \
		 the {colamd} B COLAMD workspace it has to run in"
	);
	let qr_largest = largest_during(|| {
		qr::factorize_symbolic_qr_with_column_ordering(
			a.as_ref(),
			qr_params,
			None,
		)
		.unwrap();
	});
	assert!(
		qr_largest >= colamd,
		"symbolic QR without an ordering allocated {qr_largest} B, less than \
		 the {colamd} B COLAMD workspace it has to run in"
	);
}
