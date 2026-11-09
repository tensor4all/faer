#![cfg(feature = "perf-warn")]
//! The `perf-warn` call site of `lu::partial_pivoting`: `lu_in_place` warns once per process,
//! and only for a matrix whose strides are neither of the two layouts it recommends.
//!
//! The warn-once flag is a process-wide atomic, so a unit test next to the predicate cannot own
//! it: whichever other test trips it first decides. This file is its own integration-test binary,
//! which is the isolation the flag needs, and it drives the call site rather than the predicate
//! -- restoring the old "not column-major" condition at the call site keeps the predicate test
//! green and fails here.

use std::sync::{Arc, Mutex, OnceLock};

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::factor::{
	self as plu, PartialPivLuParams,
};
use faer::{Auto, MatMut, Par};

/// The `faer_perf` messages this process produced.
struct Capture(Arc<Mutex<Vec<String>>>);

impl log::Log for Capture {
	fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
		metadata.target() == "faer_perf"
	}

	fn log(&self, record: &log::Record<'_>) {
		if self.enabled(record.metadata()) {
			self.0.lock().unwrap().push(record.args().to_string());
		}
	}

	fn flush(&self) {}
}

/// Install the only logger in this test binary, once.
fn messages() -> &'static Arc<Mutex<Vec<String>>> {
	static RECORDS: OnceLock<Arc<Mutex<Vec<String>>>> = OnceLock::new();
	RECORDS.get_or_init(|| {
		let records = Arc::new(Mutex::new(Vec::new()));
		log::set_boxed_logger(Box::new(Capture(records.clone())))
			.expect("this binary installs the process logger once");
		log::set_max_level(log::LevelFilter::Warn);
		records
	})
}

/// Factor an `nrows x ncols` matrix held at the given strides.
fn factor(
	data: &mut [f64],
	nrows: usize,
	ncols: usize,
	row_stride: isize,
	col_stride: isize,
) {
	let n = Ord::min(nrows, ncols);
	let mut perm = vec![0usize; n];
	let mut perm_inv = vec![0usize; n];
	let par = Par::Seq;
	let params = <PartialPivLuParams as Auto<f64>>::auto();
	let mut mem = MemBuffer::new(plu::lu_in_place_scratch::<usize, f64>(
		nrows,
		ncols,
		par.shape(),
		params.into(),
	));
	// SAFETY: each caller passes a buffer that holds every index the strides reach.
	let a = unsafe {
		MatMut::from_raw_parts_mut(
			data.as_mut_ptr(),
			nrows,
			ncols,
			row_stride,
			col_stride,
		)
	};
	let _ = plu::lu_in_place(
		a,
		&mut perm,
		&mut perm_inv,
		par,
		MemStack::new(&mut mem),
		params.into(),
	);
}

#[test]
fn the_warning_fires_once_and_only_for_generic_strides() {
	let records = messages();
	records.lock().unwrap().clear();

	// Contiguous row-major first, while the warn-once flag is still clear: this is the layout the
	// old call-site condition warned on, and the regression this file exists for.
	let mut row_major: Vec<f64> = (1..=9).map(|v| v as f64).collect();
	factor(&mut row_major, 3, 3, 3, 1);
	assert_eq!(
		records.lock().unwrap().len(),
		0,
		"contiguous row-major must stay silent"
	);

	// Contiguous column-major.
	let mut col_major = vec![1.0f64; 9];
	factor(&mut col_major, 3, 3, 1, 3);
	assert_eq!(
		records.lock().unwrap().len(),
		0,
		"contiguous column-major must stay silent"
	);

	// Neither stride walks a line: a 3x3 view with row stride 2 and column stride 6.
	let mut generic = vec![1.0f64; 17];
	factor(&mut generic, 3, 3, 2, 6);
	{
		let seen = records.lock().unwrap();
		assert_eq!(
			seen.len(),
			1,
			"a generic-strided input must warn exactly once: {seen:?}"
		);
		assert!(
			seen[0].contains("generic strides"),
			"unexpected message: {:?}",
			seen[0]
		);
	}

	// Warn once per process: a second generic-strided factor adds nothing.
	factor(&mut generic, 3, 3, 2, 6);
	assert_eq!(
		records.lock().unwrap().len(),
		1,
		"the warning is emitted once per process"
	);
}
