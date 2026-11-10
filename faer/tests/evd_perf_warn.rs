#![cfg(feature = "perf-warn")]
//! The `perf-warn` call site of `self_adjoint_evd` (tensor4all/faer#314): a row-major `u` is
//! computed in a column-major stand-in and no longer costs a strided write per entry, so it must
//! not warn; a `u` with generic strides still takes the in-place path and warns, once per process.
//!
//! The warn-once flag is a process-wide atomic, so this file is its own integration-test binary
//! (as `lu_perf_warn.rs` is for the LU call site), and it drives the call site itself: restoring
//! the old "not column-major" condition fails here.

use std::sync::{Arc, Mutex, OnceLock};

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{
	ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch,
};
use faer::{Mat, MatMut, Par};

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

/// The eigendecomposition of a fixed symmetric `3 x 3` matrix into a `u` held in `data` at the
/// given strides.
fn evd(data: &mut [f64], row_stride: isize, col_stride: isize) {
	let n = 3;
	let a = Mat::<f64>::from_fn(n, n, |i, j| {
		if i == j {
			4.0 + i as f64
		} else {
			1.0 / (1.0 + (i + j) as f64)
		}
	});
	let mut s = Diag::<f64>::zeros(n);
	let params = Default::default();
	let mut mem = MemBuffer::new(self_adjoint_evd_scratch::<f64>(
		n,
		ComputeEigenvectors::Yes,
		Par::Seq.shape(),
		params,
	));
	// SAFETY: each caller passes a buffer that holds every index the strides reach.
	let u = unsafe {
		MatMut::from_raw_parts_mut(
			data.as_mut_ptr(),
			n,
			n,
			row_stride,
			col_stride,
		)
	};
	self_adjoint_evd(
		a.as_ref(),
		s.as_mut(),
		Some(u),
		Par::Seq,
		MemStack::new(&mut mem),
		params,
	)
	.unwrap();
}

#[test]
fn the_warning_fires_once_and_only_for_generic_strides() {
	let records = messages();
	records.lock().unwrap().clear();

	// Contiguous row-major first, while the warn-once flag is still clear: this is the layout the
	// old call-site condition warned on, and the regression this file exists for.
	let mut row_major = vec![0.0f64; 9];
	evd(&mut row_major, 3, 1);
	{
		// one guard: a second `lock` in the message would deadlock exactly when this fails
		let seen = records.lock().unwrap();
		assert_eq!(
			seen.len(),
			0,
			"a contiguous row-major u must stay silent: {seen:?}"
		);
	}

	// Contiguous column-major.
	let mut col_major = vec![0.0f64; 9];
	evd(&mut col_major, 1, 3);
	assert_eq!(
		records.lock().unwrap().len(),
		0,
		"a contiguous column-major u must stay silent"
	);

	// Neither stride walks a line: a 3x3 view with row stride 2 and column stride 6.
	let mut generic = vec![0.0f64; 17];
	evd(&mut generic, 2, 6);
	{
		let seen = records.lock().unwrap();
		assert_eq!(
			seen.len(),
			1,
			"a generic-strided u must warn exactly once: {seen:?}"
		);
		assert!(
			seen[0].contains("generic strides"),
			"unexpected message: {:?}",
			seen[0]
		);
	}

	// Warn once per process: a second generic-strided call adds nothing.
	evd(&mut generic, 2, 6);
	assert_eq!(
		records.lock().unwrap().len(),
		1,
		"the warning is emitted once per process"
	);
}
