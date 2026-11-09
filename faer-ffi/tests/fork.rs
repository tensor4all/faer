//! A child forked from a process that has used a team handle creates its own handle and uses it
//! (tensor4all/faer#333).
//!
//! The handle's pool has dedicated workers, so `destroy` joins them and nothing is left in
//! threadteam's process-wide cache for a child to inherit. Before that, `destroy` parked the
//! workers in the cache, the child's `create` took the parent's cached Rayon pool (whose threads do
//! not exist in the child) and waited for them in the affinity broadcast forever.
//!
//! Each child runs under `alarm`, so a regression is a child killed by `SIGALRM`, reported as a
//! failure, rather than a hung test runner. Linux only: the fork is plain POSIX, but the hang being
//! guarded is in threadteam's Linux affinity path.
#![cfg(target_os = "linux")]

use core::ffi::c_void;
use faer_ffi::linalg::libfaer_v0_23_matmul_f64;
use faer_ffi::{
	Accum, MatMut, MatRef, Scalar, Team, libfaer_v0_23_team_create,
	libfaer_v0_23_team_destroy,
};

unsafe extern "C" {
	fn fork() -> i32;
	fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
	fn alarm(seconds: u32) -> u32;
	fn _exit(status: i32) -> !;
}

const N: usize = 192;
const WIDTH: usize = 4;

/// `c = a * a` on `team`, and whether every entry matches the sequential product.
fn product_is_right(team: *const Team) -> bool {
	let a = faer::Mat::<f64>::from_fn(N, N, |i, j| {
		((i * 7 + j * 3) % 11) as f64 - 5.0
	});
	let mut want = faer::Mat::<f64>::zeros(N, N);
	faer::linalg::matmul::matmul(
		want.as_mut(),
		faer::Accum::Replace,
		a.as_ref(),
		a.as_ref(),
		1.0,
		faer::Par::Seq,
	);
	let mut c = faer::Mat::<f64>::zeros(N, N);
	let alpha = 1.0f64;
	let a_ref = MatRef {
		ptr: a.as_ptr() as *const c_void,
		nrows: N,
		ncols: N,
		row_stride: a.row_stride(),
		col_stride: a.col_stride(),
	};
	let c_mut = MatMut {
		ptr: c.as_ptr_mut() as *mut c_void,
		nrows: N,
		ncols: N,
		row_stride: c.row_stride(),
		col_stride: c.col_stride(),
	};
	unsafe {
		libfaer_v0_23_matmul_f64(
			c_mut,
			Accum::Replace,
			a_ref,
			a_ref,
			&raw const alpha as *const Scalar,
			team,
		)
	};
	(0..N).all(|i| (0..N).all(|j| c[(i, j)] == want[(i, j)]))
}

/// Forks; the child creates a handle of its own, multiplies on it, destroys it and exits 0 when
/// the product is right. Returns the child's raw wait status.
fn child_creates_its_own_handle() -> i32 {
	let pid = unsafe { fork() };
	assert!(pid >= 0, "fork failed");
	if pid == 0 {
		unsafe { alarm(20) };
		let team = unsafe { libfaer_v0_23_team_create(WIDTH) };
		let ok = product_is_right(team);
		unsafe { libfaer_v0_23_team_destroy(team) };
		unsafe { _exit(if ok { 0 } else { 1 }) };
	}
	let mut status = 0;
	assert_eq!(unsafe { waitpid(pid, &mut status, 0) }, pid);
	status
}

/// A wait status as the test reads it: `exit 0`, `exit k` or `killed by signal s`.
fn describe(status: i32) -> String {
	if status & 0x7F == 0 {
		format!("exit {}", (status >> 8) & 0xFF)
	} else {
		format!("killed by signal {}", status & 0x7F)
	}
}

/// Threads of this process, from `/proc/self/task`.
fn threads() -> usize {
	std::fs::read_dir("/proc/self/task").unwrap().count()
}

/// Waits, for at most `bound`, until the process is back to `want` threads and reports how long that
/// took; `None` if it never got there.
///
/// `destroy` joins the workers, but a joined task's `/proc/self/task` entry outlives the join: the
/// kernel still has the worker's exit path to run, and `std`'s `join` returns once the worker's
/// closure has finished, not once the task is gone. A single read straight after `destroy`
/// therefore sometimes samples that teardown instead of a leak and reports "3 != 2"
/// (tensor4all/faer#348: 20 of 3000 create/use/destroy cycles, the entry gone 4 us later at the
/// median and 2 ms at the worst). Waiting for the count to come back keeps what the check is for: a
/// pool left parked in threadteam's cache never leaves, so this times out instead of passing.
fn threads_settle(
	want: usize,
	bound: std::time::Duration,
) -> Option<std::time::Duration> {
	let start = std::time::Instant::now();
	loop {
		if threads() == want {
			return Some(start.elapsed());
		}
		if start.elapsed() > bound {
			return None;
		}
		std::hint::spin_loop();
	}
}

#[test]
fn a_forked_child_creates_and_uses_a_handle_of_its_own() {
	// `destroy` joins the handle's workers: the process is back to the threads it had. A pool of
	// shared workers would leave them parked in threadteam's cache instead, which is exactly what a
	// child forked afterwards inherits without the threads.
	let before = threads();
	let team = unsafe { libfaer_v0_23_team_create(WIDTH) };
	assert!(product_is_right(team));
	assert!(
		threads() > before,
		"a {WIDTH}-wide handle started no thread"
	);
	unsafe { libfaer_v0_23_team_destroy(team) };
	assert!(
		threads_settle(before, std::time::Duration::from_secs(1)).is_some(),
		"destroy left the handle's workers running: {} threads, {before} before the handle",
		threads()
	);

	// a handle that is still alive in the parent when it forks: the child cannot use that one
	// (its workers are not in the child), but a handle it creates itself is independent of it
	let live = unsafe { libfaer_v0_23_team_create(WIDTH) };
	assert!(product_is_right(live));
	let status = child_creates_its_own_handle();
	assert_eq!(
		describe(status),
		"exit 0",
		"live handle at fork: the child's own handle"
	);
	unsafe { libfaer_v0_23_team_destroy(live) };

	// the review's sequence: create, use, destroy, fork, create in the child
	let team = unsafe { libfaer_v0_23_team_create(WIDTH) };
	assert!(product_is_right(team));
	unsafe { libfaer_v0_23_team_destroy(team) };
	let status = child_creates_its_own_handle();
	assert_eq!(
		describe(status),
		"exit 0",
		"destroy, fork, create: the child's handle"
	);
}
