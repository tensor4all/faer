//! #271: `eigenvectors_from_schur_real` / `_cplx` on a caller-held team of two, with the output,
//! `q` and the scratch sized by `eigenvectors_from_schur_scratch` built beforehand, allocate
//! nothing on a warm call. The recursive shifted triangular solve calls `matmul` on one column
//! per recursion step; on a column-major (F) `t` that is the column-major matrix-vector product,
//! whose team branch built its `nrows x width` partial sums with a `MemBuffer::new` on every
//! call (`linalg/matmul/mod.rs`, `matvec_colmajor`). A row-major (C) `t` takes the row-major
//! product, which has no partials, so it is covered here as the control.
//!
//! Whether a product of a given size runs on the team is the cost model's decision, which
//! depends on the host class and the scalar: on the Zen 5 / AVX2 hosts of the issue `n = 257`
//! dispatches for f64, c32 and c64 (not f32, whose product is faster and stays serial), on the
//! Apple M5 only from about `n = 512`. `n = 1024` dispatches for all four on both, so it is the
//! size that catches the defect for f32 and on aarch64.
//!
//! The allocator counts every allocation of the process while the call runs; the inputs, the
//! output, the scratch and the residual checks are outside the counted interval.
//!
//! The back-transform by `q` is a full `matmul`, which off x86_64 (and on x86_64 without AVX2 and
//! FMA) falls back to the `gemm` crate, whose packing buffer is allocated per call above its
//! retention cap (#56). The case with `q` is therefore ignored off x86_64, as in
//! `evd_svd_vectors_no_alloc.rs`; the case without `q` runs everywhere.
#![cfg(all(feature = "std", feature = "threadteam"))]
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd;
use faer::traits::ComplexField;
use faer::traits::math_utils::*;
use faer::{Mat, MatMut, MatRef, Par, c32, c64};
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

/// The tests of this binary share the counter, so they run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

trait Scalar: ComplexField {
	/// The upper triangular fixture of the issue: diagonal `i + 1` (plus `0.1 i` imaginary),
	/// `0.03` (`0.03 + 0.02i`) above it.
	fn point(i: usize, j: usize) -> Self;
	fn run(
		t: MatRef<'_, Self>,
		q: Option<MatRef<'_, Self>>,
		x: MatMut<'_, Self>,
		par: Par<'_>,
		stack: &mut MemStack,
	);
}
macro_rules! real {
	($t:ty) => {
		impl Scalar for $t {
			fn point(i: usize, j: usize) -> Self {
				if i == j {
					(i + 1) as Self
				} else if i < j {
					0.03
				} else {
					0.0
				}
			}

			fn run(
				t: MatRef<'_, Self>,
				q: Option<MatRef<'_, Self>>,
				x: MatMut<'_, Self>,
				par: Par<'_>,
				stack: &mut MemStack,
			) {
				evd::eigenvectors_from_schur_real(
					t,
					q,
					x,
					par,
					stack,
					Default::default(),
				)
				.unwrap();
			}
		}
	};
}
macro_rules! complex {
	($t:ty, $r:ty) => {
		impl Scalar for $t {
			fn point(i: usize, j: usize) -> Self {
				if i == j {
					Self::new((i + 1) as $r, 0.1 as $r * i as $r)
				} else if i < j {
					Self::new(0.03, 0.02)
				} else {
					Self::new(0.0, 0.0)
				}
			}

			fn run(
				t: MatRef<'_, Self>,
				q: Option<MatRef<'_, Self>>,
				x: MatMut<'_, Self>,
				par: Par<'_>,
				stack: &mut MemStack,
			) {
				evd::eigenvectors_from_schur_cplx(
					t,
					q,
					x,
					par,
					stack,
					Default::default(),
				)
				.unwrap();
			}
		}
	};
}
real!(f32);
real!(f64);
complex!(c32, f32);
complex!(c64, f64);

/// Allocations of the second of two identical calls on the team (the first is the warm-up), for
/// a column-major (`row_major == false`) or row-major `t` and `x`. Checks the eigenvectors of
/// the counted call independently: `t x - x diag(t)` against `256 n eps |t|` and unit columns.
fn warm_allocations<T: Scalar>(
	n: usize,
	row_major: bool,
	with_q: bool,
	par: &mut Par<'_>,
) -> usize {
	let t = Mat::<T>::from_fn(n, n, T::point);
	let t_rm = Mat::<T>::from_fn(n, n, |i, j| T::point(j, i));
	let q = Mat::<T>::identity(n, n);
	let mut x = Mat::<T>::zeros(n, n);
	let mut mem = MemBuffer::new(evd::eigenvectors_from_schur_scratch::<T>(n));
	let mut call = |x: &mut Mat<T>, par: &mut Par<'_>| {
		let t = if row_major {
			t_rm.as_ref().transpose()
		} else {
			t.as_ref()
		};
		let x = if row_major {
			x.as_mut().transpose_mut()
		} else {
			x.as_mut()
		};
		T::run(
			t,
			with_q.then(|| q.as_ref()),
			x,
			par.reborrow(),
			MemStack::new(&mut mem),
		);
	};
	call(&mut x, par);
	ALLOCATIONS.store(0, Ordering::SeqCst);
	COUNTING.store(true, Ordering::SeqCst);
	call(&mut x, par);
	COUNTING.store(false, Ordering::SeqCst);
	let count = ALLOCATIONS.load(Ordering::SeqCst);

	let x = if row_major {
		x.as_ref().transpose()
	} else {
		x.as_ref()
	};
	let mut residual = &t * x;
	for j in 0..n {
		for i in 0..n {
			residual[(i, j)] =
				sub(&residual[(i, j)], &mul(&x[(i, j)], &t[(j, j)]));
		}
	}
	assert!(
		residual.norm_l2()
			<= from_f64::<T::Real>(256.0 * n as f64)
				* eps::<T::Real>()
				* t.norm_l2()
	);
	for j in 0..n {
		assert!(
			abs(&sub(&x.col(j).norm_l2(), &one::<T::Real>()))
				< from_f64::<T::Real>(256.0) * eps::<T::Real>()
		);
	}
	count
}

fn check_all(with_q: bool) {
	let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
	faer::team::with_team(2, |mut par| {
		let mut failures = Vec::new();
		for n in [64, 257, 1024] {
			for row_major in [false, true] {
				let counts = [
					(
						"f32",
						warm_allocations::<f32>(n, row_major, with_q, &mut par),
					),
					(
						"f64",
						warm_allocations::<f64>(n, row_major, with_q, &mut par),
					),
					(
						"c32",
						warm_allocations::<c32>(n, row_major, with_q, &mut par),
					),
					(
						"c64",
						warm_allocations::<c64>(n, row_major, with_q, &mut par),
					),
				];
				let layout = if row_major { "C" } else { "F" };
				println!("n = {n} {layout}: {counts:?}");
				for (name, count) in counts {
					if count != 0 {
						failures.push(format!(
							"{name} n = {n} {layout}: {count} allocations"
						));
					}
				}
			}
		}
		assert!(failures.is_empty(), "warm calls allocated: {failures:#?}");
	});
}

/// `q = None`: the eigenvectors of `t` itself, no full product.
#[test]
fn schur_eigenvectors_on_a_team_allocate_nothing() {
	check_all(false);
}

/// With `q`: the back-transform is a full `matmul`, which reaches the `gemm` crate off x86_64.
#[test]
#[cfg_attr(
	not(target_arch = "x86_64"),
	ignore = "off x86_64 the back-transform reaches the gemm crate, which allocates per call (#56)"
)]
fn schur_eigenvectors_with_q_on_a_team_allocate_nothing() {
	#[cfg(target_arch = "x86_64")]
	if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
		// libtest has no skip result, so this prints and returns: the harness reports the
		// test as passed without having exercised the kernel path. CI runs on x86_64 with
		// AVX2 and does exercise it; a local run on a host without AVX2/FMA carries no
		// information about the allocation count, which is why the line below is printed
		// rather than swallowed. Do not read a green run here as a measurement.
		eprintln!(
			"SKIPPED schur_eigenvectors_with_q_on_a_team_allocate_nothing: this host has no \
			 avx2/fma, so the back-transform takes the gemm crate's allocating path and the \
			 zero-allocation claim is not observable here"
		);
		return;
	}
	check_all(true);
}
