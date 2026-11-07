//! #270: all temporary matrices, including range scaling, come from the query.
//! Setup/output/Q/scratch and independent residual/norm checks are outside the
//! counter. Numerical fixtures and oracle are the deterministic issue #270
//! reproducer.
#![cfg(feature = "std")]
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd;
use faer::traits::ComplexField;
use faer::traits::math_utils::*;
use faer::{Mat, MatMut, MatRef, Par, c32, c64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicUsize};
static ACTIVE: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
struct Probe;
unsafe impl GlobalAlloc for Probe {
	unsafe fn alloc(&self, l: Layout) -> *mut u8 {
		if ACTIVE.load(Relaxed) {
			COUNT.fetch_add(1, Relaxed);
			BYTES.fetch_add(l.size(), Relaxed);
		}
		unsafe { System.alloc(l) }
	}

	unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
		if ACTIVE.load(Relaxed) {
			COUNT.fetch_add(1, Relaxed);
			BYTES.fetch_add(l.size(), Relaxed);
		}
		unsafe { System.alloc_zeroed(l) }
	}

	unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
		if ACTIVE.load(Relaxed) {
			COUNT.fetch_add(1, Relaxed);
			BYTES.fetch_add(n, Relaxed);
		}
		unsafe { System.realloc(p, l, n) }
	}

	unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
		unsafe { System.dealloc(p, l) }
	}
}
#[global_allocator]
static ALLOC: Probe = Probe;
trait Scalar: ComplexField {
	fn point(i: usize, j: usize) -> Self;
	fn run(
		t: MatRef<'_, Self>,
		q: MatRef<'_, Self>,
		x: MatMut<'_, Self>,
		par: Par<'_>,
		s: &mut MemStack,
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
				q: MatRef<'_, Self>,
				x: MatMut<'_, Self>,
				par: Par<'_>,
				s: &mut MemStack,
			) {
				evd::eigenvectors_from_schur_real(
					t,
					Some(q),
					x,
					par,
					s,
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
				q: MatRef<'_, Self>,
				x: MatMut<'_, Self>,
				par: Par<'_>,
				s: &mut MemStack,
			) {
				evd::eigenvectors_from_schur_cplx(
					t,
					Some(q),
					x,
					par,
					s,
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
fn check<T: Scalar>(
	name: &str,
	scales: [f64; 3],
	n: usize,
	row_major: bool,
	mut par: Par<'_>,
) {
	let normal = Mat::<T>::from_fn(n, n, T::point);
	let q = Mat::<T>::identity(n, n);
	let mut x = Mat::<T>::zeros(n, n);
	let mut mem = MemBuffer::new(evd::eigenvectors_from_schur_scratch::<T>(n));
	for (case, scale) in ["ordinary", "large", "small"].into_iter().zip(scales)
	{
		let scale = from_f64::<T::Real>(scale);
		let a = Mat::<T>::from_fn(n, n, |i, j| {
			mul_real(&normal[if row_major { (j, i) } else { (i, j) }], &scale)
		});
		T::run(
			if row_major {
				a.as_ref().transpose()
			} else {
				a.as_ref()
			},
			q.as_ref(),
			if row_major {
				x.as_mut().transpose_mut()
			} else {
				x.as_mut()
			},
			par.reborrow(),
			MemStack::new(&mut mem),
		);
		for rep in 0..3 {
			COUNT.store(0, Relaxed);
			BYTES.store(0, Relaxed);
			ACTIVE.store(true, Relaxed);
			T::run(
				if row_major {
					a.as_ref().transpose()
				} else {
					a.as_ref()
				},
				q.as_ref(),
				if row_major {
					x.as_mut().transpose_mut()
				} else {
					x.as_mut()
				},
				par.reborrow(),
				MemStack::new(&mut mem),
			);
			ACTIVE.store(false, Relaxed);
			let (count, bytes) = (COUNT.load(Relaxed), BYTES.load(Relaxed));
			let x = if row_major {
				x.as_ref().transpose()
			} else {
				x.as_ref()
			};
			let mut residual = &normal * x;
			for j in 0..n {
				for i in 0..n {
					residual[(i, j)] = sub(
						&residual[(i, j)],
						&mul(&x[(i, j)], &normal[(j, j)]),
					);
				}
			}
			assert!(
				residual.norm_l2()
					<= from_f64::<T::Real>(256.0 * n as f64)
						* eps::<T::Real>()
						* normal.norm_l2()
			);
			for j in 0..n {
				assert!(
					abs(&sub(&x.col(j).norm_l2(), &one::<T::Real>()))
						< from_f64::<T::Real>(256.0) * eps::<T::Real>()
				);
			}
			assert_eq!(count, 0);
			println!(
				"{name} n={n} row_major={row_major} {case} rep={rep} \
				 allocations={count} requested_bytes={bytes}"
			);
		}
	}
}
fn short_workspace_preserves_output<T: Scalar>() {
	let n = 4;
	let scale = from_f64::<T::Real>(1e30);
	let a = Mat::<T>::from_fn(n, n, |i, j| mul_real(&T::point(i, j), &scale));
	let q = Mat::<T>::identity(n, n);
	let before = Mat::<T>::from_fn(n, n, T::point);
	let mut x = before.clone();
	let req = evd::eigenvectors_from_schur_scratch::<T>(n);
	let mut mem = MemBuffer::new(req);
	let outcome =
		std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			T::run(
				a.as_ref(),
				q.as_ref(),
				x.as_mut(),
				Par::Seq,
				MemStack::new(&mut mem[..req.size_bytes() - 1]),
			);
		}));
	assert!(outcome.is_err(), "one-byte-short scratch must panic");
	assert_eq!((&x - &before).norm_l2(), zero::<T::Real>());
}
#[test]
#[cfg_attr(
	not(target_arch = "x86_64"),
	ignore = "fallback gemm packing is tracked separately in #56"
)]
fn schur_scaling_uses_declared_workspace() {
	#[cfg(target_arch = "x86_64")]
	if !std::is_x86_feature_detected!("avx2")
		|| !std::is_x86_feature_detected!("fma")
	{
		return;
	}
	short_workspace_preserves_output::<f32>();
	short_workspace_preserves_output::<f64>();
	short_workspace_preserves_output::<c32>();
	short_workspace_preserves_output::<c64>();
	for n in [0, 1, 3, 17, 32] {
		for row_major in [false, true] {
			check::<f32>("f32", [1.0, 1e30, 1e-30], n, row_major, Par::Seq);
			check::<f64>("f64", [1.0, 1e300, 1e-300], n, row_major, Par::Seq);
			check::<c32>("c32", [1.0, 1e30, 1e-30], n, row_major, Par::Seq);
			check::<c64>("c64", [1.0, 1e300, 1e-300], n, row_major, Par::Seq);
		}
	}
}

// Retain the stronger allocation contract as an executable known-gap probe.
// This fails unchanged baseline too; do not turn the observed counts into an
// accepted allocation budget for the #270 fix.
#[test]
#[cfg(feature = "threadteam")]
#[ignore = "tensor4all/faer#271: recursive held-team Schur path still allocates"]
fn held_team_schur_eigenvectors_allocate_nothing() {
	#[cfg(target_arch = "x86_64")]
	if !std::is_x86_feature_detected!("avx2")
		|| !std::is_x86_feature_detected!("fma")
	{
		return;
	}
	faer::team::with_team(2, |mut par| {
		for row_major in [false, true] {
			check::<f32>(
				"f32 team2",
				[1.0, 1e30, 1e-30],
				257,
				row_major,
				par.reborrow(),
			);
			check::<f64>(
				"f64 team2",
				[1.0, 1e300, 1e-300],
				257,
				row_major,
				par.reborrow(),
			);
			check::<c32>(
				"c32 team2",
				[1.0, 1e30, 1e-30],
				257,
				row_major,
				par.reborrow(),
			);
			check::<c64>(
				"c64 team2",
				[1.0, 1e300, 1e-300],
				257,
				row_major,
				par.reborrow(),
			);
		}
	});
}
