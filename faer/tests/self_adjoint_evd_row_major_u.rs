//! tensor4all/faer#314: `self_adjoint_evd` with a row-major `u` computes the eigenvectors in a
//! column-major stand-in and copies them into `u` once, so the eigenvectors and the eigenvalues
//! are the column-major call's bit for bit. Before, the eigenvector solver and the
//! back-transformation wrote the row-major `u` in place, a column at a time with a stride of `n`
//! (1.4-8.5x the column-major call at f64 64-256), and most entries of `u` differed from the
//! column-major ones in their last bits (f64 n = 256: the eigenvalues too). The bit-for-bit
//! comparison is what observes the path: only the column-major arithmetic gives these bits.
//!
//! Both sides of the solver's dispatch (`n < recursion_threshold`: QR iteration; otherwise divide
//! and conquer), sequential and on a team, f32/f64/c32/c64, and a NaN input, whose failed solve
//! must leave the same `u` (and the same error) in both layouts. A generic-strided `u` keeps the
//! in-place path and is checked against the column-major one to a tolerance.
#![cfg(feature = "std")]
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{
	ComputeEigenvectors, EvdError, SelfAdjointEvdParams, self_adjoint_evd,
	self_adjoint_evd_scratch,
};
use faer::traits::ComplexField;
use faer::traits::math_utils::{abs, conj, from_f64, zero};
use faer::{Auto, Mat, MatMut, MatRef, Par, Spec, c32, c64};

/// The bytes of a native scalar (`f32`, `f64`, `c32`, `c64`: no padding), to compare two results
/// bit for bit, NaN payloads included.
fn bits<T>(x: &T) -> &[u8] {
	// SAFETY: a native scalar is plain data of `size_of::<T>()` initialised bytes
	unsafe {
		core::slice::from_raw_parts(
			x as *const T as *const u8,
			core::mem::size_of::<T>(),
		)
	}
}

/// A deterministic value in `[-0.5, 0.5)` for the entry `k`.
fn rnd(k: u64) -> f64 {
	(k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64 / (1u64 << 53) as f64
		- 0.5
}

/// One call's outcome: the result, the eigenvalues, and `u` as an owned matrix.
type Outcome<T> = (Result<(), EvdError>, Vec<T>, Mat<T>);

/// `self_adjoint_evd` into a column-major, a row-major and a generic-strided `u` (each zeroed),
/// from the same input and the same scratch request.
fn three_layouts<T: ComplexField>(
	a: MatRef<'_, T>,
	par: Par,
) -> [Outcome<T>; 3] {
	let n = a.nrows();
	// QR iteration below 8, divide and conquer from 8: both sides at small sizes
	let params = Spec::new(SelfAdjointEvdParams {
		recursion_threshold: 8,
		..<SelfAdjointEvdParams as Auto<T>>::auto()
	});
	let mut mem = MemBuffer::new(self_adjoint_evd_scratch::<T>(
		n,
		ComputeEigenvectors::Yes,
		par.shape(),
		params,
	));
	let mut par = par;
	let mut run = |u: MatMut<'_, T>| {
		let mut s = Diag::<T>::zeros(n);
		let r = self_adjoint_evd(
			a,
			s.as_mut(),
			Some(u),
			par.reborrow(),
			MemStack::new(&mut mem),
			params,
		);
		(r, (0..n).map(|i| s[i].clone()).collect::<Vec<_>>())
	};
	let mut col = Mat::<T>::zeros(n, n);
	let (rc, sc) = run(col.as_mut());
	let mut row = vec![zero::<T>(); n * n];
	let (rr, sr) = run(MatMut::from_row_major_slice_mut(&mut row, n, n));
	let row = MatRef::from_row_major_slice(&row, n, n).to_owned();
	// every other row of a column-major `2n x n` buffer: neither stride is 1
	let mut buf = Mat::<T>::zeros(2 * n, n);
	let cs = buf.col_stride();
	// SAFETY: rows 0, 2, .., 2n - 2 of the `2n x n` column-major buffer
	let generic =
		unsafe { MatMut::from_raw_parts_mut(buf.as_ptr_mut(), n, n, 2, cs) };
	let (rg, sg) = run(generic);
	let generic = Mat::<T>::from_fn(n, n, |i, j| buf[(2 * i, j)].clone());
	[(rc, sc, col), (rr, sr, row), (rg, sg, generic)]
}

fn check<T: ComplexField>(a: MatRef<'_, T>, finite: bool, par: Par) {
	let n = a.nrows();
	let threads = par.degree();
	let what = format!(
		"{} n = {n}, {threads} thread(s)",
		core::any::type_name::<T>()
	);
	let [(rc, sc, uc), (rr, sr, ur), (rg, sg, ug)] = three_layouts(a, par);
	assert_eq!(rc, rr, "{what}: the row-major call's result differs");
	assert_eq!(rc, rg, "{what}: the generic-strided call's result differs");
	for i in 0..n {
		assert!(
			bits(&sc[i]) == bits(&sr[i]),
			"{what}: eigenvalue {i} differs: {:?} (column-major u) against {:?} (row-major u)",
			sc[i],
			sr[i],
		);
		for j in 0..n {
			assert!(
				bits(&uc[(i, j)]) == bits(&ur[(i, j)]),
				"{what}: u[({i}, {j})] differs: {:?} (column-major) against {:?} (row-major)",
				uc[(i, j)],
				ur[(i, j)],
			);
		}
	}
	// the generic-strided `u` keeps the in-place path: a decomposition of `a` to a tolerance
	if finite {
		assert!(rc.is_ok(), "{what}: {rc:?}");
		let tol = from_f64::<T::Real>(64.0 * (n as f64 + 1.0))
			* faer::traits::math_utils::eps::<T::Real>();
		let norm = from_f64::<T::Real>(1.0) + a.norm_max();
		let s = Mat::<T>::from_fn(n, n, |i, j| {
			if i == j { sg[i].clone() } else { zero() }
		});
		let err = (&ug * &s * ug.adjoint() - a).norm_max();
		assert!(
			err <= &tol * &norm,
			"{what}: generic-strided u reconstructs to {err:?}"
		);
		for i in 0..n {
			let d = abs(&(&sg[i] - &sc[i]));
			assert!(
				d <= &tol * &norm,
				"{what}: generic-strided eigenvalue {i} off by {d:?}"
			);
		}
	}
}

fn check_type<T: ComplexField>(make: impl Fn(f64, f64) -> T) {
	for n in [1usize, 2, 5, 17, 40] {
		let hermitian = Mat::<T>::from_fn(n, n, |i, j| {
			let (p, q) = if i >= j { (i, j) } else { (j, i) };
			let k = (p * 1000 + q) as u64;
			let v = make(rnd(2 * k), if p == q { 0.0 } else { rnd(2 * k + 1) });
			if i >= j { v } else { conj(&v) }
		});
		let mut poisoned = hermitian.clone();
		poisoned[(n / 2, n / 2)] = make(f64::NAN, 0.0);
		for (a, finite) in
			[(hermitian.as_ref(), true), (poisoned.as_ref(), false)]
		{
			check(a, finite, Par::Seq);
			#[cfg(feature = "threadteam")]
			faer::team::with_team(4, |par| check(a, finite, par));
		}
	}
}

#[test]
fn row_major_u_is_the_column_major_u_bit_for_bit() {
	check_type::<f64>(|re, _| re);
	check_type::<f32>(|re, _| re as f32);
	check_type::<c64>(c64::new);
	check_type::<c32>(|re, im| c32::new(re as f32, im as f32));
}
