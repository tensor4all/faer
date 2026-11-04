//! faer#220: eigenvectors of a real pencil whose spectrum holds a repeated
//! complex-conjugate pair.
//!
//! The pencils are `A = diag(R, R, 3)` with `R = [[2, 1], [-1, 2]]` (eigenvalues
//! `2 +- i`), an optional coupling `A[1, 3]` between the two copies of `R`, and
//! `B = I`. The eigenvector substitution for one copy of the pair meets the
//! other copy shifted by the same eigenvalue, an exactly singular 2x2 system.
//! The solve replaces a second pivot below `smin` by `smin`, as LAPACK `dlaln2`
//! does for `dtgevc`.
//!
//! With the coupling the pair is defective, so its second eigenvector is close
//! to the first (LAPACK `dggev` returns the same); the tests check finiteness, a
//! nonzero column and the normwise residual. Without the coupling the pair is
//! semisimple and the two vectors for `2 + i` must also be independent.
//!
//! The two-block f64 tests produce NaN without the pivot protection. Longer
//! coupled chains also need scaling during substitution, in both precisions.
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{GevdParams, gevd_real, gevd_scratch};
use faer::linalg::matmul::matmul;
use faer::traits::RealField;
use faer::{Accum, Col, ColRef, Mat, MatRef, Par, ParShape, Spec, c64};

fn pencil(coupling: f32) -> (Mat<f32>, Mat<f32>) {
	let mut a = Mat::<f32>::zeros(5, 5);
	for o in [0usize, 2] {
		a[(o, o)] = 2.0;
		a[(o, o + 1)] = 1.0;
		a[(o + 1, o)] = -1.0;
		a[(o + 1, o + 1)] = 2.0;
	}
	a[(4, 4)] = 3.0;
	a[(1, 3)] = coupling;
	(a, Mat::<f32>::identity(5, 5))
}

fn pair(yes: bool) -> ComputeEigenvectors {
	if yes {
		ComputeEigenvectors::Yes
	} else {
		ComputeEigenvectors::No
	}
}

struct RealGevd {
	alpha: Vec<c64>,
	beta: Vec<f64>,
	/// `None` when the side was not requested
	left: Option<Mat<c64>>,
	right: Option<Mat<c64>>,
}

/// Columns `j`, `j + 1` of a real eigenvector matrix hold the real and
/// imaginary parts of the vector for `alpha[j]` (`im > 0`); `alpha[j + 1]` gets
/// the conjugate.
fn complex_vectors(u: MatRef<'_, f64>, alpha: &[c64]) -> Mat<c64> {
	let n = u.nrows();
	let mut out = Mat::<c64>::zeros(n, n);
	let mut j = 0;
	while j < n {
		if alpha[j].im == 0.0 {
			for i in 0..n {
				out[(i, j)] = c64::new(u[(i, j)], 0.0);
			}
			j += 1;
		} else {
			assert!(alpha[j].im > 0.0, "pair at {j} starts with im < 0");
			for i in 0..n {
				out[(i, j)] = c64::new(u[(i, j)], u[(i, j + 1)]);
				out[(i, j + 1)] = c64::new(u[(i, j)], -u[(i, j + 1)]);
			}
			j += 2;
		}
	}
	out
}

/// Runs `gevd_real::<T>` on a copy of `(a, b)`, returning the outputs widened
/// to `f64` and the vectors in complex form.
fn gevd_real_parts<T: RealField + Copy + From<f32> + Into<f64>>(
	a: &Mat<f32>,
	b: &Mat<f32>,
	left: bool,
	right: bool,
) -> RealGevd {
	let n = a.nrows();
	let mut a = Mat::<T>::from_fn(n, n, |i, j| a[(i, j)].into());
	let mut b = Mat::<T>::from_fn(n, n, |i, j| b[(i, j)].into());
	let mut s_re = Diag::<T>::zeros(n);
	let mut s_im = Diag::<T>::zeros(n);
	let mut beta = Diag::<T>::zeros(n);
	let mut u_left = Mat::<T>::zeros(n, n);
	let mut u_right = Mat::<T>::zeros(n, n);
	let mut mem = MemBuffer::new(gevd_scratch::<T>(
		n,
		pair(left),
		pair(right),
		ParShape::Seq,
		Spec::<GevdParams, T>::default(),
	));
	gevd_real(
		a.as_mut(),
		b.as_mut(),
		s_re.as_mut(),
		s_im.as_mut(),
		beta.as_mut(),
		if left { Some(u_left.as_mut()) } else { None },
		if right { Some(u_right.as_mut()) } else { None },
		Par::Seq,
		MemStack::new(&mut mem),
		Spec::<GevdParams, T>::default(),
	)
	.unwrap();
	let widen = |u: &Mat<T>| Mat::<f64>::from_fn(n, n, |i, j| u[(i, j)].into());
	let alpha: Vec<c64> = (0..n)
		.map(|j| {
			c64::new(
				s_re.column_vector()[j].into(),
				s_im.column_vector()[j].into(),
			)
		})
		.collect();
	let beta: Vec<f64> =
		(0..n).map(|j| beta.column_vector()[j].into()).collect();
	for (side, requested, u) in
		[("left", left, &u_left), ("right", right, &u_right)]
	{
		if requested {
			let u = widen(u);
			assert!(u.is_all_finite(), "{side} vectors: {u:?}");
		}
	}
	let left = left.then(|| complex_vectors(widen(&u_left).as_ref(), &alpha));
	let right =
		right.then(|| complex_vectors(widen(&u_right).as_ref(), &alpha));
	RealGevd {
		alpha,
		beta,
		left,
		right,
	}
}

fn widen(a: &Mat<f32>) -> Mat<c64> {
	Mat::<c64>::from_fn(a.nrows(), a.ncols(), |i, j| {
		c64::new(a[(i, j)] as f64, 0.0)
	})
}

fn mul(a: MatRef<'_, c64>, v: ColRef<'_, c64>) -> Col<c64> {
	let mut out = Col::<c64>::zeros(a.nrows());
	matmul(
		out.as_mat_mut(),
		Accum::Replace,
		a,
		v.as_mat(),
		c64::new(1.0, 0.0),
		Par::Seq,
	);
	out
}

/// `||beta*M*v - alpha*N*v|| / ((|beta| ||M|| + |alpha| ||N||) ||v||)`, max
/// norms.
fn relative_residual(
	m: MatRef<'_, c64>,
	n: MatRef<'_, c64>,
	v: ColRef<'_, c64>,
	alpha: c64,
	beta: f64,
) -> f64 {
	let mv = mul(m, v);
	let nv = mul(n, v);
	let r = (0..m.nrows())
		.map(|i| (mv[i] * beta - nv[i] * alpha).norm())
		.fold(0.0_f64, f64::max);
	let scale = beta.abs() * m.norm_max() + alpha.norm() * n.norm_max();
	r / (scale * v.norm_max())
}

/// Every requested column is nonzero and satisfies `beta*A*v = alpha*B*v`
/// (right) or `beta*A^T*y = alpha*B^T*y` (left) to `tol`.
///
/// The left condition is the one `gevd_real` meets for a complex pair: `y^T A
/// = lambda y^T B`, the conjugate of LAPACK's `y^H A = lambda y^H B`, which
/// `gevd_cplx` meets. The two coincide for a real eigenvalue.
fn check_vectors(a: &Mat<f32>, b: &Mat<f32>, eig: &RealGevd, tol: f64) {
	let (a, b) = (widen(a), widen(b));
	let n = a.nrows();
	if let Some(u) = &eig.right {
		for j in 0..n {
			let v = u.col(j);
			assert!(v.norm_max() > 0.0, "right column {j} is zero");
			let r = relative_residual(
				a.as_ref(),
				b.as_ref(),
				v,
				eig.alpha[j],
				eig.beta[j],
			);
			assert!(r < tol, "right column {j}: residual {r:e}");
		}
	}
	if let Some(u) = &eig.left {
		for j in 0..n {
			let y = u.col(j);
			assert!(y.norm_max() > 0.0, "left column {j} is zero");
			let r = relative_residual(
				a.transpose(),
				b.transpose(),
				y,
				eig.alpha[j],
				eig.beta[j],
			);
			assert!(r < tol, "left column {j}: residual {r:e}");
		}
	}
}

/// Each block contributes `2 +- i`, followed by the final eigenvalue `3`.
fn check_spectrum(eig: &RealGevd, tol: f64) {
	let n = eig.alpha.len();
	for j in 0..n {
		let expected = if j + 1 == n {
			c64::new(3.0, 0.0)
		} else {
			c64::new(2.0, if j % 2 == 0 { 1.0 } else { -1.0 })
		};
		let lambda = eig.alpha[j] / eig.beta[j];
		assert!(
			(lambda - expected).norm() < tol,
			"eigenvalue {j}: {lambda:?}"
		);
	}
}

fn check_all_requests<T: RealField + Copy + From<f32> + Into<f64>>(
	coupling: f32,
	tol: f64,
) {
	let (a, b) = pencil(coupling);
	for (left, right) in [(true, false), (false, true), (true, true)] {
		let eig = gevd_real_parts::<T>(&a, &b, left, right);
		check_spectrum(&eig, tol);
		check_vectors(&a, &b, &eig, tol);
	}
}

/// The reported reproduction, through the public API.
#[test]
fn reported_pencil_right_vectors_are_finite() {
	let (a, b) = pencil(0.75);
	let a = Mat::<f64>::from_fn(5, 5, |i, j| a[(i, j)] as f64);
	let b = Mat::<f64>::from_fn(5, 5, |i, j| b[(i, j)] as f64);
	let eig = a.generalized_eigen(&b, Par::Seq).unwrap();
	let u = eig.U();
	assert!(u.is_all_finite(), "right vectors: {u:?}");
	let a = Mat::<c64>::from_fn(5, 5, |i, j| c64::new(a[(i, j)], 0.0));
	let b = Mat::<c64>::from_fn(5, 5, |i, j| c64::new(b[(i, j)], 0.0));
	let s_a = eig.S_a().column_vector();
	let s_b = eig.S_b().column_vector();
	for j in 0..5 {
		let v = u.col(j);
		assert!(v.norm_max() > 0.0, "right column {j} is zero");
		// `S_b` is real for a real pencil
		assert_eq!(s_b[j].im, 0.0);
		let r = relative_residual(a.as_ref(), b.as_ref(), v, s_a[j], s_b[j].re);
		assert!(r < 1e-13, "right column {j}: residual {r:e}");
	}
}

#[test]
fn defective_repeated_pair_f64() {
	check_all_requests::<f64>(0.75, 1e-13);
}

#[test]
fn defective_repeated_pair_f32() {
	check_all_requests::<f32>(0.75, 1e-5);
}

#[test]
fn semisimple_repeated_pair_f64() {
	check_all_requests::<f64>(0.0, 1e-13);
}

#[test]
fn semisimple_repeated_pair_f32() {
	check_all_requests::<f32>(0.0, 1e-5);
}

/// Each additional coupled block amplifies an unscaled substitution by about
/// `1 / epsilon`. These chains overflow even with protected nonzero pivots.
fn check_repeated_pair_chain<T: RealField + Copy + From<f32> + Into<f64>>(
	blocks: usize,
	tol: f64,
) {
	let n = 2 * blocks + 1;
	let mut a = Mat::<f32>::zeros(n, n);
	for k in 0..blocks {
		let o = 2 * k;
		a[(o, o)] = 2.0;
		a[(o, o + 1)] = 1.0;
		a[(o + 1, o)] = -1.0;
		a[(o + 1, o + 1)] = 2.0;
		if k + 1 < blocks {
			a[(o + 1, o + 3)] = 0.75;
		}
	}
	a[(n - 1, n - 1)] = 3.0;
	let b = Mat::<f32>::identity(n, n);
	for (left, right) in [(true, false), (false, true), (true, true)] {
		let eig = gevd_real_parts::<T>(&a, &b, left, right);
		check_spectrum(&eig, tol);
		check_vectors(&a, &b, &eig, tol);
	}
}

#[test]
fn repeated_pair_chain_f32() {
	check_repeated_pair_chain::<f32>(8, 1e-5);
}

#[test]
fn repeated_pair_chain_f64() {
	check_repeated_pair_chain::<f64>(22, 1e-13);
}

/// Without the coupling `2 + i` has two independent eigenvectors on each side
/// (columns 0 and 2); a perturbed solve that returned a copy of the first
/// would still pass the residual check.
#[test]
fn semisimple_repeated_pair_vectors_are_independent() {
	let (a, b) = pencil(0.0);
	let eig = gevd_real_parts::<f64>(&a, &b, true, true);
	for (side, u) in [("left", &eig.left), ("right", &eig.right)] {
		let u = u.as_ref().unwrap();
		let (x, y) = (u.col(0), u.col(2));
		// `|x^H y| / (|x| |y|)`, the cosine of the angle between them
		let cos = (x.adjoint() * y).norm() / (x.norm_l2() * y.norm_l2());
		assert!(cos < 1e-12, "{side}: cos = {cos:e}");
	}
}
