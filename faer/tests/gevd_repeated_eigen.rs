//! faer#217: `generalized_eigen` returned NaN eigenvectors for an exactly
//! repeated generalized eigenvalue.
//!
//! The eigenvector back-substitution solves `d = ca*A(j,j) - w*B(j,j)` and
//! protects a denominator below the safe minimum. It scaled the right hand side
//! by `1/smin` and then still divided by `d`, so `d == 0` with a zero right
//! hand side gave `0 * inf = NaN`. `A = B = I2` reproduced it, with `U = [[1,
//! NaN], [0, NaN]]`.
//!
//! The protected solve now replaces the denominator by `smin` (`d = max(|d|,
//! smin)`, as LAPACK `dtgevc`/`ztgevc` do) instead of dividing by the shift
//! twice.
//!
//! Each test checks the columns that are eigenvectors of the pencil: finite
//! entries, a nonzero column, `||beta*A*v - alpha*B*v||` small for right
//! vectors and `||beta*A^H*y - conj(alpha)*B^H*y||` small for left vectors.
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{GevdParams, gevd_cplx, gevd_real, gevd_scratch};
use faer::linalg::matmul::matmul;
use faer::traits::ext::*;
use faer::traits::math_utils::{
	abs, conj, eps, from_f64, from_real, one, real, zero,
};
use faer::traits::{ComplexField, RealField};
use faer::{Accum, Col, Mat, Par, ParShape, Spec, c32, c64};

fn pair(yes: bool) -> ComputeEigenvectors {
	if yes {
		ComputeEigenvectors::Yes
	} else {
		ComputeEigenvectors::No
	}
}

/// `beta*A*v - alpha*B*v`, normwise, for a real pencil and a real column.
fn residual_right_real(
	a: &Mat<f64>,
	b: &Mat<f64>,
	v: faer::ColRef<'_, f64>,
	alpha: f64,
	beta: f64,
) -> f64 {
	let mut av = Col::<f64>::zeros(a.nrows());
	matmul(
		av.as_mat_mut(),
		Accum::Replace,
		a.as_ref(),
		v.as_mat(),
		1.0,
		Par::Seq,
	);
	let mut bv = Col::<f64>::zeros(a.nrows());
	matmul(
		bv.as_mat_mut(),
		Accum::Replace,
		b.as_ref(),
		v.as_mat(),
		1.0,
		Par::Seq,
	);
	(0..a.nrows())
		.map(|i| abs(&(av[i] * beta - bv[i] * alpha)))
		.fold(0.0_f64, f64::max)
}

/// `beta*A^T*y - alpha*B^T*y` for a real pencil and a real left column.
fn residual_left_real(
	a: &Mat<f64>,
	b: &Mat<f64>,
	y: faer::ColRef<'_, f64>,
	alpha: f64,
	beta: f64,
) -> f64 {
	let mut ay = Col::<f64>::zeros(a.nrows());
	matmul(
		ay.as_mat_mut(),
		Accum::Replace,
		a.transpose(),
		y.as_mat(),
		1.0,
		Par::Seq,
	);
	let mut by = Col::<f64>::zeros(a.nrows());
	matmul(
		by.as_mat_mut(),
		Accum::Replace,
		b.transpose(),
		y.as_mat(),
		1.0,
		Par::Seq,
	);
	(0..a.nrows())
		.map(|i| (ay[i] * beta - by[i] * alpha).abs())
		.fold(0.0_f64, f64::max)
}

/// `beta*A*v - alpha*B*v` for a complex pencil and a complex column.
fn residual_right_cplx(
	a: &Mat<c64>,
	b: &Mat<c64>,
	v: faer::ColRef<'_, c64>,
	alpha: c64,
	beta: c64,
) -> f64 {
	let mut av = Col::<c64>::zeros(a.nrows());
	matmul(
		av.as_mat_mut(),
		Accum::Replace,
		a.as_ref(),
		v.as_mat(),
		c64::new(1.0, 0.0),
		Par::Seq,
	);
	let mut bv = Col::<c64>::zeros(a.nrows());
	matmul(
		bv.as_mat_mut(),
		Accum::Replace,
		b.as_ref(),
		v.as_mat(),
		c64::new(1.0, 0.0),
		Par::Seq,
	);
	(0..a.nrows())
		.map(|i| abs(&(av[i] * beta - bv[i] * alpha)))
		.fold(0.0_f64, f64::max)
}

/// `beta*A^H*y - conj(alpha)*B^H*y` for a complex pencil and a complex left
/// column.
fn residual_left_cplx(
	a: &Mat<c64>,
	b: &Mat<c64>,
	y: faer::ColRef<'_, c64>,
	alpha: c64,
	beta: c64,
) -> f64 {
	let mut ay = Col::<c64>::zeros(a.nrows());
	matmul(
		ay.as_mat_mut(),
		Accum::Replace,
		a.adjoint(),
		y.as_mat(),
		c64::new(1.0, 0.0),
		Par::Seq,
	);
	let mut by = Col::<c64>::zeros(a.nrows());
	matmul(
		by.as_mat_mut(),
		Accum::Replace,
		b.adjoint(),
		y.as_mat(),
		c64::new(1.0, 0.0),
		Par::Seq,
	);
	(0..a.nrows())
		.map(|i| abs(&(ay[i] * beta - by[i] * alpha.conj())))
		.fold(0.0_f64, f64::max)
}

/// Runs `gevd_real` on a copy of `(a, b)` and returns the eigenvalues and the
/// requested eigenvector matrices. Panics if any entry of an output is not
/// finite.
fn gevd_real_parts(
	a: &Mat<f64>,
	b: &Mat<f64>,
	left: bool,
	right: bool,
) -> (Vec<f64>, Vec<f64>, Mat<f64>, Mat<f64>) {
	let n = a.nrows();
	let mut a = a.clone();
	let mut b = b.clone();
	let mut s_re = Diag::<f64>::zeros(n);
	let mut s_im = Diag::<f64>::zeros(n);
	let mut beta = Diag::<f64>::zeros(n);
	let mut u_left = Mat::<f64>::zeros(n, n);
	let mut u_right = Mat::<f64>::zeros(n, n);
	let mut mem = MemBuffer::new(gevd_scratch::<f64>(
		n,
		pair(left),
		pair(right),
		ParShape::Seq,
		Spec::<GevdParams, f64>::default(),
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
		Spec::<GevdParams, f64>::default(),
	)
	.unwrap();
	let s_re: Vec<f64> = s_re.column_vector().iter().copied().collect();
	let s_im: Vec<f64> = s_im.column_vector().iter().copied().collect();
	(s_re, s_im, u_left, u_right)
}

/// The real identity pencil: every eigenvalue is 1, so every shift in the back
/// substitution is zero.
fn identity_pencil_real() -> (Mat<f64>, Mat<f64>) {
	(Mat::<f64>::identity(3, 3), Mat::<f64>::identity(3, 3))
}

/// A nonsymmetric pencil with a repeated diagonal pair: upper triangular, so
/// `A` is not symmetric, with the eigenvalue `1` twice on the diagonal and
/// couplings above it.
fn repeated_block_nonsymmetric() -> (Mat<f64>, Mat<f64>) {
	let n = 4;
	let a = Mat::<f64>::from_fn(n, n, |i, j| {
		if i == j {
			[1.0, 1.0, 4.0, 9.0][i]
		} else if i < j {
			0.5 * ((i + j) as f64)
		} else {
			0.0
		}
	});
	(a, Mat::<f64>::identity(n, n))
}

/// A near repeated pair: the two small eigenvalues differ by `1e-16` relative
/// and the pencil is scaled by `1e-284`, so the shift of the second one is
/// below the protected denominator but not zero. Dividing by the shift *and* by
/// `smin` overflows here.
fn near_repeated_tiny_shift_pencil() -> (Mat<f64>, Mat<f64>) {
	let s = 1e-284_f64;
	let e = 1e-16_f64;
	let a = Mat::<f64>::from_fn(3, 3, |i, j| {
		if i == j {
			s * [1.0, 1.0 + e, 5.0][i]
		} else {
			0.0
		}
	});
	let b = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
	(a, b)
}

#[test]
fn identity_pencil_right_vectors_are_finite() {
	let (a, b) = identity_pencil_real();
	let eig = a.generalized_eigen(&b, Par::Seq).unwrap();
	let u = eig.U();
	assert!(u.is_all_finite(), "right vectors: {u:?}");
	let s_a: Vec<c64> = eig.S_a().column_vector().iter().copied().collect();
	let s_b: Vec<c64> = eig.S_b().column_vector().iter().copied().collect();
	let a_cplx = Mat::<c64>::from_fn(3, 3, |i, j| c64::new(a[(i, j)], 0.0));
	let b_cplx = Mat::<c64>::from_fn(3, 3, |i, j| c64::new(b[(i, j)], 0.0));
	for j in 0..3 {
		let v = u.col(j);
		assert!(v.norm_l2() > 0.0, "right column {j} is zero");
		let r = residual_right_cplx(&a_cplx, &b_cplx, v, s_a[j], s_b[j]);
		assert!(r < 1e-12, "right column {j} residual {r:?}");
	}
}

#[test]
fn identity_pencil_left_vectors_are_finite() {
	let (a, b) = identity_pencil_real();
	let (s_re, s_im, u_left, _) = gevd_real_parts(&a, &b, true, false);
	assert!(u_left.is_all_finite(), "left vectors: {u_left:?}");
	for j in 0..3 {
		assert_eq!(s_im[j], 0.0, "column {j} is a complex pair");
		let y = u_left.col(j);
		assert!(y.norm_l2() > 0.0, "left column {j} is zero");
		let r = residual_left_real(&a, &b, y, s_re[j], 1.0);
		assert!(r < 1e-12, "left column {j} residual {r:?}");
	}
}

#[test]
fn identity_pencil_both_sides_are_finite() {
	let (a, b) = identity_pencil_real();
	let (s_re, s_im, u_left, u_right) = gevd_real_parts(&a, &b, true, true);
	assert!(u_left.is_all_finite(), "left vectors: {u_left:?}");
	assert!(u_right.is_all_finite(), "right vectors: {u_right:?}");
	for j in 0..3 {
		assert_eq!(s_im[j], 0.0);
		assert!(u_left.col(j).norm_l2() > 0.0, "left column {j} is zero");
		assert!(u_right.col(j).norm_l2() > 0.0, "right column {j} is zero");
		let rl = residual_left_real(&a, &b, u_left.col(j), s_re[j], 1.0);
		let rr = residual_right_real(&a, &b, u_right.col(j), s_re[j], 1.0);
		assert!(rl < 1e-12, "left column {j} residual {rl:e}");
		assert!(rr < 1e-12, "right column {j} residual {rr:e}");
	}
}

/// The reported case at single precision: the pencil is `A = B = I`, so every
/// shift is zero.
#[test]
fn f32_identity_pencil_right_vectors_are_finite() {
	let a = Mat::<f32>::identity(2, 2);
	let b = Mat::<f32>::identity(2, 2);
	let eig = a.generalized_eigen(&b, Par::Seq).unwrap();
	let u = eig.U();
	assert!(u.is_all_finite(), "right vectors: {u:?}");
	let s_a: Vec<c64> = eig
		.S_a()
		.column_vector()
		.iter()
		.map(|z| c64::new(z.re as f64, z.im as f64))
		.collect();
	let s_b: Vec<c64> = eig
		.S_b()
		.column_vector()
		.iter()
		.map(|z| c64::new(z.re as f64, z.im as f64))
		.collect();
	let a_cplx =
		Mat::<c64>::from_fn(2, 2, |i, j| c64::new(a[(i, j)] as f64, 0.0));
	let b_cplx =
		Mat::<c64>::from_fn(2, 2, |i, j| c64::new(b[(i, j)] as f64, 0.0));
	let u_cplx = Mat::<c64>::from_fn(2, 2, |i, j| {
		c64::new(u[(i, j)].re as f64, u[(i, j)].im as f64)
	});
	for j in 0..2 {
		assert!(u_cplx.col(j).norm_l2() > 0.0, "right column {j} is zero");
		let r = residual_right_cplx(
			&a_cplx,
			&b_cplx,
			u_cplx.col(j),
			s_a[j],
			s_b[j],
		);
		assert!(r < 1e-6, "right column {j} residual {r:?}");
	}
}

#[test]
fn repeated_block_in_nonsymmetric_pencil() {
	let (a, b) = repeated_block_nonsymmetric();
	let n = a.nrows();
	// sanity: the pencil really is nonsymmetric and the eigenvalue 1 is
	// repeated
	assert!((a[(0, 1)] - a[(1, 0)]).abs() > 1e-3, "pencil is symmetric");
	let eig = a.generalized_eigen(&b, Par::Seq).unwrap();
	let u = eig.U();
	assert!(u.is_all_finite(), "right vectors: {u:?}");
	let s_a: Vec<c64> = eig.S_a().column_vector().iter().copied().collect();
	let s_b: Vec<c64> = eig.S_b().column_vector().iter().copied().collect();
	let a_cplx = Mat::<c64>::from_fn(n, n, |i, j| c64::new(a[(i, j)], 0.0));
	let b_cplx = Mat::<c64>::from_fn(n, n, |i, j| c64::new(b[(i, j)], 0.0));
	let mut repeated = 0;
	for j in 0..n {
		assert!(u.col(j).norm_l2() > 0.0, "right column {j} is zero");
		let alpha = s_a[j] / s_b[j];
		if (alpha.re - 1.0).abs() < 1e-6 {
			repeated += 1;
		}
		let r = residual_right_cplx(&a_cplx, &b_cplx, u.col(j), s_a[j], s_b[j]);
		assert!(r < 1e-10, "right column {j} residual {r:?}");
	}
	assert_eq!(repeated, 2, "the repeated eigenvalue should appear twice");

	let (s_re, s_im, u_left, _) = gevd_real_parts(&a, &b, true, false);
	assert!(u_left.is_all_finite(), "left vectors: {u_left:?}");
	for j in 0..n {
		assert_eq!(s_im[j], 0.0);
		assert!(u_left.col(j).norm_l2() > 0.0, "left column {j} is zero");
		let r = residual_left_real(&a, &b, u_left.col(j), s_re[j], 1.0);
		assert!(r < 1e-10, "left column {j} residual {r:?}");
	}
}

#[test]
fn near_repeated_tiny_shift() {
	let (a, b) = near_repeated_tiny_shift_pencil();
	let eig = a.generalized_eigen(&b, Par::Seq).unwrap();
	let u = eig.U();
	assert!(u.is_all_finite(), "right vectors: {u:?}");
	let s_a: Vec<c64> = eig.S_a().column_vector().iter().copied().collect();
	let s_b: Vec<c64> = eig.S_b().column_vector().iter().copied().collect();
	let a_cplx = Mat::<c64>::from_fn(3, 3, |i, j| c64::new(a[(i, j)], 0.0));
	let b_cplx = Mat::<c64>::from_fn(3, 3, |i, j| c64::new(b[(i, j)], 0.0));
	for j in 0..3 {
		let v = u.col(j);
		assert!(v.norm_l2() > 0.0, "right column {j} is zero");
		// the residual is scaled by the magnitude of the pencil
		let r = residual_right_cplx(&a_cplx, &b_cplx, v, s_a[j], s_b[j]);
		assert!(r < 1e-12 * 1e-284 * 8.0, "right column {j} residual {r:?}");
	}

	let (s_re, s_im, u_left, _) = gevd_real_parts(&a, &b, true, false);
	assert!(u_left.is_all_finite(), "left vectors: {u_left:?}");
	for j in 0..3 {
		assert_eq!(s_im[j], 0.0);
		assert!(u_left.col(j).norm_l2() > 0.0, "left column {j} is zero");
		let r = residual_left_real(&a, &b, u_left.col(j), s_re[j], 1.0);
		assert!(r < 1e-12 * 1e-284 * 8.0, "left column {j} residual {r:?}");
	}
}

// complex pencil: the same identity pair, so every shift is zero here too
fn identity_pencil_cplx() -> (Mat<c64>, Mat<c64>) {
	(Mat::<c64>::identity(3, 3), Mat::<c64>::identity(3, 3))
}

fn gevd_cplx_parts(
	a: &Mat<c64>,
	b: &Mat<c64>,
	left: bool,
	right: bool,
) -> (Vec<c64>, Mat<c64>, Mat<c64>) {
	let n = a.nrows();
	let mut a = a.clone();
	let mut b = b.clone();
	let mut s = Diag::<c64>::zeros(n);
	let mut beta = Diag::<c64>::zeros(n);
	let mut u_left = Mat::<c64>::zeros(n, n);
	let mut u_right = Mat::<c64>::zeros(n, n);
	let mut mem = MemBuffer::new(gevd_scratch::<c64>(
		n,
		pair(left),
		pair(right),
		ParShape::Seq,
		Spec::<GevdParams, c64>::default(),
	));
	gevd_cplx(
		a.as_mut(),
		b.as_mut(),
		s.as_mut(),
		beta.as_mut(),
		if left { Some(u_left.as_mut()) } else { None },
		if right { Some(u_right.as_mut()) } else { None },
		Par::Seq,
		MemStack::new(&mut mem),
		Spec::<GevdParams, c64>::default(),
	)
	.unwrap();
	let s: Vec<c64> = s.column_vector().iter().copied().collect();
	(s, u_left, u_right)
}

#[test]
fn complex_identity_pencil_right_vectors_are_finite() {
	let (a, b) = identity_pencil_cplx();
	let eig = a.generalized_eigen(&b, Par::Seq).unwrap();
	let u = eig.U();
	assert!(u.is_all_finite(), "right vectors: {u:?}");
	let s_a: Vec<c64> = eig.S_a().column_vector().iter().copied().collect();
	let s_b: Vec<c64> = eig.S_b().column_vector().iter().copied().collect();
	for j in 0..3 {
		assert!(u.col(j).norm_l2() > 0.0, "right column {j} is zero");
		let r = residual_right_cplx(&a, &b, u.col(j), s_a[j], s_b[j]);
		assert!(r < 1e-12, "right column {j} residual {r:?}");
	}
}

#[test]
fn complex_identity_pencil_left_vectors_are_finite() {
	let (a, b) = identity_pencil_cplx();
	let (s, u_left, _) = gevd_cplx_parts(&a, &b, true, false);
	assert!(u_left.is_all_finite(), "left vectors: {u_left:?}");
	for j in 0..3 {
		assert!(u_left.col(j).norm_l2() > 0.0, "left column {j} is zero");
		let r =
			residual_left_cplx(&a, &b, u_left.col(j), s[j], c64::new(1.0, 0.0));
		assert!(r < 1e-12, "left column {j} residual {r:?}");
	}
}

#[test]
fn complex_identity_pencil_both_sides_are_finite() {
	let (a, b) = identity_pencil_cplx();
	let (s, u_left, u_right) = gevd_cplx_parts(&a, &b, true, true);
	assert!(u_left.is_all_finite(), "left vectors: {u_left:?}");
	assert!(u_right.is_all_finite(), "right vectors: {u_right:?}");
	for j in 0..3 {
		assert!(u_left.col(j).norm_l2() > 0.0, "left column {j} is zero");
		assert!(u_right.col(j).norm_l2() > 0.0, "right column {j} is zero");
		let rl =
			residual_left_cplx(&a, &b, u_left.col(j), s[j], c64::new(1.0, 0.0));
		let rr = residual_right_cplx(
			&a,
			&b,
			u_right.col(j),
			s[j],
			c64::new(1.0, 0.0),
		);
		assert!(rl < 1e-12, "left column {j} residual {rl:e}");
		assert!(rr < 1e-12, "right column {j} residual {rr:e}");
	}
}

/// A regular pencil whose eigenvalues are known in closed form: `B = X X^T + I`
/// is symmetric positive definite and `C` is the tridiagonal 1-D Laplacian, so
/// `A = B*C` has the `n` real, distinct eigenvalues `2 - 2*cos(k*pi/(n+1))`.
fn laplacian_pencil<R: RealField>(n: usize) -> (Mat<R>, Mat<R>) {
	let x = Mat::<R>::from_fn(n, n, |i, j| {
		from_f64::<R>((1 + i + j * n) as f64 / (n * n) as f64)
	});
	let b = &x * x.transpose() + Mat::<R>::identity(n, n);
	let c = Mat::<R>::from_fn(n, n, |i, j| {
		from_f64::<R>(if i == j {
			2.0
		} else if i.abs_diff(j) == 1 {
			-1.0
		} else {
			0.0
		})
	});
	(&b * &c, b)
}

fn expected_spectrum(n: usize) -> Vec<f64> {
	let mut v: Vec<f64> = (0..n)
		.map(|k| {
			let theta = std::f64::consts::PI * (k + 1) as f64 / (n + 1) as f64;
			2.0 - 2.0 * theta.cos()
		})
		.collect();
	v.sort_by(f64::total_cmp);
	v
}

/// `1e-8` is the bound this test used for `f64`; a scalar whose `epsilon` is
/// larger than `1e-12` gets `1e4 * epsilon` instead, so `f32` and `c32` are not
/// held to a `f64` bound.
fn tolerance<R: RealField>() -> R {
	from_f64::<R>(1e-8).fmax(eps::<R>() * from_f64::<R>(1e4))
}

/// `beta*A^T*y - alpha*B^T*y`, normwise, for a real pencil and a real left
/// column.
fn residual_left_real_generic<R: RealField>(
	a: &Mat<R>,
	b: &Mat<R>,
	y: faer::ColRef<'_, R>,
	alpha: R,
	beta: R,
) -> R {
	let mut ay = Col::<R>::zeros(a.nrows());
	matmul(
		ay.as_mat_mut(),
		Accum::Replace,
		a.transpose(),
		y.as_mat(),
		one::<R>(),
		Par::Seq,
	);
	let mut by = Col::<R>::zeros(a.nrows());
	matmul(
		by.as_mat_mut(),
		Accum::Replace,
		b.transpose(),
		y.as_mat(),
		one::<R>(),
		Par::Seq,
	);
	(0..a.nrows())
		.map(|i| {
			abs(&(ay[i].clone() * beta.clone() - by[i].clone() * alpha.clone()))
		})
		.fold(zero::<R>(), |m, x| m.fmax(x))
}

/// `beta*A^H*y - conj(alpha)*B^H*y`, normwise, for a complex pencil and a
/// complex left column.
fn residual_left_generic<C: ComplexField>(
	a: &Mat<C>,
	b: &Mat<C>,
	y: faer::ColRef<'_, C>,
	alpha: C,
	beta: C,
) -> C::Real {
	let mut ay = Col::<C>::zeros(a.nrows());
	matmul(
		ay.as_mat_mut(),
		Accum::Replace,
		a.adjoint(),
		y.as_mat(),
		one::<C>(),
		Par::Seq,
	);
	let mut by = Col::<C>::zeros(a.nrows());
	matmul(
		by.as_mat_mut(),
		Accum::Replace,
		b.adjoint(),
		y.as_mat(),
		one::<C>(),
		Par::Seq,
	);
	(0..a.nrows())
		.map(|i| {
			abs(&(ay[i].clone() * beta.clone() - by[i].clone() * conj(&alpha)))
		})
		.fold(zero::<C::Real>(), |m, x| m.fmax(x))
}

/// Same as [`gevd_real_parts`], but also returns `beta`, which the eigenvalue
/// is `alpha / beta` for. The pencils in this file are not normalized, so the
/// callers cannot assume `beta == 1`.
fn gevd_real_with_beta<R: RealField>(
	a: &Mat<R>,
	b: &Mat<R>,
	left: bool,
	right: bool,
) -> (Vec<R>, Vec<R>, Vec<R>, Mat<R>, Mat<R>) {
	let n = a.nrows();
	let mut a = a.clone();
	let mut b = b.clone();
	let mut s_re = Diag::<R>::zeros(n);
	let mut s_im = Diag::<R>::zeros(n);
	let mut beta = Diag::<R>::zeros(n);
	let mut u_left = Mat::<R>::zeros(n, n);
	let mut u_right = Mat::<R>::zeros(n, n);
	let mut mem = MemBuffer::new(gevd_scratch::<R>(
		n,
		pair(left),
		pair(right),
		ParShape::Seq,
		Spec::<GevdParams, R>::default(),
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
		Spec::<GevdParams, R>::default(),
	)
	.unwrap();
	let s_re: Vec<R> = s_re.column_vector().iter().cloned().collect();
	let s_im: Vec<R> = s_im.column_vector().iter().cloned().collect();
	let beta: Vec<R> = beta.column_vector().iter().cloned().collect();
	(s_re, s_im, beta, u_left, u_right)
}

/// The complex routine on the same (real valued) pencil, so `c32` and `c64`
/// cover the complex QZ path with the same known spectrum.
fn gevd_cplx_with_beta<C: ComplexField>(
	a: &Mat<C>,
	b: &Mat<C>,
	left: bool,
	right: bool,
) -> (Vec<C>, Vec<C>, Mat<C>, Mat<C>) {
	let n = a.nrows();
	let mut a = a.clone();
	let mut b = b.clone();
	let mut s = Diag::<C>::zeros(n);
	let mut beta = Diag::<C>::zeros(n);
	let mut u_left = Mat::<C>::zeros(n, n);
	let mut u_right = Mat::<C>::zeros(n, n);
	let mut mem = MemBuffer::new(gevd_scratch::<C>(
		n,
		pair(left),
		pair(right),
		ParShape::Seq,
		Spec::<GevdParams, C>::default(),
	));
	gevd_cplx(
		a.as_mut(),
		b.as_mut(),
		s.as_mut(),
		beta.as_mut(),
		if left { Some(u_left.as_mut()) } else { None },
		if right { Some(u_right.as_mut()) } else { None },
		Par::Seq,
		MemStack::new(&mut mem),
		Spec::<GevdParams, C>::default(),
	)
	.unwrap();
	let s: Vec<C> = s.column_vector().iter().cloned().collect();
	let beta: Vec<C> = beta.column_vector().iter().cloned().collect();
	(s, beta, u_left, u_right)
}

fn lift<C: ComplexField>(m: &Mat<C::Real>) -> Mat<C> {
	Mat::<C>::from_fn(m.nrows(), m.ncols(), |i, j| from_real::<C>(&m[(i, j)]))
}

const MODES: [(bool, bool); 4] =
	[(false, false), (true, false), (false, true), (true, true)];

/// faer#219 through the real QZ path: every request must return the finite,
/// analytic spectrum.
fn check_real_pencil<R: RealField>(n: usize) {
	let (a, b) = laplacian_pencil::<R>(n);
	let anorm = a.norm_max();
	let bnorm = b.norm_max();
	let tol = tolerance::<R>() * (anorm.clone() + bnorm.clone());
	let expected: Vec<R> = expected_spectrum(n)
		.iter()
		.map(|x| from_f64::<R>(*x))
		.collect();
	let mut spectra: Vec<Vec<R>> = Vec::new();
	for (left, right) in MODES {
		let (s_re, s_im, beta, u_left, u_right) =
			gevd_real_with_beta::<R>(&a, &b, left, right);
		for j in 0..n {
			assert!(
				s_re[j].is_finite()
					&& s_im[j].is_finite()
					&& beta[j].is_finite(),
				"n={n} left={left} right={right}: index {j} is not finite",
			);
			assert!(
				beta[j] != zero::<R>(),
				"n={n} left={left} right={right}: index {j} has beta == 0",
			);
		}
		let mut eig: Vec<R> =
			(0..n).map(|j| s_re[j].clone() / beta[j].clone()).collect();
		eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
		for j in 0..n {
			let err = abs(&(eig[j].clone() - expected[j].clone()));
			assert!(
				err < tol,
				"n={n} left={left} right={right}: eigenvalue {j} is {:?} not \
				 {:?} ({err:?})",
				eig[j],
				expected[j],
			);
		}
		assert!(
			u_left.is_all_finite() && u_right.is_all_finite(),
			"n={n} left={left} right={right}: vectors are not finite",
		);
		if left {
			for j in 0..n {
				let y = u_left.col(j);
				assert!(
					y.norm_l2() > zero::<R>(),
					"n={n}: left column {j} is zero"
				);
				let r = residual_left_real_generic(
					&a,
					&b,
					y,
					s_re[j].clone(),
					beta[j].clone(),
				);
				// normwise, against the size of the pencil in that column
				let bound = tolerance::<R>()
					* (abs(&beta[j]) * anorm.clone()
						+ abs(&s_re[j]) * bnorm.clone());
				assert!(
					r < bound,
					"n={n}: left column {j} residual {r:?} against {bound:?}",
				);
			}
		}
		spectra.push(eig);
	}
	for s in &spectra[1..] {
		for j in 0..n {
			let err = abs(&(s[j].clone() - spectra[0][j].clone()));
			assert!(
				err < tol,
				"n={n}: eigenvalue {j} depends on the requested side ({err:?})",
			);
		}
	}
}

/// faer#219 through the complex QZ path, on the same pencil lifted to
/// `C::Real`.
fn check_cplx_pencil<C: ComplexField>(n: usize) {
	let (ar, br) = laplacian_pencil::<C::Real>(n);
	let anorm = ar.norm_max();
	let bnorm = br.norm_max();
	let a = lift::<C>(&ar);
	let b = lift::<C>(&br);
	let tol = tolerance::<C::Real>() * (anorm.clone() + bnorm.clone());
	let expected: Vec<C::Real> = expected_spectrum(n)
		.iter()
		.map(|x| from_f64::<C::Real>(*x))
		.collect();
	let mut spectra: Vec<Vec<C::Real>> = Vec::new();
	for (left, right) in MODES {
		let (s, beta, u_left, u_right) =
			gevd_cplx_with_beta::<C>(&a, &b, left, right);
		for j in 0..n {
			assert!(
				abs(&s[j]).is_finite() && abs(&beta[j]).is_finite(),
				"n={n} left={left} right={right}: index {j} is not finite",
			);
			assert!(
				abs(&beta[j]) > zero::<C::Real>(),
				"n={n} left={left} right={right}: index {j} has beta == 0",
			);
		}
		let mut eig: Vec<C::Real> = (0..n)
			.map(|j| {
				let lambda = s[j].clone() / beta[j].clone();
				assert!(
					abs(&real(&lambda)) < tol
						|| abs(&real(&lambda)) >= zero::<C::Real>(),
					"n={n}: index {j} is not finite",
				);
				real(&lambda)
			})
			.collect();
		eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
		for j in 0..n {
			let err = abs(&(eig[j].clone() - expected[j].clone()));
			assert!(
				err < tol,
				"n={n} left={left} right={right}: eigenvalue {j} is {:?} not \
				 {:?} ({err:?})",
				eig[j],
				expected[j],
			);
		}
		assert!(
			u_left.is_all_finite() && u_right.is_all_finite(),
			"n={n} left={left} right={right}: vectors are not finite",
		);
		if left {
			for j in 0..n {
				let y = u_left.col(j);
				assert!(
					abs(&y.norm_l2()) > zero::<C::Real>(),
					"n={n}: left column {j} is zero"
				);
				let r = residual_left_generic(
					&a,
					&b,
					y,
					s[j].clone(),
					beta[j].clone(),
				);
				let bound = tolerance::<C::Real>()
					* (abs(&beta[j]) * anorm.clone()
						+ abs(&s[j]) * bnorm.clone());
				assert!(
					r < bound,
					"n={n}: left column {j} residual {r:?} against {bound:?}",
				);
			}
		}
		spectra.push(eig);
	}
	for s in &spectra[1..] {
		for j in 0..n {
			let err = abs(&(s[j].clone() - spectra[0][j].clone()));
			assert!(
				err < tol,
				"n={n}: eigenvalue {j} depends on the requested side ({err:?})",
			);
		}
	}
}

/// faer#219: requesting only the left eigenvectors, or the eigenvalues alone,
/// changed the finite spectrum into nonfinite values, while the right-only and
/// the both-sides requests succeeded on the same pencil.
///
/// The generalized Hessenberg reduction cleared the stored rotations of the
/// first block with a rectangle that also covered the already reduced columns,
/// whenever no right-hand side output (`Z`) was requested. `A` and `B` reached
/// the QZ iteration with that data destroyed, so the iteration never deflated
/// and `alpha`/`beta` kept their initial zeros for most indices.
///
/// `n = 24` exercises the unblocked reduction; `33` is the first blocked size
/// (`block_size` is 32), and `40` and `64` more blocks. `96` is dropped from
/// this grid because it costs twice the whole grid again under coverage and
/// reaches the same branch. The four tests are the same pencil through the real
/// (`f32`/`f64`) and the complex (`c32`/`c64`) QZ path; on the reduction
/// that predates the fix every `n > 32` with `right = false` returns `beta ==
/// 0` for `n - 4` indices.
#[test]
fn spectrum_does_not_depend_on_which_eigenvectors_are_requested() {
	for n in [24usize, 33, 40, 64] {
		check_real_pencil::<f64>(n);
	}
}

#[test]
fn spectrum_does_not_depend_on_which_eigenvectors_are_requested_f32() {
	for n in [24usize, 33, 40, 64] {
		check_real_pencil::<f32>(n);
	}
}

#[test]
fn spectrum_does_not_depend_on_which_eigenvectors_are_requested_c64() {
	for n in [24usize, 33, 40, 64] {
		check_cplx_pencil::<c64>(n);
	}
}

#[test]
fn spectrum_does_not_depend_on_which_eigenvectors_are_requested_c32() {
	for n in [24usize, 33, 40, 64] {
		check_cplx_pencil::<c32>(n);
	}
}
