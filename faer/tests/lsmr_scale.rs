use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::{IdentityPrecond, InitialGuessStatus, lsmr::*};
use faer::traits::{
	ComplexField,
	math_utils::{from_f64, is_finite},
};
use faer::{Mat, Par, c32, c64};

/// A scalar multiple of the identity has condition number one and the representable
/// solution `x = b / scale`, so LSMR must return it whatever the scale is. Before the
/// bidiagonalisation coefficients were carried at a power of two near `|A|`, this
/// reported a zero residual with `x = inf` at `scale = 1e-300` and `x = 0` at
/// `scale = 1e300` (tensor4all/faer#245).
fn check<T: ComplexField>(scale: f64, value: fn(f64) -> T, tol: f64) {
	let a =
		Mat::<T>::from_fn(
			4,
			4,
			|i, j| {
				if i == j { value(scale) } else { value(0.0) }
			},
		);
	let b = Mat::<T>::from_fn(4, 1, |_, _| value(1.0));
	let mut x = Mat::<T>::zeros(4, 1);
	let m = IdentityPrecond { dim: 4 };
	let mut mem = MemBuffer::new(lsmr_scratch(m, &a, 1, Par::Seq.shape()));
	let info = lsmr(
		x.as_mut(),
		m,
		&a,
		b.as_ref(),
		LsmrParams {
			max_iters: 2,
			..Default::default()
		},
		|_| {},
		Par::Seq,
		MemStack::new(&mut mem),
	)
	.unwrap();
	let expected = from_f64::<T>(1.0 / scale);
	let got = x[(0, 0)].clone();
	let error = Mat::<T>::from_fn(1, 1, |_, _| &got - &expected).norm_max();
	let reference = Mat::<T>::from_fn(1, 1, |_, _| expected.clone()).norm_max();
	assert!(
		error <= reference * from_f64::<T::Real>(tol),
		"scale={scale:e}: x00 must be b/scale, got {:?}",
		got
	);
	assert!(
		x.as_ref().is_all_finite(),
		"scale={scale:e}: x is not finite"
	);
	// The estimate `info.abs_residual` is the true residual here: `A` is a multiple of
	// the identity, so an exact iterate leaves only rounding.
	let true_residual = (&a * &x - &b).norm_l2();
	let abs = from_f64::<T::Real>(tol);
	assert!(
		true_residual <= abs,
		"scale={scale:e}: true residual {true_residual:?}"
	);
	// `abs_residual` is the preconditioned normal residual, whose magnitude is the
	// scale of `A` times the least-squares one: it is checked for finiteness here, and
	// its normalized form below
	assert!(
		is_finite(&info.abs_residual),
		"scale={scale:e}: reported residual {:?}",
		info.abs_residual
	);
	assert!(
		info.rel_residual <= abs,
		"scale={scale:e}: reported relative residual {:?}",
		info.rel_residual
	);
}

#[test]
fn real_double_precision() {
	for scale in [1e-300, 1e-150, 1.0, 1e150, 1e300] {
		check::<f64>(scale, |x| x, 1e-12);
	}
}

#[test]
fn real_single_precision() {
	for scale in [1e-30, 1e-15, 1.0, 1e15, 1e30] {
		check::<f32>(scale, |x| x as f32, 1e-4);
	}
}

#[test]
fn complex_double_precision() {
	// a nonzero imaginary part keeps the coefficients complex in the recurrence too
	for scale in [1e-300, 1.0, 1e300] {
		check::<c64>(scale, |x| c64::new(x, 0.5 * x), 1e-12);
	}
}

#[test]
fn complex_single_precision() {
	for scale in [1e-30, 1.0, 1e30] {
		check::<c32>(scale, |x| c32::new(x as f32, 0.5 * x as f32), 1e-4);
	}
}

/// The normalization is in the operator, so an explicit initial guess at either end of
/// the range follows the same path: `x0` is given in the caller's units and the update is
/// added to it.
#[test]
fn nonzero_initial_guess_at_both_range_ends() {
	for scale in [1e-300, 1e300] {
		let a =
			Mat::<f64>::from_fn(4, 4, |i, j| if i == j { scale } else { 0.0 });
		let b = Mat::<f64>::from_fn(4, 1, |_, _| 1.0);
		let x0 = 0.5 / scale;
		let mut x = Mat::<f64>::from_fn(4, 1, |_, _| x0);
		let m = IdentityPrecond { dim: 4 };
		let mut mem = MemBuffer::new(lsmr_scratch(m, &a, 1, Par::Seq.shape()));
		let info = lsmr(
			x.as_mut(),
			m,
			&a,
			b.as_ref(),
			LsmrParams {
				initial_guess: InitialGuessStatus::MaybeNonZero,
				max_iters: 2,
				..Default::default()
			},
			|_| {},
			Par::Seq,
			MemStack::new(&mut mem),
		)
		.unwrap();
		let true_residual = (&a * &x - &b).norm_l2();
		assert!(
			true_residual <= 1e-12,
			"scale={scale:e}: x00={:?} residual={true_residual:e} info={info:?}",
			x[(0, 0)]
		);
	}
}
