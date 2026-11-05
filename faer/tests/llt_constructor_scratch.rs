//! Canonical owning LLT construction consumes the declared caller workspace.
mod common;
use common::*;
use faer::Side;
use faer::linalg::solvers::Llt;
trait Value: ComplexField {
	fn value(re: f64, im: f64) -> Self;
}
impl Value for f32 {
	fn value(re: f64, _: f64) -> Self {
		re as f32
	}
}
impl Value for f64 {
	fn value(re: f64, _: f64) -> Self {
		re
	}
}
impl Value for c32 {
	fn value(re: f64, im: f64) -> Self {
		Self::new(re as f32, im as f32)
	}
}
impl Value for c64 {
	fn value(re: f64, im: f64) -> Self {
		Self::new(re, im)
	}
}
fn check<T: Value>(mut par: Par<'_>) {
	for n in [0, 1, 3, 65, 257] {
		let x = Mat::<T>::from_fn(n, n, |i, j| {
			T::value(sample(i, j), 0.3 * sample(j, i))
		});
		let mut a = &x * x.adjoint();
		for i in 0..n {
			a[(i, i)] += from_f64::<T>(n as f64 + 1.0);
		}
		for side in [Side::Lower, Side::Upper] {
			for row_major in [false, true] {
				let input = if row_major {
					a.transpose().to_owned()
				} else {
					a.clone()
				};
				let input = if row_major {
					input.as_ref().transpose()
				} else {
					input.as_ref()
				};
				let req = Llt::<T>::new_scratch(n, par.shape());
				let mut mem = MemBuffer::new(req);
				let f = Llt::new(
					input,
					side,
					par.reborrow(),
					MemStack::new(&mut mem),
				)
				.unwrap();
				let l = f.L();
				let err = (&l * l.adjoint() - &a).norm_max();
				let bound = eps::<T::Real>()
					* from_f64::<T::Real>(128.0 * n.max(1) as f64)
					* a.norm_max();
				assert!(err <= bound);
				assert_eq!((input - &a).norm_max(), zero::<T::Real>());
				if req.size_bytes() > 0 {
					let mut short = vec![
						core::mem::MaybeUninit::<u8>::uninit();
						req.size_bytes() - 1
					];
					assert!(
						std::panic::catch_unwind(std::panic::AssertUnwindSafe(
							|| {
								Llt::new(
									input,
									side,
									par.reborrow(),
									MemStack::new(&mut short),
								)
								.unwrap();
							}
						))
						.is_err()
					);
					assert_eq!((input - &a).norm_max(), zero::<T::Real>());
				}
			}
		}
	}
}
#[test]
fn exact_caller_workspace_four_scalars_and_layouts() {
	check::<f32>(Par::Seq);
	check::<f64>(Par::Seq);
	check::<c32>(Par::Seq);
	check::<c64>(Par::Seq);
	#[cfg(feature = "threadteam")]
	faer::team::with_team(2, |mut par| {
		check::<f32>(par.reborrow());
		check::<f64>(par.reborrow());
		check::<c32>(par.reborrow());
		check::<c64>(par);
	});
}

fn triangles_and_errors<T: Value>() {
	let a = Mat::<T>::from_fn(3, 3, |i, j| {
		if i == j {
			T::value(4.0, 0.0)
		} else if i > j {
			T::value(0.2, 0.1)
		} else {
			conj(&T::value(0.2, 0.1))
		}
	});
	for side in [Side::Lower, Side::Upper] {
		let input = Mat::<T>::from_fn(3, 3, |i, j| {
			if (side == Side::Lower && i < j) || (side == Side::Upper && i > j)
			{
				T::value(f64::NAN, f64::NAN)
			} else {
				a[(i, j)].clone()
			}
		});
		let req = Llt::<T>::new_scratch(3, ParShape::Seq);
		let mut mem = MemBuffer::new(req);
		let f =
			Llt::new(input.as_ref(), side, Par::Seq, MemStack::new(&mut mem))
				.unwrap();
		close((&f.L() * f.L().adjoint()).as_ref(), a.as_ref());
		let g = Llt::new(
			input.as_ref().conjugate(),
			side,
			Par::Seq,
			MemStack::new(&mut mem),
		)
		.unwrap();
		close(
			(&g.L() * g.L().adjoint()).as_ref(),
			a.as_ref().conjugate().to_owned().as_ref(),
		);
	}
	let bad = Mat::<T>::from_fn(2, 2, |i, j| {
		if i != j {
			zero()
		} else {
			T::value(if i == 0 { 1.0 } else { -1.0 }, 0.0)
		}
	});
	let mut mem = MemBuffer::new(Llt::<T>::new_scratch(2, ParShape::Seq));
	assert!(matches!(
		Llt::new(bad.as_ref(), Side::Lower, Par::Seq, MemStack::new(&mut mem)),
		Err(faer::linalg::solvers::LltError::NonPositivePivot { index: 1 })
	));
	assert_eq!(bad[(1, 1)], T::value(-1.0, 0.0));
}
#[test]
fn selected_triangle_conjugation_and_nonpositive_pivots() {
	triangles_and_errors::<f32>();
	triangles_and_errors::<f64>();
	triangles_and_errors::<c32>();
	triangles_and_errors::<c64>();
}
