//! Caller-sized upgrade scratch preserves scalar/layout/previous-factor
//! contracts.
mod common;
use common::*;
use faer::linalg::householder::{
	upgrade_householder_factor, upgrade_householder_factor_scratch,
};
use reborrow::*;
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
fn gram<T: ComplexField>(a: MatRef<'_, T>, i: usize, j: usize) -> T {
	let mut sum = zero::<T>();
	for row in 0..a.nrows() {
		sum += conj(&a[(row, i)]) * &a[(row, j)];
	}
	sum
}
fn check<T: Value>(mut par: Par<'_>) {
	for (m, n) in [
		(0, 0),
		(1, 1),
		(3, 2),
		(96, 9),
		(384, 32),
		(2048, 16),
		(96, 64),
	] {
		for row_major in [false, true] {
			for prev in [1, 8] {
				let b = n.max(1);
				let basis = Mat::<T>::from_fn(m, n, |i, j| {
					if i < j {
						zero()
					} else if i == j {
						one()
					} else {
						T::value(0.02 * sample(i, j), 0.01 * sample(j, i))
					}
				});
				let mut input = if row_major {
					basis.transpose().to_owned()
				} else {
					basis.clone()
				};
				let mut out = Mat::<T>::full(n, n, T::value(19.0, 0.0));
				let logical_basis = if row_major {
					input.as_mut().transpose_mut()
				} else {
					input.as_mut()
				};
				let mut logical_out = if row_major {
					out.as_mut().transpose_mut()
				} else {
					out.as_mut()
				};
				for j in 0..n {
					for i in 0..=j {
						if i == j {
							logical_out[(i, j)] =
								gram(basis.as_ref(), i, j) * from_f64::<T>(0.5);
						} else if i / prev == j / prev {
							logical_out[(i, j)] = gram(basis.as_ref(), i, j);
						} else {
							logical_out[(i, j)] = T::value(23.0, 0.0);
						}
					}
				}
				let req = upgrade_householder_factor_scratch::<T>(
					m,
					n,
					b,
					prev,
					par.shape(),
				);
				let mut memory = MemBuffer::new(req);
				upgrade_householder_factor(
					logical_out.rb_mut(),
					logical_basis.rb(),
					b,
					prev,
					par.reborrow(),
					MemStack::new(&mut memory),
				);
				let bound = eps::<T::Real>()
					* from_f64::<T::Real>(32.0 * m.max(1) as f64);
				for j in 0..n {
					for i in 0..n {
						if i > j {
							assert_eq!(
								logical_out[(i, j)],
								T::value(19.0, 0.0)
							);
							continue;
						}
						let expected = gram(basis.as_ref(), i, j)
							* if i == j { from_f64::<T>(0.5) } else { one() };
						assert!(
							abs(&(logical_out[(i, j)].clone() - expected))
								<= bound,
							"({m},{n}) ({i},{j}) prev={prev}"
						);
					}
				}
				if matches!(par.shape(), ParShape::Seq) || prev >= 8 || n <= 8 {
					assert_eq!(req.size_bytes(), 0);
				}
			}
		}
	}
}
#[test]
fn caller_scratch_matches_four_scalar_layouts() {
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

fn multiple_blocks<T: Value>(mut par: Par<'_>) {
	for (m, n, b, prev) in [
		(9, 9, 2, 1),
		(17, 17, 4, 1),
		(96, 65, 16, 8),
		(2048, 33, 16, 1),
		(2048, 65, 16, 1),
		(2048, 65, 16, 8),
	] {
		let basis = Mat::<T>::from_fn(m, n, |i, j| {
			if i < j {
				zero()
			} else if i == j {
				one()
			} else {
				T::value(0.02 * sample(i, j), 0.01 * sample(j, i))
			}
		});
		let mut factor = Mat::<T>::full(n, n, T::value(19.0, 0.0));
		for j in 0..n {
			for i in 0..=j {
				factor[(i, j)] = if i == j {
					gram(basis.as_ref(), i, j) * from_f64::<T>(0.5)
				} else if i / prev == j / prev {
					gram(basis.as_ref(), i, j)
				} else {
					T::value(23.0, 0.0)
				};
			}
		}
		let req =
			upgrade_householder_factor_scratch::<T>(m, n, b, prev, par.shape());
		let mut memory = MemBuffer::new(req);
		upgrade_householder_factor(
			factor.as_mut(),
			basis.as_ref(),
			b,
			prev,
			par.reborrow(),
			MemStack::new(&mut memory),
		);
		let bound =
			eps::<T::Real>() * from_f64::<T::Real>(32.0 * m.max(1) as f64);
		for j in 0..n {
			for i in 0..n {
				if i > j {
					assert_eq!(factor[(i, j)], T::value(19.0, 0.0));
					continue;
				}
				if i / b != j / b {
					assert_eq!(factor[(i, j)], T::value(23.0, 0.0));
					continue;
				}
				let expected = gram(basis.as_ref(), i, j)
					* if i == j { from_f64::<T>(0.5) } else { one() };
				assert!(
					abs(&(factor[(i, j)].clone() - expected)) <= bound,
					"m={m} n={n} block={b} prev={prev} at ({i},{j})"
				);
			}
		}
	}
}
#[test]
fn multiple_blocks_keep_column_aligned_boundaries() {
	multiple_blocks::<f32>(Par::Seq);
	multiple_blocks::<f64>(Par::Seq);
	multiple_blocks::<c32>(Par::Seq);
	multiple_blocks::<c64>(Par::Seq);
	#[cfg(feature = "threadteam")]
	for width in [2, 8] {
		faer::team::with_team(width, |mut par| {
			multiple_blocks::<f32>(par.reborrow());
			multiple_blocks::<f64>(par.reborrow());
			multiple_blocks::<c32>(par.reborrow());
			multiple_blocks::<c64>(par);
		});
	}
}

#[cfg(feature = "threadteam")]
#[test]
fn short_workspace_panics_before_factor_update() {
	faer::team::with_team(2, |mut par| {
		let (m, n) = (8192, 32);
		let basis = Mat::<f64>::from_fn(m, n, |i, j| {
			if i < j {
				0.0
			} else if i == j {
				1.0
			} else {
				0.01
			}
		});
		let mut factor = Mat::<f64>::full(n, n, 19.0);
		for j in 0..n {
			factor[(j, j)] = (1.0 + 0.0001 * (m - j - 1) as f64) / 2.0;
		}
		let before = factor.clone();
		let req =
			upgrade_householder_factor_scratch::<f64>(m, n, n, 1, par.shape());
		assert!(
			req.size_bytes() > 0,
			"test must exercise the actual deep parallel branch"
		);
		let short = faer::dyn_stack::StackReq::new_aligned::<u8>(
			req.size_bytes() - 1,
			req.align_bytes(),
		);
		let mut memory = MemBuffer::new(short);
		let result =
			std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
				upgrade_householder_factor(
					factor.as_mut(),
					basis.as_ref(),
					n,
					1,
					par.reborrow(),
					MemStack::new(&mut memory),
				)
			}));
		assert!(result.is_err());
		assert_eq!(factor, before);
		let mut memory = MemBuffer::new(req);
		upgrade_householder_factor(
			factor.as_mut(),
			basis.as_ref(),
			n,
			1,
			par,
			MemStack::new(&mut memory),
		);
		assert!(
			(factor[(0, 1)] - (0.01 + 0.0001 * (m - 2) as f64)).abs() < 1e-10
		);
	});
}
