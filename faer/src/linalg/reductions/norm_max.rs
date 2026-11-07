use super::LINEAR_IMPL_THRESHOLD;
use crate::internal_prelude::*;
use faer_traits::RealReg;
use num_complex::Complex;
#[inline(always)]
fn norm_max_simd<'N, T: ComplexField>(
	data: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
) -> T::Real {
	struct Impl<'a, 'N, T: ComplexField> {
		data: ColRef<'a, T, Dim<'N>, ContiguousFwd>,
	}
	impl<'N, T: ComplexField> pulp::WithSimd for Impl<'_, 'N, T> {
		type Output = T::Real;

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self { data } = self;
			let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), data.nrows());
			let mut acc = [RealReg(simd.zero()); 4];
			// `simd.max` keeps the non-NaN operand, so NaN is detected on a
			// separate sum of the magnitudes: every term is `>= 0`, `+inf` or
			// NaN, so the sum is NaN exactly when some entry is NaN (#316).
			let mut nan_acc = [RealReg(simd.zero()); 4];
			simd_iter!(for (IDX, i) in [simd.batch_indices(); 4] {
				let x = simd.abs_max(simd.read(data, i));
				acc[IDX] = (*simd).max(acc[IDX], x);
				nan_acc[IDX] = RealReg(simd.add(nan_acc[IDX].0, x.0));
			});
			let acc0 = simd.max(acc[0], acc[1]);
			let acc2 = simd.max(acc[2], acc[3]);
			let acc0 = simd.max(acc0, acc2);
			let nan0 = RealReg(simd.add(nan_acc[0].0, nan_acc[1].0));
			let nan2 = RealReg(simd.add(nan_acc[2].0, nan_acc[3].0));
			let nan0 = RealReg(simd.add(nan0.0, nan2.0));
			if is_nan(&simd.reduce_sum_real(nan0)) {
				nan()
			} else {
				simd.reduce_max_real(acc0)
			}
		}
	}
	dispatch!(Impl { data }, Impl, T)
}
/// `fmax` that returns NaN when either operand is NaN.
#[inline(always)]
fn max_propagate_nan<R: RealField>(lhs: R, rhs: R) -> R {
	if is_nan(&lhs) || is_nan(&rhs) {
		nan()
	} else {
		lhs.fmax(rhs)
	}
}
fn norm_max_simd_pairwise_rows<T: ComplexField>(
	data: ColRef<'_, T, usize, ContiguousFwd>,
) -> T::Real {
	if data.nrows() <= LINEAR_IMPL_THRESHOLD {
		with_dim!(N, data.nrows());
		norm_max_simd(data.as_row_shape(N))
	} else {
		let split_point = ((data.nrows() + 1) / 2).next_power_of_two();
		let (head, tail) = data.split_at_row(split_point);
		let acc0 = norm_max_simd_pairwise_rows(head);
		let acc1 = norm_max_simd_pairwise_rows(tail);
		max_propagate_nan(acc0, acc1)
	}
}
fn norm_max_simd_pairwise_cols<T: ComplexField>(
	data: MatRef<'_, T, usize, usize, ContiguousFwd>,
) -> T::Real {
	if data.ncols() == 1 {
		norm_max_simd_pairwise_rows(data.col(0))
	} else {
		let split_point = ((data.ncols() + 1) / 2).next_power_of_two();
		let (head, tail) = data.split_at_col(split_point);
		let acc0 = norm_max_simd_pairwise_cols(head);
		let acc1 = norm_max_simd_pairwise_cols(tail);
		max_propagate_nan(acc0, acc1)
	}
}
/// returns the largest absolute value of the entries of `mat` (for a complex
/// entry, the larger of the absolute values of its real and imaginary parts).
///
/// NaN propagates: if any entry (or either part of a complex entry) is NaN,
/// the result is NaN, as with `norm_l1` and `norm_l2` and as in LAPACK's
/// `xLANGE('M')`; `t4a-faer` 0.32.0 and earlier skipped NaN entries. An empty
/// matrix returns zero.
pub fn norm_max<T: ComplexField>(mut mat: MatRef<'_, T>) -> T::Real {
	if mat.ncols() > 1 && mat.col_stride().unsigned_abs() == 1 {
		mat = mat.transpose();
	}
	if mat.row_stride() < 0 {
		mat = mat.reverse_rows();
	}
	if mat.nrows() == 0 || mat.ncols() == 0 {
		zero()
	} else {
		let m = mat.nrows();
		let n = mat.ncols();
		if const { T::SIMD_CAPABILITIES.is_simd() } {
			if let Some(mat) = mat.try_as_col_major() {
				if const { T::IS_NATIVE_C32 } {
					let mat: MatRef<
						'_,
						Complex<f32>,
						usize,
						usize,
						ContiguousFwd,
					> = unsafe { crate::hacks::coerce(mat) };
					let mat = unsafe {
						MatRef::<'_, f32, usize, usize, ContiguousFwd>::from_raw_parts(
							mat.as_ptr() as *const f32,
							2 * mat.nrows(),
							mat.ncols(),
							ContiguousFwd,
							mat.col_stride().wrapping_mul(2),
						)
					};
					return unsafe {
						crate::hacks::coerce(
							norm_max_simd_pairwise_cols::<f32>(mat),
						)
					};
				} else if const { T::IS_NATIVE_C64 } {
					let mat: MatRef<
						'_,
						Complex<f64>,
						usize,
						usize,
						ContiguousFwd,
					> = unsafe { crate::hacks::coerce(mat) };
					let mat = unsafe {
						MatRef::<'_, f64, usize, usize, ContiguousFwd>::from_raw_parts(
							mat.as_ptr() as *const f64,
							2 * mat.nrows(),
							mat.ncols(),
							ContiguousFwd,
							mat.col_stride().wrapping_mul(2),
						)
					};
					return unsafe {
						crate::hacks::coerce(
							norm_max_simd_pairwise_cols::<f64>(mat),
						)
					};
				} else {
					return norm_max_simd_pairwise_cols(mat);
				}
			}
		}
		let mut acc = zero::<T::Real>();
		for j in 0..n {
			for i in 0..m {
				let val = &mat[(i, j)];
				if is_nan(val) {
					return nan();
				}
				acc = acc.fmax(val.real().abs().fmax(val.imag().abs()));
			}
		}
		acc
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Mat, assert};
	#[test]
	fn test_norm_max() {
		let relative_err =
			|a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());
		for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
			for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
				let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
				let mut target = 0.0;
				zip!(mat.as_ref()).for_each(|unzip!(x)| {
					target = f64::max(target, x.abs());
				});
				if factor == 0.0 {
					assert!(norm_max(mat.as_ref()) == target);
				} else {
					assert!(
						relative_err(norm_max(mat.as_ref()), target) < 1e-14
					);
				}
			}
		}
	}

	/// the contiguous copy of `mat` (SIMD kernel for SIMD types) and a view with
	/// row stride 2 (scalar loop), compared through every public entry point.
	fn norm_max_both_paths<T: ComplexField>(
		mat: &Mat<T>,
	) -> (T::Real, T::Real) {
		let (m, n) = mat.shape();
		let buf = Mat::<T>::from_fn(2 * m, n, |i, j| {
			if i % 2 == 0 {
				mat[(i / 2, j)].clone()
			} else {
				zero()
			}
		});
		// SAFETY: `(i, j)` with `i < m`, `j < n` maps to offset
		// `2 * i + j * buf.col_stride()`, which is entry `(2 * i, j)` of the
		// column-major `2m x n` matrix `buf`; `buf` outlives the view.
		let strided = unsafe {
			MatRef::<'_, T>::from_raw_parts(
				buf.as_ptr(),
				m,
				n,
				2,
				buf.col_stride(),
			)
		};
		assert!(strided.row_stride() == 2);
		let simd = norm_max(mat.as_ref());
		let scalar = norm_max(strided);
		let same =
			|a: &T::Real, b: &T::Real| (is_nan(a) && is_nan(b)) || a == b;
		assert!(same(&mat.norm_max(), &simd));
		assert!(same(&strided.norm_max(), &scalar));
		if n == 1 {
			assert!(same(&mat.col(0).norm_max(), &simd));
			assert!(same(&mat.col(0).transpose().norm_max(), &simd));
		}
		if m == 1 {
			assert!(same(&mat.row(0).norm_max(), &simd));
			assert!(same(&mat.row(0).transpose().norm_max(), &simd));
		}
		(simd, scalar)
	}

	/// shapes on both sides of the SIMD tail and of the pairwise split
	/// (`LINEAR_IMPL_THRESHOLD = 128`).
	const SHAPES: [(usize, usize); 7] =
		[(1, 1), (2, 1), (1, 2), (7, 1), (37, 3), (129, 2), (300, 1)];

	fn check_nan_propagates<T: ComplexField>(nan_entry: T) {
		for (m, n) in SHAPES {
			for (ni, nj) in [(0, 0), (m - 1, n - 1), (m / 2, n / 2)] {
				let mat = Mat::<T>::from_fn(m, n, |i, j| {
					if (i, j) == (ni, nj) {
						nan_entry.clone()
					} else {
						from_f64::<T>((i + 3 * j + 1) as f64)
					}
				});
				let (simd, scalar) = norm_max_both_paths(&mat);
				assert!(is_nan(&simd));
				assert!(is_nan(&scalar));
				assert!(is_nan(&mat.norm_l1()));
				assert!(is_nan(&mat.norm_l2()));
			}
			let all_nan = Mat::<T>::from_fn(m, n, |_, _| nan_entry.clone());
			let (simd, scalar) = norm_max_both_paths(&all_nan);
			assert!(is_nan(&simd));
			assert!(is_nan(&scalar));
		}
	}

	fn check_finite_and_inf_agree<T: ComplexField>(scale: T) {
		for (m, n) in SHAPES {
			let mat = Mat::<T>::from_fn(m, n, |i, j| {
				scale.mul_real(from_f64::<T::Real>(
					((7 * i + 3 * j) % 11) as f64 - 5.0,
				))
			});
			let (simd, scalar) = norm_max_both_paths(&mat);
			assert!(simd == scalar);
			assert!(!is_nan(&simd));
			// an infinite entry is not NaN, and the NaN check must not turn
			// `inf` into NaN either
			let mut inf = mat.clone();
			inf[(m - 1, n - 1)] = scale.mul_real(infinity::<T::Real>());
			let (simd, scalar) = norm_max_both_paths(&inf);
			assert!(simd == infinity::<T::Real>());
			assert!(scalar == infinity::<T::Real>());
		}
	}

	#[test]
	fn test_norm_max_propagates_nan() {
		// #316: NaN in a real entry, the real part or the imaginary part of a
		// complex entry
		check_nan_propagates::<f64>(f64::NAN);
		check_nan_propagates::<f32>(f32::NAN);
		check_nan_propagates::<c64>(c64::new(f64::NAN, 0.0));
		check_nan_propagates::<c64>(c64::new(0.0, f64::NAN));
		check_nan_propagates::<c32>(c32::new(f32::NAN, 1.0));
		check_nan_propagates::<c32>(c32::new(1.0, f32::NAN));

		// the reproducer of #316
		let a = mat![[1.0f64, f64::NAN]];
		assert!(a.norm_max().is_nan());
		assert!(a.norm_l1().is_nan() && a.norm_l2().is_nan());
	}

	#[test]
	fn test_norm_max_finite_and_inf_paths_agree() {
		check_finite_and_inf_agree::<f64>(1.0);
		check_finite_and_inf_agree::<f32>(1.0);
		check_finite_and_inf_agree::<c64>(c64::new(0.5, -2.0));
		check_finite_and_inf_agree::<c32>(c32::new(-3.0, 0.25));
	}

	#[test]
	fn test_norm_max_empty_is_zero() {
		for (m, n) in [(0, 0), (0, 3), (3, 0)] {
			assert!(Mat::<f64>::zeros(m, n).norm_max() == 0.0);
			assert!(Mat::<c64>::zeros(m, n).norm_max() == 0.0);
		}
	}
}
