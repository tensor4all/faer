use crate::assert;
use crate::internal_prelude::*;
use faer_traits::RealReg;
/// Specifies how missing values should be handled in mean and variance
/// computations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NanHandling {
	/// NaNs are passed as-is to arithmetic operators.
	Propagate,
	/// NaNs are skipped, and they're not included in the total count of
	/// entries.
	Ignore,
}
#[inline(always)]
fn from_usize<T: RealField>(n: usize) -> T {
	from_f64::<T>(n as u32 as f64)
		+ from_f64::<T>((n as u64 - (n as u32 as u64)) as f64)
}
#[inline(always)]
fn reduce<T: ComplexField, S: pulp::Simd>(
	non_nan_count: T::SimdIndex<S>,
) -> usize {
	let slice: &[T::Index] =
		bytemuck::cast_slice(core::slice::from_ref(&non_nan_count));
	let mut acc = 0usize;
	for &count in slice {
		acc += count.zx();
	}
	acc
}
fn col_mean_row_major_ignore_nan<T: ComplexField>(
	out: ColMut<'_, T>,
	mat: MatRef<'_, T, usize, usize, isize, ContiguousFwd>,
) {
	struct Impl<'a, T: ComplexField> {
		out: ColMut<'a, T>,
		mat: MatRef<'a, T, usize, usize, isize, ContiguousFwd>,
	}
	impl<T: ComplexField> pulp::WithSimd for Impl<'_, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self { out, mat } = self;
			with_dim!(M, mat.nrows());
			with_dim!(N, mat.ncols());
			let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), N);
			let mut out = out.as_row_shape_mut(M);
			let mat = mat.as_shape(M, N);
			let indices = simd.batch_indices::<4>();
			let nan = simd.splat(&nan::<T>());
			for i in M.indices() {
				let row = mat.row(i).transpose();
				let (head, mut body4, body1, tail) = indices.clone();
				let mut non_nan_count_total = 0usize;
				#[inline(always)]
				fn process<'M, T: ComplexField, S: pulp::Simd>(
					simd: SimdCtx<'M, T, S>,
					acc: T::SimdVec<S>,
					non_nan_count: T::SimdIndex<S>,
					val: T::SimdVec<S>,
				) -> (T::SimdVec<S>, T::SimdIndex<S>) {
					let is_not_nan = (*simd).eq(val, val);
					(
						simd.select(is_not_nan, simd.add(acc, val), acc),
						simd.iselect(
							is_not_nan,
							simd.iadd(
								non_nan_count,
								simd.isplat(T::Index::truncate(1)),
							),
							non_nan_count,
						),
					)
				}
				let mut sum0 = simd.splat(&zero::<T>());
				let mut sum1 = simd.splat(&zero::<T>());
				let mut sum2 = simd.splat(&zero::<T>());
				let mut sum3 = simd.splat(&zero::<T>());
				let mut non_nan_count0 = simd.isplat(T::Index::truncate(0));
				let mut non_nan_count1 = simd.isplat(T::Index::truncate(0));
				let mut non_nan_count2 = simd.isplat(T::Index::truncate(0));
				let mut non_nan_count3 = simd.isplat(T::Index::truncate(0));
				if let Some(i) = head {
					(sum0, non_nan_count0) = process(
						simd,
						sum0,
						non_nan_count0,
						simd.select(simd.head_mask(), simd.read(row, i), nan),
					);
					non_nan_count_total += reduce::<T, S>(non_nan_count0);
					non_nan_count0 = simd.isplat(T::Index::truncate(0));
				}
				loop {
					if body4.len() == 0 {
						break;
					}
					for [i0, i1, i2, i3] in (&mut body4).take(256) {
						(sum0, non_nan_count0) = process(
							simd,
							sum0,
							non_nan_count0,
							simd.read(row, i0),
						);
						(sum1, non_nan_count1) = process(
							simd,
							sum1,
							non_nan_count1,
							simd.read(row, i1),
						);
						(sum2, non_nan_count2) = process(
							simd,
							sum2,
							non_nan_count2,
							simd.read(row, i2),
						);
						(sum3, non_nan_count3) = process(
							simd,
							sum3,
							non_nan_count3,
							simd.read(row, i3),
						);
					}
					non_nan_count0 = simd.iadd(non_nan_count0, non_nan_count1);
					non_nan_count2 = simd.iadd(non_nan_count2, non_nan_count3);
					non_nan_count0 = simd.iadd(non_nan_count0, non_nan_count2);
					non_nan_count_total += reduce::<T, S>(non_nan_count0);
					non_nan_count0 = simd.isplat(T::Index::truncate(0));
					non_nan_count1 = simd.isplat(T::Index::truncate(0));
					non_nan_count2 = simd.isplat(T::Index::truncate(0));
					non_nan_count3 = simd.isplat(T::Index::truncate(0));
				}
				for i in body1 {
					(sum0, non_nan_count0) =
						process(simd, sum0, non_nan_count0, simd.read(row, i));
				}
				if let Some(i) = tail {
					(sum0, non_nan_count0) = process(
						simd,
						sum0,
						non_nan_count0,
						simd.select(simd.tail_mask(), simd.read(row, i), nan),
					);
				}
				non_nan_count_total += reduce::<T, S>(non_nan_count0);
				sum0 = simd.add(sum0, sum1);
				sum2 = simd.add(sum2, sum3);
				sum0 = simd.add(sum0, sum2);
				let sum = simd.reduce_sum(sum0);
				out[i] = sum.mul_real(
					from_usize::<T::Real>(non_nan_count_total).recip(),
				);
			}
		}
	}
	T::Arch::default().dispatch(Impl { out, mat });
}
fn col_varm_row_major_ignore_nan<T: ComplexField>(
	out: ColMut<'_, T::Real>,
	mat: MatRef<'_, T, usize, usize, isize, ContiguousFwd>,
	col_mean: ColRef<'_, T>,
) {
	struct Impl<'a, T: ComplexField> {
		out: ColMut<'a, T::Real>,
		mat: MatRef<'a, T, usize, usize, isize, ContiguousFwd>,
		col_mean: ColRef<'a, T>,
	}
	impl<T: ComplexField> pulp::WithSimd for Impl<'_, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self { out, mat, col_mean } = self;
			with_dim!(M, mat.nrows());
			with_dim!(N, mat.ncols());
			let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), N);
			let mut out = out.as_row_shape_mut(M);
			let mat = mat.as_shape(M, N);
			let col_mean = col_mean.as_row_shape(M);
			let indices = simd.batch_indices::<4>();
			let nan_v = simd.splat(&nan::<T>());
			for i in M.indices() {
				let row = mat.row(i).transpose();
				let mean = simd.splat(&col_mean[i]);
				let (head, mut body4, body1, tail) = indices.clone();
				let mut non_nan_count = 0usize;
				#[inline(always)]
				fn process<'M, T: ComplexField, S: pulp::Simd>(
					simd: SimdCtx<'M, T, S>,
					mean: T::SimdVec<S>,
					acc: RealReg<T::SimdVec<S>>,
					non_nan_count: T::SimdIndex<S>,
					val: T::SimdVec<S>,
				) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>) {
					let is_not_nan = (*simd).eq(val, val);
					let diff = simd.sub(val, mean);
					(
						RealReg(simd.select(
							is_not_nan,
							simd.abs2_add(diff, acc).0,
							acc.0,
						)),
						simd.iselect(
							is_not_nan,
							simd.iadd(
								non_nan_count,
								simd.isplat(T::Index::truncate(1)),
							),
							non_nan_count,
						),
					)
				}
				let mut sum0 = RealReg(simd.splat(&zero::<T>()));
				let mut sum1 = RealReg(simd.splat(&zero::<T>()));
				let mut sum2 = RealReg(simd.splat(&zero::<T>()));
				let mut sum3 = RealReg(simd.splat(&zero::<T>()));
				let mut non_nan_count0 = simd.isplat(T::Index::truncate(0));
				let mut non_nan_count1 = simd.isplat(T::Index::truncate(0));
				let mut non_nan_count2 = simd.isplat(T::Index::truncate(0));
				let mut non_nan_count3 = simd.isplat(T::Index::truncate(0));
				if let Some(i) = head {
					(sum0, non_nan_count0) = process(
						simd,
						mean,
						sum0,
						non_nan_count0,
						simd.select(simd.head_mask(), simd.read(row, i), nan_v),
					);
					non_nan_count += reduce::<T, S>(non_nan_count0);
					non_nan_count0 = simd.isplat(T::Index::truncate(0));
				}
				loop {
					if body4.len() == 0 {
						break;
					}
					for [i0, i1, i2, i3] in (&mut body4).take(256) {
						(sum0, non_nan_count0) = process(
							simd,
							mean,
							sum0,
							non_nan_count0,
							simd.read(row, i0),
						);
						(sum1, non_nan_count1) = process(
							simd,
							mean,
							sum1,
							non_nan_count1,
							simd.read(row, i1),
						);
						(sum2, non_nan_count2) = process(
							simd,
							mean,
							sum2,
							non_nan_count2,
							simd.read(row, i2),
						);
						(sum3, non_nan_count3) = process(
							simd,
							mean,
							sum3,
							non_nan_count3,
							simd.read(row, i3),
						);
					}
					non_nan_count0 = simd.iadd(non_nan_count0, non_nan_count1);
					non_nan_count2 = simd.iadd(non_nan_count2, non_nan_count3);
					non_nan_count0 = simd.iadd(non_nan_count0, non_nan_count2);
					non_nan_count += reduce::<T, S>(non_nan_count0);
					non_nan_count0 = simd.isplat(T::Index::truncate(0));
					non_nan_count1 = simd.isplat(T::Index::truncate(0));
					non_nan_count2 = simd.isplat(T::Index::truncate(0));
					non_nan_count3 = simd.isplat(T::Index::truncate(0));
				}
				for i in body1 {
					(sum0, non_nan_count0) = process(
						simd,
						mean,
						sum0,
						non_nan_count0,
						simd.read(row, i),
					);
				}
				if let Some(i) = tail {
					(sum0, non_nan_count0) = process(
						simd,
						mean,
						sum0,
						non_nan_count0,
						simd.select(simd.tail_mask(), simd.read(row, i), nan_v),
					);
				}
				non_nan_count += reduce::<T, S>(non_nan_count0);
				sum0 = RealReg(simd.add(sum0.0, sum1.0));
				sum2 = RealReg(simd.add(sum2.0, sum3.0));
				sum0 = RealReg(simd.add(sum0.0, sum2.0));
				let sum = simd.reduce_sum(sum0.0).real();
				if non_nan_count == 0 {
					out[i] = nan();
				} else if non_nan_count == 1 {
					out[i] = zero();
				} else {
					out[i] = sum.mul_real(
						from_usize::<T::Real>(non_nan_count - 1).recip(),
					);
				}
			}
		}
	}
	T::Arch::default().dispatch(Impl { out, mat, col_mean });
}
fn col_mean_row_major_propagate_nan<T: ComplexField>(
	out: ColMut<'_, T>,
	mat: MatRef<'_, T, usize, usize, isize, ContiguousFwd>,
) {
	struct Impl<'a, T: ComplexField> {
		out: ColMut<'a, T>,
		mat: MatRef<'a, T, usize, usize, isize, ContiguousFwd>,
	}
	impl<T: ComplexField> pulp::WithSimd for Impl<'_, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self { out, mat } = self;
			with_dim!(M, mat.nrows());
			with_dim!(N, mat.ncols());
			let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), N);
			let mut out = out.as_row_shape_mut(M);
			let mat = mat.as_shape(M, N);
			let indices = simd.batch_indices::<4>();
			let ref n = from_usize::<T::Real>(*N).recip();
			for i in M.indices() {
				let row = mat.row(i).transpose();
				let (head, body4, body1, tail) = indices.clone();
				let mut sum0 = simd.splat(zero::<T>());
				let mut sum1 = simd.splat(zero::<T>());
				let mut sum2 = simd.splat(zero::<T>());
				let mut sum3 = simd.splat(zero::<T>());
				if let Some(i) = head {
					sum0 = simd.add(sum0, simd.read(row, i));
				}
				for [i0, i1, i2, i3] in body4 {
					sum0 = simd.add(sum0, simd.read(row, i0));
					sum1 = simd.add(sum1, simd.read(row, i1));
					sum2 = simd.add(sum2, simd.read(row, i2));
					sum3 = simd.add(sum3, simd.read(row, i3));
				}
				for i in body1 {
					sum0 = simd.add(sum0, simd.read(row, i));
				}
				if let Some(i) = tail {
					sum0 = simd.add(sum0, simd.read(row, i));
				}
				sum0 = simd.add(sum0, sum1);
				sum2 = simd.add(sum2, sum3);
				sum0 = simd.add(sum0, sum2);
				let sum = simd.reduce_sum(sum0);
				out[i] = sum.mul_real(n);
			}
		}
	}
	T::Arch::default().dispatch(Impl { out, mat });
}
fn col_varm_row_major_propagate_nan<T: ComplexField>(
	out: ColMut<'_, T::Real>,
	mat: MatRef<'_, T, usize, usize, isize, ContiguousFwd>,
	col_mean: ColRef<'_, T>,
) {
	struct Impl<'a, T: ComplexField> {
		out: ColMut<'a, T::Real>,
		mat: MatRef<'a, T, usize, usize, isize, ContiguousFwd>,
		col_mean: ColRef<'a, T>,
	}
	impl<T: ComplexField> pulp::WithSimd for Impl<'_, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self { out, mat, col_mean } = self;
			with_dim!(M, mat.nrows());
			with_dim!(N, mat.ncols());
			let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), N);
			let mut out = out.as_row_shape_mut(M);
			let mat = mat.as_shape(M, N);
			let col_mean = col_mean.as_row_shape(M);
			let indices = simd.batch_indices::<4>();
			let n = *N;
			if n == 0 {
				out.fill(nan());
			} else if n == 1 {
				out.fill(zero());
			} else {
				let ref n = from_usize::<T::Real>(n - 1).recip();
				for i in M.indices() {
					let row = mat.row(i).transpose();
					let mean = simd.splat(&col_mean[i]);
					let (head, body4, body1, tail) = indices.clone();
					let mut sum0 = simd.splat(zero::<T>());
					let mut sum1 = simd.splat(zero::<T>());
					let mut sum2 = simd.splat(zero::<T>());
					let mut sum3 = simd.splat(zero::<T>());
					if let Some(i0) = head {
						sum0 = simd.select(
							simd.head_mask(),
							simd.abs2_add(
								simd.sub(simd.read(row, i0), mean),
								RealReg(sum0),
							)
							.0,
							sum0,
						);
					}
					for [i0, i1, i2, i3] in body4 {
						sum0 = simd
							.abs2_add(
								simd.sub(simd.read(row, i0), mean),
								RealReg(sum0),
							)
							.0;
						sum1 = simd
							.abs2_add(
								simd.sub(simd.read(row, i1), mean),
								RealReg(sum1),
							)
							.0;
						sum2 = simd
							.abs2_add(
								simd.sub(simd.read(row, i2), mean),
								RealReg(sum2),
							)
							.0;
						sum3 = simd
							.abs2_add(
								simd.sub(simd.read(row, i3), mean),
								RealReg(sum3),
							)
							.0;
					}
					for i0 in body1 {
						sum0 = simd
							.abs2_add(
								simd.sub(simd.read(row, i0), mean),
								RealReg(sum0),
							)
							.0;
					}
					if let Some(i0) = tail {
						sum0 = simd.select(
							simd.tail_mask(),
							simd.abs2_add(
								simd.sub(simd.read(row, i0), mean),
								RealReg(sum0),
							)
							.0,
							sum0,
						);
					}
					sum0 = simd.add(sum0, sum1);
					sum2 = simd.add(sum2, sum3);
					sum0 = simd.add(sum0, sum2);
					let sum = simd.reduce_sum(sum0).real();
					out[i] = sum * n;
				}
			}
		}
	}
	T::Arch::default().dispatch(Impl { out, mat, col_mean });
}
fn col_mean_ignore_nan_fallback<T: ComplexField>(
	out: ColMut<'_, T>,
	mat: MatRef<'_, T>,
) {
	with_dim!(M, mat.nrows());
	with_dim!(N, mat.ncols());
	let non_nan_count = &mut *alloc::vec![0usize; * M];
	let non_nan_count = Array::from_mut(non_nan_count, M);
	let mut out = out.as_row_shape_mut(M);
	let mat = mat.as_shape(M, N);
	out.fill(zero());
	for j in N.indices() {
		for i in M.indices() {
			let val = mat[(i, j)].copy();
			let nan = val.is_nan();
			let val = if nan { zero::<T>() } else { val };
			non_nan_count[i] += (!nan) as usize;
			out[i] += val;
		}
	}
	for i in M.indices() {
		out[i] =
			out[i].mul_real(from_usize::<T::Real>(non_nan_count[i]).recip());
	}
}
fn col_varm_ignore_nan_fallback<T: ComplexField>(
	out: ColMut<'_, T::Real>,
	mat: MatRef<'_, T>,
	col_mean: ColRef<'_, T>,
) {
	with_dim!(M, mat.nrows());
	with_dim!(N, mat.ncols());
	let non_nan_count = &mut *alloc::vec![0usize; * M];
	let non_nan_count = Array::from_mut(non_nan_count, M);
	let mut out = out.as_row_shape_mut(M);
	let mat = mat.as_shape(M, N);
	let col_mean = col_mean.as_row_shape(M);
	out.fill(zero());
	for j in N.indices() {
		for i in M.indices() {
			let val = mat[(i, j)].copy();
			let col_mean = col_mean[i].copy();
			let nan = val.is_nan();
			let val = if nan {
				zero::<T::Real>()
			} else {
				(val - col_mean).abs2()
			};
			non_nan_count[i] += (!nan) as usize;
			out[i] += val;
		}
	}
	for i in M.indices() {
		let non_nan_count = non_nan_count[i];
		if non_nan_count == 0 {
			out[i] = nan();
		} else if non_nan_count == 1 {
			out[i] = zero();
		} else {
			out[i] = out[i]
				.mul_real(from_usize::<T::Real>(non_nan_count - 1).recip());
		}
	}
}
fn col_mean_propagate_nan_fallback<T: ComplexField>(
	out: ColMut<'_, T>,
	mat: MatRef<'_, T>,
) {
	with_dim!(M, mat.nrows());
	with_dim!(N, mat.ncols());
	let mut out = out.as_row_shape_mut(M);
	let mat = mat.as_shape(M, N);
	out.fill(zero());
	for j in N.indices() {
		for i in M.indices() {
			out[i] += &mat[(i, j)];
		}
	}
	let ref n = from_usize::<T::Real>(*N).recip();
	for i in M.indices() {
		out[i] = out[i].mul_real(n);
	}
}
fn col_varm_propagate_nan_fallback<T: ComplexField>(
	out: ColMut<'_, T::Real>,
	mat: MatRef<'_, T>,
	col_mean: ColRef<'_, T>,
) {
	with_dim!(M, mat.nrows());
	with_dim!(N, mat.ncols());
	let mut out = out.as_row_shape_mut(M);
	let mat = mat.as_shape(M, N);
	let col_mean = col_mean.as_row_shape(M);
	out.fill(zero());
	for j in N.indices() {
		for i in M.indices() {
			let val = (&mat[(i, j)] - &col_mean[i]).abs2();
			out[i] += val;
		}
	}
	let n = *N;
	if n == 0 {
		out.fill(nan());
	} else if n == 1 {
		out.fill(zero());
	} else {
		let ref n = from_usize::<T::Real>(*N - 1).recip();
		for i in M.indices() {
			out[i] = out[i].mul_real(n);
		}
	}
}
fn col_mean_ignore<T: ComplexField>(out: ColMut<'_, T>, mat: MatRef<'_, T>) {
	let mat = if mat.col_stride() >= 0 {
		mat
	} else {
		mat.reverse_cols()
	};
	// rows are the output axis: reverse `out` with them, or entry `i` lands at
	// `nrows - 1 - i` (#317)
	let (mat, out) = if mat.row_stride() >= 0 {
		(mat, out)
	} else {
		(mat.reverse_rows(), out.reverse_rows_mut())
	};
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if mat.ncols() > 1 && mat.col_stride() == 1 {
			col_mean_row_major_ignore_nan(out, mat.try_as_row_major().unwrap());
		} else {
			col_mean_ignore_nan_fallback(out, mat);
		}
	} else {
		col_mean_ignore_nan_fallback(out, mat);
	}
}
fn col_varm_ignore<T: ComplexField>(
	out: ColMut<'_, T::Real>,
	mat: MatRef<'_, T>,
	col_mean: ColRef<'_, T>,
) {
	let mat = if mat.col_stride() >= 0 {
		mat
	} else {
		mat.reverse_cols()
	};
	// rows are the output axis: reverse `out` and `col_mean` with them, or
	// entry `i` lands at `nrows - 1 - i` (#317)
	let (mat, out, col_mean) = if mat.row_stride() >= 0 {
		(mat, out, col_mean)
	} else {
		(
			mat.reverse_rows(),
			out.reverse_rows_mut(),
			col_mean.reverse_rows(),
		)
	};
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if mat.ncols() > 1 && mat.col_stride() == 1 {
			col_varm_row_major_ignore_nan(
				out,
				mat.try_as_row_major().unwrap(),
				col_mean,
			);
		} else {
			col_varm_ignore_nan_fallback(out, mat, col_mean);
		}
	} else {
		col_varm_ignore_nan_fallback(out, mat, col_mean);
	}
}
fn col_mean_propagate<T: ComplexField>(out: ColMut<'_, T>, mat: MatRef<'_, T>) {
	let mat = if mat.col_stride() >= 0 {
		mat
	} else {
		mat.reverse_cols()
	};
	// rows are the output axis: reverse `out` with them, or entry `i` lands at
	// `nrows - 1 - i` (#317)
	let (mat, out) = if mat.row_stride() >= 0 {
		(mat, out)
	} else {
		(mat.reverse_rows(), out.reverse_rows_mut())
	};
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if mat.ncols() > 1 && mat.col_stride() == 1 {
			col_mean_row_major_propagate_nan(
				out,
				mat.try_as_row_major().unwrap(),
			);
		} else {
			col_mean_propagate_nan_fallback(out, mat);
		}
	} else {
		col_mean_propagate_nan_fallback(out, mat);
	}
}
fn col_varm_propagate<T: ComplexField>(
	out: ColMut<'_, T::Real>,
	mat: MatRef<'_, T>,
	col_mean: ColRef<'_, T>,
) {
	let mat = if mat.col_stride() >= 0 {
		mat
	} else {
		mat.reverse_cols()
	};
	// rows are the output axis: reverse `out` and `col_mean` with them, or
	// entry `i` lands at `nrows - 1 - i` (#317)
	let (mat, out, col_mean) = if mat.row_stride() >= 0 {
		(mat, out, col_mean)
	} else {
		(
			mat.reverse_rows(),
			out.reverse_rows_mut(),
			col_mean.reverse_rows(),
		)
	};
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if mat.ncols() > 1 && mat.col_stride() == 1 {
			col_varm_row_major_propagate_nan(
				out,
				mat.try_as_row_major().unwrap(),
				col_mean,
			);
		} else {
			col_varm_propagate_nan_fallback(out, mat, col_mean);
		}
	} else {
		col_varm_propagate_nan_fallback(out, mat, col_mean);
	}
}
/// computes the mean of the columns of `mat` and stores the result in `out`
#[track_caller]
pub fn col_mean<T: ComplexField>(
	out: ColMut<'_, T>,
	mat: MatRef<'_, T>,
	nan: NanHandling,
) {
	assert!(all(out.nrows() == mat.nrows()));
	match nan {
		NanHandling::Propagate => col_mean_propagate(out, mat),
		NanHandling::Ignore => col_mean_ignore(out, mat),
	}
}
/// computes the mean of the rows of `mat` and stores the result in `out`
#[track_caller]
pub fn row_mean<T: ComplexField>(
	out: RowMut<'_, T>,
	mat: MatRef<'_, T>,
	nan: NanHandling,
) {
	assert!(all(out.ncols() == mat.ncols()));
	col_mean(out.transpose_mut(), mat.transpose(), nan);
}
/// computes the variance of the columns of `mat` and stores the result in `out`
#[track_caller]
pub fn col_varm<T: ComplexField>(
	out: ColMut<'_, T::Real>,
	mat: MatRef<'_, T>,
	col_mean: ColRef<'_, T>,
	nan: NanHandling,
) {
	assert!(all(
		out.nrows() == mat.nrows(),
		col_mean.nrows() == mat.nrows()
	));
	match nan {
		NanHandling::Propagate => col_varm_propagate(out, mat, col_mean),
		NanHandling::Ignore => col_varm_ignore(out, mat, col_mean),
	}
}
/// computes the variance of the rows of `mat` and stores the result in `out`
#[track_caller]
pub fn row_varm<T: ComplexField>(
	out: RowMut<'_, T::Real>,
	mat: MatRef<'_, T>,
	row_mean: RowRef<'_, T>,
	nan: NanHandling,
) {
	assert!(all(
		out.ncols() == mat.ncols(),
		row_mean.ncols() == mat.ncols()
	));
	col_varm(
		out.transpose_mut(),
		mat.transpose(),
		row_mean.transpose(),
		nan,
	);
}
#[cfg(test)]
mod tests {
	use super::*;
	use equator::assert;
	#[test]
	fn test_meanvar_propagate() {
		let c32 = c32::new;
		let A = mat![
			[c32(1.2, 2.3), c32(3.4, 1.2)],
			[c32(1.7, -1.0), c32(-3.8, 1.95)],
		];
		let mut row_mean = Row::zeros(A.ncols());
		let mut row_var = Row::zeros(A.ncols());
		super::row_mean(row_mean.as_mut(), A.as_ref(), NanHandling::Propagate);
		super::row_varm(
			row_var.as_mut(),
			A.as_ref(),
			row_mean.as_ref(),
			NanHandling::Propagate,
		);
		let mut col_mean = Col::zeros(A.nrows());
		let mut col_var = Col::zeros(A.nrows());
		super::col_mean(col_mean.as_mut(), A.as_ref(), NanHandling::Propagate);
		super::col_varm(
			col_var.as_mut(),
			A.as_ref(),
			col_mean.as_ref(),
			NanHandling::Propagate,
		);
		assert!(
			row_mean
				== row![
					(A[(0, 0)] + A[(1, 0)]) / 2.0,
					(A[(0, 1)] + A[(1, 1)]) / 2.0,
				]
		);
		assert!(
			row_var
				== row![
					(A[(0, 0)] - row_mean[0]).abs2()
						+ (A[(1, 0)] - row_mean[0]).abs2(),
					(A[(0, 1)] - row_mean[1]).abs2()
						+ (A[(1, 1)] - row_mean[1]).abs2(),
				]
		);
		assert!(
			col_mean
				== col![
					(A[(0, 0)] + A[(0, 1)]) / 2.0,
					(A[(1, 0)] + A[(1, 1)]) / 2.0,
				]
		);
		assert!(
			col_var
				== col![
					(A[(0, 0)] - col_mean[0]).abs2()
						+ (A[(0, 1)] - col_mean[0]).abs2(),
					(A[(1, 0)] - col_mean[1]).abs2()
						+ (A[(1, 1)] - col_mean[1]).abs2(),
				]
		);
	}
	#[test]
	fn test_meanvar_ignore_nan_nonan_c32() {
		let c32 = c32::new;
		let A = mat![
			[c32(1.2, 2.3), c32(3.4, 1.2)],
			[c32(1.7, -1.0), c32(-3.8, 1.95)],
		];
		let mut row_mean = Row::zeros(A.ncols());
		let mut row_var = Row::zeros(A.ncols());
		super::row_mean(row_mean.as_mut(), A.as_ref(), NanHandling::Ignore);
		super::row_varm(
			row_var.as_mut(),
			A.as_ref(),
			row_mean.as_ref(),
			NanHandling::Ignore,
		);
		let mut col_mean = Col::zeros(A.nrows());
		let mut col_var = Col::zeros(A.nrows());
		super::col_mean(col_mean.as_mut(), A.as_ref(), NanHandling::Ignore);
		super::col_varm(
			col_var.as_mut(),
			A.as_ref(),
			col_mean.as_ref(),
			NanHandling::Ignore,
		);
		assert!(
			row_mean
				== row![
					(A[(0, 0)] + A[(1, 0)]) / 2.0,
					(A[(0, 1)] + A[(1, 1)]) / 2.0,
				]
		);
		assert!(
			row_var
				== row![
					(A[(0, 0)] - row_mean[0]).abs2()
						+ (A[(1, 0)] - row_mean[0]).abs2(),
					(A[(0, 1)] - row_mean[1]).abs2()
						+ (A[(1, 1)] - row_mean[1]).abs2(),
				]
		);
		assert!(
			col_mean
				== col![
					(A[(0, 0)] + A[(0, 1)]) / 2.0,
					(A[(1, 0)] + A[(1, 1)]) / 2.0,
				]
		);
		assert!(
			col_var
				== col![
					(A[(0, 0)] - col_mean[0]).abs2()
						+ (A[(0, 1)] - col_mean[0]).abs2(),
					(A[(1, 0)] - col_mean[1]).abs2()
						+ (A[(1, 1)] - col_mean[1]).abs2(),
				]
		);
	}
	#[test]
	fn test_meanvar_ignore_nan_yesnan_c32() {
		let c32 = c32::new;
		let nan = f32::NAN;
		let A = mat![
			[c32(1.2, nan), c32(3.4, 1.2)],
			[c32(1.7, -1.0), c32(-3.8, 1.95)],
		];
		let mut row_mean = Row::zeros(A.ncols());
		let mut row_var = Row::zeros(A.ncols());
		super::row_mean(row_mean.as_mut(), A.as_ref(), NanHandling::Ignore);
		super::row_varm(
			row_var.as_mut(),
			A.as_ref(),
			row_mean.as_ref(),
			NanHandling::Ignore,
		);
		let mut col_mean = Col::zeros(A.nrows());
		let mut col_var = Col::zeros(A.nrows());
		super::col_mean(col_mean.as_mut(), A.as_ref(), NanHandling::Ignore);
		super::col_varm(
			col_var.as_mut(),
			A.as_ref(),
			col_mean.as_ref(),
			NanHandling::Ignore,
		);
		assert!(
			row_mean == row![A[(1, 0)] / 1.0, (A[(0, 1)] + A[(1, 1)]) / 2.0,]
		);
		assert!(
			row_var
				== row![
					(A[(1, 0)] - row_mean[0]).abs2(),
					(A[(0, 1)] - row_mean[1]).abs2()
						+ (A[(1, 1)] - row_mean[1]).abs2(),
				]
		);
		assert!(
			col_mean == col![A[(0, 1)] / 1.0, (A[(1, 0)] + A[(1, 1)]) / 2.0,]
		);
		assert!(
			col_var
				== col![
					(A[(0, 1)] - col_mean[0]).abs2(),
					(A[(1, 0)] - col_mean[1]).abs2()
						+ (A[(1, 1)] - col_mean[1]).abs2(),
				]
		);
	}
	#[test]
	fn test_meanvar_ignore_nan_nonan_c64() {
		let c64 = c64::new;
		let A = mat![
			[c64(1.2, 2.3), c64(3.4, 1.2)],
			[c64(1.7, -1.0), c64(-3.8, 1.95)],
		];
		let mut row_mean = Row::zeros(A.ncols());
		let mut row_var = Row::zeros(A.ncols());
		super::row_mean(row_mean.as_mut(), A.as_ref(), NanHandling::Ignore);
		super::row_varm(
			row_var.as_mut(),
			A.as_ref(),
			row_mean.as_ref(),
			NanHandling::Ignore,
		);
		let mut col_mean = Col::zeros(A.nrows());
		let mut col_var = Col::zeros(A.nrows());
		super::col_mean(col_mean.as_mut(), A.as_ref(), NanHandling::Ignore);
		super::col_varm(
			col_var.as_mut(),
			A.as_ref(),
			col_mean.as_ref(),
			NanHandling::Ignore,
		);
		assert!(
			row_mean
				== row![
					(A[(0, 0)] + A[(1, 0)]) / 2.0,
					(A[(0, 1)] + A[(1, 1)]) / 2.0,
				]
		);
		assert!(
			row_var
				== row![
					(A[(0, 0)] - row_mean[0]).abs2()
						+ (A[(1, 0)] - row_mean[0]).abs2(),
					(A[(0, 1)] - row_mean[1]).abs2()
						+ (A[(1, 1)] - row_mean[1]).abs2(),
				]
		);
		assert!(
			col_mean
				== col![
					(A[(0, 0)] + A[(0, 1)]) / 2.0,
					(A[(1, 0)] + A[(1, 1)]) / 2.0,
				]
		);
		assert!(
			col_var
				== col![
					(A[(0, 0)] - col_mean[0]).abs2()
						+ (A[(0, 1)] - col_mean[0]).abs2(),
					(A[(1, 0)] - col_mean[1]).abs2()
						+ (A[(1, 1)] - col_mean[1]).abs2(),
				]
		);
	}
	#[test]
	fn test_meanvar_ignore_nan_yesnan_c64() {
		let c64 = c64::new;
		let nan = f64::NAN;
		let A = mat![
			[c64(1.2, nan), c64(3.4, 1.2)],
			[c64(1.7, -1.0), c64(-3.8, 1.95)],
		];
		let mut row_mean = Row::zeros(A.ncols());
		let mut row_var = Row::zeros(A.ncols());
		super::row_mean(row_mean.as_mut(), A.as_ref(), NanHandling::Ignore);
		super::row_varm(
			row_var.as_mut(),
			A.as_ref(),
			row_mean.as_ref(),
			NanHandling::Ignore,
		);
		let mut col_mean = Col::zeros(A.nrows());
		let mut col_var = Col::zeros(A.nrows());
		super::col_mean(col_mean.as_mut(), A.as_ref(), NanHandling::Ignore);
		super::col_varm(
			col_var.as_mut(),
			A.as_ref(),
			col_mean.as_ref(),
			NanHandling::Ignore,
		);
		assert!(
			row_mean == row![A[(1, 0)] / 1.0, (A[(0, 1)] + A[(1, 1)]) / 2.0,]
		);
		assert!(
			row_var
				== row![
					(A[(1, 0)] - row_mean[0]).abs2(),
					(A[(0, 1)] - row_mean[1]).abs2()
						+ (A[(1, 1)] - row_mean[1]).abs2(),
				]
		);
		assert!(
			col_mean == col![A[(0, 1)] / 1.0, (A[(1, 0)] + A[(1, 1)]) / 2.0,]
		);
		assert!(
			col_var
				== col![
					(A[(0, 1)] - col_mean[0]).abs2(),
					(A[(1, 0)] - col_mean[1]).abs2()
						+ (A[(1, 1)] - col_mean[1]).abs2(),
				]
		);
	}

	/// NaN-aware comparison with a relative tolerance: the view and the owned
	/// copy may take different kernels (row-major SIMD vs. fallback).
	fn close<T: ComplexField>(a: &T, b: &T) -> bool {
		if is_nan(a) || is_nan(b) {
			return is_nan(a) && is_nan(b);
		}
		let scale = abs(a).fmax(abs(b)).fmax(one::<T::Real>());
		abs(&(a - b)) <= scale * from_f64::<T::Real>(1e-5)
	}

	/// distinct per-row values (so a reversed output is visible) with one NaN
	fn sample<T: ComplexField>(m: usize, n: usize) -> Mat<T> {
		Mat::from_fn(m, n, |i, j| {
			if (i, j) == (1, 2) {
				nan()
			} else {
				from_f64::<T>(((i + 1) * (j + 2) + 3 * i) as f64)
			}
		})
	}

	/// `col_*` on `reverse_rows` and `row_*` on `reverse_cols` agree with the
	/// same matrix copied to an owned, positively strided one (#317).
	fn check_reversed_output_axis<T: ComplexField>() {
		let (m, n) = (5, 9);
		for row_major in [false, true] {
			let base = sample::<T>(m, n);
			let row_major_base = base.transpose().to_owned();
			let a = if row_major {
				row_major_base.transpose()
			} else {
				base.as_ref()
			};
			for nan in [NanHandling::Propagate, NanHandling::Ignore] {
				// col_* with a negative row stride
				let view = a.reverse_rows();
				assert!(view.row_stride() < 0);
				let owned = view.to_owned();
				let mut mean_owned = Col::<T>::zeros(m);
				col_mean(mean_owned.as_mut(), owned.as_ref(), nan);
				let mut mean_view = Col::<T>::zeros(m);
				col_mean(mean_view.as_mut(), view, nan);
				let mut var_owned = Col::<T::Real>::zeros(m);
				col_varm(
					var_owned.as_mut(),
					owned.as_ref(),
					mean_owned.as_ref(),
					nan,
				);
				let mut var_view = Col::<T::Real>::zeros(m);
				col_varm(var_view.as_mut(), view, mean_owned.as_ref(), nan);
				for i in 0..m {
					assert!(close(&mean_view[i], &mean_owned[i]));
					assert!(close(&var_view[i], &var_owned[i]));
				}

				// row_* with a negative column stride
				let view = a.transpose().reverse_cols();
				assert!(view.col_stride() < 0);
				let owned = view.to_owned();
				let mut mean_owned = Row::<T>::zeros(m);
				row_mean(mean_owned.as_mut(), owned.as_ref(), nan);
				let mut mean_view = Row::<T>::zeros(m);
				row_mean(mean_view.as_mut(), view, nan);
				let mut var_owned = Row::<T::Real>::zeros(m);
				row_varm(
					var_owned.as_mut(),
					owned.as_ref(),
					mean_owned.as_ref(),
					nan,
				);
				let mut var_view = Row::<T::Real>::zeros(m);
				row_varm(var_view.as_mut(), view, mean_owned.as_ref(), nan);
				for j in 0..m {
					assert!(close(&mean_view[j], &mean_owned[j]));
					assert!(close(&var_view[j], &var_owned[j]));
				}
			}
		}
	}

	#[test]
	fn test_meanvar_reversed_output_axis() {
		check_reversed_output_axis::<f64>();
		check_reversed_output_axis::<f32>();
		check_reversed_output_axis::<c64>();
		check_reversed_output_axis::<c32>();
	}

	#[test]
	fn test_row_meanvar_reverse_cols_issue_317() {
		// the reproducer of #317: column j holds [0, 2(j+1)]
		let a = Mat::<f64>::from_fn(2, 4, |i, j| {
			(j + 1) as f64 * if i == 0 { 0.0 } else { 2.0 }
		});
		let r = a.reverse_cols();
		for nan in [NanHandling::Propagate, NanHandling::Ignore] {
			let mut mean = Row::<f64>::zeros(4);
			row_mean(mean.as_mut(), r, nan);
			assert!(mean == row![4.0, 3.0, 2.0, 1.0]);
			let mut var = Row::<f64>::zeros(4);
			row_varm(var.as_mut(), r, mean.as_ref(), nan);
			assert!(var == row![32.0, 18.0, 8.0, 2.0]);
		}
	}
}
