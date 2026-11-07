pub use super::super::no_pivoting::factor::recommended_block_size;
use crate::assert;
use crate::internal_prelude::*;
use crate::utils::cost::{ParThreshold, rate};
use crate::utils::thread::with_width;
use crate::perm::swap_cols_idx;
use linalg::householder::{self, HouseholderInfo};
use pulp::Simd;
fn update_mat_and_dot_simd<T: ComplexField>(
	norm: RowMut<'_, T>,
	dot: RowMut<'_, T>,
	B01: RowMut<'_, T>,
	B11: MatMut<'_, T, usize, usize, ContiguousFwd>,
	A10: ColRef<'_, T, usize, ContiguousFwd>,
	B10: ColRef<'_, T, usize, ContiguousFwd>,
	l: T,
	tau_inv: T::Real,
	align: usize,
) {
	struct Impl<'a, 'M, 'N, T: ComplexField> {
		norm: RowMut<'a, T, Dim<'N>>,
		dot: RowMut<'a, T, Dim<'N>>,
		B01: RowMut<'a, T, Dim<'N>>,
		B11: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		A10: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		B10: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		l: T,
		tau_inv: T::Real,
		align: usize,
	}
	impl<'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'_, 'M, 'N, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let Self {
				mut norm,
				mut dot,
				B01: mut u,
				mut B11,
				A10,
				B10,
				l,
				ref tau_inv,
				align,
			} = self;
			let m = B11.nrows();
			let n = B11.ncols();
			let simd =
				SimdCtx::<'_, T, S>::new_align(T::simd_ctx(simd), m, align);
			let indices4 = simd.batch_indices::<4>();

			for j in n.indices() {
				let b0 = dot[j].copy();
				let rhs0 = simd.splat(&b0);
				let mut acc = [simd.zero(); 4];
				simd_iter!(for (IDX, i) in [indices4; 4] {
					let lhs0 = simd.read(A10, i);
					let lhs1 = simd.read(B10, i);
					let mut dst0 = simd.read(B11.rb().col(j), i);
					dst0 = simd.mul_add(lhs0, rhs0, dst0);
					acc[IDX] = simd.conj_mul_add(lhs1, dst0, acc[IDX]);
					simd.write(B11.rb_mut().col_mut(j), i, dst0);
				});
				let acc = simd
					.add(simd.add(acc[0], acc[2]), simd.add(acc[1], acc[3]));
				let tmp = &u[j] + &l * b0;
				let d0 = (&tmp + simd.reduce_sum(acc)).mul_real(-tau_inv);
				u[j] = tmp + &d0;
				dot[j] = d0;
				norm[j] = (norm[j].abs2() - u[j].abs2()).sqrt().to_cplx();
			}
		}
	}
	with_dim!(M, B11.nrows());
	with_dim!(N, B11.ncols());
	dispatch!(
		Impl {
			norm: norm.as_col_shape_mut(N),
			dot: dot.as_col_shape_mut(N),
			B01: B01.as_col_shape_mut(N),
			B11: B11.as_shape_mut(M, N),
			A10: A10.as_row_shape(M),
			B10: B10.as_row_shape(M),
			l,
			tau_inv,
			align
		},
		Impl,
		T
	)
}
/// $QR$ factorization with column pivoting tuning parameters
///
/// # Rank truncation
///
/// `max_rank`, `abs_tol` and `rel_tol` are all `None` by default, and then the
/// factorization runs to `min(m, n)` exactly as without them and reports
/// `rank = min(m, n)`. Setting any of them reports the number of accepted pivots in
/// [`ColPivQrInfo::rank`]:
///
/// - `max_rank = Some(r)`: only the first `r` steps are factored (clamped to `min(m, n)`),
///   and the remaining steps' trailing updates are skipped, so the work is `O(m n r)`
///   instead of `O(m n min(m, n))`. Rows `0..r` of the stored factor are the full
///   factorization's first `r` rows, and the `(m - r) x (n - r)` block below them is the
///   Schur complement after those steps.
/// - `abs_tol`, `rel_tol`: the pivot of step `k` is accepted while
///   `|R[k, k]| > max(abs_tol, rel_tol * |R[0, 0]|)`, where `|.|` is the modulus and an
///   absent tolerance counts as `0`; `rank` is the index of the first rejected diagonal
///   entry. `rel_tol` is measured against the **first** diagonal entry `R[0, 0]`, **not**
///   against the previous one. Column pivoting picks the largest remaining column norm at
///   every step, so this diagonal test is rank-revealing only up to that rule (LAPACK's
///   `xgeqp3` + `xlaic1` estimate is incremental); the tolerance is checked on the computed
///   factor, so it does not shorten the factorization; only `max_rank` does.
/// - Non-finite input: if the matrix holds a NaN or an infinity, the tolerances are
///   ignored and `rank` is `min(m, n)` capped by `max_rank`, so a non-finite entry never
///   shows up as a truncated rank. A NaN diagonal entry is never rejected, and a tolerance
///   is ignored when the threshold itself is not finite.
///
/// On `rank = r`: the first `r` entries of the permutation are the selected columns,
/// `R[:r, :]` (with everything below its diagonal zero) and the first `r` reflectors are
/// the rank-`r` factorization. The solve, inverse and reconstruct routines of this module
/// assume the full factorization and must not be given a truncated one.
#[derive(Copy, Clone, Debug)]
pub struct ColPivQrParams {
	/// threshold at which blocking algorithms should be disabled
	pub blocking_threshold: usize,
	/// floor of the parallel regions: [`ParThreshold::Derived`] (the default) runs a step in
	/// parallel when its serial time pays for one section; [`ParThreshold::Fixed`] counts
	/// elements of the trailing block `(m-k-1)*(n-k-1)`
	pub par_threshold: ParThreshold,
	/// factor at most this many steps (see
	/// [rank truncation](ColPivQrParams#rank-truncation))
	pub max_rank: Option<usize>,
	/// absolute diagonal tolerance (see [rank truncation](ColPivQrParams#rank-truncation))
	pub abs_tol: Option<f64>,
	/// diagonal tolerance relative to the first one (see
	/// [rank truncation](ColPivQrParams#rank-truncation))
	pub rel_tol: Option<f64>,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
impl<T: ComplexField> Auto<T> for ColPivQrParams {
	#[inline]
	fn auto() -> Self {
		Self {
			blocking_threshold: 48 * 48,
			par_threshold: ParThreshold::Derived,
			max_rank: None,
			abs_tol: None,
			rel_tol: None,
			non_exhaustive: NonExhaustive(()),
		}
	}
}
#[track_caller]
fn qr_in_place_unblocked<'out, I: Index, T: ComplexField>(
	A: MatMut<'_, T>,
	H: RowMut<'_, T>,
	col_perm: &'out mut [I],
	col_perm_inv: &'out mut [I],
	par: Par,
	stack: &mut MemStack,
	params: Spec<ColPivQrParams, T>,
) -> (ColPivQrInfo, PermRef<'out, I>) {
	let m = A.nrows();
	let n = A.ncols();
	let size = H.ncols();
	let params = params.config;
	// a non-finite input switches the tolerances off (see the params' rank truncation
	// section), and `max_rank` factors only the leading `r` steps
	let has_tol = params.abs_tol.is_some() || params.rel_tol.is_some();
	let reveal = has_tol && A.rb().is_all_finite();
	let steps = params.max_rank.map_or(size, |r| Ord::min(r, size));
	let mut A = A;
	let mut H = H;
	let mut par = par;
	// `Never` keeps the whole call sequential (the steps decide their own width otherwise)
	if matches!(params.par_threshold, ParThreshold::Never) {
		par = Par::Seq;
	}
	assert!(size == Ord::min(m, n));
	for j in 0..n {
		col_perm[j] = I::truncate(j);
	}
	let mut n_trans = 0;
	'main: {
		if size == 0 {
			break 'main;
		}
		let (mut dot, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
		let (mut norm, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
		let _ = stack;
		let mut dot = dot.as_mat_mut().col_mut(0).transpose_mut();
		let mut norm = norm.as_mat_mut().col_mut(0).transpose_mut();
		let mut best = zero();
		let ref threshold = eps::<T::Real>().sqrt();
		for j in 0..n {
			let val = A.rb().col(j).norm_l2();
			norm[j] = val.to_cplx();
			if val > best {
				best = val;
			}
		}
		// scaling by the largest column norm keeps the reduction in range; a zero
		// matrix has nothing to scale, and LAPACK's `dlarfg` makes an all-zero column
		// the identity reflector rather than a division by its norm.
		let ref scale_fwd = if best > zero() { best.copy() } else { one() };
		let ref scale_bwd = scale_fwd.recip();
		zip!(A.rb_mut()).for_each(|unzip!(a)| *a = a.mul_real(scale_bwd));
		for j in 0..n {
			norm[j] = (norm[j].real() * scale_bwd).to_cplx();
		}
		best = best * scale_bwd;
		let mut best_threshold = best * threshold;
		// the width the previous step's decision chose: it also covers the product that the
		// next step opens with when the update was not delayed
		let mut width = par.degree();
		'unscale: {
			for k in 0..steps {
				let mut new_best = zero::<T::Real>();
				let mut best_col = k;
				for j in k..n {
					let val = norm[j].real();
					if val > new_best {
						new_best = val;
						best_col = j;
					}
				}
				let delayed_update = T::SIMD_CAPABILITIES.is_simd()
					&& A.row_stride() == 1
					&& k > 0 && new_best >= best_threshold;
				if k > 0 && !delayed_update {
					let (_, _, A10, mut A11) = A.rb_mut().split_at_mut(k, k);
					let dot = dot.rb().get(k..);
					let A10 = A10.rb().col(k - 1);
					with_width(par.reborrow(), width, |par| {
						linalg::matmul::matmul(
							A11.rb_mut(),
							Accum::Add,
							A10,
							dot,
							one(),
							par,
						);
					});
					best = zero();
					for j in k..n {
						let val = A11.rb().col(j - k).norm_l2();
						norm[j] = val.to_cplx();
						if val > best {
							best = val;
							best_col = j;
						}
					}
					best_threshold = best * threshold;
				}
				if best_col != k {
					n_trans += 1;
					col_perm.as_mut().swap(best_col, k);
					swap_cols_idx(A.rb_mut(), best_col, k);
					swap_cols_idx(dot.rb_mut().as_mat_mut(), best_col, k);
					swap_cols_idx(norm.rb_mut().as_mat_mut(), best_col, k);
				}
				let (_, _, A10, mut A11) = A.rb_mut().split_at_mut(k, k);
				let A10 = A10.rb();
				let dot0 = dot.rb_mut().get_mut(k..);
				let (mut B00, B01, B10, mut B11) =
					A11.rb_mut().split_at_mut(1, 1);
				let B00 = &mut B00[(0, 0)];
				let mut B01 = B01.row_mut(0);
				let mut B10 = B10.col_mut(0);
				let l = if delayed_update {
					let A10 = A10.col(k - 1);
					A10[0].copy()
				} else {
					zero()
				};
				let r = dot0[0].copy();
				let mut dot = dot.rb_mut().get_mut(k + 1..);
				let mut norm = norm.rb_mut().get_mut(k + 1..);
				if delayed_update {
					let A10 = A10.col(k - 1).get(1..);
					*B00 += &l * &r;
					zip!(B10.rb_mut(), A10).for_each(|unzip!(x, y)| {
						*x += &r * y;
					});
				}
				let HouseholderInfo { tau, .. } =
					householder::make_householder_in_place(B00, B10.rb_mut());
				let ref tau_inv = tau.recip();
				H[k] = tau.to_cplx();
				// the last step of the full factorization has nothing left to update;
				// a truncated one stops after this step's update instead (below), so
				// its rows are the full factorization's first `steps` rows
				if k + 1 == size {
					if delayed_update {
						zip!(B01.rb_mut(), dot.rb()).for_each(
							|unzip!(x, y)| {
								*x += &l * y;
							},
						);
					}
					break 'unscale;
				}
				// One section per step: the fused update of the trailing block and the
				// norms (or, without the delayed update, the product, which the same
				// decision covers); the unit is an element of the trailing block.
				width = params.par_threshold.region_width::<T>(
					&par,
					(m - k - 1) * (n - k - 1),
					rate::qr_col_pivot(),
					1,
				);
				if delayed_update {
					let A10 = A10.col(k - 1).get(1..);
					with_width(par.reborrow(), width, |par| match par {
						Par::Seq => {
							update_mat_and_dot_simd(
								norm.rb_mut(),
								dot.rb_mut(),
								B01.rb_mut(),
								B11.rb_mut().try_as_col_major_mut().unwrap(),
								A10.try_as_col_major().unwrap(),
								B10.rb().try_as_col_major().unwrap(),
								l.copy(),
								tau_inv.copy(),
								simd_align(k + 1),
							);
						},
						#[cfg(feature = "rayon")]
						_ => {
							let nthreads =
								core::num::NonZeroUsize::new(par.degree())
									.unwrap();
							let nthreads = nthreads.get();
							use rayon::prelude::*;
							crate::utils::thread::for_each(
								par,
								nthreads,
								norm.par_partition_mut(nthreads)
									.zip(dot.par_partition_mut(nthreads))
									.zip(B01.par_partition_mut(nthreads))
									.zip(B11.par_col_partition_mut(nthreads)),
								|(((norm, dot), B01), B11)| {
									update_mat_and_dot_simd(
										norm,
										dot,
										B01,
										B11.try_as_col_major_mut().unwrap(),
										A10.try_as_col_major().unwrap(),
										B10.rb().try_as_col_major().unwrap(),
										l.copy(),
										tau_inv.copy(),
										simd_align(k + 1),
									);
								},
							);
						},
					});
				} else {
					dot.copy_from(B01.rb());
					with_width(par.reborrow(), width, |par| {
						linalg::matmul::matmul(
							dot.rb_mut(),
							Accum::Add,
							B10.rb().adjoint(),
							B11.rb(),
							one(),
							par,
						);
					});
					zip!(B01.rb_mut(), dot.rb_mut(), norm.rb_mut()).for_each(
						|unzip!(a, dot, norm)| {
							*dot = -dot.mul_real(tau_inv);
							*a += &*dot;
							*norm = (norm.abs2() - a.abs2()).sqrt().to_cplx();
						},
					);
				}
				// the truncated factorization has factored its last step: the steps
				// below would only update the block this one just wrote
				if k + 1 == steps {
					break 'unscale;
				}
			}
		}
		zip!(A.rb_mut()).for_each_triangular_upper(
			linalg::zip::Diag::Include,
			|unzip!(a)| *a = a.mul_real(scale_fwd),
		);
	}
	for j in 0..n {
		col_perm_inv[col_perm[j].zx()] = I::truncate(j);
	}
	// the diagonal of the unscaled factor, with the tolerances of the first step
	let mut rank = steps;
	if reveal && steps > 0 {
		let threshold = linalg::lu::rank_threshold::<T>(
			params.abs_tol,
			params.rel_tol,
			&A[(0, 0)].abs(),
		);
		if let Some(k) = (0..steps).find(|&k| {
			linalg::lu::rank_rejects::<T>(&A[(k, k)].abs(), &threshold)
		}) {
			rank = k;
		}
	}
	(
		ColPivQrInfo {
			transposition_count: n_trans,
			rank,
		},
		unsafe { PermRef::new_unchecked(col_perm, col_perm_inv, n) },
	)
}
/// computes the layout of required workspace for performing a qr decomposition
/// with column pivoting
pub fn qr_in_place_scratch<I: Index, T: ComplexField>(
	nrows: usize,
	ncols: usize,
	block_size: usize,
	par: ParShape,
	params: Spec<ColPivQrParams, T>,
) -> StackReq {
	let _ = nrows;
	let _ = ncols;
	let _ = par;
	let _ = block_size;
	let _ = &params;
	linalg::temp_mat_scratch::<T>(ncols, 2)
}
/// information about the resulting $QR$ factorization.
///
/// The type is `#[non_exhaustive]`, so a later version can add a field without breaking
/// callers that only read the ones they need.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct ColPivQrInfo {
	/// number of transpositions that were performed, can be used to compute
	/// the determinant of $P$.
	pub transposition_count: usize,
	/// number of accepted steps: `min(m, n)` unless a
	/// [rank truncation](ColPivQrParams#rank-truncation) option is set
	pub rank: usize,
}
#[track_caller]
pub fn qr_in_place<'out, I: Index, T: ComplexField>(
	A: MatMut<'_, T>,
	Q_coeff: MatMut<'_, T>,
	col_perm: &'out mut [I],
	col_perm_inv: &'out mut [I],
	mut par: Par,
	stack: &mut MemStack,
	params: Spec<ColPivQrParams, T>,
) -> (ColPivQrInfo, PermRef<'out, I>) {
	// Row-major input: the pivot search walks every one of the `n` columns of
	// length `m` (`for j in 0..n { A.col(j).norm_l2() }`, and again per step when
	// the incremental-norm path is closed), and that path is column-major only --
	// it is gated on `A.row_stride() == 1` and unwraps a column-major view. So a
	// row-major matrix touches `m * n` cache lines where the copy below moves
	// `m * n` elements contiguously twice, on every shape rather than only on the
	// tall ones. Factor a column-major copy and write the factor back, which is
	// what LAPACK's row-major interface pays internally; the arithmetic is the
	// copy's, so the factor, the `tau` matrix and the permutations are the
	// column-major ones.
	let mut A = A;
	// `Never` keeps the whole call sequential, the compact-factor upgrade included
	if matches!(params.config.par_threshold, ParThreshold::Never) {
		par = Par::Seq;
	}
	if A.row_stride().unsigned_abs() != 1
		&& A.nrows() > 1
		&& A.ncols() > 1
		&& linalg::row_major_copy_pays(A.nrows(), A.ncols())
	{
		let (m, n) = A.shape();
		let mut work = linalg::column_major_copy(A.rb(), par.reborrow());
		let mut work = MatMut::from_column_major_slice_mut(&mut work, m, n);
		let ret = qr_in_place(
			work.rb_mut(),
			Q_coeff,
			col_perm,
			col_perm_inv,
			par.reborrow(),
			stack,
			params,
		);
		linalg::copy_across_layouts(A, work.rb(), par);
		return ret;
	}
	let mut H = Q_coeff;
	let size = H.ncols();
	let block_size = H.nrows();
	let ret = qr_in_place_unblocked(
		A.rb_mut(),
		H.rb_mut().row_mut(0),
		col_perm,
		col_perm_inv,
		par.reborrow(),
		stack,
		params,
	);
	// a truncated factorization computed only the first `steps` reflectors, so the
	// conversion below stops there: the compact form of a reflector that was not
	// computed would read the caller's buffer as if it held one
	let steps = params.config.max_rank.map_or(size, |r| Ord::min(r, size));
	let mut j = 0;
	while j < steps {
		let block_size = Ord::min(block_size, steps - j);
		let mut H = H
			.rb_mut()
			.subcols_mut(j, block_size)
			.subrows_mut(0, block_size);
		for j in 0..block_size {
			H[(j, j)] = H[(0, j)].copy();
		}
		let A = A.rb().get(j.., j..j + block_size);
		householder::upgrade_householder_factor(
			H.rb_mut(),
			A,
			block_size,
			1,
			par.reborrow(),
		);
		j += block_size;
	}
	ret
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, assert, c32, c64};
	use dyn_stack::MemBuffer;
	#[test]
	fn test_unblocked_qr() {
		let rng = &mut StdRng::seed_from_u64(0);
		let mut run = |mut par: Par<'_>| {
			let par_shape = par.shape();
			for n in [2, 3, 4, 8, 16, 24, 32, 128, 255] {
				let bs = 15;
				let approx_eq = CwiseMat(ApproxEq {
					abs_tol: 1e-10,
					rel_tol: 1e-10,
				});
				let A = CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: ComplexDistribution::new(
						StandardNormal,
						StandardNormal,
					),
				}
				.rand::<Mat<c64>>(rng);
				let A = A.as_ref();
				let mut QR = A.cloned();
				let mut H = Mat::zeros(bs, n);
				let col_perm = &mut *vec![0usize; n];
				let col_perm_inv = &mut *vec![0usize; n];
				let q = qr_in_place(
					QR.as_mut(),
					H.as_mut(),
					col_perm,
					col_perm_inv,
					par.reborrow(),
					MemStack::new(&mut MemBuffer::new(qr_in_place_scratch::<
						usize,
						c64,
					>(
						n,
						n,
						bs,
						par_shape,
						default(),
					))),
					default(),
				)
				.1;
				let mut Q = Mat::<c64, _, _>::zeros(n, n);
				let mut R = QR.as_ref().cloned();
				for j in 0..n {
					Q[(j, j)] = c64::ONE;
				}
				householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
					QR.as_ref(),
					H.as_ref(),
					Conj::No,
					Q.as_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(
						householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<c64>(n, bs, n),
					)),
				);
				for j in 0..n {
					for i in j + 1..n {
						R[(i, j)] = c64::ZERO;
					}
				}
				assert!(Q * R * q ~ A);
			}
			let n = 20;
			for m in [2, 3, 4, 8, 16, 24, 32, 128, 255] {
				let bs = 15;
				let size = Ord::min(m, n);
				let approx_eq = CwiseMat(ApproxEq {
					abs_tol: 1e-10,
					rel_tol: 1e-10,
				});
				let A = CwiseMatDistribution {
					nrows: m,
					ncols: n,
					dist: ComplexDistribution::new(
						StandardNormal,
						StandardNormal,
					),
				}
				.rand::<Mat<c64>>(rng);
				let A = A.as_ref();
				let mut QR = A.cloned();
				let mut H = Mat::zeros(bs, size);
				let col_perm = &mut *vec![0usize; n];
				let col_perm_inv = &mut *vec![0usize; n];
				let q = qr_in_place(
					QR.as_mut(),
					H.as_mut(),
					col_perm,
					col_perm_inv,
					par.reborrow(),
					MemStack::new(&mut MemBuffer::new(qr_in_place_scratch::<
						usize,
						c64,
					>(
						m,
						n,
						bs,
						par_shape,
						default(),
					))),
					default(),
				)
				.1;
				let mut Q = Mat::<c64, _, _>::zeros(m, m);
				let mut R = QR.as_ref().cloned();
				for j in 0..m {
					Q[(j, j)] = c64::ONE;
				}
				householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
					QR.as_ref().subcols(0, size),
					H.as_ref(),
					Conj::No,
					Q.as_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(
						householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<c64>(m, bs, m),
					)),
				);
				for j in 0..n {
					for i in j + 1..m {
						R[(i, j)] = c64::ZERO;
					}
				}
				assert!(Q * R * q ~ A);
			}
		};
		run(Par::Seq);
		crate::team::with_team(8, |par| run(par));
	}
	/// A nonempty zero matrix has a valid factorization -- zero `R` and a unitary `Q` --
	/// so scaling by the largest column norm must not divide by a zero pivot scale.
	#[test]
	fn zero_matrix_reconstructs() {
		// 64x64 crosses the blocking threshold, the rest keep the panel path short
		for (m, n) in [(1, 1), (3, 2), (8, 8), (64, 64)] {
			let run = |mut par: Par<'_>| {
				check_zero_matrix::<f32>(m, n, par.reborrow());
				check_zero_matrix::<c32>(m, n, par.reborrow());
				check_zero_matrix::<f64>(m, n, par.reborrow());
				check_zero_matrix::<c64>(m, n, par);
			};
			run(Par::Seq);
			crate::team::with_team(8, |par| run(par));
		}
	}
	fn check_zero_matrix<T: ComplexField>(m: usize, n: usize, mut par: Par) {
		let par_shape = par.shape();
		let size = Ord::min(m, n);
		let bs = 15;
		let approx_eq = CwiseMat(ApproxEq {
			abs_tol: from_f64::<T::Real>(1e-10),
			rel_tol: from_f64::<T::Real>(1e-10),
		});
		let a = Mat::<T>::zeros(m, n);
		let a = a.as_ref();
		let mut QR = a.cloned();
		let mut H = Mat::<T>::zeros(bs, size);
		let col_perm = &mut *vec![0usize; n];
		let col_perm_inv = &mut *vec![0usize; n];
		let perm = qr_in_place(
			QR.as_mut(),
			H.as_mut(),
			col_perm,
			col_perm_inv,
			par.reborrow(),
			MemStack::new(&mut MemBuffer::new(
				qr_in_place_scratch::<usize, T>(m, n, bs, par_shape, default()),
			)),
			default(),
		)
		.1;
		assert!(QR.as_ref().is_all_finite());
		assert!(QR.as_ref().norm_max() == zero::<T::Real>());
		let mut q = Mat::<T, _, _>::zeros(m, m);
		for j in 0..m {
			q[(j, j)] = one();
		}
		householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
			QR.as_ref().subcols(0, size),
			H.as_ref(),
			Conj::No,
			q.as_mut(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(
				householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(m, bs, m),
			)),
		);
		// `R` is exactly zero, so the factor itself stands for it -- which makes the
		// reconstruction true for any finite `Q`, so `Q`'s unitarity is checked separately
		assert!(
			(q.adjoint() * &q - Mat::<T>::identity(m, m)).norm_max()
				<= eps::<T::Real>() * from_f64::<T::Real>(32.0)
		);
		assert!(q * QR.as_ref() * perm ~ a);
	}
}
