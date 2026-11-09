use crate::internal_prelude::*;
use crate::utils::cost::{ParThreshold, rate};
use crate::perm::{swap_cols_idx, swap_rows_idx};
use crate::utils::thread::par_split_indices;
use crate::utils::thread::with_width;
use faer_traits::{Real, RealReg};
use linalg::matmul::matmul;
use pulp::Simd;

#[inline(always)]
fn best_value<T: ComplexField, S: Simd>(
	simd: &SimdCtx<T, S>,
	best_value: RealReg<T::SimdVec<S>>,
	best_indices: T::SimdIndex<S>,
	value: T::SimdVec<S>,
	indices: T::SimdIndex<S>,
) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>) {
	let value = simd.abs1(value);
	let is_better = (**simd).gt(value, best_value);
	(
		RealReg(simd.select(is_better, value.0, best_value.0)),
		simd.iselect(is_better, indices, best_indices),
	)
}
#[inline(always)]
fn best_score<T: ComplexField, S: Simd>(
	simd: &SimdCtx<T, S>,
	best_score: RealReg<T::SimdVec<S>>,
	best_indices: T::SimdIndex<S>,
	score: RealReg<T::SimdVec<S>>,
	indices: T::SimdIndex<S>,
) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>) {
	// merges two accumulators of one column, which hold interleaved rows: on equal
	// scores the smaller row wins, whichever accumulator holds it, so a tie goes to
	// the smallest row as in the scalar search (tensor4all/faer#342)
	let is_better = (**simd).or_mask(
		(**simd).gt(score, best_score),
		(**simd).and_mask(
			(**simd).ge(score, best_score),
			T::simd_index_less_than(&simd.0, indices, best_indices),
		),
	);
	(
		RealReg(simd.select(is_better, score.0, best_score.0)),
		simd.iselect(is_better, indices, best_indices),
	)
}
#[inline(always)]
fn best_score_2d<T: ComplexField, S: Simd>(
	simd: &SimdCtx<T, S>,
	best_score: RealReg<T::SimdVec<S>>,
	best_row: T::SimdIndex<S>,
	best_col: T::SimdIndex<S>,
	score: RealReg<T::SimdVec<S>>,
	row: T::SimdIndex<S>,
	col: T::SimdIndex<S>,
) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>, T::SimdIndex<S>) {
	let is_better = (**simd).gt(score, best_score);
	(
		RealReg(simd.select(is_better, score.0, best_score.0)),
		simd.iselect(is_better, row, best_row),
		simd.iselect(is_better, col, best_col),
	)
}
#[inline(always)]
fn reduce_2d<T: ComplexField, S: Simd>(
	simd: &SimdCtx<T, S>,
	best_values: RealReg<T::SimdVec<S>>,
	best_row: T::SimdIndex<S>,
	best_col: T::SimdIndex<S>,
) -> (usize, usize, Real<T>) {
	let best_val = simd.reduce_max_real(best_values);
	let best_val_splat = simd.splat_real(&best_val);
	let is_best = (**simd).ge(best_values, best_val_splat);
	// The lanes that hold the maximum tie with each other (a matrix of small integers ties a
	// lot), and which lane comes first in the vector depends on the lane count: with 8 lanes an
	// AVX-512 build picks another row than an AVX2 build for the same matrix, and the elimination
	// then follows another path. Break the tie the way the scalar search does - the smallest
	// column, then the smallest row - so the pivots do not depend on the vector width.
	let sentinel = simd.isplat(T::Index::truncate(usize::MAX));
	let rows_tied = simd.iselect(is_best, best_row, sentinel);
	let cols_tied = simd.iselect(is_best, best_col, sentinel);
	let row_lanes = bytemuck::cast_slice::<T::SimdIndex<S>, T::Index>(
		core::slice::from_ref(&rows_tied),
	);
	let col_lanes = bytemuck::cast_slice::<T::SimdIndex<S>, T::Index>(
		core::slice::from_ref(&cols_tied),
	);
	let mut best: Option<(usize, usize)> = None;
	for lane in 0..row_lanes.len() {
		let (row, col) = (row_lanes[lane].zx(), col_lanes[lane].zx());
		if row == usize::MAX || col == usize::MAX {
			continue;
		}
		if best.is_none_or(|(brow, bcol)| (col, row) < (bcol, brow)) {
			best = Some((row, col));
		}
	}
	let (best_row, best_col) = best.expect("the maximum is held by some lane");
	(best_row, best_col, best_val)
}
#[inline(always)]
fn best_in_col_simd<'M, T: ComplexField, S: Simd>(
	simd: SimdCtx<'M, T, S>,
	data: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>) {
	let iota = T::simd_iota(&simd.0);
	let lane_count =
		core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>();
	let inc = simd.isplat(T::Index::truncate(lane_count));

	let indices = simd.batch_indices::<4>();
	let mut best_val = [RealReg(simd.zero()); 4];
	let mut best_idx = [simd.izero(); 4];

	let mut idx = simd.iadd(
		iota,
		simd.isplat(T::Index::truncate(simd.offset().wrapping_neg())),
	);
	simd_iter!(for (IDX, i) in [indices; 4] {
		(best_val[IDX], best_idx[IDX]) = best_value(
			&simd,
			best_val[IDX],
			best_idx[IDX],
			simd.read(data, i),
			idx,
		);
		idx = simd.iadd(idx, inc);
	});
	let _ = idx;

	let (best_val0, best_idx0) =
		best_score(&simd, best_val[0], best_idx[0], best_val[1], best_idx[1]);
	let (best_val2, best_idx2) =
		best_score(&simd, best_val[2], best_idx[2], best_val[3], best_idx[3]);
	best_score(&simd, best_val0, best_idx0, best_val2, best_idx2)
}
#[inline(always)]
fn update_and_best_in_col_simd<'M, T: ComplexField, S: Simd>(
	simd: SimdCtx<'M, T, S>,
	data: ColMut<'_, T, Dim<'M>, ContiguousFwd>,
	lhs: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
	rhs: T,
) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>) {
	let mut data = data;
	let iota = T::simd_iota(&simd.0);
	let lane_count =
		core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>();
	let inc = simd.isplat(T::Index::truncate(lane_count));
	let mut best_val = [RealReg(simd.zero()); 3];
	let mut best_idx = [simd.izero(); 3];

	let mut idx = simd.iadd(
		iota,
		simd.isplat(T::Index::truncate(simd.offset().wrapping_neg())),
	);

	let indices = simd.batch_indices::<3>();
	let rhs = simd.splat(&-rhs);
	simd_iter!(for (IDX, i) in [indices; 3] {
		let mut x0 = simd.read(data.rb(), i);
		let l0 = simd.read(lhs, i);
		x0 = simd.mul_add(l0, rhs, x0);
		(best_val[IDX], best_idx[IDX]) =
			best_value(&simd, best_val[IDX], best_idx[IDX], x0, idx);
		simd.write(data.rb_mut(), i, x0);
		idx = simd.iadd(idx, inc);
	});
	let _ = idx;

	let (best_val0, best_idx0) =
		best_score(&simd, best_val[0], best_idx[0], best_val[1], best_idx[1]);
	let (best_val2, best_idx2) = (best_val[2], best_idx[2]);
	best_score(&simd, best_val0, best_idx0, best_val2, best_idx2)
}
#[inline(always)]
fn best_in_mat_simd<T: ComplexField>(
	data: MatRef<'_, T, usize, usize, ContiguousFwd>,
) -> (usize, usize, Real<T>) {
	struct Impl<'a, 'M, 'N, T: ComplexField> {
		data: MatRef<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
	}
	impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
		type Output = (usize, usize, Real<T>);

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let Self { data } = self;
			let M = data.nrows();
			let N = data.ncols();
			let simd = SimdCtx::<'_, T, S>::new(T::simd_ctx(simd), M);
			let mut best_row = simd.isplat(T::Index::truncate(0));
			let mut best_col = simd.isplat(T::Index::truncate(0));
			let mut best_val = simd.splat_real(&zero());
			for j in N.indices() {
				let col = data.col(j);
				let (best_val_j, best_row_j) = best_in_col_simd(simd, col);
				(best_val, best_row, best_col) = best_score_2d(
					&simd,
					best_val,
					best_row,
					best_col,
					best_val_j,
					best_row_j,
					simd.isplat(T::Index::truncate(*j)),
				);
			}
			reduce_2d(&simd, best_val, best_row, best_col)
		}
	}
	with_dim!(M, data.nrows());
	with_dim!(N, data.ncols());
	dispatch!(
		Impl {
			data: data.as_shape(M, N)
		},
		Impl,
		T
	)
}
#[inline(always)]
fn update_and_best_in_mat_simd<T: ComplexField>(
	data: MatMut<'_, T, usize, usize, ContiguousFwd>,
	lhs: ColRef<'_, T, usize, ContiguousFwd>,
	rhs: RowRef<'_, T, usize>,
	align: usize,
) -> (usize, usize, Real<T>) {
	struct Impl<'a, 'M, 'N, T: ComplexField> {
		data: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		lhs: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		rhs: RowRef<'a, T, Dim<'N>>,
		align: usize,
	}
	impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
		type Output = (usize, usize, Real<T>);

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let Self {
				data,
				lhs,
				rhs,
				align,
			} = self;
			let M = data.nrows();
			let N = data.ncols();
			let simd =
				SimdCtx::<'_, T, S>::new_align(T::simd_ctx(simd), M, align);
			let mut best_row = simd.isplat(T::Index::truncate(0));
			let mut best_col = simd.isplat(T::Index::truncate(0));
			let mut best_val = simd.splat_real(&zero());
			let mut data = data;
			for j in N.indices() {
				let data = data.rb_mut().col_mut(j);
				let rhs = rhs[j].copy();
				let (best_val_j, best_row_j) =
					update_and_best_in_col_simd(simd, data, lhs, rhs);
				(best_val, best_row, best_col) = best_score_2d(
					&simd,
					best_val,
					best_row,
					best_col,
					best_val_j,
					best_row_j,
					simd.isplat(T::Index::truncate(*j)),
				);
			}
			reduce_2d(&simd, best_val, best_row, best_col)
		}
	}
	with_dim!(M, data.nrows());
	with_dim!(N, data.ncols());
	dispatch!(
		Impl {
			data: data.as_shape_mut(M, N),
			lhs: lhs.as_row_shape(M),
			rhs: rhs.as_col_shape(N),
			align,
		},
		Impl,
		T
	)
}
fn best_in_matrix_fallback<T: ComplexField>(
	data: MatRef<'_, T>,
) -> (usize, usize, Real<T>) {
	let mut max = zero();
	let mut row = 0;
	let mut col = 0;
	let (m, n) = data.shape();
	for j in 0..n {
		for i in 0..m {
			let abs = data[(i, j)].abs1();
			if abs > max {
				row = i;
				col = j;
				max = abs;
			}
		}
	}
	(row, col, max)
}
fn best_in_matrix<T: ComplexField>(
	data: MatRef<'_, T>,
) -> (usize, usize, Real<T>) {
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if let Some(dst) = data.try_as_col_major() {
			best_in_mat_simd(dst)
		} else {
			best_in_matrix_fallback(data)
		}
	} else {
		best_in_matrix_fallback(data)
	}
}
fn rank_one_update_and_best_in_matrix<T: ComplexField>(
	mut dst: MatMut<'_, T>,
	lhs: ColRef<'_, T>,
	rhs: RowRef<'_, T>,
	align: usize,
) -> (usize, usize, Real<T>) {
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if let (Some(dst), Some(lhs)) =
			(dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major())
		{
			update_and_best_in_mat_simd(dst, lhs, rhs, align)
		} else {
			matmul(
				dst.rb_mut(),
				Accum::Add,
				lhs.as_mat(),
				rhs.as_mat(),
				-one::<T>(),
				Par::Seq,
			);
			best_in_matrix(dst.rb())
		}
	} else {
		matmul(
			dst.rb_mut(),
			Accum::Add,
			lhs.as_mat(),
			rhs.as_mat(),
			-one::<T>(),
			Par::Seq,
		);
		best_in_matrix(dst.rb())
	}
}
fn lu_in_place_unblocked<T: ComplexField>(
	A: MatMut<'_, T>,
	row_trans: &mut [usize],
	col_trans: &mut [usize],
	best: &mut [(usize, usize, T::Real)],
	par: Par,
	transpose: bool,
	params: Spec<FullPivLuParams, T>,
) -> (usize, usize) {
	// the parallel arm is the only reader (absent without `rayon`)
	let _ = &best;
	let params = params.config;
	// with any truncation option set the rank is revealed, unless the input holds a non-finite
	// entry: then the tolerances are off and the rank is not truncated below `max_rank` (one pass
	// over the input, skipped when no option is set)
	let reveal = (params.max_rank.is_some()
		|| params.abs_tol.is_some()
		|| params.rel_tol.is_some())
		&& A.is_all_finite();
	let mut n_trans = 0;
	let (m, n) = A.shape();
	if m == 0 || n == 0 {
		return (0, 0);
	}
	let size = Ord::min(m, n);
	// `max_rank` caps the steps; without it the loop runs to `size` as before
	let max_rank = params.max_rank.map_or(size, |r| Ord::min(r, size));
	let mut rank = max_rank;
	let mut threshold = None;
	let mut par = par;
	// `Never` keeps the whole call sequential (the steps decide their own width otherwise)
	if matches!(params.par_threshold, ParThreshold::Never) {
		par = Par::Seq;
	}
	let mut A = A;
	let (mut max_row, mut max_col, mut max_score) = best_in_matrix(A.rb());
	for k in 0..Ord::min(m, n) {
		// The stop of the rank truncation: `max_rank` pivots accepted, or the next pivot (the
		// largest entry of the Schur complement, already found by the previous step's fused
		// update) fails the tolerance. The Schur complement is left in the trailing block and
		// the remaining steps -- their rank-one updates -- are not run.
		let mut stop = k == max_rank;
		if reveal && !stop && max_score >= min_positive() {
			let pivot = A[(max_row, max_col)].abs();
			if k == 0 {
				threshold = linalg::lu::rank_threshold::<T>(
					params.abs_tol,
					params.rel_tol,
					&pivot,
				);
			}
			stop = linalg::lu::rank_rejects::<T>(&pivot, &threshold);
		}
		if stop || max_score < min_positive() {
			// an exactly zero (or subnormal) Schur complement stopped the factorization before
			// truncation existed; it counts as the stop when the rank is revealed
			if reveal {
				rank = k;
			}
			for (i, (row, col)) in
				core::iter::zip(&mut row_trans[k..], &mut col_trans[k..])
					.enumerate()
			{
				*row = i + k;
				*col = i + k;
			}
			break;
		}
		row_trans[k] = max_row;
		col_trans[k] = max_col;
		if max_row != k {
			swap_rows_idx(A.rb_mut(), k, max_row);
			n_trans += 1;
		}
		if max_col != k {
			swap_cols_idx(A.rb_mut(), k, max_col);
			n_trans += 1;
		}
		let ref inv = A[(k, k)].recip();
		if transpose {
			for j in k + 1..n {
				A[(k, j)] *= inv;
			}
		} else {
			for i in k + 1..m {
				A[(i, k)] *= inv;
			}
		}
		if k + 1 == Ord::min(m, n) {
			break;
		}
		// One section per step (the fused rank-one update and pivot search over column
		// blocks); the unit is an element of the trailing block.
		let width = params.par_threshold.region_width::<T>(
			&par,
			(m - k - 1) * (n - k - 1),
			rate::lu_full_pivot(),
			1,
		);
		let (_, A01, A10, mut A11) = A.rb_mut().split_at_mut(k + 1, k + 1);
		let lhs = A10.col(k);
		let rhs = A01.row(k);
		with_width(par.reborrow(), width, |par| {
			match par {
				Par::Seq => {
					(max_row, max_col, max_score) =
						rank_one_update_and_best_in_matrix(
							A11.rb_mut(),
							lhs,
							rhs,
							simd_align(k + 1),
						);
				},
		#[cfg(feature = "rayon")]
				_ => {
				let nthreads = core::num::NonZeroUsize::new(par.degree()).unwrap();
					use rayon::prelude::*;
					let nthreads = nthreads.get();
					// one entry per block, from the call's scratch (no allocation per step, #111); every
					// entry is written by its block
					let best = &mut best[..nthreads];
					let full_cols = A11.ncols();
					crate::utils::thread::for_each(par, 
						nthreads,
						best.par_iter_mut()
							.zip_eq(A11.rb_mut().par_col_partition_mut(nthreads))
							.zip_eq(rhs.par_partition(nthreads))
							.enumerate(),
						|(idx, (((max_row, max_col, max_score), A11), rhs))| {
							(*max_row, *max_col, *max_score) = {
								let (a, mut b, c) =
									rank_one_update_and_best_in_matrix(
										A11,
										lhs,
										rhs,
										simd_align(k + 1),
									);
								b += par_split_indices(full_cols, idx, nthreads).0;
								(a, b, c)
							};
						},
					);
					max_row = 0;
					max_col = 0;
					max_score = zero();
					for (row, col, val) in best.iter() {
						if *val > max_score {
							max_row = *row;
							max_col = *col;
							max_score = val.copy();
						}
					}
				},
			}
		});
		max_row += k + 1;
		max_col += k + 1;
	}
	(n_trans, rank)
}
/// $LU$ factorization tuning parameters
///
/// # Rank truncation
///
/// `max_rank`, `abs_tol` and `rel_tol` are all `None` by default, and then the factorization runs
/// to `min(m, n)` exactly as without them and reports `rank = min(m, n)`. Setting any of them stops
/// the factorization after `rank` pivots and reports that number in [`FullPivLuInfo::rank`]:
///
/// - `max_rank`: at most this many pivots are accepted (clamped to `min(m, n)`).
/// - `abs_tol`, `rel_tol`: the pivot of step `k` is accepted while
///   `|pivot_k| > max(abs_tol, rel_tol * |pivot_0|)`, where `|.|` is the modulus and an absent
///   tolerance counts as `0`. `rel_tol` is measured against the **first** pivot `pivot_0`, which
///   full pivoting picks as the largest entry of the matrix, **not** against the previous pivot;
///   relative to the previous pivot would be a different stopping rule. The first rejected pivot
///   ends the factorization. Since each pivot is the largest entry of the current Schur
///   complement, a stop at step `r` bounds the residual `A - P^T L U Q^T` entrywise by the
///   rejected pivot (up to the `abs1` versus modulus factor `sqrt(2)` for complex entries).
/// - An exactly zero or subnormal Schur complement has always stopped this factorization; with
///   any option set it is reported as the stop (`rank = k`), without them `rank` stays
///   `min(m, n)` as before.
/// - Non-finite input: if the matrix holds a NaN or an infinity, the tolerances are ignored and
///   `rank` is `min(m, n)` capped by `max_rank`, so a non-finite entry never shows up as a
///   truncated rank; it reaches the factors as in the untruncated factorization. (A tolerance is
///   also ignored when the threshold itself is not finite.)
///
/// On a stop at `rank = r`: the first `r` entries of the row and column permutations are the
/// selected pivots and the rest are the identity on the remaining indices, `L[:, :r]` (unit lower)
/// and `U[:r, :]` are the factors, and the trailing `(m - r) x (n - r)` block holds the Schur
/// complement `A22 - L21 U12` of the permuted matrix, which is the residual of the rank-`r`
/// approximation. The solve, inverse and reconstruct routines of this module assume the full
/// factorization and must not be given a truncated one.
#[derive(Copy, Clone, Debug)]
pub struct FullPivLuParams {
	/// floor of the parallel regions: [`ParThreshold::Derived`] (the default) runs a step in
	/// parallel when its serial time pays for one section; [`ParThreshold::Fixed`] counts
	/// elements of the trailing block `(m-k-1)*(n-k-1)`
	pub par_threshold: ParThreshold,
	/// stop after at most this many pivots (see [rank truncation](FullPivLuParams#rank-truncation))
	pub max_rank: Option<usize>,
	/// absolute pivot tolerance (see [rank truncation](FullPivLuParams#rank-truncation))
	pub abs_tol: Option<f64>,
	/// pivot tolerance relative to the first pivot (see
	/// [rank truncation](FullPivLuParams#rank-truncation))
	pub rel_tol: Option<f64>,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
impl<T: ComplexField> Auto<T> for FullPivLuParams {
	#[inline]
	fn auto() -> Self {
		Self {
			par_threshold: ParThreshold::Derived,
			max_rank: None,
			abs_tol: None,
			rel_tol: None,
			non_exhaustive: NonExhaustive(()),
		}
	}
}
#[inline]
pub fn lu_in_place_scratch<I: Index, T: ComplexField>(
	nrows: usize,
	ncols: usize,
	par: ParShape,
	params: Spec<FullPivLuParams, T>,
) -> StackReq {
	_ = params;
	let size = Ord::min(nrows, ncols);
	// the transpositions, and one pivot candidate per block of a parallel step
	StackReq::new::<usize>(size)
		.array(2)
		.and(StackReq::new::<(usize, usize, T::Real)>(par.degree()))
}
/// Whether `lu_in_place` factors a matrix with these strides through a
/// column-major copy rather than in place.
///
/// The copy is [`linalg::row_major_copy_pays`]'s (no unit row stride, tall or
/// square, at least `64 * 64` elements), less one layout: a square matrix below
/// `128` whose rows are contiguous (`col_stride == 1`, any row stride, so a
/// padded or reversed row-major view counts). That one goes through the
/// transpose dual (`lu_in_place_unblocked(.., transpose = true)`), whose fused
/// SIMD update needs exactly that unit column stride. A generic-stride view
/// (neither stride `1`, or a column stride of `-1`) keeps the copy: in place it
/// would fall to the separate matrix update and pivot search.
///
/// The square exclusion is what was measured (tensor4all/faer#314): `64 x 64`
/// f64 one thread costs 1.24x a column-major input through the copy and 1.01x
/// through the dual on Apple M5 Max, 1.32x and 1.03x on Zen 5. The dual was
/// timed at `64`, `80`, `96`, `112`, `120` and `127` on M5 and at `64` and `96`
/// on Zen 5; the sizes in between are taken on the strength of their
/// neighbours. From `128` up the dual's row scaling `A[(k, j)] *= inv` walks a
/// strided row every step, one cache line per element, and the dual leaves the
/// 1.05x band on at least one machine: 1.09x at `512` and `1024` on four
/// threads (M5), 1.11x at `128` and 1.08-1.27x from `256` on Zen 5 (the scaling
/// alone goes from 0.22 to 3.26 ms at `1024`, one thread). On a tall matrix the
/// dual also makes the long swap strided and the trailing block wide
/// (1.4-2.1x), and off the square it leaves the band early (`80 x 64`: 1.06x),
/// so only the square shapes skip the copy.
#[inline]
pub(crate) fn row_major_takes_the_copy(
	nrows: usize,
	ncols: usize,
	row_stride: isize,
	col_stride: isize,
) -> bool {
	row_stride.unsigned_abs() != 1
		&& nrows >= ncols
		&& ncols > 1
		&& linalg::row_major_copy_pays(nrows, ncols)
		&& !(col_stride == 1 && nrows == ncols && ncols < 128)
}
#[derive(Copy, Clone, Debug)]
pub struct FullPivLuInfo {
	pub transposition_count: usize,
	/// number of accepted pivots: `min(m, n)` unless a
	/// [rank truncation](FullPivLuParams#rank-truncation) option stopped the factorization
	pub rank: usize,
}
/// # Pivot ties
///
/// When several entries of the remaining block share the largest magnitude,
/// the pivot depends on the layout, because the search runs on one of two
/// matrices:
///
/// 1. If `row_major_takes_the_copy` holds (in this module's source: `|row
///    stride| != 1`, tall or square, at least `64 * 64` elements, not a square
///    matrix below `128` with `col_stride == 1`), the input is copied to
///    column-major and the copy is searched as it is.
/// 2. Otherwise the input itself is searched as it is when `|row_stride| <
///    |col_stride|`, and its transpose is searched when `|row_stride| >=
///    |col_stride|`.
///
/// Among the tied entries of the matrix that is searched, the pivot is the one
/// in the smallest column, then the smallest row, of the remaining block at
/// that step (positions after the earlier swaps). So a column-major input, or a
/// row-major one that is copied, breaks a tie by column first, and a row-major
/// input that is not copied (a square one below `128`, a wide one, one below
/// `64 * 64` elements) breaks it by row first, on the SIMD search as on the
/// scalar one.
///
/// For example, a zero `64 x 64` f64 matrix with `A[(1, 2)] = A[(0, 4)] = 2`
/// and `max_rank = Some(1)` pivots on `(1, 2)` (two transpositions) when
/// column-major and on `(0, 4)` (one transposition) when row-major; so does an
/// `8 x 16` matrix with the same two entries. Both are valid full-pivoting
/// factorizations of the same matrix, but the permutations, the transposition
/// count and so the factors differ between the two layouts.
pub fn lu_in_place<'out, I: Index, T: ComplexField>(
	mat: MatMut<'_, T>,
	row_perm: &'out mut [I],
	row_perm_inv: &'out mut [I],
	col_perm: &'out mut [I],
	col_perm_inv: &'out mut [I],
	mut par: Par,
	stack: &mut MemStack,
	params: Spec<FullPivLuParams, T>,
) -> (FullPivLuInfo, PermRef<'out, I>, PermRef<'out, I>) {
	#[cfg(feature = "perf-warn")]
	if (mat.col_stride().unsigned_abs() == 1
		|| mat.row_stride().unsigned_abs() != 1)
		&& crate::__perf_warn!(LU_WARN)
	{
		log::warn!(
			target : "faer_perf",
			"LU with full pivoting prefers column-major or row-major matrix. Found matrix with generic strides."
		);
	}
	// Same row-major adaptation as the partial-pivoting factorisation above, except
	// on the small square shapes with contiguous rows the transpose dual factors
	// in place at the column-major speed.
	let (M, N) = mat.shape();
	if row_major_takes_the_copy(M, N, mat.row_stride(), mat.col_stride()) {
		let mut work = linalg::column_major_copy(mat.rb(), par.reborrow());
		let mut work = MatMut::from_column_major_slice_mut(&mut work, M, N);
		let ret = lu_in_place(
			work.rb_mut(),
			row_perm,
			row_perm_inv,
			col_perm,
			col_perm_inv,
			par.reborrow(),
			stack,
			params,
		);
		linalg::copy_across_layouts(mat, work.rb(), par);
		return ret;
	}
	let size = Ord::min(M, N);
	let (mut row_transpositions, stack) = stack.make_with(size, |_| 0);
	let row_transpositions = row_transpositions.as_mut();
	let (mut col_transpositions, stack) = stack.make_with(size, |_| 0);
	let col_transpositions = col_transpositions.as_mut();
	let (mut best, _) = stack.make_with(par.degree(), |_| (0usize, 0usize, zero::<T::Real>()));
	let best = best.as_mut();
	let (n_transpositions, rank) = if mat.row_stride().abs() < mat.col_stride().abs() {
		lu_in_place_unblocked(
			mat,
			row_transpositions,
			col_transpositions,
			best,
			par,
			false,
			params,
		)
	} else {
		lu_in_place_unblocked(
			mat.transpose_mut(),
			col_transpositions,
			row_transpositions,
			best,
			par,
			true,
			params,
		)
	};
	for i in 0..M {
		row_perm[i] = I::truncate(i);
	}
	for (i, t) in row_transpositions.iter().copied().enumerate() {
		row_perm.as_mut().swap(i, t);
	}
	for i in 0..M {
		row_perm_inv[row_perm[i].zx()] = I::truncate(i);
	}
	for j in 0..N {
		col_perm[j] = I::truncate(j);
	}
	for (i, t) in col_transpositions.iter().copied().enumerate() {
		col_perm.as_mut().swap(i, t);
	}
	for j in 0..N {
		col_perm_inv[col_perm[j].zx()] = I::truncate(j);
	}
	unsafe {
		(
			FullPivLuInfo {
				transposition_count: n_transpositions,
				rank,
			},
			PermRef::new_unchecked(row_perm, row_perm_inv, M),
			PermRef::new_unchecked(col_perm, col_perm_inv, N),
		)
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, assert, c64};
	use dyn_stack::MemBuffer;
	#[test]
	fn test_flu() {
		let rng = &mut StdRng::seed_from_u64(0);
		let mut run = |mut par: Par<'_>| {
			let par_shape = par.shape();
			for m in [8, 16, 24, 32, 128, 255, 256, 257] {
				let n = 8;
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
				let mut LU = A.cloned();
				let row_perm = &mut *vec![0usize; m];
				let row_perm_inv = &mut *vec![0usize; m];
				let col_perm = &mut *vec![0usize; n];
				let col_perm_inv = &mut *vec![0usize; n];
				let (_, p, q) = lu_in_place(
					LU.as_mut(),
					row_perm,
					row_perm_inv,
					col_perm,
					col_perm_inv,
					par.reborrow(),
					MemStack::new(&mut MemBuffer::new(lu_in_place_scratch::<
						usize,
						c64,
					>(
						n, n, par_shape, default()
					))),
					default(),
				);
				let mut L = LU.as_ref().cloned();
				let mut U = LU.as_ref().cloned();
				for j in 0..n {
					for i in 0..j {
						L[(i, j)] = c64::ZERO;
					}
					L[(j, j)] = c64::ONE;
				}
				for j in 0..n {
					for i in j + 1..m {
						U[(i, j)] = c64::ZERO;
					}
				}
				let L = L.as_ref();
				let U = U.as_ref();
				let U = U.subrows(0, n);
				assert!(p.inverse() * L * U * q ~ A);
			}
			for n in [16, 24, 32, 128, 255, 256, 257] {
				let approx_eq = CwiseMat(ApproxEq {
					abs_tol: 1e-10,
					rel_tol: 1e-10,
				});
				let A = CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: StandardNormal,
				}
				.rand::<Mat<f64>>(rng);
				let A = A.as_ref();
				let mut LU = A.cloned();
				let row_perm = &mut *vec![0usize; n];
				let row_perm_inv = &mut *vec![0usize; n];
				let col_perm = &mut *vec![0usize; n];
				let col_perm_inv = &mut *vec![0usize; n];
				let (_, p, q) = lu_in_place(
					LU.as_mut(),
					row_perm,
					row_perm_inv,
					col_perm,
					col_perm_inv,
					par.reborrow(),
					MemStack::new(&mut MemBuffer::new(lu_in_place_scratch::<
						usize,
						f64,
					>(
						n, n, par_shape, default()
					))),
					default(),
				);
				let mut L = LU.as_ref().cloned();
				let mut U = LU.as_ref().cloned();
				for j in 0..n {
					for i in 0..j {
						L[(i, j)] = 0.0;
					}
					L[(j, j)] = 1.0;
				}
				for j in 0..n {
					for i in j + 1..n {
						U[(i, j)] = 0.0;
					}
				}
				let L = L.as_ref();
				let U = U.as_ref();
				assert!(p.inverse() * L * U * q ~ A);
			}
		};
		run(Par::Seq);
		crate::team::with_team(8, |par| run(par));
	}
	/// The pivot search breaks a tie the way the scalar search does: the smallest column, then the
	/// smallest row. Without that, which lane wins depends on the lane count, so an AVX2 build and
	/// an AVX-512 build eliminate in a different order for the same matrix (and on a singular
	/// fixture one of the two ends on an exact zero pivot, which the solve turns into `NaN`).
	#[test]
	fn pivot_search_breaks_ties_by_column_then_row() {
		// column 2 holds the maximum twice (rows 1 and 5): both routes must take row 1, and a
		// maximum in column 2 must win over an equal one in a later column
		let a = crate::Mat::<f64>::from_fn(8, 6, |i, j| match (j, i) {
			(2, 1) | (2, 5) => 3.0,
			(4, 0) | (4, 7) => 3.0,
			(1, _) => 0.5,
			_ => 0.0,
		});
		assert_eq!(best_in_matrix(a.as_ref()), (1, 2, 3.0));
		assert_eq!(best_in_matrix_fallback(a.as_ref()), (1, 2, 3.0));

		// a matrix of small integers, where every column ties: the whole search must agree with
		// the scalar one, whatever the lane count is
		let mut rng = core::num::Wrapping(7u64);
		for (m, n) in [(8usize, 8usize), (27, 13), (16, 40), (1, 9), (9, 1)] {
			let mut a = crate::Mat::<f64>::zeros(m, n);
			for j in 0..n {
				for i in 0..m {
					rng += core::num::Wrapping(6364136223846793005u64);
					a[(i, j)] = ((rng.0 >> 33) % 3) as f64;
				}
			}
			assert_eq!(
				best_in_matrix(a.as_ref()),
				best_in_matrix_fallback(a.as_ref()),
				"{m}x{n}"
			);
		}
	}

	/// Within one column the SIMD search splits the rows over four accumulators (three in the
	/// fused update), lane by lane. Two equal maxima in one column can land in different
	/// accumulators with the larger row in the one merged first; the merge must still keep the
	/// smaller row (tensor4all/faer#342 review: `A[(6, 2)] = A[(8, 2)]` returned row 8 on a
	/// two-lane build). Every pair of rows is tried, so the test holds whatever the lane count.
	#[test]
	fn pivot_search_keeps_the_smaller_row_across_accumulators() {
		let m = 64;
		for r0 in 0..m {
			for r1 in r0 + 1..m {
				let mut a = crate::Mat::<f64>::zeros(m, 3);
				a[(r0, 1)] = 2.0;
				a[(r1, 1)] = 2.0;
				assert_eq!(
					best_in_matrix(a.as_ref()),
					(r0, 1, 2.0),
					"rows {r0}, {r1}"
				);

				// the fused update with a zero update leaves the matrix as it is
				let lhs = crate::Col::<f64>::zeros(m);
				let rhs = crate::Row::<f64>::zeros(3);
				for align in [0, 1, 3] {
					let mut b = a.clone();
					assert_eq!(
						rank_one_update_and_best_in_matrix(
							b.as_mut(),
							lhs.as_ref(),
							rhs.as_ref(),
							align
						),
						(r0, 1, 2.0),
						"update, rows {r0}, {r1}, align {align}"
					);
				}
			}
		}
	}

	#[test]
	fn test_gh238() {
		#[rustfmt::skip]
        let A = [
            [
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.0,
                0.0,
                1.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ];
		let A = MatRef::from_row_major_array(&A).to_owned();
		let b = [
			-0.0, -0.0, -2.0, -2.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
			-0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
		];
		let b = ColRef::from_slice(&b);
		let lu = A.full_piv_lu(Par::Seq);
		assert!(lu.solve(b, Par::Seq).is_all_finite());
	}
}
