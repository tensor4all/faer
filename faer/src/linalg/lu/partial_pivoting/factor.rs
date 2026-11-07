use crate::internal_prelude::*;
use crate::perm::swap_rows_idx;
use crate::{assert, debug_assert};
#[inline]
pub(crate) fn swap_elems<T: ComplexField>(
	col: ColMut<'_, T>,
	i: usize,
	j: usize,
) {
	debug_assert!(all(i < col.nrows(), j < col.nrows()));
	let rs = col.row_stride();
	let col = col.as_ptr_mut();
	unsafe {
		let a = col.offset(i as isize * rs);
		let b = col.offset(j as isize * rs);
		core::ptr::swap(a, b);
	}
}
/// The first row of the largest `abs1` among the lanes of `val` (each lane holds the first row at
/// which it saw its maximum, so the smallest tied row is the first row of the column): the scalar
/// search's choice, whatever the lane count. 0 when every value is zero (or NaN), as there.
#[inline(always)]
fn first_of_max<T: ComplexField, S: pulp::Simd>(
	simd: &SimdCtx<'_, T, S>,
	val: faer_traits::RealReg<T::SimdVec<S>>,
	idx: T::SimdIndex<S>,
) -> usize {
	let best = simd.reduce_max_real(val);
	let is_best = (**simd).ge(val, simd.splat_real(&best));
	let tied = simd.iselect(is_best, idx, simd.isplat(T::Index::truncate(usize::MAX)));
	let lanes = bytemuck::cast_slice::<T::SimdIndex<S>, T::Index>(core::slice::from_ref(&tied));
	lanes.iter().map(|i| i.zx()).min().unwrap_or(0)
}

/// One column step of the leaf on contiguous columns, in one pass per column: `lhs` (rows below
/// the pivot of the pivot column) is scaled by `inv` when given, then `rest[.., c] -= lhs *
/// urow[c]` for every column of `rest`, and the first row of `rest[.., 0]` with the largest `abs1`
/// is returned (the next pivot, found while that column is written). With `lhs` empty of columns
/// to update, only the search runs when `search_lhs`: the first column of a leaf.
fn leaf_step_simd<T: ComplexField>(
	lhs: ColMut<'_, T, usize, ContiguousFwd>,
	rest: MatMut<'_, T, usize, usize, ContiguousFwd>,
	urow: RowRef<'_, T, usize>,
	inv: Option<T>,
	search_lhs: bool,
) -> usize {
	struct Impl<'a, 'M, 'N, T: ComplexField> {
		lhs: ColMut<'a, T, Dim<'M>, ContiguousFwd>,
		rest: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		urow: RowRef<'a, T, Dim<'N>>,
		inv: Option<T>,
		search_lhs: bool,
	}
	impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
		type Output = usize;

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> usize {
			let Self { mut lhs, mut rest, urow, inv, search_lhs } = self;
			let M = lhs.nrows();
			let N = rest.ncols();
			let simd = SimdCtx::<'_, T, S>::new(T::simd_ctx(simd), M);
			let lanes = core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>();
			let inc = simd.isplat(T::Index::truncate(lanes));
			let iota = T::simd_iota(&simd.0);
			let indices = simd.indices();
			let mut best_val = faer_traits::RealReg(simd.zero());
			let mut best_idx = simd.izero();
			let track = |x: T::SimdVec<S>,
			             idx: T::SimdIndex<S>,
			             best_val: &mut faer_traits::RealReg<T::SimdVec<S>>,
			             best_idx: &mut T::SimdIndex<S>| {
				let v = simd.abs1(x);
				let better = (*simd).gt(v, *best_val);
				*best_val = faer_traits::RealReg(simd.select(better, v.0, best_val.0));
				*best_idx = simd.iselect(better, idx, *best_idx);
			};
			// `x - l * u` in one fused multiply-add: the rounding of the scalar leaf's rank one
			// `matmul`, so the factor, and the pivots of near ties, do not depend on the layout
			// (a strided panel takes the scalar leaf)
			let update = |l: T::SimdVec<S>, u: T::SimdVec<S>, x: T::SimdVec<S>| simd.mul_add(l, u, x);
			if let Some(inv) = inv {
				let inv = simd.splat(&inv);
				simd_iter!(for i in [indices] {
					let x = simd.read(lhs.rb(), i);
					simd.write(lhs.rb_mut(), i, simd.mul(x, inv));
				});
			}
			if search_lhs {
				let mut idx = iota;
				simd_iter!(for i in [indices] {
					track(simd.read(lhs.rb(), i), idx, &mut best_val, &mut best_idx);
					idx = simd.iadd(idx, inc);
				});
				let _ = idx;
				return first_of_max(&simd, best_val, best_idx);
			}
			let lhs = lhs.rb();
			for j in N.indices() {
				let u = simd.splat(&-urow[j].copy());
				let mut dst = rest.rb_mut().col_mut(j);
				if *j == 0 {
					let mut idx = iota;
					simd_iter!(for i in [indices] {
						let x = update(simd.read(lhs, i), u, simd.read(dst.rb(), i));
						simd.write(dst.rb_mut(), i, x);
						track(x, idx, &mut best_val, &mut best_idx);
						idx = simd.iadd(idx, inc);
					});
					let _ = idx;
				} else {
					simd_iter!(for i in [indices] {
						let x = update(simd.read(lhs, i), u, simd.read(dst.rb(), i));
						simd.write(dst.rb_mut(), i, x);
					});
				}
			}
			if *N == 0 { 0 } else { first_of_max(&simd, best_val, best_idx) }
		}
	}
	with_dim!(M, lhs.nrows());
	with_dim!(N, rest.ncols());
	dispatch!(
		Impl {
			lhs: lhs.as_row_shape_mut(M),
			rest: rest.as_shape_mut(M, N),
			urow: urow.as_col_shape(N),
			inv,
			search_lhs,
		},
		Impl,
		T
	)
}

/// The leaf on contiguous columns: the scalar leaf's steps (search, swap of the whole row, scaling,
/// rank one update of the leaf's columns) with the scaling, the update and the next column's search
/// in one SIMD pass per column, and no `matmul` call per column.
fn lu_in_place_unblocked_simd<I: Index, T: ComplexField>(
	matrix: MatMut<'_, T>,
	start: usize,
	end: usize,
	trans: &mut [I],
) -> usize {
	let mut matrix = matrix;
	let m = matrix.nrows();
	let mut n_trans = 0;
	let mut imax = {
		let (col, _) = matrix.rb_mut().get_mut(.., start..end).split_at_col_mut(1);
		let col = col.col_mut(0).try_as_col_major_mut().unwrap();
		let empty = MatMut::from_column_major_slice_mut(&mut [], m, 0)
			.try_as_col_major_mut()
			.unwrap();
		leaf_step_simd(col, empty, RowRef::from_slice(&[]), None, true)
	};
	for j in start..end {
		let row = j - start;
		trans[row] = I::truncate(imax - row);
		if imax != row {
			swap_rows_idx(matrix.rb_mut(), row, imax);
			n_trans += 1;
		}
		let panel = matrix.rb_mut().get_mut(.., j..end);
		let (top, bottom) = panel.split_at_row_mut(row + 1);
		let top = top.rb();
		let pivot = top[(row, 0)].copy();
		let (mut lhs, rest) = bottom.split_at_col_mut(1);
		let mut inv = None;
		// An exactly zero pivot column is a singular factor, not a division: OpenBLAS's
		// `getf2_k` leaves the (all zero) column alone instead of scaling it by infinity.
		if pivot != zero() {
			if pivot.abs() >= min_positive() {
				inv = Some(pivot.recip());
			} else {
				// see the scalar leaf: a subnormal pivot divides elementwise, rescaled
				let ref shift = max_positive::<T::Real>();
				let ref pivot = pivot.mul_real(shift);
				for i in 0..lhs.nrows() {
					lhs[(i, 0)] = &lhs[(i, 0)].mul_real(shift) / pivot;
				}
			}
		}
		let urow = top.row(row).subcols(1, end - j - 1);
		let next = leaf_step_simd(
			lhs.col_mut(0).try_as_col_major_mut().unwrap(),
			rest.try_as_col_major_mut().unwrap(),
			urow,
			inv,
			false,
		);
		imax = row + 1 + next;
	}
	n_trans
}

fn lu_in_place_unblocked<I: Index, T: ComplexField>(
	matrix: MatMut<'_, T>,
	start: usize,
	end: usize,
	trans: &mut [I],
) -> usize {
	if start == end {
		return 0;
	}
	// real types only: the lane indices of the complex search were not right (the scalar leaf
	// stays the complex one)
	if const { T::SIMD_CAPABILITIES.is_simd() && T::IS_REAL } && matrix.row_stride() == 1 {
		lu_in_place_unblocked_simd(matrix, start, end, trans)
	} else {
		lu_in_place_unblocked_scalar(matrix, start, end, trans)
	}
}
fn lu_in_place_unblocked_scalar<I: Index, T: ComplexField>(
	matrix: MatMut<'_, T>,
	start: usize,
	end: usize,
	trans: &mut [I],
) -> usize {
	let mut matrix = matrix;
	let m = matrix.nrows();
	let mut n_trans = 0;
	for j in start..end {
		let col = j;
		let row = j - start;
		let t = &mut trans[row];
		let mut imax = row;
		let mut max = zero();
		for i in imax..m {
			let abs = matrix[(i, col)].abs1();
			if abs > max {
				max = abs;
				imax = i;
			}
		}
		*t = I::truncate(imax - row);
		if imax != row {
			swap_rows_idx(matrix.rb_mut(), row, imax);
			n_trans += 1;
		}
		let mut matrix = matrix.rb_mut().get_mut(.., start..end);
		// An exactly zero pivot column is a singular factor, not a division: OpenBLAS's
		// `getf2_k` leaves the (all zero) column alone instead of scaling it by infinity.
		let pivot = matrix[(row, row)].copy();
		if pivot != zero() {
			if pivot.abs() >= min_positive() {
				let ref inv = pivot.recip();
				for i in row + 1..m {
					matrix[(i, row)] *= inv;
				}
			} else {
				// the reciprocal of a subnormal pivot overflows, and the complex quotient
				// forms its squared denominator, so rescale both operands instead: a power
				// of two keeps the scaling exact and the squares in range (`dgetf2`
				// divides elementwise below `SFMIN`)
				let ref shift = max_positive::<T::Real>();
				let ref pivot = pivot.mul_real(shift);
				for i in row + 1..m {
					matrix[(i, row)] =
						&matrix[(i, row)].mul_real(shift) / pivot;
				}
			}
		}
		let (_, A01, A10, A11) = matrix.rb_mut().split_at_mut(row + 1, row + 1);
		let A01 = A01.row(row);
		let A10 = A10.col(row);
		linalg::matmul::matmul(
			A11,
			Accum::Add,
			A10.as_mat(),
			A01.as_mat(),
			-one::<T>(),
			Par::Seq,
		);
	}
	n_trans
}
/// Sections of the trailing update of one recursion level: the left half's row swaps with the
/// triangular solve (one section over column chunks of the right half), then the product; the
/// second runs on caches the first filled or evicted, so the two are counted as the cold-region
/// rule of the thresholds design (provisional).
const TRAILING_UPDATE_SECTIONS: u32 = 2;
/// Sections of the swap stage of one level when both the columns left of the panel and the
/// columns right of it are swapped: a split of the two blocks over the parallelism, then a
/// split of the columns of each; memory bound, so a section costs more than its share of the
/// work (provisional, from the thresholds design).
const SWAP_STAGE_SECTIONS: u32 = 4;
/// Sections of the swap stage when one side holds every column (the first panel of a wide
/// matrix, the last of a tall one): one split of the columns over the whole budget.
const SWAP_ONE_SIDE_SECTIONS: u32 = 1;
use crate::utils::cost::{gemm_ns_per_mac, max_region_width, par_width_time_upto, rate, serial_time};
use crate::utils::thread::with_width;

pub use crate::utils::cost::ParThreshold;

/// The width the trailing update (`b` row swaps over `cols` columns, `b x cols` solve,
/// `rows x cols x b` product) of a level runs
/// at (1: sequential). [`ParThreshold::Derived`] is the widest width of `par` that pays for the
/// serial time; [`Fixed`](ParThreshold::Fixed) is an explicit floor in `f64` elements of the
/// updated block, counted in flops (a complex element carries four times the arithmetic), and
/// takes the whole degree above it.
fn trailing_update_width<T: ComplexField>(
	par: &Par,
	par_threshold: ParThreshold,
	rows: usize,
	cols: usize,
	b: usize,
) -> usize {
	let floor = match par_threshold {
		ParThreshold::Never => return 1,
		ParThreshold::Fixed(n) => n,
		ParThreshold::Derived => {
			let macs = cols.saturating_mul(
				b.saturating_mul(b) / 2 + rows.saturating_mul(b),
			);
			// the region also applies the left half's `b` row swaps to its `cols` columns
			return par_width_time_upto(
				par,
				serial_time(macs, gemm_ns_per_mac::<T>())
					+ serial_time(b.saturating_mul(cols), rate::lu_swap().of::<T>()),
				TRAILING_UPDATE_SECTIONS,
				max_region_width(par.degree()),
			);
		},
	};
	let work = rows
		.saturating_mul(cols)
		.saturating_mul(if T::IS_REAL { 2 } else { 8 });
	if work > floor.saturating_mul(2) { par.degree() } else { 1 }
}

/// The width the swap stage (`n` row swaps over `cols` columns of an `m` row matrix) runs at;
/// `par_threshold` as in [`trailing_update_width`], where an explicit value counts `m * cols`
/// elements.
fn swap_stage_width<T: ComplexField>(
	par: &Par,
	par_threshold: ParThreshold,
	m: usize,
	n: usize,
	cols: usize,
	one_side: bool,
) -> usize {
	match par_threshold {
		ParThreshold::Never => 1,
		ParThreshold::Fixed(floor) => {
			if m.saturating_mul(cols) > floor { par.degree() } else { 1 }
		},
		ParThreshold::Derived => par_width_time_upto(
			par,
			serial_time(n.saturating_mul(cols), rate::lu_swap().of::<T>()),
			if one_side { SWAP_ONE_SIDE_SECTIONS } else { SWAP_STAGE_SECTIONS },
			max_region_width(par.degree()),
		),
	}
}
/// Applies the transpositions `trans` (row `j` with row `j + trans[j]`, in order) to every
/// column of `mat`.
fn apply_transpositions<I: Index, T: ComplexField>(mat: MatMut<'_, T>, trans: &[I]) {
	let mut mat = mat;
	for j in 0..mat.ncols() {
		let mut col = mat.rb_mut().col_mut(j);
		for (j, &t) in trans.iter().enumerate() {
			swap_elems(col.rb_mut(), j, t.zx() + j);
		}
	}
}
pub(crate) fn lu_in_place_recursion<I: Index, T: ComplexField>(
	A: MatMut<'_, T>,
	start: usize,
	end: usize,
	trans: &mut [I],
	mut par: Par,
	params: Spec<PartialPivLuParams, T>,
) -> usize {
	let params = params.config;
	let mut A = A;
	let m = A.nrows();
	let ncols = A.ncols();
	let n = end - start;
	if n <= params.recursion_threshold {
		return lu_in_place_unblocked(A, start, end, trans);
	}
	let half = n / 2;
	let pow = Ord::min(16, half.next_power_of_two());
	let block_size = half.next_multiple_of(pow);
	let mut n_trans = 0;
	assert!(n <= m);
	// The left half sees only its own columns: its transpositions reach the right half in the
	// trailing update below, inside the same section as the solve, so each worker swaps the rows
	// of the columns it then solves instead of the caller swapping them alone beforehand.
	n_trans += lu_in_place_recursion(
		A.rb_mut().get_mut(.., start..start + block_size),
		0,
		block_size,
		&mut trans[..block_size],
		par.reborrow(),
		params.into(),
	);
	{
		let mut A = A.rb_mut().get_mut(.., start..end);
		let (A_l, A_r) = A.rb_mut().split_at_col_mut(block_size);
		let (A00, A10) = A_l.rb().split_at_row(block_size);
		let left_trans = &trans[..block_size];
		// The panel solve and the trailing product below are this level's whole
		// parallel work, and the block size halves every level of the recursion, so
		// past a few levels they are a fork and a join around a handful of thousands
		// of flops. The region is decided by its serial time against the cost of its
		// sections: the solve (`b * b / 2` MACs per column of `A01`) and the product
		// (`b` MACs per element of `A11`) at the dtype's GEMM rate, two sections
		// (one join for the solve, one for the product), the second of them on
		// cold caches. Below it take `Par::Seq`; the recursion keeps the caller's
		// `par`, so every level decides for its own trailing update.
		let (rows1, cols1, b) = (A10.nrows(), A_r.ncols(), A10.ncols());
		let width = trailing_update_width::<T>(&par, params.par_threshold, rows1, cols1, b);
		with_width(par.reborrow(), width, |mut par_here| {
			// the swap and the solve of one column chunk of the right half are the chunk's
			// worker's alone: one section, sequential work on every member
			let swap_and_solve = |mut chunk: MatMut<'_, T>| {
				apply_transpositions(chunk.rb_mut(), left_trans);
				linalg::triangular_solve::solve_unit_lower_triangular_in_place(
					A00,
					chunk.get_mut(..block_size, ..),
					Par::Seq,
				);
			};
			let mut A_r = A_r;
			match par_here {
				Par::Seq => swap_and_solve(A_r.rb_mut()),
				#[cfg(feature = "rayon")]
				_ => {
					let k = par_here.degree();
					crate::utils::thread::for_each(
						par_here.reborrow(),
						k,
						A_r.rb_mut().par_col_partition_mut(k),
						|chunk| swap_and_solve(chunk),
					);
				},
			}
			let (A01, A11) = A_r.split_at_row_mut(block_size);
			let A01 = A01.rb();
			let mut A11 = A11;
			linalg::matmul::matmul(
				A11.rb_mut(),
				Accum::Add,
				A10.rb(),
				A01.rb(),
				-one::<T>(),
				par_here,
			);
		});
		n_trans += lu_in_place_recursion(
			A.rb_mut().get_mut(block_size..m, ..),
			block_size,
			n,
			&mut trans[block_size..n],
			par.reborrow(),
			params.into(),
		);
	}
	let swap = |mat: MatMut<'_, T>| apply_transpositions(mat, &trans[..n]);
	let (A_left, A_right) = A.rb_mut().split_at_col_mut(start);
	let A_right = A_right.get_mut(.., end - start..ncols - start);
	let width = swap_stage_width::<T>(
		&par,
		params.par_threshold,
		m,
		n,
		ncols - n,
		start == 0 || end == ncols,
	);
	with_width(par.reborrow(), width, |par| {
		match par {
			Par::Seq => {
				swap(A_left);
				swap(A_right);
			},
		#[cfg(feature = "rayon")]
			_ => {
			let nthreads = core::num::NonZeroUsize::new(par.degree()).unwrap();
				let nthreads = nthreads.get();
				let len = (A_left.ncols() + A_right.ncols()) as f64;
				let left_threads = Ord::min(
					(nthreads as f64 * (A_left.ncols() as f64 / len)) as usize,
					nthreads,
				);
				let (left_cols, right_cols) = (A_left.ncols(), A_right.ncols());
				if left_cols == 0 || right_cols == 0 {
					// one side holds every column: it takes the whole budget
					let (A, _) = if left_cols == 0 { (A_right, A_left) } else { (A_left, A_right) };
					crate::utils::thread::for_each(
						par,
						nthreads,
						A.par_col_partition_mut(nthreads),
						|A| swap(A),
					);
				} else {
					// the two sides run on the two halves of the split, each on the
					// parallelism it is handed
					crate::utils::thread::join_sized(
						par,
						left_threads,
						|par| {
							let n = par.degree();
							crate::utils::thread::for_each(
								par,
								n,
								A_left.par_col_partition_mut(n),
								|A| swap(A),
							)
						},
						|par| {
							let n = par.degree();
							crate::utils::thread::for_each(
								par,
								n,
								A_right.par_col_partition_mut(n),
								|A| swap(A),
							)
						},
					);
				}
			},
		}
	});
	n_trans
}
/// $LU$ factorization tuning parameters
///
/// # Rank truncation
///
/// `max_rank`, `abs_tol` and `rel_tol` are all `None` by default, and then the factorization runs
/// to `min(m, n)` exactly as without them and reports `rank = min(m, n)`. Setting any of them
/// reports the number of accepted pivots in [`PartialPivLuInfo::rank`]:
///
/// - `max_rank = Some(r)`: only the first `r` columns are factored (clamped to `min(m, n)`): the
///   `m x r` panel runs through the same recursion as the full factorization, its row swaps are
///   applied to the other columns, and `U[:r, r:]` is one triangular solve. The Schur complement
///   update of the trailing block is skipped, which is the saving: `O(m r^2 + r^2 (n - r))`
///   instead of `O(m n min(m, n))`.
/// - `abs_tol`, `rel_tol`: the pivot of step `k` is accepted while
///   `|U[k, k]| > max(abs_tol, rel_tol * |U[0, 0]|)`, where `|.|` is the modulus and an absent
///   tolerance counts as `0`; `rank` is the index of the first rejected pivot. `rel_tol` is
///   measured against the **first** pivot `U[0, 0]`, **not** against the previous pivot;
///   relative to the previous pivot would be a different stopping rule. Partial pivoting picks
///   the largest entry of each column, so `|U[0, 0]|` is the largest entry of the first column,
///   and its pivots are not rank-revealing in general: the rule is the same as the full pivoting
///   one, its guarantee is weaker. The tolerance is checked on the computed factor, so it does
///   not shorten the factorization; only `max_rank` does. An exactly zero pivot is rejected by
///   any tolerance; with `max_rank` alone it is accepted, as in the full factorization.
/// - Non-finite input: if the matrix holds a NaN or an infinity, the tolerances are ignored and
///   `rank` is `min(m, n)` capped by `max_rank`, so a non-finite entry never shows up as a
///   truncated rank. A NaN pivot is never rejected, and a tolerance is ignored when the threshold
///   itself is not finite.
///
/// On `rank = r`: the first `r` entries of the permutation are the selected rows, and
/// `L[:, :r]` (unit lower) and `U[:r, :]` are the factors. With `max_rank`, the trailing
/// `(m - r) x (n - r)` block holds the row-permuted input, **not** the Schur complement. With a
/// tolerance stop below `max_rank` the factorization of the columns past `r` is present but is
/// not part of the rank-`r` factorization. The solve, inverse and reconstruct routines of this
/// module assume the full factorization and must not be given a truncated one.
#[derive(Copy, Clone, Debug)]
pub struct PartialPivLuParams {
	/// threshold at which the implementation should stop recursing
	pub recursion_threshold: usize,
	/// blocking variant step size
	pub block_size: usize,
	/// floor of the parallel regions; [`ParThreshold::Derived`] (the default) derives them from
	/// the serial time of each region and the dispatch cost of the parallelism
	/// ([`ParShape::dispatch_cost`]), per region: the trailing update of a recursion level
	/// (2 sections) and the swap stage (4 sections). [`ParThreshold::Fixed`] restores a fixed
	/// floor, [`ParThreshold::Never`] disables parallelism. The trailing update and the swap
	/// stage keep separate rules. A derived width is also capped at twice the largest width the
	/// host's dispatch row measures (`cost::max_region_width`): a factorization opens hundreds of
	/// sections, and past the row's last anchor a section's cost is an extrapolation that
	/// misses its tail.
	pub par_threshold: ParThreshold,
	/// factor at most this many columns (see
	/// [rank truncation](PartialPivLuParams#rank-truncation))
	pub max_rank: Option<usize>,
	/// absolute pivot tolerance (see [rank truncation](PartialPivLuParams#rank-truncation))
	pub abs_tol: Option<f64>,
	/// pivot tolerance relative to the first pivot (see
	/// [rank truncation](PartialPivLuParams#rank-truncation))
	pub rel_tol: Option<f64>,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
/// information about the resulting $LU$ factorization
#[derive(Copy, Clone, Debug)]
pub struct PartialPivLuInfo {
	/// number of transpositions that were performed, can be used to compute
	/// the determinant of $P$
	pub transposition_count: usize,
	/// number of accepted pivots: `min(m, n)` unless a
	/// [rank truncation](PartialPivLuParams#rank-truncation) option is set
	pub rank: usize,
}
/// error in the $LU$ factorization
#[derive(Copy, Clone, Debug)]
pub enum LdltError {
	ZeroPivot { index: usize },
}
/// Width of the unblocked leaf panel for `f64`. Below it the factorization runs a column at a time,
/// sequentially, so on a team the leaves are the serial part of the call: at n = 512 on 8 threads
/// they were half of it once the `U` solve stopped aliasing (`docs/worklogs/2026-11-05-lu-leaf-width.md`).
/// Median of 31 calls, ms, the widths interleaved; evo2 (Zen 5), `f64`, a team of `T`:
///
/// | n, T    | 4     | 8     | 16    | 32    |
/// |---------|-------|-------|-------|-------|
/// | 256, 1  | 0.308 | 0.304 | 0.327 | 0.453 |
/// | 256, 8  | 0.249 | 0.245 | 0.265 | 0.381 |
/// | 512, 1  | 1.777 | 1.791 | 1.970 | 2.362 |
/// | 512, 8  | 0.915 | 0.930 | 1.104 | 1.490 |
/// | 1024, 1 | 11.01 | 11.07 | 11.53 | 12.56 |
/// | 1024, 8 | 3.425 | 3.486 | 3.931 | 4.985 |
/// | 2048, 1 | 81.66 | 82.05 | 83.23 | 86.26 |
/// | 2048, 8 | 16.96 | 17.19 | 18.57 | 21.79 |
/// | 4096, 1 | 614.7 | 615.3 | 619.9 | 630.2 |
/// | 4096, 8 | 101.7 | 102.6 | 107.6 | 119.5 |
///
/// 8 wins or ties everywhere (4 is within 2 %, ahead at large n and behind at 256). The other scalar
/// types keep 16: they were not scanned.
const F64_RECURSION_THRESHOLD: usize = 8;
impl<T: ComplexField> Auto<T> for PartialPivLuParams {
	#[inline]
	fn auto() -> Self {
		Self {
			recursion_threshold: if const { T::IS_NATIVE_F64 } {
				F64_RECURSION_THRESHOLD
			} else {
				16
			},
			block_size: 64,
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
	params: Spec<PartialPivLuParams, T>,
) -> StackReq {
	_ = par;
	_ = params;
	StackReq::new::<I>(Ord::min(nrows, ncols))
}
pub fn lu_in_place<'out, I: Index, T: ComplexField>(
	A: MatMut<'_, T>,
	perm: &'out mut [I],
	perm_inv: &'out mut [I],
	mut par: Par,
	stack: &mut MemStack,
	params: Spec<PartialPivLuParams, T>,
) -> (PartialPivLuInfo, PermRef<'out, I>) {
	let _ = &params;
	let truncate = I::truncate;
	#[cfg(feature = "perf-warn")]
	if (A.col_stride().unsigned_abs() == 1
		|| A.row_stride().unsigned_abs() != 1)
		&& crate::__perf_warn!(LU_WARN)
	{
		log::warn!(
			target : "faer_perf",
			"LU with partial pivoting prefers column-major or row-major matrix. Found matrix with generic strides."
		);
	}
	let mut matrix = A;
	let mut stack = stack;
	let m = matrix.nrows();
	let n = matrix.ncols();
	// Row-major input: the unblocked core walks each pivot column twice (the
	// search, then the scaling) and the swap stage swaps elements inside a
	// column, which is a cache line per element when the columns are not
	// contiguous -- `size * m` touches that way against `m * n` elements the copy
	// moves contiguously twice. Factor a column-major copy and write the factor
	// back, which is what LAPACK's row-major interface pays internally; the
	// arithmetic is the copy's, so the factor and the permutation are the
	// column-major ones. The guard is the panel's own cost model, the same one
	// `qr_in_place` states: a wide matrix has `size == m` over `n` columns, where
	// the copy would cost more than it saved.
	if matrix.row_stride().unsigned_abs() != 1
		&& matrix.nrows() >= matrix.ncols()
		&& matrix.ncols() > 1
		&& linalg::row_major_copy_pays(matrix.nrows(), matrix.ncols())
	{
		let mut work = linalg::column_major_copy(matrix.rb(), par.reborrow());
		let mut work = MatMut::from_column_major_slice_mut(&mut work, m, n);
		let ret = lu_in_place(
			work.rb_mut(),
			perm,
			perm_inv,
			par.reborrow(),
			stack,
			params,
		);
		linalg::copy_across_layouts(matrix, work.rb(), par);
		return ret;
	}
	let size = Ord::min(n, m);
	let config = params.config;
	let has_tol = config.abs_tol.is_some() || config.rel_tol.is_some();
	// a non-finite input switches the tolerances off (see the params' rank truncation section)
	let reveal = has_tol && matrix.is_all_finite();
	// `max_rank` factors only the leading `r` columns; without it `r = size`, as before
	let r = config.max_rank.map_or(size, |r| Ord::min(r, size));
	for i in 0..m {
		let p = &mut perm[i];
		*p = truncate(i);
	}
	let (mut transpositions, _) =
		stack.rb_mut().make_with(size, |_| truncate(0));
	let transpositions = transpositions.as_mut();
	// the recursion applies its row swaps to the columns past `r` but leaves them otherwise
	// untouched: the Schur complement update of a truncated factorization is never run
	let n_transpositions = lu_in_place_recursion(
		matrix.rb_mut(),
		0,
		r,
		&mut transpositions[..r],
		par.reborrow(),
		params,
	);
	for idx in 0..r {
		let t = transpositions[idx];
		perm.as_mut().swap(idx, idx + t.zx());
	}
	// `U[:r, r:]`: for `r = size` this is the wide matrix's solve (`m < n`) and nothing otherwise
	if r < n {
		let (left, right) = matrix.rb_mut().split_at_col_mut(r);
		linalg::triangular_solve::solve_unit_lower_triangular_in_place(
			left.rb().get(..r, ..),
			right.get_mut(..r, ..),
			par,
		);
	}
	for i in 0..m {
		perm_inv[perm[i].zx()] = truncate(i);
	}
	let mut rank = r;
	if reveal && r > 0 {
		let threshold = linalg::lu::rank_threshold::<T>(
			config.abs_tol,
			config.rel_tol,
			&matrix[(0, 0)].abs(),
		);
		if let Some(k) = (0..r)
			.find(|&k| linalg::lu::rank_rejects::<T>(&matrix[(k, k)].abs(), &threshold))
		{
			rank = k;
		}
	}
	(
		PartialPivLuInfo {
			transposition_count: n_transpositions,
			rank,
		},
		unsafe { PermRef::new_unchecked(perm, perm_inv, m) },
	)
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, assert, c32, c64};
	use crate::utils::cost::{pays_at, rate, serial_time};
	use dyn_stack::MemBuffer;
	#[cfg(feature = "rayon")]
	#[test]
	fn region_rules_follow_the_serial_time() {
		use core::num::NonZeroUsize;
		let seq = Par::Seq;
		// sequential never goes wider than one, whatever the size
		assert_eq!(trailing_update_width::<f64>(&seq, ParThreshold::Derived, 4096, 4096, 256), 1);
		assert_eq!(swap_stage_width::<f64>(&seq, ParThreshold::Derived, 4096, 256, 4096, false), 1);
		crate::team::with_team(6, |par| {
			// tiny regions stay serial, huge ones take the whole team
			assert_eq!(trailing_update_width::<f64>(&par, ParThreshold::Derived, 16, 16, 8), 1);
			assert_eq!(trailing_update_width::<f64>(&par, ParThreshold::Derived, 2048, 2048, 128), 6);
			assert_eq!(swap_stage_width::<f64>(&par, ParThreshold::Derived, 64, 16, 48, false), 1);
			assert_eq!(swap_stage_width::<f64>(&par, ParThreshold::Derived, 4096, 256, 4096, false), 6);
			// a complex trailing update reaches the floor at fewer elements than a real one
			let (r, c) = (160, 160);
			let real = trailing_update_width::<f64>(&par, ParThreshold::Derived, r, c, 32);
			let cplx = trailing_update_width::<c64>(&par, ParThreshold::Derived, r, c, 32);
			assert!(cplx >= real);
			// an explicit floor restores the fixed rule (the whole team) and `usize::MAX` disables
			assert_eq!(trailing_update_width::<f64>(&par, ParThreshold::Fixed(1), 64, 64, 8), 6);
			assert_eq!(trailing_update_width::<f64>(&par, ParThreshold::Never, 4096, 4096, 256), 1);
			assert_eq!(swap_stage_width::<f64>(&par, ParThreshold::Never, 4096, 256, 4096, false), 1);
		});
		// the swap stage of a 64 x 1024 matrix (64 swaps over 960 columns, every column on one
		// side: one section) takes the widest width at which the rule holds on the tables of the
		// host class, for both dtypes; the same region counted as the two sided split (four
		// sections) is what kept it serial. The verdict depends on the class, so the test states
		// the rule.
		for w in [4, 8] {
			crate::team::with_team(w, |par| {
				let shape = par.shape();
				let rule = |ns: f64| {
					(2..=w)
						.rev()
						.find(|&k| pays_at(serial_time(64 * 960, ns), SWAP_ONE_SIDE_SECTIONS, shape.dispatch_cost(NonZeroUsize::new(k).unwrap())))
						.unwrap_or(1)
				};
				assert_eq!(swap_stage_width::<f64>(&par, ParThreshold::Derived, 64, 64, 960, true), rule(rate::lu_swap().of::<f64>()), "f64 {w}");
				assert_eq!(swap_stage_width::<c64>(&par, ParThreshold::Derived, 64, 64, 960, true), rule(rate::lu_swap().of::<c64>()), "c64 {w}");
			});
		}
	}

	/// The wide shape of the merge gate (`64 x 1024`): every column of the swap stage is on one
	/// side (one section), and the right block goes through the flat split of the triangular
	/// solve, at the widths the caller reserves.
	#[cfg(feature = "rayon")]
	#[test]
	fn wide_matrix_parallel_agrees_with_serial() {
		fn check<T: ComplexField<Real = f64>>(m: usize, n: usize) {
			let a = Mat::<T>::from_fn(m, n, |i, j| {
				let h = (i * 2654435761usize).wrapping_add(j * 40503) % 1009;
				from_f64::<T>(h as f64 / 504.0 - 1.0)
			});
			let run = |mut par: Par| {
				let mut lu = a.clone();
				let mut perm = vec![0usize; m];
				let mut perm_inv = vec![0usize; m];
				let mut mem = MemBuffer::new(lu_in_place_scratch::<usize, T>(m, n, par.shape(), default()));
				lu_in_place(lu.as_mut(), &mut perm, &mut perm_inv, par.reborrow(), MemStack::new(&mut mem), default());
				(lu, perm)
			};
			let (want, perm_want) = run(Par::Seq);
			for threads in [4, 8] {
				{
					let (got, perm_got) = crate::team::with_team(threads, run);
					assert_eq!(perm_want, perm_got, "{m}x{n} {threads} threads");
					let mut d: f64 = 0.0;
					for j in 0..n {
						for i in 0..m {
							d = d.max((want[(i, j)].clone() - got[(i, j)].clone()).abs());
						}
					}
					assert!(d < 1e-9, "{m}x{n} {threads} threads: {d}");
				}
			}
		}
		check::<f64>(64, 1024);
		check::<c64>(64, 1024);
		check::<f64>(48, 700);
	}

	#[test]
	fn derived_gates_keep_the_factorization() {
		let rng = &mut StdRng::seed_from_u64(7);
		for n in [96, 200, 384] {
			let A = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let mut want = A.clone();
			let mut got = A.clone();
			let mut perm_a = vec![0usize; n];
			let mut perm_ai = vec![0usize; n];
			let mut perm_b = vec![0usize; n];
			let mut perm_bi = vec![0usize; n];
			let mut mem = MemBuffer::new(lu_in_place_scratch::<usize, f64>(n, n, Par::Seq.shape(), default()));
			lu_in_place(want.as_mut(), &mut perm_a, &mut perm_ai, Par::Seq, MemStack::new(&mut mem), default());
			crate::team::with_team(4, |par| {
				lu_in_place(got.as_mut(), &mut perm_b, &mut perm_bi, par, MemStack::new(&mut mem), default())
			});
			assert_eq!(perm_a, perm_b, "n = {n}");
			let mut d: f64 = 0.0;
			for j in 0..n {
				for i in 0..n {
					d = d.max((want[(i, j)] - got[(i, j)]).abs());
				}
			}
			assert!(d < 1e-9, "n = {n}: {d}");
		}
	}

	#[test]
	fn test_plu() {
		let rng = &mut StdRng::seed_from_u64(0);
		let approx_eq = CwiseMat(ApproxEq {
			abs_tol: 1e-13,
			rel_tol: 1e-13,
		});
		for n in [1, 2, 3, 128, 255, 256, 257] {
			let A = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let A = A.as_ref();
			let mut LU = A.cloned();
			let perm = &mut *vec![0usize; n];
			let perm_inv = &mut *vec![0usize; n];
			let params = PartialPivLuParams {
				recursion_threshold: 2,
				block_size: 2,
				..auto!(f64)
			};
			let p = lu_in_place(
				LU.as_mut(),
				perm,
				perm_inv,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(lu_in_place_scratch::<
					usize,
					f64,
				>(
					n,
					n,
					ParShape::Seq,
					params.into(),
				))),
				params.into(),
			)
			.1;
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
			assert!(p.inverse() * L * U ~ A);
		}
		for m in [8, 128, 255, 256, 257] {
			let n = 8;
			let A = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let A = A.as_ref();
			let mut LU = A.cloned();
			let perm = &mut *vec![0usize; m];
			let perm_inv = &mut *vec![0usize; m];
			let p = lu_in_place(
				LU.as_mut(),
				perm,
				perm_inv,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(lu_in_place_scratch::<
					usize,
					f64,
				>(
					n, n, ParShape::Seq, default()
				))),
				default(),
			)
			.1;
			let mut L = LU.as_ref().cloned();
			let mut U = LU.as_ref().cloned();
			for j in 0..n {
				for i in 0..j {
					L[(i, j)] = 0.0;
				}
				L[(j, j)] = 1.0;
			}
			for j in 0..n {
				for i in j + 1..m {
					U[(i, j)] = 0.0;
				}
			}
			let L = L.as_ref();
			let U = U.as_ref();
			let U = U.subrows(0, n);
			assert!(p.inverse() * L * U ~ A);
		}
	}
	fn factor_lu(A: &Mat<f64>, par: Par) -> (Mat<f64>, Vec<usize>, usize) {
		let n = A.nrows();
		let mut LU = A.cloned();
		let perm = &mut *vec![0usize; n];
		let perm_inv = &mut *vec![0usize; n];
		let params: Spec<PartialPivLuParams, f64> = default();
		let stack =
			&mut MemBuffer::new(lu_in_place_scratch::<usize, f64>(n, n, par.shape(), params));
		let stack = MemStack::new(stack);
		let (info, _) = lu_in_place(LU.as_mut(), perm, perm_inv, par, stack, params);
		(LU, perm.to_vec(), info.transposition_count)
	}
	fn factor_lu_c64(A: &Mat<c64>, par: Par) -> (Mat<c64>, Vec<usize>, usize) {
		let n = A.nrows();
		let mut LU = A.cloned();
		let perm = &mut *vec![0usize; n];
		let perm_inv = &mut *vec![0usize; n];
		let params: Spec<PartialPivLuParams, c64> = default();
		let stack =
			&mut MemBuffer::new(lu_in_place_scratch::<usize, c64>(n, n, par.shape(), params));
		let stack = MemStack::new(stack);
		let (info, _) = lu_in_place(LU.as_mut(), perm, perm_inv, par, stack, params);
		(LU, perm.to_vec(), info.transposition_count)
	}
	/// The two products at a recursion level take `Par::Seq` while the trailing update's work is
	/// below the floor, instead of the caller's hint. That is a choice of how to schedule the
	/// same arithmetic, so both sides of the floor have to return the same factorization: `96`,
	/// `128` and `192` are below `128 * 128` `f64` elements at the top level, `256` and `512`
	/// above it, and every level in between halves down through it. A `c64` element is four real
	/// MACs, so the same floor is reached at a quarter of the area -- `128` is below it and `256`
	/// above it for `c64` too, which is the case the element-only comparison got wrong.
	#[test]
	fn threaded_lu_matches_the_sequential_one() {
		let rng = &mut StdRng::seed_from_u64(0);
		let approx_eq = CwiseMat(ApproxEq {
			abs_tol: 1e-12,
			rel_tol: 1e-12,
		});
		for n in [96, 128, 192, 256, 512] {
			let A = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let (seq, perm_seq, trans_seq) = factor_lu(&A, Par::Seq);
			let (par, perm_par, trans_par) =
				crate::team::with_team(4, |par| factor_lu(&A, par));
			assert!(seq.as_ref() ~ par.as_ref());
			assert!(perm_seq == perm_par);
			assert!(trans_seq == trans_par);
		}
		for n in [96, 128, 192, 256, 512] {
			let real = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let imag = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let A = Mat::<c64>::from_fn(n, n, |i, j| c64::new(real[(i, j)], imag[(i, j)]));
			let (seq, perm_seq, trans_seq) = factor_lu_c64(&A, Par::Seq);
			let (par, perm_par, trans_par) =
				crate::team::with_team(4, |par| factor_lu_c64(&A, par));
			for j in 0..n {
				for i in 0..n {
					let diff = (seq[(i, j)] - par[(i, j)]).abs();
					assert!(diff <= 1e-10, "({i}, {j}) differs by {diff:e}");
				}
			}
			assert!(perm_seq == perm_par);
			assert!(trans_seq == trans_par);
		}
	}
	/// An exactly zero pivot column is a singular matrix, not a failed factorization: the
	/// factors stay finite and still reconstruct the input, with the singularity left in `U`.
	/// A subnormal pivot must not be scaled by its infinite reciprocal either. OpenBLAS's
	/// `getf2_k` skips the column when the pivot is exactly zero, and reference `dgetf2`
	/// divides elementwise below `SFMIN`.
	#[test]
	fn zero_pivot_column_reconstructs() {
		fn zero_pivot<T: ComplexField>(n: usize) -> Mat<T> {
			Mat::<T>::from_fn(n, n, |i, j| {
				if j > 0 {
					from_f64::<T>((i + 2 * j) as f64)
				} else {
					zero()
				}
			})
		}
		// the pivot is the largest entry of a column, so a subnormal pivot needs a
		// subnormal fixture: `tiny` is the smallest useful one for each dtype
		fn subnormal<T: ComplexField>(tiny: f64) -> Mat<T> {
			let tiny = from_f64::<T>(tiny);
			Mat::<T>::from_fn(2, 2, |i, j| match (i, j) {
				(0, 0) => tiny.copy(),
				(0, 1) => one(),
				(1, 0) => tiny.mul_real(from_f64::<T::Real>(0.5)),
				(1, 1) => from_f64::<T>(2.0),
				_ => zero(),
			})
		}
		for n in [1usize, 2, 3, 128] {
			check_factor::<f32>(zero_pivot::<f32>(n).as_ref(), None);
			check_factor::<c32>(zero_pivot::<c32>(n).as_ref(), None);
			check_factor::<f64>(zero_pivot::<f64>(n).as_ref(), None);
			check_factor::<c64>(zero_pivot::<c64>(n).as_ref(), None);
		}
		// a power of two so that the subnormal and its half are both exact and the
		// expected multiplier is exactly `0.5`
		check_factor::<f32>(
			subnormal::<f32>(2f64.powi(-147)).as_ref(),
			Some(0.5f32),
		);
		check_factor::<c32>(
			subnormal::<c32>(2f64.powi(-147)).as_ref(),
			Some(c32::new(0.5, 0.0)),
		);
		check_factor::<f64>(
			subnormal::<f64>(2f64.powi(-1072)).as_ref(),
			Some(0.5f64),
		);
		check_factor::<c64>(
			subnormal::<c64>(2f64.powi(-1072)).as_ref(),
			Some(c64::new(0.5, 0.0)),
		);
	}
	fn check_factor<T: ComplexField>(a: MatRef<'_, T>, l10: Option<T>) {
		let tol = eps::<T::Real>() * from_f64::<T::Real>(64.0);
		let approx_eq = CwiseMat(ApproxEq {
			abs_tol: tol.copy(),
			rel_tol: tol,
		});
		let n = a.nrows();
		let run = |mut par: Par| {
			let par_shape = par.shape();
			let mut LU = a.cloned();
			let perm = &mut *vec![0usize; n];
			let perm_inv = &mut *vec![0usize; n];
			let params = PartialPivLuParams {
				recursion_threshold: 2,
				block_size: 2,
				..auto!(T)
			};
			let p = lu_in_place(
				LU.as_mut(),
				perm,
				perm_inv,
				par.reborrow(),
				MemStack::new(&mut MemBuffer::new(lu_in_place_scratch::<
					usize,
					T,
				>(
					n, n, par_shape, params.into()
				))),
				params.into(),
			)
			.1;
			assert!(LU.as_ref().is_all_finite());
			let mut L = LU.as_ref().cloned();
			let mut U = LU.as_ref().cloned();
			for j in 0..n {
				for i in 0..j {
					L[(i, j)] = zero();
				}
				L[(j, j)] = one();
			}
			for j in 0..n {
				for i in j + 1..n {
					U[(i, j)] = zero();
				}
			}
			assert!(p.inverse() * L.as_ref() * U.as_ref() ~ a);
			if let Some(ref x) = l10 {
				assert!(
					(L[(1, 0)].copy() - x.copy()).abs()
						<= eps::<T::Real>() * from_f64::<T::Real>(4.0)
				);
			}
		};
		run(Par::Seq);
		crate::team::with_team(8, run);
	}

	/// The `f64` default leaf is 8 columns (the other types keep 16), and it factors shapes whose
	/// width leaves a leaf tail, sequentially and on a team: `P A = L U` to rounding.
	fn simd_leaf_picks_the_scalar_pivots_for<T: ComplexField<Real = T> + Copy + PartialEq>(
		from: impl Fn(f64) -> T,
	) {
		// the same panels through both leaves: the transpositions must be identical, ties
		// included (a matrix of small integers ties a lot, and the scalar search takes the
		// first row), and the factor bit for bit (the update is the same fused multiply-add as
		// the scalar leaf's `matmul`); the panel sits between other columns, which the row
		// swaps must reach too
		let rng = &mut StdRng::seed_from_u64(0x5EAF);
		let shapes = [(1usize, 1usize), (8, 8), (9, 8), (16, 8), (17, 8), (33, 5), (100, 8), (257, 16), (64, 3)];
		for (m, w) in shapes {
			for ties in [false, true] {
				let ncols = w + 6;
				let R = CwiseMatDistribution {
					nrows: m,
					ncols,
					dist: StandardNormal,
				}
				.rand::<Mat<f64>>(rng);
				let A = Mat::from_fn(m, ncols, |i, j| {
					// small integers: exact ties first, near ties after a step
					from(if ties { ((i * 7 + j * 3) % 5) as f64 + (R[(i, j)] * 2.0).round() } else { R[(i, j)] })
				});
				let (start, end) = (3, 3 + w);
				let (mut simd, mut scalar) = (A.clone(), A.clone());
				let (mut t_simd, mut t_scalar) = (vec![0usize; w], vec![0usize; w]);
				let n_simd =
					lu_in_place_unblocked_simd(simd.as_mut(), start, end, &mut t_simd);
				let n_scalar =
					lu_in_place_unblocked_scalar(scalar.as_mut(), start, end, &mut t_scalar);
				assert!(all(t_simd == t_scalar, n_simd == n_scalar));
				assert!(simd == scalar);
			}
		}
	}
	#[test]
	fn simd_leaf_picks_the_scalar_pivots() {
		simd_leaf_picks_the_scalar_pivots_for::<f64>(|x| x);
		simd_leaf_picks_the_scalar_pivots_for::<f32>(|x| x as f32);
	}
	#[test]
	fn f64_default_leaf_factors() {
		let rt_f64 =
			<PartialPivLuParams as Auto<f64>>::auto().recursion_threshold;
		let rt_f32 =
			<PartialPivLuParams as Auto<f32>>::auto().recursion_threshold;
		let rt_c64 =
			<PartialPivLuParams as Auto<c64>>::auto().recursion_threshold;
		assert!(all(
			rt_f64 == F64_RECURSION_THRESHOLD,
			rt_f32 == 16,
			rt_c64 == 16
		));
		let rng = &mut StdRng::seed_from_u64(0x1EAF);
		for n in [7usize, 8, 9, 17, 100, 257] {
			let A = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			for team in [1usize, 4] {
				let mut LU = A.clone();
				let (mut perm, mut inv) = (vec![0usize; n], vec![0usize; n]);
				crate::team::with_team(team, |par| {
					lu_in_place(
						LU.as_mut(),
						&mut perm,
						&mut inv,
						par,
						MemStack::new(&mut MemBuffer::new(
							lu_in_place_scratch::<usize, f64>(
								n,
								n,
								ParShape::Seq,
								default(),
							),
						)),
						default(),
					);
				});
				let L = Mat::<f64>::from_fn(n, n, |i, j| {
					if i > j {
						LU[(i, j)]
					} else if i == j {
						1.0
					} else {
						0.0
					}
				});
				let U = Mat::<f64>::from_fn(n, n, |i, j| {
					if i <= j { LU[(i, j)] } else { 0.0 }
				});
				let PA = Mat::<f64>::from_fn(n, n, |i, j| A[(perm[i], j)]);
				let err = (&PA - &L * &U).norm_max();
				let tol = 64.0 * n as f64 * f64::EPSILON * A.norm_max();
				assert!(
					err <= tol,
					"n={n} team={team}: |PA - LU| = {err:e} > {tol:e}"
				);
			}
		}
	}
}
