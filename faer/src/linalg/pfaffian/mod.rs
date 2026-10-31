//! Pfaffians of real or complex skew-symmetric matrices.
//!
//! Partial, pivoted Parlett–Reid reduction, with delayed rank-two updates:
//! M. Wimmer, *Algorithm 923: Efficient Numerical Computation of the Pfaffian
//! for Dense and Banded Skew-Symmetric Matrices*, ACM TOMS 38(4), 2012,
//! [§II.B and Appendix B](https://arxiv.org/abs/1102.3440).
//! For pivoted panels and the serial-panel/parallel-update distinction, see
//! I. Satyarth et al., *Skew-Symmetric Matrix Decompositions on Shared-Memory
//! Architectures*, [§§3.6–3.7](https://arxiv.org/html/2411.09859v2).
//! This is an independent implementation of the published equations; no
//! PfaPack or mvmc source code was used.
//!
//! The panel and pivot decisions are ordered. Completed panels update disjoint
//! trailing column blocks on the caller's team; sequential execution uses the
//! same update kernel. Transposes here are **not conjugate transposes**.

use crate::internal_prelude::*;
use crate::utils::cost::{ParThreshold, rate};
use crate::utils::partition::lower_area_split;
use crate::utils::thread::{Ptr, with_width};
use faer_traits::RealReg;
use pulp::Simd;

/// Failure to compute a finite, nonzero Pfaffian.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack, StackReq};
/// use faer::linalg::pfaffian::{PfaffianError, pfaffian};
/// use faer::{Mat, Par};
/// let a = Mat::<f64>::zeros(3, 3);
/// let mut memory = MemBuffer::new(StackReq::EMPTY);
/// let result = pfaffian(a.as_ref(), Par::Seq, MemStack::new(&mut memory));
/// assert_eq!(result, Err(PfaffianError::OddDimension));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PfaffianError {
	/// The input is not square.
	NonSquare,
	/// The input dimension is odd.
	OddDimension,
	/// An exactly zero pivot was encountered at this zero-based column.
	ZeroPivot { index: usize },
	/// Nonfinite input or arithmetic, or a nonzero product underflowing to
	/// zero.
	NumericalFailure,
}
impl core::fmt::Display for PfaffianError {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		match self {
			Self::NonSquare => f.write_str("Pfaffian input is not square"),
			Self::OddDimension => {
				f.write_str("Pfaffian input has odd dimension")
			},
			Self::ZeroPivot { index } => {
				write!(f, "zero Pfaffian pivot at column {index}")
			},
			Self::NumericalFailure => {
				f.write_str("Pfaffian arithmetic is nonfinite or underflowed")
			},
		}
	}
}
impl core::error::Error for PfaffianError {}

/// Workspace for [`pfaffian`] on a matrix of dimension `dim`.
///
/// # Examples
/// ```
/// use dyn_stack::MemBuffer;
/// use faer::linalg::pfaffian::pfaffian_scratch;
/// let memory = MemBuffer::new(pfaffian_scratch::<f64>(4));
/// assert!(!memory.is_empty());
/// ```
pub fn pfaffian_scratch<T: ComplexField>(dim: usize) -> StackReq {
	linalg::temp_mat_scratch::<T>(dim, dim)
		.and(linalg::temp_mat_scratch::<T>(dim / 2, 1))
}

// Every other elimination is sufficient for the Pfaffian (Wimmer §II.B).
const PANEL: usize = 32;

/// Computes the Pfaffian without modifying `a` or allocating working memory.
///
/// `a` must be skew-symmetric: `a[i,j] = -a[j,i]`, with zero diagonal, **not**
/// skew-Hermitian. As with the adjacent symmetric factorizations, this is a
/// caller precondition, not a tolerance-based check: only the strict lower
/// triangle is authoritative. The upper triangle and diagonal are ignored.
/// The empty matrix has Pfaffian one.
///
/// # Errors
/// Returns [`PfaffianError::NonSquare`] or [`PfaffianError::OddDimension`]
/// before using scratch for invalid shapes. An exactly zero pivot gives
/// [`PfaffianError::ZeroPivot`], rather than silently returning zero.
/// [`PfaffianError::NumericalFailure`] reports nonfinite lower-triangle input,
/// nonfinite arithmetic, or product underflow. There is no near-singularity
/// tolerance or scaled/logarithmic output; extreme scales may fail even when
/// the mathematical Pfaffian is representable.
///
/// # Panics
/// Panics if `stack` is smaller than [`pfaffian_scratch`].
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::linalg::pfaffian::{pfaffian, pfaffian_scratch};
/// use faer::{Par, mat};
/// let a = mat![[0.0, -3.0], [3.0, 0.0]];
/// let mut memory = MemBuffer::new(pfaffian_scratch::<f64>(2));
/// let p = pfaffian(a.as_ref(), Par::Seq, MemStack::new(&mut memory))?;
/// assert_eq!(p, -3.0);
/// # Ok::<(), faer::linalg::pfaffian::PfaffianError>(())
/// ```
pub fn pfaffian<T: ComplexField>(
	a: MatRef<'_, T>,
	mut par: Par<'_>,
	stack: &mut MemStack,
) -> Result<T, PfaffianError> {
	let n = a.nrows();
	if n != a.ncols() {
		return Err(PfaffianError::NonSquare);
	}
	if n % 2 != 0 {
		return Err(PfaffianError::OddDimension);
	}
	if n == 0 {
		return Ok(one());
	}
	for j in 0..n {
		for i in j + 1..n {
			if !a[(i, j)].is_finite() {
				return Err(PfaffianError::NumericalFailure);
			}
		}
	}
	// SAFETY: the input copy initializes every matrix entry before it is read.
	let (mut work, stack) =
		unsafe { linalg::temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut work = work.as_mat_mut();
	work.copy_from(a);
	let mut work = work.try_as_col_major_mut().unwrap();
	// SAFETY: each pivot entry is written before utu2pfa reads the column.
	let (mut pivots, _) =
		unsafe { linalg::temp_mat_uninit::<T, _, _>(n / 2, 1, stack) };
	let mut pivots = pivots.as_mat_mut();
	let mut odd_permutation = false;
	for start in (0..n).step_by(PANEL) {
		let end = (start + PANEL).min(n);
		for k in (start..end).step_by(2) {
			update_panel_column(work.rb_mut(), start, k);
			let pivot =
				k + 1 + pivot_index(work.rb().col(k).subrows(k + 1, n - k - 1));
			if pivot != k + 1 {
				swap_lower(work.rb_mut(), k + 1, pivot);
				odd_permutation = !odd_permutation;
			}
			let d = work[(k + 1, k)].copy();
			if !d.is_finite() {
				return Err(PfaffianError::NumericalFailure);
			}
			if d == zero() {
				return Err(PfaffianError::ZeroPivot { index: k });
			}
			pivots[(k / 2, 0)] = -&d;
			update_panel_column(work.rb_mut(), start, k + 1);
			for i in k + 2..n {
				work[(i, k)] = &work[(i, k)] / &d;
			}
		}
		if end < n {
			let (prefix, trailing) = work.rb_mut().split_at_col_mut(end);
			let factors = prefix.rb().subcols(start, end - start);
			let trailing = trailing.subrows_mut(end, n - end);
			update_trailing(trailing, factors, end, par.reborrow());
		}
	}
	let value = utu2pfa(pivots.col_mut(0));
	if !value.is_finite() || value == zero() {
		return Err(PfaffianError::NumericalFailure);
	}
	Ok(if odd_permutation { -value } else { value })
}

// A symmetric permutation, including rows of already stored multipliers.
fn swap_lower<T: ComplexField>(
	mut a: MatMut<'_, T, usize, usize, ContiguousFwd>,
	p: usize,
	q: usize,
) {
	for j in 0..p {
		let x = a[(p, j)].copy();
		a[(p, j)] = a[(q, j)].copy();
		a[(q, j)] = x;
	}
	for j in p + 1..q {
		let x = a[(j, p)].copy();
		a[(j, p)] = -&a[(q, j)];
		a[(q, j)] = -x;
	}
	a[(q, p)] = -&a[(q, p)];
	for i in q + 1..a.nrows() {
		let x = a[(i, p)].copy();
		a[(i, p)] = a[(i, q)].copy();
		a[(i, q)] = x;
	}
}

fn pivot_index<T: ComplexField>(
	col: ColRef<'_, T, usize, ContiguousFwd>,
) -> usize {
	struct Impl<'a, T: ComplexField> {
		col: ColRef<'a, T, usize, ContiguousFwd>,
	}
	impl<T: ComplexField> pulp::WithSimd for Impl<'_, T> {
		type Output = usize;

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> usize {
			let col = self.col;
			if const { T::SIMD_CAPABILITIES.is_simd() } {
				with_dim!(N, col.nrows());
				let col = col.as_row_shape(N);
				let ctx = SimdCtx::<T, S>::new_force_mask(T::simd_ctx(simd), N);
				let mut largest = RealReg(ctx.zero());
				let mut at = ctx.izero();
				let mut visit = |value, offset| {
					let index = ctx.iadd(
						T::simd_iota(&ctx.0),
						ctx.isplat(T::Index::truncate(offset)),
					);
					let value = ctx.abs_max(value);
					let better = (*ctx).gt(value, largest);
					largest = RealReg(ctx.select(better, value.0, largest.0));
					at = ctx.iselect(better, index, at);
				};
				let (_, body, tail) = ctx.indices();
				for i in body {
					visit(ctx.read(col, i), i.offset() as usize);
				}
				if let Some(i) = tail {
					let lanes = core::mem::size_of::<T::SimdVec<S>>()
						/ core::mem::size_of::<T>();
					visit(
						ctx.read(col, i),
						(col.nrows().unbound() - 1) / lanes * lanes,
					);
				}
				let value = ctx.reduce_max_real(largest);
				let ties = (*ctx).ge(largest, ctx.splat_real(&value));
				let at = ctx.iselect(
					ties,
					at,
					ctx.isplat(T::Index::truncate(usize::MAX)),
				);
				// The smallest tied row is independent of SIMD width, including
				// masked tails.
				bytemuck::cast_slice::<T::SimdIndex<S>, T::Index>(
					core::slice::from_ref(&at),
				)
				.iter()
				.map(|i| i.zx())
				.min()
				.unwrap()
			} else {
				let mut at = 0;
				for i in 1..col.nrows() {
					if col[i].absmax() > col[at].absmax() {
						at = i;
					}
				}
				at
			}
		}
	}
	dispatch!(Impl { col }, Impl, T)
}

fn update_panel_column<T: ComplexField>(
	mut work: MatMut<'_, T, usize, usize, ContiguousFwd>,
	start: usize,
	col: usize,
) {
	// The odd column must not include its own, not-yet-normalized pivot pair.
	let pairs_end = col - col % 2;
	if pairs_end == start {
		return;
	}
	let (prefix, remaining) = work.rb_mut().split_at_col_mut(col);
	update_columns(
		remaining.subcols_mut(0, 1),
		prefix.rb().subcols(start, pairs_end - start),
		col,
		0,
	);
}

// Apply the two rank-one terms X Y^T - Y X^T, keeping every entry's sum order
// fixed.
fn update_columns<T: ComplexField>(
	a: MatMut<'_, T, usize, usize, ContiguousFwd>,
	factors: MatRef<'_, T, usize, usize, ContiguousFwd>,
	col_start: usize,
	row_start: usize,
) {
	struct Impl<'a, T: ComplexField> {
		a: MatMut<'a, T, usize, usize, ContiguousFwd>,
		factors: MatRef<'a, T, usize, usize, ContiguousFwd>,
		col_start: usize,
		row_start: usize,
	}
	impl<T: ComplexField> pulp::WithSimd for Impl<'_, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) {
			let Self {
				mut a,
				factors,
				col_start,
				row_start,
			} = self;
			for j in 0..a.ncols() {
				let column = col_start + j;
				let first = column + 1;
				let len = factors.nrows() - first;
				if len == 0 {
					continue;
				}
				let mut dst =
					a.rb_mut().col_mut(j).subrows_mut(first - row_start, len);
				if const { T::SIMD_CAPABILITIES.is_simd() } {
					with_dim!(N, len);
					let mut dst = dst.as_row_shape_mut(N);
					let ctx =
						SimdCtx::<T, S>::new_force_mask(T::simd_ctx(simd), N);
					let (head, body, rest, tail) = ctx.batch_indices::<4>();
					// Four vectors hide the rank-two FMA dependency chain.
					for indices in body {
						let mut values = indices.map(|i| ctx.read(dst.rb(), i));
						for k in (0..factors.ncols()).step_by(2) {
							let x = factors
								.col(k)
								.subrows(first, len)
								.as_row_shape(N);
							let y = factors
								.col(k + 1)
								.subrows(first, len)
								.as_row_shape(N);
							let alpha = ctx.splat(&factors[(column, k + 1)]);
							let beta = ctx.splat(&(-&factors[(column, k)]));
							for b in 0..4 {
								values[b] = ctx.mul_add(
									ctx.read(x, indices[b]),
									alpha,
									values[b],
								);
								values[b] = ctx.mul_add(
									ctx.read(y, indices[b]),
									beta,
									values[b],
								);
							}
						}
						for b in 0..4 {
							ctx.write(dst.rb_mut(), indices[b], values[b]);
						}
					}
					simd_iter!(for i in [(head, rest, tail)] {
						let mut value = ctx.read(dst.rb(), i);
						for k in (0..factors.ncols()).step_by(2) {
							let x = factors
								.col(k)
								.subrows(first, len)
								.as_row_shape(N);
							let y = factors
								.col(k + 1)
								.subrows(first, len)
								.as_row_shape(N);
							value = ctx.mul_add(
								ctx.read(x, i),
								ctx.splat(&factors[(column, k + 1)]),
								value,
							);
							value = ctx.mul_add(
								ctx.read(y, i),
								ctx.splat(&(-&factors[(column, k)])),
								value,
							);
						}
						ctx.write(dst.rb_mut(), i, value);
					});
				} else {
					for i in 0..len {
						for k in (0..factors.ncols()).step_by(2) {
							dst[i] = &dst[i]
								+ &factors[(first + i, k)]
									* &factors[(column, k + 1)];
							dst[i] = &dst[i]
								- &factors[(first + i, k + 1)]
									* &factors[(column, k)];
						}
					}
				}
			}
		}
	}
	dispatch!(
		Impl {
			a,
			factors,
			col_start,
			row_start
		},
		Impl,
		T
	)
}

fn update_trailing<T: ComplexField>(
	a: MatMut<'_, T, usize, usize, ContiguousFwd>,
	factors: MatRef<'_, T, usize, usize, ContiguousFwd>,
	origin: usize,
	par: Par<'_>,
) {
	let n = a.nrows();
	let units = n.saturating_mul(n).saturating_mul(factors.ncols() / 2);
	let width = ParThreshold::Derived
		.region_width::<T>(&par, units, rate::bunch_kaufman(), 1)
		.min(n);
	with_width(par, width, |par| {
		let ptr = Ptr(a.as_ptr_mut());
		let col_stride = a.col_stride();
		let run = &|rank: usize, parts: usize| {
			let ptr = ptr;
			let j0 = lower_area_split(n - 1, n, rank, parts, 1);
			let j1 = lower_area_split(n - 1, n, rank + 1, parts, 1);
			// SAFETY: monotone column boundaries partition this contiguous-row
			// matrix. Each joined team member owns distinct columns, disjoint
			// from `factors`.
			let block = unsafe {
				MatMut::from_raw_parts_mut(
					ptr.0.wrapping_offset(j0 as isize * col_stride),
					n,
					j1 - j0,
					1,
					col_stride,
				)
			}
			.try_as_col_major_mut()
			.unwrap();
			update_columns(block, factors, origin + j0, origin);
		};
		match par {
			Par::Seq => run(0, 1),
			#[cfg(feature = "threadteam")]
			Par::Team(team) => crate::utils::team::run(team, width, run),
			#[cfg(not(feature = "threadteam"))]
			Par::__Lifetime(x, _) => match x {},
		}
	});
}

// Pf(T_2r) = t[0,1] t[2,3] ... t[2r-2,2r-1]. A product tree supplies
// SIMD accumulation without a lane-width-dependent horizontal reduction.
fn utu2pfa<T: ComplexField>(pivots: ColMut<'_, T>) -> T {
	struct Impl<'a, T: ComplexField> {
		pivots: ColMut<'a, T>,
	}
	impl<T: ComplexField> pulp::WithSimd for Impl<'_, T> {
		type Output = T;

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> T {
			let mut pivots = self.pivots;
			let mut len = pivots.nrows();
			while len > 1 {
				let mid = len.div_ceil(2);
				let pairs = len - mid;
				let (left, right) =
					pivots.rb_mut().subrows_mut(0, len).split_at_row_mut(mid);
				let mut left = left.subrows_mut(0, pairs);
				if const { T::SIMD_CAPABILITIES.is_simd() } {
					with_dim!(N, pairs);
					let mut left = left
						.try_as_col_major_mut()
						.unwrap()
						.as_row_shape_mut(N);
					let right =
						right.rb().try_as_col_major().unwrap().as_row_shape(N);
					let ctx =
						SimdCtx::<T, S>::new_force_mask(T::simd_ctx(simd), N);
					simd_iter!(for i in [ctx.indices()] {
						let value =
							ctx.mul(ctx.read(left.rb(), i), ctx.read(right, i));
						ctx.write(left.rb_mut(), i, value);
					});
				} else {
					for i in 0..pairs {
						left[i] = &left[i] * &right[i];
					}
				}
				len = mid;
			}
			pivots[0].copy()
		}
	}
	dispatch!(Impl { pivots }, Impl, T)
}
