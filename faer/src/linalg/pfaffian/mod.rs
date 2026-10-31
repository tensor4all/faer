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
//! trailing row blocks on the caller's team using a shared compact panel.
//! Pivots exchange logical indices only; tile stores maintain both triangles.
//! Sequential execution degenerates to the same kernels.
//! Transposes here are **not conjugate transposes**.

use crate::internal_prelude::*;
use crate::utils::cost::{ParThreshold, rate};
use crate::utils::partition::lower_area_split;
use crate::utils::thread::{Ptr, with_width};
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
	linalg::temp_mat_scratch::<T>(
		work_rows::<T>(dim),
		if dim > PANEL { dim } else { 0 },
	)
	.and(linalg::temp_mat_scratch::<T>(
		work_rows::<T>(dim),
		PANEL.min(dim),
	))
	.and(linalg::temp_mat_scratch::<T>(
		work_rows::<T>(dim),
		if dim > PANEL { PANEL } else { 0 },
	))
	.and(linalg::temp_mat_scratch::<T>(
		work_rows::<T>(PANEL),
		tile_columns(dim),
	))
	.and(linalg::temp_mat_scratch::<T>(dim / 2, 1))
	.and(StackReq::new::<usize>(dim))
}

// Like SVD's work matrix, avoid cache-set aliasing in strided row accesses.
fn work_rows<T>(n: usize) -> usize {
	let size = core::mem::size_of::<T>();
	if size > 0 && (n * size) % 1024 == 0 {
		n + 64usize.div_ceil(size)
	} else {
		n
	}
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
	if !valid_input(a, par.reborrow()) {
		return Err(PfaffianError::NumericalFailure);
	}
	// SAFETY: the first panel reads input. Completed updates initialize both
	// orientations of surviving pairs; later panels gather only live entries.
	let (mut work, stack) = unsafe {
		linalg::temp_mat_uninit::<T, _, _>(
			work_rows::<T>(n),
			if n > PANEL { n } else { 0 },
			stack,
		)
	};
	let mut work = work
		.as_mat_mut()
		.subrows_mut(0, n)
		.try_as_col_major_mut()
		.unwrap();
	// SAFETY: every panel column is filled before use; padding is never read.
	let (mut factors, stack) = unsafe {
		linalg::temp_mat_uninit::<T, _, _>(
			work_rows::<T>(n),
			PANEL.min(n),
			stack,
		)
	};
	let mut factors = factors
		.as_mat_mut()
		.subrows_mut(0, n)
		.try_as_col_major_mut()
		.unwrap();
	// SAFETY: packing fills the complete compact panel before the joined team
	// reads it. No trailing update is needed when n <= PANEL.
	let (mut packed, stack) = unsafe {
		linalg::temp_mat_uninit::<T, _, _>(
			work_rows::<T>(n),
			if n > PANEL { PANEL } else { 0 },
			stack,
		)
	};
	let mut packed = packed
		.as_mat_mut()
		.subrows_mut(0, n)
		.try_as_col_major_mut()
		.unwrap();
	// SAFETY: each task gathers its complete tile before GEMM reads it.
	let (mut tiles, stack) = unsafe {
		linalg::temp_mat_uninit::<T, _, _>(
			work_rows::<T>(PANEL),
			tile_columns(n),
			stack,
		)
	};
	let mut tiles = tiles.as_mat_mut().try_as_col_major_mut().unwrap();
	// SAFETY: every pivot is written before utu2pfa reads it.
	let (mut pivots, stack) =
		unsafe { linalg::temp_mat_uninit::<T, _, _>(n / 2, 1, stack) };
	let mut pivots = pivots.as_mat_mut();
	let (mut perm, _) = stack.make_with(n, |i| i);
	let mut odd_permutation = false;
	for start in (0..n).step_by(PANEL) {
		let end = (start + PANEL).min(n);
		for k in (start..end).step_by(2) {
			let c = perm[k];
			let source = if start == 0 {
				a
			} else {
				work.rb().as_dyn_stride()
			};
			update_panel_column(
				factors.rb_mut(),
				source,
				c,
				k - start,
				&perm[k..],
				start == 0,
			);
			let pivot = k
				+ 1
				+ pivot_index(factors.rb().col(k - start), &perm[k + 1..])?;
			if pivot != k + 1 {
				perm.swap(k + 1, pivot);
				odd_permutation = !odd_permutation;
			}
			let q = perm[k + 1];
			let d = factors[(q, k - start)].copy();
			if !d.is_finite() {
				return Err(PfaffianError::NumericalFailure);
			}
			if d == zero() {
				return Err(PfaffianError::ZeroPivot { index: k });
			}
			pivots[(k / 2, 0)] = -&d;
			update_panel_column(
				factors.rb_mut(),
				source,
				q,
				k + 1 - start,
				&perm[k..],
				start == 0,
			);
			for &i in &perm[k + 2..] {
				factors[(i, k - start)] = &factors[(i, k - start)] / &d;
			}
		}
		if end < n {
			let remaining = &perm[end..];
			let mut compact = packed.rb_mut().submatrix_mut(
				0,
				0,
				remaining.len(),
				end - start,
			);
			for j in 0..compact.ncols() {
				for (i, &physical) in remaining.iter().enumerate() {
					compact[(i, j)] = factors[(physical, j)].copy();
				}
			}
			if start == 0 {
				update_trailing::<true, T>(
					work.rb_mut(),
					compact.rb(),
					a,
					remaining,
					tiles.rb_mut(),
					par.reborrow(),
				);
			} else {
				update_trailing::<false, T>(
					work.rb_mut(),
					compact.rb(),
					a,
					remaining,
					tiles.rb_mut(),
					par.reborrow(),
				);
			}
		}
	}
	let value = utu2pfa(pivots.col_mut(0));
	if !value.is_finite() || value == zero() {
		return Err(PfaffianError::NumericalFailure);
	}
	Ok(if odd_permutation { -value } else { value })
}

fn tile_columns(n: usize) -> usize {
	if n > PANEL {
		n.div_ceil(PANEL) * PANEL
	} else {
		0
	}
}

fn valid_input<T: ComplexField>(a: MatRef<'_, T>, par: Par<'_>) -> bool {
	use core::sync::atomic::{AtomicBool, Ordering};
	let invalid = AtomicBool::new(false);
	let n = a.nrows();
	let width = par.degree().min(n);
	with_width(par, width, |par| {
		let run = &|rank: usize, parts: usize| {
			let mut valid = true;
			for j in n * rank / parts..n * (rank + 1) / parts {
				for i in j + 1..n {
					valid &= a[(i, j)].is_finite();
				}
			}
			if !valid {
				invalid.store(true, Ordering::Relaxed);
			}
		};
		match par {
			Par::Seq => run(0, 1),
			#[cfg(feature = "threadteam")]
			Par::Team(team) => crate::utils::team::run(team, width, run),
			#[cfg(not(feature = "threadteam"))]
			Par::__Lifetime(x, _) => match x {},
		}
	});
	!invalid.load(Ordering::Relaxed)
}

fn pivot_index<T: ComplexField>(
	col: ColRef<'_, T, usize, ContiguousFwd>,
	perm: &[usize],
) -> Result<usize, PfaffianError> {
	let mut at = 0;
	for i in 0..perm.len() {
		if !col[perm[i]].is_finite() {
			return Err(PfaffianError::NumericalFailure);
		}
		if col[perm[i]].absmax() > col[perm[at]].absmax() {
			at = i;
		}
	}
	Ok(at)
}

fn update_panel_column<T: ComplexField>(
	mut factors: MatMut<'_, T, usize, usize, ContiguousFwd>,
	a: MatRef<'_, T>,
	column: usize,
	at: usize,
	live: &[usize],
	lower_only: bool,
) {
	let (prefix, remaining) = factors.rb_mut().split_at_col_mut(at);
	let mut dst = remaining.col_mut(0);
	dst.rb_mut().fill(zero());
	if lower_only {
		for &i in live {
			dst[i] = if i > column {
				a[(i, column)].copy()
			} else if i < column {
				-&a[(column, i)]
			} else {
				zero()
			};
		}
	} else {
		for &i in live {
			dst[i] = if i == column {
				zero()
			} else {
				a[(i, column)].copy()
			};
		}
	}
	// The odd column must exclude its own, not-yet-normalized pivot pair.
	let pairs_end = at - at % 2;
	if pairs_end > 0 {
		update_column(dst, prefix.rb().subcols(0, pairs_end), column, 0);
	}
}

// The dependent panel columns retain the ordered rank-two SIMD recurrence.
fn update_column<T: ComplexField>(
	dst: ColMut<'_, T, usize, ContiguousFwd>,
	factors: MatRef<'_, T, usize, usize, ContiguousFwd>,
	column: usize,
	first: usize,
) {
	if dst.nrows() == 0 || factors.ncols() == 0 {
		return;
	}
	struct Impl<'a, T: ComplexField> {
		dst: ColMut<'a, T, usize, ContiguousFwd>,
		factors: MatRef<'a, T, usize, usize, ContiguousFwd>,
		column: usize,
		first: usize,
	}
	impl<T: ComplexField> pulp::WithSimd for Impl<'_, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) {
			let Self {
				mut dst,
				factors,
				column,
				first,
			} = self;
			let len = dst.nrows();
			if const { T::SIMD_CAPABILITIES.is_simd() } {
				with_dim!(N, len);
				let mut dst = dst.as_row_shape_mut(N);
				let ctx = SimdCtx::<T, S>::new_force_mask(T::simd_ctx(simd), N);
				let (head, body, rest, tail) = ctx.batch_indices::<4>();
				// Four vectors hide the rank-two FMA dependency chain.
				for indices in body {
					let mut values = indices.map(|i| ctx.read(dst.rb(), i));
					for k in (0..factors.ncols()).step_by(2) {
						let x =
							factors.col(k).subrows(first, len).as_row_shape(N);
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
						let x =
							factors.col(k).subrows(first, len).as_row_shape(N);
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
	dispatch!(
		Impl {
			dst,
			factors,
			column,
			first
		},
		Impl,
		T
	)
}

fn update_trailing<const INITIAL: bool, T: ComplexField>(
	a: MatMut<'_, T, usize, usize, ContiguousFwd>,
	factors: MatRef<'_, T, usize, usize, ContiguousFwd>,
	input: MatRef<'_, T>,
	perm: &[usize],
	tiles: MatMut<'_, T, usize, usize, ContiguousFwd>,
	par: Par<'_>,
) {
	let m = perm.len();
	let blocks = m.div_ceil(PANEL);
	let units = m.saturating_mul(m).saturating_mul(factors.ncols() / 2);
	let width = ParThreshold::Derived
		.region_width::<T>(&par, units, rate::bunch_kaufman(), 1)
		.min(blocks);
	// SAFETY: the compact panel is fully initialized, and even/odd views each
	// select half its columns without extending the underlying allocation.
	let (x, y) = unsafe {
		let stride = factors.col_stride();
		(
			MatRef::from_raw_parts(
				factors.as_ptr(),
				m,
				factors.ncols() / 2,
				1,
				2 * stride,
			),
			MatRef::from_raw_parts(
				factors.as_ptr().offset(stride),
				m,
				factors.ncols() / 2,
				1,
				2 * stride,
			),
		)
	};
	with_width(par, width, |par| {
		let ptr = Ptr(a.as_ptr_mut());
		let stride = a.col_stride();
		let tile_ptr = Ptr(tiles.as_ptr_mut());
		let tile_stride = tiles.col_stride();
		let run = &|rank: usize, parts: usize| {
			let ptr = ptr;
			let tile_ptr = tile_ptr;
			// Reverse the decreasing column-area weights for increasing row
			// work.
			let first = blocks
				- lower_area_split(blocks - 1, blocks, rank + 1, parts, 1);
			let last =
				blocks - lower_area_split(blocks - 1, blocks, rank, parts, 1);
			for block in first..last {
				let i0 = block * PANEL;
				let rows = PANEL.min(m - i0);
				for j0 in (0..=i0).step_by(PANEL) {
					let cols = PANEL.min(m - j0);
					// SAFETY: a fixed logical row block owns its private
					// scratch tile and every unordered pair (i,j), i>j, it
					// scatters. A bijective perm preserves disjoint pair+mirror
					// ownership. Never lend a whole work view that overlaps
					// another mirror.
					unsafe {
						let mut tile = MatMut::from_raw_parts_mut(
							tile_ptr
								.0
								.offset((block * PANEL) as isize * tile_stride),
							rows,
							cols,
							1,
							tile_stride,
						);
						for j in 0..cols {
							for i in 0..rows {
								let (r, c) = (perm[i0 + i], perm[j0 + j]);
								tile[(i, j)] = if i0 + i <= j0 + j {
									zero()
								} else if INITIAL {
									if r > c {
										input[(r, c)].copy()
									} else {
										-&input[(c, r)]
									}
								} else {
									(&*ptr.0.offset(c as isize * stride).add(r))
										.copy()
								};
							}
						}
						linalg::matmul::matmul(
							tile.rb_mut(),
							Accum::Add,
							x.subrows(i0, rows),
							y.subrows(j0, cols).transpose(),
							one(),
							Par::Seq,
						);
						linalg::matmul::matmul(
							tile.rb_mut(),
							Accum::Add,
							y.subrows(i0, rows),
							x.subrows(j0, cols).transpose(),
							-one::<T>(),
							Par::Seq,
						);
						for j in 0..cols {
							for i in 0..rows {
								let (r, c) = (perm[i0 + i], perm[j0 + j]);
								if i0 + i > j0 + j {
									*ptr.0.offset(c as isize * stride).add(r) =
										tile[(i, j)].copy();
									*ptr.0.offset(r as isize * stride).add(c) =
										-&tile[(i, j)];
								} else if i0 + i == j0 + j {
									*ptr.0.offset(c as isize * stride).add(r) =
										zero();
								}
							}
						}
					}
				}
			}
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

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use faer_traits::{cx128, fx128};

	fn check_updates<T: ComplexField>(
		entry: impl Fn(f64, f64) -> T,
		tolerance: f64,
	) {
		for n in [6, 34, 36, 64, 66, 258] {
			let input = Mat::from_fn(n, n, |i, j| {
				if i > j {
					entry(
						(i * 13 + j * 7 + 1) as f64 / 1024.0,
						(i * 3 + j * 11 + 2) as f64 / 2048.0,
					)
				} else {
					entry(f64::NAN, f64::NAN)
				}
			});
			let perm: Vec<_> = (0..n - 2).map(|i| (i * 5 + 1) % n).collect();
			let m = perm.len();
			let owned: Vec<_> = (0..n).map(|i| perm.contains(&i)).collect();
			let factors = Mat::from_fn(m, PANEL, |i, j| {
				entry(
					(i * 7 + j * 11 + 1) as f64 / 1024.0,
					(i * 3 + j * 5 + 2) as f64 / 2048.0,
				)
			});
			let check = |mut par: Par<'_>| {
				let mut work = Mat::from_fn(work_rows::<T>(n), n, |i, j| {
					if i < n && owned[i] && owned[j] {
						entry(f64::NAN, f64::NAN)
					} else {
						entry((i + 1) as f64 / 64.0, (j + 1) as f64 / 128.0)
					}
				});
				let mut tiles = Mat::from_fn(
					work_rows::<T>(PANEL),
					m.div_ceil(PANEL) * PANEL,
					|_, _| entry(f64::NAN, f64::NAN),
				);
				for initial in [true, false] {
					let mut expected = work.clone();
					for j in 0..m {
						expected[(perm[j], perm[j])] = zero();
						for i in j + 1..m {
							let (r, c) = (perm[i], perm[j]);
							let mut value = if initial {
								if r > c {
									input[(r, c)].copy()
								} else {
									-&input[(c, r)]
								}
							} else {
								work[(r, c)].copy()
							};
							for k in (0..PANEL).step_by(2) {
								value = value
									+ &factors[(i, k)] * &factors[(j, k + 1)];
								value = value
									- &factors[(i, k + 1)] * &factors[(j, k)];
							}
							expected[(r, c)] = value.copy();
							expected[(c, r)] = -value;
						}
					}
					let matrix = work
						.as_mut()
						.subrows_mut(0, n)
						.try_as_col_major_mut()
						.unwrap();
					let f = factors.as_ref().try_as_col_major().unwrap();
					let t = tiles.as_mut().try_as_col_major_mut().unwrap();
					#[cfg(feature = "threadteam")]
					let trace =
						n == 258 && T::IS_NATIVE_F64 && par.degree() > 1;
					#[cfg(feature = "threadteam")]
					if trace {
						crate::utils::team::stats::enable(true);
						let _ = crate::utils::team::stats::take();
					}
					if initial {
						update_trailing::<true, T>(
							matrix,
							f,
							input.as_ref(),
							&perm,
							t,
							par.reborrow(),
						);
					} else {
						update_trailing::<false, T>(
							matrix,
							f,
							input.as_ref(),
							&perm,
							t,
							par.reborrow(),
						);
					}
					#[cfg(feature = "threadteam")]
					if trace {
						let stats = crate::utils::team::stats::take();
						crate::utils::team::stats::enable(false);
						assert!(
							stats.run.iter().skip(2).sum::<usize>() > 0,
							"trailing row blocks were not parallel: {stats:?}"
						);
					}
					for j in 0..n {
						for i in 0..work.nrows() {
							let got = &work[(i, j)];
							let want = &expected[(i, j)];
							if i < n && owned[i] && owned[j] {
								assert!(want.is_finite());
								assert!(got.is_finite());
								assert!(
									(got - want).abs()
										<= from_f64::<T::Real>(tolerance)
											* (one::<T::Real>() + want.abs()),
									"n={n}, initial={initial}, ({i},{j})"
								);
							} else {
								assert_eq!(
									got, want,
									"unowned/padding entry overwritten"
								);
							}
						}
					}
				}
			};
			check(Par::Seq);
			#[cfg(feature = "threadteam")]
			for width in [3, 8, 70] {
				crate::team::with_team(width, |par| check(par));
			}
		}
	}

	#[test]
	fn mirrored_tiled_updates_match_dense_skew_oracle() {
		check_updates(|r, _| r, 2e-12);
		check_updates(|r, _| r as f32, 2e-5);
		check_updates(c64::new, 2e-12);
		check_updates(|r, i| c32::new(r as f32, i as f32), 2e-5);
		check_updates(|r, _| from_f64::<fx128>(r), 1e-25);
		check_updates(|r, i| cx128::new(from_f64(r), from_f64(i)), 1e-25);
	}
}
