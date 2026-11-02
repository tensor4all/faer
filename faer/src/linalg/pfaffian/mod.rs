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
const PANEL: usize = 24;

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
	// The panel is held in compact order: row `i` of the working panel is the
	// physical row `perm[start + i]`, so a step only touches the rows that are
	// still live and both the pivot search and the rank-two recurrence run over
	// contiguous memory. The trailing rows of the panel (`cols..`) are already
	// the shared compact panel the update reads, so no second gather is needed.
	// SAFETY: a panel column is gathered over the live rows it is read at, and
	// dead rows are never read; padding is never read.
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
		let cols = end - start;
		let live = n - start;
		let lower_only = start == 0;
		let source = if lower_only {
			a
		} else {
			work.rb().as_dyn_stride()
		};
		for k in (start..end).step_by(2) {
			let ki = k - start;
			{
				let mut pf =
					factors.rb_mut().subrows_mut(0, live).subcols_mut(0, cols);
				let (prefix, rest) = pf.rb_mut().split_at_col_mut(ki);
				let mut col = rest.col_mut(0);
				let dst = col.rb_mut().subrows_mut(ki, live - ki);
				let dst = gather_panel_column(
					dst,
					source,
					perm[k],
					&perm[k..],
					lower_only,
				);
				if ki > 0 {
					update_column(dst, prefix.rb(), ki, ki);
				}
			}
			let pivot = {
				let pf = factors.rb().subrows(0, live).subcols(0, cols);
				k + 1 + pivot_index(pf.col(ki).subrows(ki + 1, live - ki - 1))?
			};
			if pivot != k + 1 {
				perm.swap(k + 1, pivot);
				odd_permutation = !odd_permutation;
				// Relabel the finished panel columns: compact row `i` must stay
				// the physical row `perm[start + i]`. Rows `ki + 1..` of those
				// columns are initialized, so nothing uninitialized moves.
				let mut pf = factors
					.rb_mut()
					.subrows_mut(0, live)
					.subcols_mut(0, ki + 1);
				let (mut lo, mut hi) =
					pf.rb_mut().two_rows_mut(ki + 1, pivot - start);
				for j in 0..=ki {
					core::mem::swap(&mut lo[j], &mut hi[j]);
				}
			}
			let q = perm[k + 1];
			let d = factors.rb().subrows(0, live).subcols(0, cols)
				[(ki + 1, ki)]
				.copy();
			if !d.is_finite() {
				return Err(PfaffianError::NumericalFailure);
			}
			if d == zero() {
				return Err(PfaffianError::ZeroPivot { index: k });
			}
			pivots[(k / 2, 0)] = -&d;
			{
				let mut pf =
					factors.rb_mut().subrows_mut(0, live).subcols_mut(0, cols);
				let (prefix, rest) = pf.rb_mut().split_at_col_mut(ki + 1);
				let mut col = rest.col_mut(0);
				let dst = col.rb_mut().subrows_mut(ki, live - ki);
				// `q` carries the exchange, so this column is re-read from the
				// source rather than moved.
				let dst =
					gather_panel_column(dst, source, q, &perm[k..], lower_only);
				// The pair being completed is excluded from the recurrence.
				if ki > 0 {
					update_column(dst, prefix.rb().subcols(0, ki), ki + 1, ki);
				}
			}
			// One reciprocal per step instead of one division per entry, as
			// LAPACK's unblocked `getf2` does, for the native scalars. The
			// extended-precision scalars keep the exactly rounded division their
			// accuracy contract is written against, and so does a reciprocal
			// that would itself overflow, which keeps the reachable exponent
			// range unchanged.
			let native = const {
				T::IS_NATIVE_F32
					|| T::IS_NATIVE_F64
					|| T::IS_NATIVE_C32
					|| T::IS_NATIVE_C64
			};
			let inv = crate::math_utils::recip(&d);
			let mut pf =
				factors.rb_mut().subrows_mut(0, live).subcols_mut(0, cols);
			let mut dst = pf.rb_mut().col_mut(ki);
			if native && inv.is_finite() {
				for i in ki + 2..live {
					dst[i] = &dst[i] * &inv;
				}
			} else {
				for i in ki + 2..live {
					dst[i] = &dst[i] / &d;
				}
			}
		}
		if end < n {
			let remaining = &perm[end..];
			// Rows `cols..live` of the compact panel are the trailing rows in
			// pivot order: exactly the shared panel the tiled update reads.
			let compact = factors
				.rb()
				.subrows(0, live)
				.subcols(0, cols)
				.subrows(cols, live - cols);
			if lower_only {
				update_trailing::<true, T>(
					work.rb_mut(),
					compact,
					a,
					remaining,
					tiles.rb_mut(),
					par.reborrow(),
				);
			} else {
				update_trailing::<false, T>(
					work.rb_mut(),
					compact,
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

/// The contiguous tile range `[first, last)` of the lower triangle, in row-major
/// tile order, that rank `rank` of `parts` owns.
///
/// Tiles are equal cost, so an equal number of them per rank is the balanced
/// split. The row-block split this replaces could not go below one row block per
/// rank, and one row block of a shallow trailing block is a large share of the
/// triangle, so the widest rank could not be helped.
fn tile_range(tiles: usize, rank: usize, parts: usize) -> (usize, usize) {
	(tiles * rank / parts, tiles * (rank + 1) / parts)
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
) -> Result<usize, PfaffianError> {
	let mut at = 0;
	for i in 0..col.nrows() {
		if !col[i].is_finite() {
			return Err(PfaffianError::NumericalFailure);
		}
		if col[i].absmax() > col[at].absmax() {
			at = i;
		}
	}
	Ok(at)
}

/// Fills one pending panel column with the current source entries of the live
/// rows, in compact order, and returns it for the rank-two recurrence.
fn gather_panel_column<'a, T: ComplexField>(
	mut dst: ColMut<'a, T, usize, ContiguousFwd>,
	a: MatRef<'_, T>,
	column: usize,
	live: &[usize],
	lower_only: bool,
) -> ColMut<'a, T, usize, ContiguousFwd> {
	for i in 0..dst.nrows() {
		let row = live[i];
		dst[i] = if lower_only {
			if row > column {
				a[(row, column)].copy()
			} else if row < column {
				-&a[(column, row)]
			} else {
				zero()
			}
		} else if row == column {
			zero()
		} else {
			a[(row, column)].copy()
		};
	}
	dst
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
	// Every trailing tile is one job, so the split follows the tile count
	// rather than the row-block count: at small `m` one row block still holds
	// most of the triangle, and splitting only rows leaves the widest rank with
	// a quarter of the work while the others idle.
	let tile_count = blocks.saturating_mul(blocks + 1) / 2;
	let units = m.saturating_mul(m).saturating_mul(factors.ncols() / 2);
	// The tile scratch holds one slot per `PANEL`-wide strip of the original
	// `n`, and that is what caps the split; the row-block count is much smaller
	// than the tile count once the trailing block gets shallow, and using it as
	// the cap left most of the team idle in the last panels.
	let width = ParThreshold::Derived
		.region_width::<T>(&par, units, rate::bunch_kaufman(), 1)
		.min(tile_count)
		.min(tiles.ncols() / PANEL);
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
			// The lower triangle in row-major tile order: rank `r` owns the
			// contiguous tile range `[tiles r / parts, tiles (r+1) / parts)`.
			// Row block `i` holds tiles `i (i+1) / 2 .. (i+1) (i+2) / 2`.
			let (first, last) = tile_range(tile_count, rank, parts);
			let mut row_at = 0usize;
			for block in 0..blocks {
				let row_first = row_at;
				row_at += block + 1;
				let lo = first.max(row_first);
				let hi = last.min(row_at);
				if lo >= hi {
					continue;
				}
				let i0 = block * PANEL;
				let rows = PANEL.min(m - i0);
				for jb in lo - row_first..hi - row_first {
					let j0 = jb * PANEL;
					let cols = PANEL.min(m - j0);
					// SAFETY: each rank owns a private scratch slot, and the
					// tiles it owns are disjoint from every other rank's, so
					// every unordered pair (i,j), i>j, and its mirror have one
					// writer. A bijective perm preserves that ownership. Never
					// lend a whole work view that overlaps another mirror.
					unsafe {
						let mut tile = MatMut::from_raw_parts_mut(
							tile_ptr
								.0
								.offset((rank * PANEL) as isize * tile_stride),
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

	/// `slots` is how many tile scratch slots the caller offers, which is what
	/// caps the trailing update's split; `usize::MAX` offers the whole triangle.
	fn check_updates<T: ComplexField>(
		entry: impl Fn(f64, f64) -> T,
		tolerance: f64,
		slots: usize,
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
				// Every slot the region could possibly ask for is present and holds
				// a finite canary, but only `offered` of them are handed to the
				// update: a rank that escapes the clamp overwrites a canary instead
				// of unrelated memory, and the assertion below catches it.
				let total = m.div_ceil(PANEL).max(1);
				let offered = slots.min(total).max(1);
				let canary = entry(7.0, -3.0);
				// Enough canary slots for the widest split the region could pick on
				// this shape, so a missing clamp fails the assertion below instead of
				// writing outside the allocation.
				let tiles_total = total * (total + 1) / 2;
				let spare = offered.max(tiles_total.min(par.degree()).max(1));
				let mut tiles = Mat::from_fn(
					work_rows::<T>(PANEL),
					spare * PANEL,
					|_, _| canary.copy(),
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
					let t = tiles
						.as_mut()
						.submatrix_mut(
							0,
							0,
							work_rows::<T>(PANEL),
							offered * PANEL,
						)
						.try_as_col_major_mut()
						.unwrap();
					// Only the whole-triangle case has enough slots for every rank.
					#[cfg(feature = "threadteam")]
					let trace = n == 258
						&& T::IS_NATIVE_F64
						&& par.degree() > 1
						&& slots == usize::MAX;
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
					for j in offered * PANEL..spare * PANEL {
						for i in 0..work_rows::<T>(PANEL) {
							assert_eq!(
								tiles[(i, j)],
								canary,
								"rank escaped its tile scratch slot \
								 (offered {offered}, n {n}, initial {initial})"
							);
						}
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
	fn trailing_tiles_have_one_owner_each_and_a_balanced_split() {
		for tiles in [1usize, 3, 6, 10, 15, 21, 28, 36, 45, 55] {
			for parts in 1..=8 {
				let ranges: Vec<_> =
					(0..parts).map(|r| tile_range(tiles, r, parts)).collect();
				assert_eq!(ranges[0].0, 0, "tiles={tiles} parts={parts}");
				assert_eq!(ranges[parts - 1].1, tiles);
				assert!(
					ranges.windows(2).all(|w| w[0].1 == w[1].0),
					"tiles={tiles} parts={parts}: {ranges:?}"
				);
				let counts: Vec<usize> =
					ranges.iter().map(|&(a, b)| b - a).collect();
				assert!(
					counts.iter().max().unwrap() - counts.iter().min().unwrap()
						<= 1,
					"tiles={tiles} parts={parts}: {counts:?}"
				);
				if parts <= tiles {
					assert!(counts.iter().all(|&count| count >= 1));
				}
			}
		}
	}

	/// Fewer tile scratch slots than the team has ranks: the split must stay
	/// inside the slots it was given. The offered slots hold the real work, the
	/// slot after them is a finite canary that a rank escaping the clamp would
	/// overwrite, and the numbers must still be right at that narrower width.
	#[test]
	fn trailing_update_keeps_ranks_inside_the_offered_slots() {
		for offered in [1usize, 2, 3] {
			check_updates(|r, _| r, 2e-12, offered);
			check_updates(c64::new, 2e-12, offered);
		}
	}

	/// A permutation that parks the partner of the first block at the far corner
	/// makes every pivot of the first panel land outside that panel: column 0 has
	/// a single nonzero entry, at row `n - 1`, so the first exchange brings in the
	/// last physical column and each later pair jumps the same way. The matrix is
	/// a permuted direct sum of `2x2` blocks with exact power-of-two entries, so
	/// the Pfaffian is `sign(P) * (-1)^pairs * prod(t)` in exact arithmetic.
	#[test]
	fn outside_panel_exchange_returns_the_exact_pfaffian() {
		let n = 2 * PANEL + 2;
		assert!(
			n - 1 >= PANEL,
			"the forced exchange must leave the first panel"
		);
		let pairs = n / 2;
		let t: Vec<f64> =
			(0..pairs).map(|k| 2.0f64.powi(-(k as i32))).collect();
		// position 0 and the far corner take the first block; the rest are spread
		// by the smallest stride coprime with the number of remaining labels.
		let rest = n - 2;
		let stride = (3..).find(|s| gcd(*s, rest) == 1).unwrap();
		let mut order = vec![0usize; n];
		order[n - 1] = 1;
		for (step, label) in (2..n).enumerate() {
			order[1 + (step * stride) % rest] = label;
		}
		let mut seen = order.clone();
		seen.sort_unstable();
		assert_eq!(seen, (0..n).collect::<Vec<_>>(), "not a permutation");
		let mut inversions = 0usize;
		for i in 0..n {
			for j in i + 1..n {
				if order[i] > order[j] {
					inversions += 1;
				}
			}
		}
		let sign = if inversions % 2 == 0 { 1.0 } else { -1.0 };
		// A block `[[0, -t], [t, 0]]` has Pfaffian `-t`, this module's convention
		// (see its 2x2 doctest), so a direct sum of `pairs` blocks is
		// `(-1)^pairs * prod(t)`.
		let expected =
			sign * (-1.0f64).powi(pairs as i32) * t.iter().product::<f64>();
		let block = |r: usize, c: usize| {
			let (p, q) = (order[r], order[c]);
			if p / 2 == q / 2 && p != q {
				let value = t[p / 2];
				if p > q { value } else { -value }
			} else {
				0.0
			}
		};
		// column 0 is the forced far-corner pivot and nothing else
		assert!(block(n - 1, 0) != 0.0);
		assert!((1..n - 1).all(|r| block(r, 0) == 0.0));
		let a = Mat::<f64>::from_fn(n, n, |i, j| block(i, j));
		let mut memory = dyn_stack::MemBuffer::new(pfaffian_scratch::<f64>(n));
		let mut check = |mut par: Par<'_>| {
			let value = pfaffian(
				a.as_ref(),
				par.reborrow(),
				dyn_stack::MemStack::new(&mut memory),
			)
			.unwrap();
			assert!(
				(value - expected).abs() <= 1e-14 * expected.abs(),
				"{value} != {expected} at n={n}"
			);
		};
		check(Par::Seq);
		#[cfg(feature = "threadteam")]
		for width in [2, 3, 8] {
			crate::team::with_team(width, |par| check(par));
		}
	}

	fn gcd(a: usize, b: usize) -> usize {
		if b == 0 { a } else { gcd(b, a % b) }
	}

	#[test]
	fn mirrored_tiled_updates_match_dense_skew_oracle() {
		check_updates(|r, _| r, 2e-12, usize::MAX);
		check_updates(|r, _| r as f32, 2e-5, usize::MAX);
		check_updates(c64::new, 2e-12, usize::MAX);
		check_updates(|r, i| c32::new(r as f32, i as f32), 2e-5, usize::MAX);
		check_updates(|r, _| from_f64::<fx128>(r), 1e-25, usize::MAX);
		check_updates(
			|r, i| cx128::new(from_f64(r), from_f64(i)),
			1e-25,
			usize::MAX,
		);
	}
}
