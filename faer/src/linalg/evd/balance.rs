use super::*;
use crate::assert;
use crate::perm::{swap_cols_idx, swap_rows_idx};

/// Balancing failed because the input contains a nonfinite entry.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::Par;
/// use faer::linalg::evd::*;
/// let mut a = faer::mat![[f64::NAN]];
/// let mut mem = MemBuffer::new(balance_scratch());
/// assert_eq!(
/// 	balance_in_place(
/// 		a.as_mut(),
/// 		&mut [0],
/// 		&mut [1.0],
/// 		Par::Seq,
/// 		MemStack::new(&mut mem)
/// 	),
/// 	Err(BalanceError::NonFiniteInput)
/// );
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BalanceError {
	/// NaN or infinity in the input; the matrix is unchanged.
	NonFiniteInput,
}

/// Which parts of the balancing to run, like the `job` argument of LAPACK's `xGEBAL`.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::Par;
/// use faer::linalg::evd::*;
/// let a = faer::mat![[1.0f64, 1024.0], [1.0 / 1024.0, 2.0]];
/// // scaling alone leaves the permutation at the identity
/// let (mut perm, mut scale) = ([0; 2], [0.0; 2]);
/// let mut mem = MemBuffer::new(balance_scratch());
/// balance_in_place_with(
/// 	a.cloned().as_mut(),
/// 	&mut perm,
/// 	&mut scale,
/// 	BalanceJob::Scale,
/// 	Par::Seq,
/// 	MemStack::new(&mut mem),
/// )
/// .unwrap();
/// assert_eq!(perm, [0, 1]);
/// assert_ne!(scale, [1.0, 1.0]);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BalanceJob {
	/// neither permute nor scale (`xGEBAL` job `N`)
	None,
	/// permute only (`xGEBAL` job `P`)
	Permute,
	/// scale only (`xGEBAL` job `S`)
	Scale,
	/// permute and scale (`xGEBAL` job `B`), what [`balance_in_place`] does
	Both,
}

/// Workspace for [`balance_in_place`]; all metadata is supplied by the caller.
///
/// # Examples
/// ```
/// assert_eq!(faer::linalg::evd::balance_scratch().size_bytes(), 0);
/// ```
pub fn balance_scratch() -> StackReq {
	StackReq::EMPTY
}

/// Permutes and balances a square matrix in place, like `xGEBAL(job=B)`.
/// Returns the zero-based, half-open active range `ilo..ihi`. Outside this
/// range the permutation isolates eigenvalues in triangular blocks.
///
/// On return `b[i,j] = a[perm[i],perm[j]] * scale[j] / scale[i]`:
/// `perm[i]` is the original index now at row/column `i`, and `scale` holds
/// positive powers of two (one outside the active range). Scaling is accepted
/// only if it reduces the sum of the active row/column 2-norms (including the
/// diagonal) by at least 5%, subject to safe-range guards. This is the LAPACK
/// 3.5+ criterion, rather than the older 1-norm rule.
///
/// Use [`balance_eigenvectors_in_place`] for right eigenvectors. For a matrix
/// function such as `exp`, undo by `f(a)[perm[i],perm[j]] =
/// scale[i] * f(b)[i,j] / scale[j]`. Powers of two introduce no rounding when
/// the scaled values stay normal; underflow can prevent bit-for-bit recovery.
///
/// # Panics
/// Panics unless `a` is square and both metadata slices have length `n`.
///
/// # Errors
/// Returns [`BalanceError::NonFiniteInput`] before modifying any output if an
/// entry is NaN or infinite.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::Par;
/// use faer::linalg::evd::*;
/// let mut a = faer::mat![[1.0f64, 1024.0], [1.0 / 1024.0, 2.0]];
/// let (mut perm, mut scale) = ([0; 2], [0.0; 2]);
/// let mut mem = MemBuffer::new(balance_scratch());
/// let active = balance_in_place(
/// 	a.as_mut(),
/// 	&mut perm,
/// 	&mut scale,
/// 	Par::Seq,
/// 	MemStack::new(&mut mem),
/// )
/// .unwrap();
/// assert_eq!(active, 0..2);
/// let row = a.row(0).norm_l2();
/// let col = a.col(0).norm_l2();
/// assert!(row <= 2.0 * col && col <= 2.0 * row);
/// assert_eq!(a[(0, 1)] * scale[0] / scale[1], 1024.0);
/// ```
pub fn balance_in_place<T: ComplexField>(
	a: MatMut<'_, T>,
	perm: &mut [usize],
	scale: &mut [T::Real],
	par: Par,
	stack: &mut MemStack,
) -> Result<core::ops::Range<usize>, BalanceError> {
	balance_in_place_with(a, perm, scale, BalanceJob::Both, par, stack)
}

/// Permutes and/or scales a square matrix in place, like `xGEBAL` with the job `job`.
/// Returns the zero-based, half-open active range `ilo..ihi`. Outside this range the
/// permutation (when `job` includes it) isolates eigenvalues in triangular blocks.
///
/// [`BalanceJob::None`] returns `0..n` and touches no output; [`BalanceJob::Scale`] scales
/// without permuting and also returns `0..n`; [`BalanceJob::Permute`] and
/// [`BalanceJob::Both`] return the isolated range.
///
/// On return `b[i,j] = a[perm[i],perm[j]] * scale[j] / scale[i]`:
/// `perm[i]` is the original index now at row/column `i`, and `scale` holds
/// positive powers of two (one outside the active range). Scaling is accepted
/// only if it reduces the sum of the active row/column 2-norms (including the
/// diagonal) by at least 5%, subject to safe-range guards. This is the LAPACK
/// 3.5+ criterion, rather than the older 1-norm rule.
///
/// Use [`balance_eigenvectors_in_place`] for right eigenvectors. For a matrix
/// function such as `exp`, undo by `f(a)[perm[i],perm[j]] =
/// scale[i] * f(b)[i,j] / scale[j]`. Powers of two introduce no rounding when
/// the scaled values stay normal; underflow can prevent bit-for-bit recovery.
///
/// # Panics
/// Panics unless `a` is square and both metadata slices have length `n`.
///
/// # Errors
/// Returns [`BalanceError::NonFiniteInput`] before modifying any output if an
/// entry is NaN or infinite, whatever `job` is.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::Par;
/// use faer::linalg::evd::*;
/// let mut a = faer::mat![[1.0f64, 1024.0], [1.0 / 1024.0, 2.0]];
/// let original = a.clone();
/// let (mut perm, mut scale) = ([0; 2], [0.0; 2]);
/// let mut mem = MemBuffer::new(balance_scratch());
/// // permute only: the entries move, the scales stay one
/// let active = balance_in_place_with(
/// 	a.as_mut(),
/// 	&mut perm,
/// 	&mut scale,
/// 	BalanceJob::Permute,
/// 	Par::Seq,
/// 	MemStack::new(&mut mem),
/// )
/// .unwrap();
/// assert_eq!(active, 0..2);
/// assert_eq!(scale, [1.0, 1.0]);
/// for i in 0..2 {
/// 	for j in 0..2 {
/// 		assert_eq!(a[(i, j)], original[(perm[i], perm[j])]);
/// 	}
/// }
/// ```
pub fn balance_in_place_with<T: ComplexField>(
	mut a: MatMut<'_, T>,
	perm: &mut [usize],
	scale: &mut [T::Real],
	job: BalanceJob,
	_par: Par,
	_stack: &mut MemStack,
) -> Result<core::ops::Range<usize>, BalanceError> {
	let n = a.nrows();
	assert!(a.ncols() == n && perm.len() == n && scale.len() == n);
	if !a.is_all_finite() {
		return Err(BalanceError::NonFiniteInput);
	}
	for i in 0..n {
		perm[i] = i;
		scale[i] = one();
	}
	let mut lo = 0;
	let mut hi = n;
	// `N` and `S` do not isolate: the active range is the whole matrix (`xGEBAL` sets
	// `ilo = 1`, `ihi = n` unless the job permutes).
	let permute = matches!(job, BalanceJob::Permute | BalanceJob::Both);
	let rescale = matches!(job, BalanceJob::Scale | BalanceJob::Both);
	if permute {
		loop {
			let row = (lo..hi)
				.rev()
				.find(|&i| (lo..hi).all(|j| i == j || a[(i, j)] == zero()));
			if let Some(i) = row {
				hi -= 1;
				swap_rows_idx(a.rb_mut(), i, hi);
				swap_cols_idx(a.rb_mut(), i, hi);
				perm.swap(i, hi);
				continue;
			}
			let col = (lo..hi)
				.find(|&j| (lo..hi).all(|i| i == j || a[(i, j)] == zero()));
			if let Some(j) = col {
				swap_rows_idx(a.rb_mut(), j, lo);
				swap_cols_idx(a.rb_mut(), j, lo);
				perm.swap(j, lo);
				lo += 1;
				continue;
			}
			break;
		}
	}
	if !rescale {
		// `P` and `N`: no scaling step
		return Ok(lo..hi);
	}
	// Own implementation of Parlett-Reinsch balancing (Numer. Math. 13,
	// 293-304, 1969), using the 2-norm convergence rule of LAPACK >= 3.5
	// xGEBAL (James-Langou-Lowery, arXiv:1401.5766). No LAPACK source is
	// copied.
	let two = from_f64::<T::Real>(2.0);
	let half = from_f64::<T::Real>(0.5);
	let factor = from_f64::<T::Real>(0.95);
	let small = min_positive::<T::Real>() / eps::<T::Real>();
	let large = small.recip();
	loop {
		let mut changed = false;
		for i in lo..hi {
			let mut c = zero::<T::Real>();
			let mut r = zero::<T::Real>();
			for j in lo..hi {
				c = c.hypot(&a[(j, i)].abs());
				r = r.hypot(&a[(i, j)].abs());
			}
			if c == zero() || r == zero() || !c.is_finite() || !r.is_finite() {
				continue;
			}
			let mut ca = zero::<T::Real>();
			let mut ra = zero::<T::Real>();
			for j in 0..n {
				ca = ca.fmax(a[(j, i)].abs());
				ra = ra.fmax(a[(i, j)].abs());
			}
			let original = c.mul_pow2(half.copy()) + r.mul_pow2(half.copy());
			let mut f = one::<T::Real>();
			while c < r.mul_pow2(half.copy())
				&& c < large.mul_pow2(half.copy())
				&& ca < large.mul_pow2(half.copy())
				&& r > &small * &two
				&& ra > &small * &two
				&& f < large.mul_pow2(half.copy())
			{
				c *= &two;
				r *= &half;
				ca *= &two;
				ra *= &half;
				f *= &two;
			}
			while r <= c.mul_pow2(half.copy())
				&& r < large.mul_pow2(half.copy())
				&& ra < large.mul_pow2(half.copy())
				&& c > &small * &two
				&& ca > &small * &two
				&& f > &small * &two
			{
				c *= &half;
				r *= &two;
				ca *= &half;
				ra *= &two;
				f *= &half;
			}
			if c.mul_pow2(half.copy()) + r.mul_pow2(half.copy())
				>= &factor * &original
			{
				continue;
			}
			if (f < one() && scale[i] < &small / &f)
				|| (f > one() && scale[i] > &large / &f)
			{
				continue;
			}
			scale[i] *= &f;
			let inv = f.recip();
			for j in 0..n {
				if j != i {
					a[(i, j)] = a[(i, j)].mul_pow2(inv.copy());
					a[(j, i)] = a[(j, i)].mul_pow2(f.copy());
				}
			}
			changed = true;
		}
		if !changed {
			break;
		}
	}
	Ok(lo..hi)
}

/// Workspace for [`balance_eigenvectors_in_place`].
///
/// # Examples
/// ```
/// let _buffer = dyn_stack::MemBuffer::new(
/// 	faer::linalg::evd::balance_eigenvectors_scratch::<f64>(3),
/// );
/// ```
pub fn balance_eigenvectors_scratch<T: ComplexField>(dim: usize) -> StackReq {
	temp_mat_scratch::<T>(dim, 1)
}

/// Back-transforms columns of right eigenvectors: `x_original = P D
/// x_balanced`. Accepts any number of columns, including the real/imaginary
/// pair encoding. The columns are not renormalized. `perm` and `scale` are the
/// metadata from [`balance_in_place`].
///
/// # Panics
/// Panics on incompatible lengths, a non-permutation, a nonpositive/nonfinite
/// scale, or insufficient scratch. Metadata must describe the balanced input.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::Par;
/// use faer::linalg::evd::*;
/// let mut x = faer::mat![[1.0f64], [2.0]];
/// let mut mem = MemBuffer::new(balance_eigenvectors_scratch::<f64>(2));
/// balance_eigenvectors_in_place(
/// 	x.as_mut(),
/// 	&[1, 0],
/// 	&[2.0, 0.5],
/// 	Par::Seq,
/// 	MemStack::new(&mut mem),
/// );
/// assert_eq!(x, faer::mat![[1.0], [2.0]]);
/// ```
pub fn balance_eigenvectors_in_place<T: ComplexField>(
	mut x: MatMut<'_, T>,
	perm: &[usize],
	scale: &[T::Real],
	_par: Par,
	stack: &mut MemStack,
) {
	let n = x.nrows();
	assert!(perm.len() == n && scale.len() == n);
	let (mut tmp, _) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
	let mut tmp = tmp.as_mat_mut().col_mut(0);
	// Use the output-sized workspace to validate the permutation once.
	for &i in perm {
		assert!(i < n && tmp[i] == zero());
		tmp[i] = one();
	}
	assert!(scale.iter().all(|s| s.is_finite() && *s > zero()));
	for j in 0..x.ncols() {
		for i in 0..n {
			tmp[perm[i]] = x[(i, j)].mul_pow2(scale[i].copy());
		}
		x.rb_mut().col_mut(j).copy_from(tmp.rb());
	}
}
