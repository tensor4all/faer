//! low level implementation of the generalized eigenvalue decomposition of a
//! square matrix pair.
//!
//! the generalized eigenvalue decomposition of a matrix pair $(A, B)$ of shape
//! $(n, n)$ is a decomposition into components $S$, $U$ such that:
//!
//! - $U$ has shape $(n, n)$ and is invertible
//! - $S$ has shape $(n, n)$ and is a diagonal matrix
//! - and finally:
//!
//! $$A = B U S U^{-1}$$
//!
//! if $A$ is self-adjoint and $B$ is positive definite, then $U$ can be made
//! $B$-orthonormal ($U^H B U = I$), and $S$ is real valued.
//!
//! the implementation can choose to compute the left and/or right generalized
//! eigenvectors separately, and the generalized eigenvalues (diagonal of $S$)
//! are represented as a ratio $\alpha / \beta$ to improve numerical stability.
//!
//! The real driver's protected small solves and propagation of their scales
//! follow the numerical contracts of Reference LAPACK v3.12.1, SRC/dlaln2.f and
//! SRC/dtgevc.f: <https://github.com/Reference-LAPACK/lapack/tree/v3.12.1/SRC>.
use crate::internal_prelude::*;
use complex::Complex;
use equator::assert;
pub use linalg::evd::ComputeEigenvectors;
mod reorder;
mod schur_eigenvalues;
pub use reorder::{
	SchurReorderError, reorder_schur_cplx, reorder_schur_real,
	reorder_schur_scratch,
};
pub use schur_eigenvalues::{SchurEigenvalueError,schur_eigenvalues_real};

/// eigendecomposition error
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GevdError {
	/// reached max iterations
	NoConvergence,
}
/// eigendecomposition error
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SelfAdjointGevdError {
	/// reached max iterations
	NoConvergence,
	/// B matrix not positive definite
	NonPositiveDefinite,
}
/// generalized schur decomposition parameters
#[derive(Clone, Copy, Debug)]
pub struct GeneralizedSchurParams {
	/// An estimate of the relative cost of flops within the near-the-diagonal
	/// shift chase compared to flops within the matmul calls of a QZ sweep.
	pub relative_cost_estimate_of_shift_chase_to_matmul: extern "C" fn(
		matrix_dimension: usize,
		active_block_dimension: usize,
	)
		-> usize,
	/// Function that returns the number of shifts to use for a given matrix
	/// size
	pub recommended_shift_count: extern "C" fn(
		matrix_dimension: usize,
		active_block_dimension: usize,
	) -> usize,
	/// Function that returns the deflation window to use for a given matrix
	/// size
	pub recommended_deflation_window: extern "C" fn(
		matrix_dimension: usize,
		active_block_dimension: usize,
	) -> usize,
	/// Threshold to switch between blocked and unblocked code
	pub blocking_threshold: usize,
	/// Threshold of percent of aggressive-early-deflation window that must
	/// converge to skip a sweep
	pub nibble_threshold: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
/// schur to eigendecomposition conversion parameters
#[derive(Clone, Copy, Debug)]
pub struct GevdFromSchurParams {
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
/// eigendecomposition tuning parameters
#[derive(Clone, Copy, Debug)]
pub struct GevdParams {
	/// hessenberg parameters
	pub hessenberg: gen_hessenberg::GeneralizedHessenbergParams,
	/// schur from hessenberg conversion parameters
	pub schur: GeneralizedSchurParams,
	/// eigendecomposition from schur conversion parameters
	pub evd_from_schur: GevdFromSchurParams,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
extern "C" fn default_relative_cost_estimate_of_shift_chase_to_matmul(
	n: usize,
	nh: usize,
) -> usize {
	_ = (n, nh);
	10
}
impl<T: ComplexField> Auto<T> for GeneralizedSchurParams {
	fn auto() -> Self {
		let schur: linalg::evd::SchurParams = auto!(T);
		Self {
			relative_cost_estimate_of_shift_chase_to_matmul:
				default_relative_cost_estimate_of_shift_chase_to_matmul,
			recommended_shift_count: schur.recommended_shift_count,
			recommended_deflation_window: schur.recommended_deflation_window,
			blocking_threshold: schur.blocking_threshold,
			nibble_threshold: schur.nibble_threshold,
			non_exhaustive: NonExhaustive(()),
		}
	}
}
impl<T: ComplexField> Auto<T> for GevdParams {
	fn auto() -> Self {
		Self {
			hessenberg: auto!(T),
			schur: auto!(T),
			evd_from_schur: auto!(T),
			non_exhaustive: NonExhaustive(()),
		}
	}
}
impl<T: ComplexField> Auto<T> for GevdFromSchurParams {
	fn auto() -> Self {
		Self {
			non_exhaustive: NonExhaustive(()),
		}
	}
}
/// hessenberg decomposition
pub mod gen_hessenberg;
/// $QZ$ decomposition for complex matrices
pub mod qz_cplx;
/// $QZ$ decomposition for real matrices
pub mod qz_real;
#[track_caller]
fn compute_gevd_generic<T: ComplexField>(
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	alpha_re: ColMut<'_, T>,
	alpha_im: ColMut<'_, T>,
	beta: ColMut<'_, T>,
	u_left: Option<MatMut<'_, T>>,
	u_right: Option<MatMut<'_, T>>,
	mut par: Par,
	stack: &mut MemStack,
	params: GevdParams,
	hessenberg_to_qz: fn(
		A: MatMut<'_, T>,
		B: MatMut<'_, T>,
		Q: Option<MatMut<'_, T>>,
		Z: Option<MatMut<'_, T>>,
		alphar: ColMut<'_, T>,
		alphai: ColMut<'_, T>,
		beta: ColMut<'_, T>,
		eigenvectors: ComputeEigenvectors,
		par: Par,
		params: GeneralizedSchurParams,
		stack: &mut MemStack,
	),
	qz_to_gevd: fn(
		A: MatRef<'_, T>,
		B: MatRef<'_, T>,
		Q: Option<MatMut<'_, T>>,
		Z: Option<MatMut<'_, T>>,
		par: Par,
		stack: &mut MemStack,
	),
) -> Result<(), GevdError> {
	let n = A.nrows();
	assert!(all(
		A.nrows() == n,
		A.ncols() == n,
		B.nrows() == n,
		B.ncols() == n,
		alpha_re.nrows() == n,
		if const { T::IS_REAL } {
			alpha_im.nrows()
		} else {
			n
		} == n,
		beta.nrows() == n,
	));
	if let Some(u_left) = u_left.rb() {
		assert!(all(u_left.nrows() == n, u_left.ncols() == n));
	}
	if let Some(u_right) = u_right.rb() {
		assert!(all(u_right.nrows() == n, u_right.ncols() == n));
	}
	if n == 0 {
		return Ok(());
	}
	#[cfg(feature = "perf-warn")]
	{
		for u in [u_left.rb(), u_right.rb()] {
			if let Some(matrix) = u.rb() {
				if matrix.row_stride().unsigned_abs() != 1
					&& crate::__perf_warn!(GEVD_WARN)
				{
					if matrix.col_stride().unsigned_abs() == 1 {
						log::warn!(
							target : "faer_perf",
							"GEVD prefers column-major eigenvector matrix. Found row-major matrix."
						);
					} else {
						log::warn!(
							target : "faer_perf",
							"GEVD prefers column-major eigenvector matrix. Found matrix with generic strides."
						);
					}
				}
			}
		}
		for M in [A.rb(), B.rb()] {
			if M.row_stride().unsigned_abs() != 1
				&& crate::__perf_warn!(GEVD_WARN)
			{
				if M.col_stride().unsigned_abs() == 1 {
					log::warn!(
						target : "faer_perf",
						"GEVD prefers column-major input matrix. Found row-major matrix."
					);
				} else {
					log::warn!(
						target : "faer_perf",
						"GEVD prefers column-major input matrix. Found matrix with generic strides."
					);
				}
			}
		}
	}
	if !(A.is_all_finite() && B.is_all_finite()) {
		{ alpha_re }.fill(nan());
		{ alpha_im }.fill(nan());
		{ beta }.fill(nan());
		for u in [u_left, u_right] {
			if let Some(mut u) = u {
				u.fill(nan());
			}
		}
		return Err(GevdError::NoConvergence);
	}
	let need_qz = u_left.is_some() || u_right.is_some();
	let mut A = A;
	let mut B = B;
	let mut u_left = u_left;
	let mut u_right = u_right;
	let mut alpha_re = alpha_re;
	let mut alpha_im = alpha_im;
	let mut beta = beta;
	for u in [u_left.rb_mut(), u_right.rb_mut()] {
		if let Some(mut u) = u {
			// the outputs may be reused buffers; initialize them as full identities,
			// which is what LAPACK's `dgges` does with `dlaset` before `dhgeqz`
			// accumulates the transformations
			u.fill(zero());
			u.diagonal_mut().column_vector_mut().fill(one());
		}
	}
	{
		let block_size =
			linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n, n);
		let (mut householder, stack) =
			unsafe { linalg::temp_mat_uninit::<T, _, _>(block_size, n, stack) };
		let mut householder = householder.as_mat_mut();
		linalg::qr::no_pivoting::factor::qr_in_place(
			B.rb_mut(),
			householder.rb_mut(),
			par.reborrow(),
			stack,
			default(),
		);
		linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
			B.rb(),
			householder.rb(),
			Conj::Yes,
			A.rb_mut(),
			par.reborrow(),
			stack,
		);
		if let Some(u_left) = u_left.rb_mut() {
			linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
				B.rb(),
				householder.rb(),
				Conj::No,
				u_left,
				par.reborrow(),
				stack,
			);
		}
		zip!(B.rb_mut())
			.for_each_triangular_lower(linalg::zip::Diag::Skip, |unzip!(x)| {
				*x = zero()
			});
	}
	gen_hessenberg::generalized_hessenberg(
		A.rb_mut(),
		B.rb_mut(),
		u_left.rb_mut(),
		u_right.rb_mut(),
		par.reborrow(),
		stack,
		params.hessenberg,
	);
	hessenberg_to_qz(
		A.rb_mut(),
		B.rb_mut(),
		u_left.rb_mut(),
		u_right.rb_mut(),
		alpha_re.rb_mut(),
		alpha_im.rb_mut(),
		beta.rb_mut(),
		if need_qz {
			ComputeEigenvectors::Yes
		} else {
			ComputeEigenvectors::No
		},
		par.reborrow(),
		params.schur,
		stack,
	);
	qz_to_gevd(
		A.rb(),
		B.rb(),
		u_left.rb_mut(),
		u_right.rb_mut(),
		par,
		stack,
	);
	Ok(())
}
fn solve_shifted_1x1<T: ComplexField>(
	smin: T::Real,
	ca: T::Real,
	A: T,
	d0: T,
	B: &mut T,
	w: T,
) {
	let ref safmin = min_positive::<T::Real>();
	let ref smlnum = safmin + safmin;
	let ref smin = smin.fmax(smlnum);
	let CR = A.mul_real(ca) - w * d0;
	let cmax = CR.abs();
	if cmax < *smin {
		// protected denominator: `d = max(|d|, smin)` instead of dividing by
		// the shift twice (LAPACK `dtgevc`: `IF( ABS( D ).LE.DMIN ) D = DMIN`)
		*B = B.mul_real(smin.recip());
		return;
	}
	*B *= CR.recip();
}
fn solve_complex_shifted_1x1<T: RealField>(
	smin: T,
	ca: T,
	A: MatRef<'_, T>,
	ref d0: T,
	mut B: MatMut<'_, T>,
	wr: T,
	wi: T,
) {
	let nw = B.ncols();
	let ref safmin = min_positive::<T>();
	let ref smlnum = safmin + safmin;
	let ref smin = smin.fmax(smlnum);
	let CR = ca * &A[(0, 0)] - wr * d0;
	let CI = -wi * d0;
	let cmax = CR.abs().fmax(CI.abs());
	if cmax < *smin {
		// protected denominator, as in `solve_shifted_1x1` (LAPACK `ztgevc`:
		// `IF( CABS1( D ).LE.DMIN ) D = CMPLX( DMIN )`); dividing by the
		// complex shift afterwards would scale it twice
		let ref smin_inv = smin.recip();
		zip!(B.rb_mut()).for_each(|unzip!(x)| *x *= smin_inv);
		return;
	}
	if nw == 1 {
		let ref C = CR.recip();
		zip!(B.rb_mut()).for_each(|unzip!(x)| *x *= C);
	} else {
		let (Br, Bi) = B.two_cols_mut(0, 1);
		let C = Complex { re: CR, im: CI }.recip();
		zip!(Br, Bi).for_each(|unzip!(re, im)| {
			(*re, *im) =
				(&*re * &C.re - &*im * &C.im, &*re * &C.im + &*im * &C.re);
		});
	}
}
// Reserve room for the two elimination/substitution steps, including complex
// products. The caller protects both pivots before using this bound.
//
// What it guarantees: with `limit = min(pivot_min, 1) / (16 * min_positive)`
// and the right hand side scaled to `|b|_max <= limit`, no intermediate of
// `solve_scaled_2x2` overflows. Complete pivoting gives `|L10| <= 1` and
// `|U01 / U00| <= 1`, so `|b1 - L10 b0| <= 2 |b|_max`, `|x1| <= 2 |b|_max /
// pivot_min` and `|x0| <= 3 |b|_max / pivot_min`, all below `3 / (16 *
// min_positive)`; the factor 16 covers this and the growth of complex
// products. `pivot_min` is capped at 1 because only a pivot below 1 can
// enlarge the solution (`dlaln2` likewise scales only when `|U11| < 1`).
//
// How it differs from `dlaln2`: Reference LAPACK bounds the eliminated
// right hand side, `BBND = max(|b0 U11 / U00|, |b1 - L10 b0|)`, against
// `BIGNUM * |U11|` with `BIGNUM = 1 / (2 * min_positive)`, after the `SMIN`
// clamp, and rescales a second time when `|x| > BIGNUM / max|C|` so that the
// caller's next update `C x` cannot overflow. Here the bound is taken on the
// unreduced `b` against the smaller pivot, which is sufficient but can scale
// where `dlaln2` would not; the second rescale is left to the caller, which
// normalizes the whole working vector after every block
// (`rescale_gevd_rhs`). A different scale only changes the working vector by
// a positive factor, which the final normalization removes; the vector is
// not bit-identical to `dtgevc`'s.
//
// Out of scope: this is an overflow guard for the 2x2 solve, nothing more.
// - Underflow: scaling down may flush small components to subnormals or zero
//   earlier than `dlaln2` would.
// - The complex divisions in `solve_scaled_2x2` (and the complex reciprocal
//   in `solve_complex_shifted_1x1`) form `|d|^2` without scaling, unlike
//   `dlaln2`'s `DLADIV`, so they lose the result for `|d|` outside about
//   `[sqrt(min_positive), sqrt(max_positive)]` (`1e-154 .. 1e154` in f64,
//   `1e-19 .. 1e19` in f32), whatever this scale is. A complex pair's
//   vectors are therefore only finite for pencils within that range: in f64,
//   `(s A, s B)` for the faer#220 pencils gives non-finite vectors for `s`
//   outside `[1e-146, 1e154]`, measured at powers of ten (faer#234). Reference LAPACK's `dggev` avoids this
//   range by scaling the pencil into `[sqrt(min_positive) / eps, eps /
//   sqrt(min_positive)]` before the QZ; `gevd_real` does not.
// - With `dtgevc`'s coefficient normalization the right hand side is about
//   `|coef| |(A, B)| xmax` and `smin >= eps |coef| |(A, B)|`, so `|b|_max /
//   pivot_min` stays below about `2 / eps`: this scale can only engage when
//   `pivot_min > 1` and `|b|_max > 1 / (16 min_positive)`, that is for
//   `|(A, B)|` near `max_positive`, beyond the range where the complex
//   divisions above already fail. No GEVD test, reference fixture or graded
//   random pencil (entries up to `1e+-300`) reaches it (faer#234).
fn shifted_2x2_scale<T: RealField>(bnorm: T, pivot_min: T) -> T {
	let limit = (min_positive::<T>() * from_f64::<T>(16.0)).recip()
		* pivot_min.fmin(one::<T>());
	if bnorm > limit { limit / bnorm } else { one() }
}

// Solve the pivoted 2x2 system with a scaled RHS. Form U01/U00 before
// multiplying by x1: U01*x1 itself can overflow even when x0 is finite.
fn solve_scaled_2x2<T: ComplexField>(
	LU: MatRef<'_, T>,
	row_perm: crate::perm::PermRef<'_, usize>,
	col_perm: crate::perm::PermRef<'_, usize>,
	mut b: ColMut<'_, T>,
) -> T::Real {
	let bnorm = b[0]
		.real()
		.abs()
		.fmax(b[0].imag().abs())
		.fmax(b[1].real().abs())
		.fmax(b[1].imag().abs());
	let scale =
		shifted_2x2_scale(bnorm, LU[(0, 0)].abs().fmin(LU[(1, 1)].abs()));
	let rows = row_perm.arrays().0;
	let cols = col_perm.arrays().0;
	let b0 = b[rows[0]].mul_real(&scale);
	let b1 = b[rows[1]].mul_real(&scale) - &LU[(1, 0)] * &b0;
	let x1 = b1 / &LU[(1, 1)];
	let x0 = b0 / &LU[(0, 0)] - (&LU[(0, 1)] / &LU[(0, 0)]) * &x1;
	b[cols[0]] = x0;
	b[cols[1]] = x1;
	scale
}

fn solve_complex_shifted_2x2<T: RealField>(
	smin: T,
	ref ca: T,
	A: MatRef<'_, T>,
	ref d0: T,
	ref d1: T,
	mut B: MatMut<'_, T>,
	ref wr: T,
	ref wi: T,
	stack: &mut MemStack,
) -> T {
	let nw = B.ncols();
	let zero = zero::<T>;
	let ref safmin = min_positive::<T>();
	let ref smlnum = safmin + safmin;
	let ref smin = smin.fmax(smlnum);
	if nw == 1 {
		let (mut C, stack) =
			unsafe { linalg::temp_mat_uninit::<T, _, _>(2, 2, stack) };
		let mut C = C.as_mat_mut();
		let mut row_perm_fwd = [0usize; 2];
		let mut row_perm_inv = [0usize; 2];
		let mut col_perm_fwd = [0usize; 2];
		let mut col_perm_inv = [0usize; 2];
		C[(0, 0)] = ca * &A[(0, 0)] - wr * d0;
		C[(1, 0)] = ca * &A[(1, 0)];
		C[(0, 1)] = ca * &A[(0, 1)];
		C[(1, 1)] = ca * &A[(1, 1)] - wr * d1;
		let cmax = C.norm_max();
		if cmax < *smin {
			// `C` is replaced by `smin * I` (LAPACK `dlaln2`), so the solve is
			// this scaling alone
			let ref smin_inv = smin.recip();
			let scale = shifted_2x2_scale(B.norm_max(), smin.copy());
			zip!(B.rb_mut()).for_each(|unzip!(x)| {
				*x = (&*x * &scale) * smin_inv;
			});
			return scale;
		}
		let (_, row_perm, col_perm) =
			linalg::lu::full_pivoting::factor::lu_in_place(
				C.rb_mut(),
				&mut row_perm_fwd,
				&mut row_perm_inv,
				&mut col_perm_fwd,
				&mut col_perm_inv,
				Par::Seq,
				stack,
				Default::default(),
			);
		// a second pivot below `smin` is replaced by `smin` (LAPACK `dlaln2`).
		// `C` is exactly singular when the block holds a copy of the
		// eigenvalue being computed, and the zero pivot would divide the right
		// hand side into `inf`/`NaN` (faer#220)
		//
		// The clamp is applied to the factorized `U11`, after `lu_in_place`
		// has formed `L10 = C10 / U00`, and `L10` is not recomputed. For a 2x2
		// this is the same perturbation as `dlaln2`'s, not an approximation of
		// it: `dlaln2` also forms `LR21 = CR21 / UR11` and `UR22 = CR22 - UR12 *
		// LR21` first and then clamps `UR22` (v3.12.1, SRC/dlaln2.f, real 2x2
		// and complex 2x2 branches). The multiplier depends only on the first
		// pivot, which complete pivoting makes the largest entry, `>= cmax >=
		// smin`, so it never needs protection; the clamp replaces the solved
		// system by `L (U + (smin - U11) e1 e1^T)`, which perturbs only the
		// pivoted (1, 1) entry of `C`, by less than `2 smin`. What can differ
		// from `dlaln2` is the first pivot itself: faer's LU may rank complex
		// entries by another magnitude than `dlaln2`'s `|re| + |im|` and break
		// ties differently. Any largest entry gives `|L10| <= 1` and the same
		// argument.
		if C[(1, 1)].abs() < *smin {
			C[(1, 1)] = smin.copy();
		}
		solve_scaled_2x2(C.rb(), row_perm, col_perm, B.rb_mut().col_mut(0))
	} else {
		let (mut C, stack) =
			unsafe { linalg::temp_mat_uninit::<Complex<T>, _, _>(2, 2, stack) };
		let mut C = C.as_mat_mut();
		let mut row_perm_fwd = [0usize; 2];
		let mut row_perm_inv = [0usize; 2];
		let mut col_perm_fwd = [0usize; 2];
		let mut col_perm_inv = [0usize; 2];
		C[(0, 0)] = Complex {
			re: ca * &A[(0, 0)] - wr * d0,
			im: -wi * d0,
		};
		C[(1, 0)] = Complex {
			re: ca * &A[(1, 0)],
			im: zero(),
		};
		C[(0, 1)] = Complex {
			re: ca * &A[(0, 1)],
			im: zero(),
		};
		C[(1, 1)] = Complex {
			re: ca * &A[(1, 1)] - wr * d1,
			im: -wi * d1,
		};
		let cmax = C.norm_max();
		if cmax < *smin {
			// `C` is replaced by `smin * I`, as in the real branch
			let ref smin_inv = smin.recip();
			let scale = shifted_2x2_scale(B.norm_max(), smin.copy());
			zip!(B.rb_mut()).for_each(|unzip!(x)| {
				*x = (&*x * &scale) * smin_inv;
			});
			return scale;
		}
		let (_, row_perm, col_perm) =
			linalg::lu::full_pivoting::factor::lu_in_place(
				C.rb_mut(),
				&mut row_perm_fwd,
				&mut row_perm_inv,
				&mut col_perm_fwd,
				&mut col_perm_inv,
				Par::Seq,
				stack,
				Default::default(),
			);
		// the second pivot is protected as in the real branch, measured by
		// `|re| + |im|` (LAPACK `dlaln2`), and for the same reason it is
		// equivalent to `dlaln2`'s clamp after elimination (see the real
		// branch above)
		if C[(1, 1)].re.abs() + C[(1, 1)].im.abs() < *smin {
			C[(1, 1)] = Complex {
				re: smin.copy(),
				im: zero(),
			};
		}
		let n = B.nrows();
		let (Br, Bi) = B.two_cols_mut(0, 1);
		let (mut B, _) =
			unsafe { linalg::temp_mat_uninit::<Complex<T>, _, _>(n, 1, stack) };
		let mut B = B.as_mat_mut().col_mut(0);
		zip!(&mut B, &Br, &Bi).for_each(|unzip!(z, re, im)| {
			*z = Complex {
				re: re.copy(),
				im: im.copy(),
			}
		});
		let scale = solve_scaled_2x2(C.rb(), row_perm, col_perm, B.rb_mut());
		zip!(Br, Bi, &B).for_each(|unzip!(re, im, z)| {
			(*re, *im) = (z.re.copy(), z.im.copy())
		});
		scale
	}
}

// Apply the solve's scale to the remaining entries, then bound the whole
// working vector before the next matrix update or the back transformation.
fn rescale_gevd_rhs<T: RealField>(
	mut rhs: MatMut<'_, T>,
	j: usize,
	na: usize,
	ref scale: T,
) -> T {
	if *scale != one() {
		zip!(rhs.rb_mut().get_mut(..j, ..)).for_each(|unzip!(x)| *x *= scale);
		zip!(rhs.rb_mut().get_mut(j + na.., ..))
			.for_each(|unzip!(x)| *x *= scale);
	}
	let limit = if rhs.ncols() == 2 {
		from_f64::<T>(0.5)
	} else {
		one()
	};
	let xmax = rhs.norm_max();
	if xmax > limit {
		let ref scale = &limit / &xmax;
		zip!(rhs).for_each(|unzip!(x)| *x *= scale);
		one()
	} else {
		xmax / limit
	}
}
fn qz_to_gevd_real<T: RealField>(
	A: MatRef<'_, T>,
	B: MatRef<'_, T>,
	Q: Option<MatMut<'_, T>>,
	Z: Option<MatMut<'_, T>>,
	mut par: Par,
	stack: &mut MemStack,
) {
	let n = A.nrows();
	if n == 0 {
		return;
	}
	let one = one::<T>;
	let zero = zero::<T>;
	let ref ulp = eps::<T>();
	let ref safmin = min_positive::<T>();
	let ref smallnum = safmin * from_f64::<T::Real>(n as f64);
	let ref small = smallnum * ulp.recip();
	let ref bignum = smallnum.recip();
	let ref big = small.recip();
	let (mut acolnorm, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
	let acolnorm = acolnorm.as_mat_mut();
	let (mut bcolnorm, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
	let bcolnorm = bcolnorm.as_mat_mut();
	let mut acolnorm = acolnorm.col_mut(0);
	let mut bcolnorm = bcolnorm.col_mut(0);
	let mut anorm = zero();
	let mut bnorm = zero();
	let mut j = 0;
	while j < n {
		let cplx = j + 1 < n && A[(j + 1, j)] != zero();
		if !cplx {
			let a = A.rb().col(j).get(..j).norm_l1();
			acolnorm[j] = a.copy();
			anorm = anorm.fmax(a + A[(j, j)].abs());
			j += 1;
		} else {
			for jc in j..j + 2 {
				let a = A.rb().col(jc).get(..j).norm_l1();
				acolnorm[jc] = a.copy();
				anorm = anorm.fmax(a + A[(j, jc)].abs() + A[(j + 1, jc)].abs());
			}
			j += 2;
		}
	}
	for j in 0..n {
		let b = B.rb().col(j).get(..j).norm_l1();
		bcolnorm[j] = b.copy();
		bnorm = bnorm.fmax(b + B[(j, j)].abs());
	}
	let ref anorm = anorm;
	let ref bnorm = bnorm;
	let ref ascale = anorm.fmax(safmin).recip();
	let ref bscale = bnorm.fmax(safmin).recip();
	if let Some(mut u) = Q {
		let mut je = 0usize;
		while je < n {
			let cplx_eigval = je + 1 < n && A[(je + 1, je)] != zero();
			let nw = if cplx_eigval { 2 } else { 1 };
			if !cplx_eigval
				&& A[(je, je)].abs().fmax(&B[(je, je)].abs()) < *safmin
			{
				u.rb_mut().col_mut(je).fill(zero());
				u[(je, je)] = one();
				je += 1;
				continue;
			}
			let mut acoef;
			let mut acoefa;
			let mut bcoefa;
			let mut bcoefr;
			let mut bcoefi;
			let mut xmax;
			let (mut rhs, stack) =
				linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
			let mut rhs = rhs.as_mat_mut();
			if !cplx_eigval {
				let ref temp = safmin.fmax(
					(A[(je, je)].abs() * ascale)
						.fmax(B[(je, je)].abs() * bscale),
				);
				let ref salfar = A[(je, je)].mul_real(temp).mul_real(ascale);
				let ref sbeta = B[(je, je)].real() * temp * bscale;
				acoef = sbeta * ascale;
				bcoefr = salfar * bscale;
				bcoefi = zero();
				let mut scale = one();
				let lsa = sbeta.abs() >= *safmin && acoef.abs() < *small;
				let lsb = salfar.abs() >= *safmin && bcoefr.abs() < *small;
				if lsa {
					scale = (small / sbeta.abs()) * anorm.fmin(big);
				}
				if lsb {
					scale =
						scale.fmax((small / salfar.abs()) * bnorm.fmin(big));
				}
				if lsa || lsb {
					scale = scale.fmin(
						one()
							/ (safmin
								* one().fmax(acoef.abs()).fmax(bcoefr.abs())),
					);
					if lsa {
						acoef = ascale * (&scale * sbeta)
					} else {
						acoef = &scale * acoef
					}
					if lsb {
						bcoefr = bscale * (&scale * salfar)
					} else {
						bcoefr = &scale * bcoefr
					}
				}
				acoefa = acoef.abs();
				bcoefa = bcoefr.abs();
				rhs[(je, 0)] = one();
				xmax = one();
			} else {
				let (ref scale, _, wr, _, wi) = qz_real::generalized_eigval_2x2(
					(
						A[(je, je)].copy(),
						A[(je, je + 1)].copy(),
						A[(je + 1, je)].copy(),
						A[(je + 1, je + 1)].copy(),
					),
					(
						B[(je, je)].copy(),
						B[(je, je + 1)].copy(),
						B[(je + 1, je)].copy(),
						B[(je + 1, je + 1)].copy(),
					),
				);
				acoef = scale.copy();
				bcoefr = wr;
				bcoefi = wi;
				acoefa = acoef.abs();
				bcoefa = bcoefr.abs() + bcoefi.abs();
				let mut scale = one();
				if &acoefa * ulp < *safmin && acoefa >= *safmin {
					scale = (safmin / ulp) / &acoefa;
				}
				if &bcoefa * ulp < *safmin && bcoefa >= *safmin {
					scale = scale.fmax((safmin / ulp) / &bcoefa);
				}
				if safmin * &acoefa > *ascale {
					scale = ascale / (safmin * &acoefa);
				}
				if safmin * &bcoefa > *bscale {
					scale = scale.fmin(bscale / (safmin * &bcoefa));
				}
				if scale != one() {
					acoef = &scale * &acoef;
					acoefa = acoef.abs();
					bcoefr = &scale * &bcoefr;
					bcoefi = &scale * &bcoefi;
					bcoefa = bcoefr.abs() + bcoefi.abs();
				}
				let ref temp = &acoef * &A[(je + 1, je)];
				let temp2r = &acoef * &A[(je, je)] - &bcoefr * &B[(je, je)];
				let temp2i = -&bcoefi * &B[(je, je)];
				if temp.abs() > temp2r.abs() + temp2i.abs() {
					rhs[(je, 0)] = one();
					rhs[(je, 1)] = zero();
					rhs[(je + 1, 0)] = -temp2r / temp;
					rhs[(je + 1, 1)] = -temp2i / temp;
				} else {
					rhs[(je + 1, 0)] = one();
					rhs[(je + 1, 1)] = zero();
					let ref temp = &acoef * &A[(je, je + 1)];
					rhs[(je, 0)] = (&bcoefr * &B[(je + 1, je + 1)]
						- &acoef * &A[(je + 1, je + 1)])
						/ temp;
					rhs[(je, 1)] = &bcoefi * &B[(je + 1, je + 1)] / temp;
				}
				xmax = (rhs[(je, 0)].abs() + rhs[(je, 1)].abs())
					.fmax(rhs[(je + 1, 0)].abs() + rhs[(je + 1, 1)].abs());
			}
			let dmin = safmin
				.fmax(ulp * &acoefa * anorm)
				.fmax(ulp * &bcoefa * bnorm);
			let mut j = je + nw;
			while j < n {
				let cplx = j + 1 < n && A[(j + 1, j)] != zero();
				let na = if cplx { 2 } else { 1 };
				// the guard below may only shrink the vector (LAPACK `dtgevc`:
				// `XSCALE = ONE / MAX( ONE, XMAX )`)
				let ref xscale = xmax.fmax(one()).recip();
				let mut temp = acolnorm[j]
					.fmax(&bcolnorm[j])
					.fmax(&acoefa * &acolnorm[j] + &bcoefa * &bcolnorm[j]);
				let b0 = B[(j, j)].copy();
				let b1;
				if cplx {
					temp = temp
						.fmax(
							&acoefa * &acolnorm[j + 1]
								+ &bcoefa * &bcolnorm[j + 1],
						)
						.fmax(&acolnorm[j + 1])
						.fmax(&bcolnorm[j + 1]);
					b1 = B[(j + 1, j + 1)].copy();
				} else {
					b1 = zero();
				}
				if temp > bignum * xscale {
					for jw in 0..nw {
						for jr in je..j {
							rhs[(jr, jw)] = xscale * &rhs[(jr, jw)];
						}
					}
				}
				let (mut sums, stack) = unsafe {
					linalg::temp_mat_uninit::<T, _, _>(na, nw, stack)
				};
				let mut sums = sums.as_mat_mut();
				let (mut sump, stack) = unsafe {
					linalg::temp_mat_uninit::<T, _, _>(na, nw, stack)
				};
				let mut sump = sump.as_mat_mut();
				for jw in 0..nw {
					for ja in 0..na {
						sums[(ja, jw)] = zero();
						sump[(ja, jw)] = zero();
						for jr in je..j {
							sums[(ja, jw)] = &sums[(ja, jw)]
								+ &A[(jr, j + ja)] * &rhs[(jr, jw)];
							sump[(ja, jw)] = &sump[(ja, jw)]
								+ &B[(jr, j + ja)] * &rhs[(jr, jw)];
						}
					}
				}
				for ja in 0..na {
					if cplx_eigval {
						rhs[(j + ja, 0)] = -&acoef * &sums[(ja, 0)]
							+ &bcoefr * &sump[(ja, 0)]
							- &bcoefi * &sump[(ja, 1)];
						rhs[(j + ja, 1)] = -&acoef * &sums[(ja, 1)]
							+ &bcoefr * &sump[(ja, 1)]
							+ &bcoefi * &sump[(ja, 0)];
					} else {
						rhs[(j + ja, 0)] =
							-&acoef * &sums[(ja, 0)] + &bcoefr * &sump[(ja, 0)];
					}
				}
				let scale = if cplx {
					solve_complex_shifted_2x2(
						dmin.copy(),
						acoef.copy(),
						A.submatrix(j, j, na, na).transpose(),
						b0,
						b1,
						rhs.rb_mut().subrows_mut(j, na),
						bcoefr.copy(),
						bcoefi.copy(),
						stack,
					)
				} else {
					solve_complex_shifted_1x1(
						dmin.copy(),
						acoef.copy(),
						A.submatrix(j, j, na, na).transpose(),
						b0,
						rhs.rb_mut().subrows_mut(j, na),
						bcoefr.copy(),
						bcoefi.copy(),
					);
					one()
				};
				// the left vector is zero above `je`, so this window (from `je`,
				// offset `j - je`) and the right vectors' form (from row 0, offset
				// `j`) give the same result; the offset only matters when the 2x2
				// solve returned `scale != 1`, which `shifted_2x2_scale` explains
				// is out of reach today.
				// `left_vectors_of_schur_pencil_*` pins this path
				xmax = rescale_gevd_rhs(
					rhs.rb_mut().get_mut(je..j + na, ..),
					j - je,
					na,
					scale,
				);
				j += na;
			}
			let (mut tmp, _) = linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
			let mut tmp = tmp.as_mat_mut();
			linalg::matmul::matmul(
				tmp.rb_mut(),
				Accum::Replace,
				u.rb().get(.., je..),
				rhs.rb().get(je.., ..),
				one(),
				par.reborrow(),
			);
			let mut u = u.rb_mut().subcols_mut(je, nw);
			u.copy_from(&tmp);
			let ref scale = u.norm_l2().recip();
			zip!(u).for_each(|unzip!(u)| {
				*u *= scale;
			});
			je += nw;
		}
	}
	if let Some(mut u) = Z {
		let mut je = n;
		while je > 0 {
			je -= 1;
			let cplx_eigval = je > 0 && A[(je, je - 1)] != zero();
			let nw = if cplx_eigval { 2 } else { 1 };
			je -= nw - 1;
			if !cplx_eigval
				&& A[(je, je)].abs().fmax(&B[(je, je)].abs()) < *safmin
			{
				u.rb_mut().col_mut(je).fill(zero());
				u[(je, je)] = one();
				continue;
			}
			let mut acoef;
			let mut acoefa;
			let mut bcoefa;
			let mut bcoefr;
			let mut bcoefi;
			let (mut rhs, stack) =
				linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
			let mut rhs = rhs.as_mat_mut();
			if !cplx_eigval {
				let ref temp = safmin.fmax(
					(A[(je, je)].abs() * ascale)
						.fmax(B[(je, je)].abs() * bscale),
				);
				let ref salfar = A[(je, je)].mul_real(temp).mul_real(ascale);
				let ref sbeta = B[(je, je)].real() * temp * bscale;
				acoef = sbeta * ascale;
				bcoefr = salfar * bscale;
				bcoefi = zero();
				let mut scale = one();
				let lsa = sbeta.abs() >= *safmin && acoef.abs() < *small;
				let lsb = salfar.abs() >= *safmin && bcoefr.abs() < *small;
				if lsa {
					scale = (small / sbeta.abs()) * anorm.fmin(big);
				}
				if lsb {
					scale =
						scale.fmax((small / salfar.abs()) * bnorm.fmin(big));
				}
				if lsa || lsb {
					scale = scale.fmin(
						one()
							/ (safmin
								* one().fmax(acoef.abs()).fmax(bcoefr.abs())),
					);
					if lsa {
						acoef = ascale * (&scale * sbeta)
					} else {
						acoef = &scale * &acoef
					}
					if lsb {
						bcoefr = bscale * (&scale * salfar)
					} else {
						bcoefr = &scale * &bcoefr
					}
				}
				acoefa = acoef.abs();
				bcoefa = bcoefr.abs();
				rhs[(je, 0)] = one();
				for jr in 0..je {
					rhs[(jr, 0)] =
						&bcoefr * &B[(jr, je)] - &acoef * &A[(jr, je)];
				}
			} else {
				let (scale, _, wr, _, wi) = qz_real::generalized_eigval_2x2(
					(
						A[(je, je)].copy(),
						A[(je, je + 1)].copy(),
						A[(je + 1, je)].copy(),
						A[(je + 1, je + 1)].copy(),
					),
					(
						B[(je, je)].copy(),
						B[(je, je + 1)].copy(),
						B[(je + 1, je)].copy(),
						B[(je + 1, je + 1)].copy(),
					),
				);
				acoef = scale;
				bcoefr = wr;
				bcoefi = wi;
				acoefa = acoef.abs();
				bcoefa = bcoefr.abs() + bcoefi.abs();
				let mut scale = one();
				if &acoefa * ulp < *safmin && acoefa >= *safmin {
					scale = (safmin / ulp) / &acoefa;
				}
				if &bcoefa * ulp < *safmin && bcoefa >= *safmin {
					scale = scale.fmax((safmin / ulp) / &bcoefa);
				}
				if safmin * &acoefa > *ascale {
					scale = ascale / (safmin * &acoefa);
				}
				if safmin * &bcoefa > *bscale {
					scale = scale.fmin(bscale / (safmin * &bcoefa));
				}
				if scale != one() {
					acoef = &scale * &acoef;
					acoefa = acoef.abs();
					bcoefr = &scale * &bcoefr;
					bcoefi = &scale * &bcoefi;
					bcoefa = bcoefr.abs() + bcoefi.abs();
				}
				let ref temp = &acoef * &A[(je + 1, je)];
				let ref temp2r = &acoef * &A[(je + 1, je + 1)]
					- &bcoefr * &B[(je + 1, je + 1)];
				let ref temp2i = -&bcoefi * &B[(je + 1, je + 1)];
				if temp.abs() > temp2r.abs() + temp2i.abs() {
					rhs[(je + 1, 0)] = one();
					rhs[(je + 1, 1)] = zero();
					rhs[(je, 0)] = -temp2r / temp;
					rhs[(je, 1)] = -temp2i / temp;
				} else {
					rhs[(je, 0)] = one();
					rhs[(je, 1)] = zero();
					let ref temp = &acoef * &A[(je, je + 1)];
					rhs[(je + 1, 0)] =
						(&bcoefr * &B[(je, je)] - &acoef * &A[(je, je)]) / temp;
					rhs[(je + 1, 1)] = &bcoefi * &B[(je, je)] / temp;
				}
				let ref creala = &acoef * &rhs[(je, 0)];
				let ref cimaga = &acoef * &rhs[(je, 1)];
				let ref crealb =
					&bcoefr * &rhs[(je, 0)] - &bcoefi * &rhs[(je, 1)];
				let ref cimagb =
					&bcoefi * &rhs[(je, 0)] + &bcoefr * &rhs[(je, 1)];
				let ref cre2a = &acoef * &rhs[(je + 1, 0)];
				let ref cim2a = &acoef * &rhs[(je + 1, 1)];
				let ref cre2b =
					&bcoefr * &rhs[(je + 1, 0)] - &bcoefi * &rhs[(je + 1, 1)];
				let ref cim2b =
					&bcoefi * &rhs[(je + 1, 0)] + &bcoefr * &rhs[(je + 1, 1)];
				for jr in 0..je {
					rhs[(jr, 0)] = -creala * &A[(jr, je)]
						+ crealb * &B[(jr, je)]
						- cre2a * &A[(jr, je + 1)]
						+ cre2b * &B[(jr, je + 1)];
					rhs[(jr, 1)] = -cimaga * &A[(jr, je)]
						+ cimagb * &B[(jr, je)]
						- cim2a * &A[(jr, je + 1)]
						+ cim2b * &B[(jr, je + 1)];
				}
			}
			let dmin = safmin
				.fmax(ulp * &acoefa * anorm)
				.fmax(ulp * &bcoefa * bnorm);
			let mut j = je;
			while j > 0 {
				j -= 1;
				let cplx = j > 0 && A[(j, j - 1)] != zero();
				let na = if cplx { 2 } else { 1 };
				j -= na - 1;
				let b0 = B[(j, j)].copy();
				let b1;
				if cplx {
					b1 = B[(j + 1, j + 1)].copy();
				} else {
					b1 = zero();
				}
				let scale = if cplx {
					solve_complex_shifted_2x2(
						dmin.copy(),
						acoef.copy(),
						A.submatrix(j, j, na, na),
						b0,
						b1,
						rhs.rb_mut().subrows_mut(j, na),
						bcoefr.copy(),
						bcoefi.copy(),
						stack,
					)
				} else {
					solve_complex_shifted_1x1(
						dmin.copy(),
						acoef.copy(),
						A.submatrix(j, j, na, na),
						b0,
						rhs.rb_mut().subrows_mut(j, na),
						bcoefr.copy(),
						bcoefi.copy(),
					);
					one()
				};
				rescale_gevd_rhs(
					rhs.rb_mut().get_mut(..je + nw, ..),
					j,
					na,
					scale,
				);
				for ja in 0..na {
					if cplx_eigval {
						let ref creala = &acoef * &rhs[(j + ja, 0)];
						let ref cimaga = &acoef * &rhs[(j + ja, 1)];
						let ref crealb = &bcoefr * &rhs[(j + ja, 0)]
							- &bcoefi * &rhs[(j + ja, 1)];
						let ref cimagb = &bcoefi * &rhs[(j + ja, 0)]
							+ &bcoefr * &rhs[(j + ja, 1)];
						for jr in 0..j {
							rhs[(jr, 0)] -= creala * &A[(jr, j + ja)]
								- crealb * &B[(jr, j + ja)];
							rhs[(jr, 1)] -= cimaga * &A[(jr, j + ja)]
								- cimagb * &B[(jr, j + ja)];
						}
					} else {
						let ref creala = &acoef * &rhs[(j + ja, 0)];
						let ref crealb = &bcoefr * &rhs[(j + ja, 0)];
						for jr in 0..j {
							rhs[(jr, 0)] -= creala * &A[(jr, j + ja)]
								- crealb * &B[(jr, j + ja)];
						}
					}
				}
			}
			let (mut tmp, _) = linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
			let mut tmp = tmp.as_mat_mut();
			linalg::matmul::matmul(
				tmp.rb_mut(),
				Accum::Replace,
				u.rb().get(.., ..je + nw),
				rhs.rb().get(..je + nw, ..),
				one(),
				par.reborrow(),
			);
			let mut u = u.rb_mut().subcols_mut(je, nw);
			u.copy_from(&tmp);
			let ref scale = u.norm_l2().recip();
			zip!(u).for_each(|unzip!(u)| {
				*u *= scale;
			});
		}
	}
}
fn qz_to_gevd_cplx<T: ComplexField>(
	A: MatRef<'_, T>,
	B: MatRef<'_, T>,
	Q: Option<MatMut<'_, T>>,
	Z: Option<MatMut<'_, T>>,
	mut par: Par,
	stack: &mut MemStack,
) {
	let n = A.nrows();
	if n == 0 {
		return;
	}
	let one = one::<T::Real>;
	let ref ulp = eps::<T::Real>();
	let ref safmin = min_positive::<T::Real>();
	let ref smallnum = safmin * from_f64::<T::Real>(n as f64);
	let ref small = smallnum * ulp.recip();
	let ref bignum = smallnum.recip();
	let ref big = small.recip();
	let (mut acolnorm, stack) =
		linalg::temp_mat_zeroed::<T::Real, _, _>(n, 1, stack);
	let acolnorm = acolnorm.as_mat_mut();
	let (mut bcolnorm, stack) =
		linalg::temp_mat_zeroed::<T::Real, _, _>(n, 1, stack);
	let bcolnorm = bcolnorm.as_mat_mut();
	let mut acolnorm = acolnorm.col_mut(0);
	let mut bcolnorm = bcolnorm.col_mut(0);
	let mut anorm = zero::<T::Real>();
	let mut bnorm = zero::<T::Real>();
	let mut j = 0;
	while j < n {
		let a = A.rb().col(j).get(..j).norm_l1();
		acolnorm[j] = a.copy();
		anorm = anorm.fmax(a + A[(j, j)].abs());
		j += 1;
	}
	for j in 0..n {
		let ref b = B.rb().col(j).get(..j).norm_l1();
		bcolnorm[j] = b.copy();
		bnorm = bnorm.fmax(b + B[(j, j)].abs());
	}
	let ref anorm = anorm;
	let ref bnorm = bnorm;
	let ref ascale = anorm.fmax(safmin).recip();
	let ref bscale = bnorm.fmax(safmin).recip();
	if let Some(mut u) = Q {
		let mut je = 0usize;
		while je < n {
			if A[(je, je)].abs().fmax(&B[(je, je)].abs()) < *safmin {
				u.rb_mut().col_mut(je).fill(zero());
				u[(je, je)] = one().to_cplx();
				je += 1;
				continue;
			}
			let mut acoef;
			let acoefa;
			let bcoefa;
			let mut bcoef;
			let mut xmax;
			let (mut rhs, stack) =
				linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
			let mut rhs = rhs.as_mat_mut().col_mut(0);
			{
				let ref temp = safmin.fmax(
					(A[(je, je)].abs() * ascale)
						.fmax(B[(je, je)].abs() * bscale),
				);
				let ref salfar = A[(je, je)].mul_real(temp).mul_real(ascale);
				let ref sbeta = B[(je, je)].real() * temp * bscale;
				acoef = sbeta * ascale;
				bcoef = salfar.mul_real(bscale);
				let mut scale = one();
				let lsa = sbeta.abs() >= *safmin && acoef.abs() < *small;
				let lsb = salfar.abs() >= *safmin && bcoef.abs() < *small;
				if lsa {
					scale = (small / sbeta.abs()) * anorm.fmin(big);
				}
				if lsb {
					scale =
						scale.fmax((small / salfar.abs()) * bnorm.fmin(big));
				}
				if lsa || lsb {
					scale = scale.fmin(
						one()
							/ (safmin
								* one().fmax(acoef.abs()).fmax(bcoef.abs())),
					);
					if lsa {
						acoef = sbeta * &scale * ascale
					} else {
						acoef = acoef * &scale
					}
					if lsb {
						bcoef = salfar.mul_real(&scale).mul_real(bscale)
					} else {
						bcoef = bcoef.mul_real(&scale)
					}
				}
				acoefa = acoef.abs();
				bcoefa = bcoef.abs();
				rhs[je] = one().to_cplx();
				xmax = one();
			}
			let dmin = safmin
				.fmax(ulp * &acoefa * anorm)
				.fmax(ulp * &bcoefa * bnorm);
			let mut j = je + 1;
			while j < n {
				let ref xscale = xmax.recip();
				let ref temp = acolnorm[j]
					.fmax(&bcolnorm[j])
					.fmax(&acoefa * &acolnorm[j] + &bcoefa * &bcolnorm[j]);
				let b0 = B[(j, j)].copy();
				if *temp > bignum * xscale {
					for jr in je..j {
						rhs[jr] = rhs[jr].mul_real(xscale);
					}
					xmax = xmax * xscale;
				}
				let mut sums = zero::<T>();
				let mut sump = zero::<T>();
				for jr in je..j {
					sums += A[(jr, j)].conj() * &rhs[jr];
					sump += B[(jr, j)].conj() * &rhs[jr];
				}
				rhs[j] = bcoef.conj() * &sump - sums.mul_real(&acoef);
				solve_shifted_1x1(
					dmin.copy(),
					acoef.copy(),
					A[(j, j)].conj(),
					b0.conj(),
					&mut rhs[j],
					bcoef.conj(),
				);
				j += 1;
			}
			let (mut tmp, _) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
			let mut tmp = tmp.as_mat_mut().col_mut(0);
			linalg::matmul::matmul(
				tmp.rb_mut(),
				Accum::Replace,
				u.rb().get(.., je..),
				rhs.rb().get(je..),
				one().to_cplx(),
				par.reborrow(),
			);
			let mut u = u.rb_mut().col_mut(je);
			u.copy_from(&tmp);
			let ref scale = u.norm_l2().recip();
			zip!(u).for_each(|unzip!(u)| {
				*u = u.mul_real(scale);
			});
			je += 1;
		}
	}
	if let Some(mut u) = Z {
		let mut je = n;
		while je > 0 {
			je -= 1;
			if A[(je, je)].abs().fmax(&B[(je, je)].abs()) < *safmin {
				u.rb_mut().col_mut(je).fill(zero());
				u[(je, je)] = one().to_cplx();
				continue;
			}
			let mut acoef;
			let acoefa;
			let bcoefa;
			let mut bcoefr;
			let (mut rhs, stack) =
				linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
			let mut rhs = rhs.as_mat_mut().col_mut(0);
			{
				let ref temp = safmin.fmax(
					(A[(je, je)].abs() * ascale)
						.fmax(B[(je, je)].abs() * bscale),
				);
				let ref salfar = A[(je, je)].mul_real(temp).mul_real(ascale);
				let ref sbeta = B[(je, je)].real() * temp * bscale;
				acoef = sbeta * ascale;
				bcoefr = salfar.mul_real(bscale);
				let mut scale = one();
				let lsa = sbeta.abs() >= *safmin && acoef.abs() < *small;
				let lsb = salfar.abs() >= *safmin && bcoefr.abs() < *small;
				if lsa {
					scale = (small / sbeta.abs()) * anorm.fmin(big);
				}
				if lsb {
					scale =
						scale.fmax((small / salfar.abs()) * bnorm.fmin(big));
				}
				if lsa || lsb {
					scale = scale.fmin(
						one()
							/ (safmin
								* one().fmax(acoef.abs()).fmax(bcoefr.abs())),
					);
					if lsa {
						acoef = sbeta * &scale * ascale
					} else {
						acoef = acoef * &scale
					}
					if lsb {
						bcoefr = salfar.mul_real(&scale).mul_real(bscale)
					} else {
						bcoefr = bcoefr.mul_real(&scale)
					}
				}
				acoefa = acoef.abs();
				bcoefa = bcoefr.abs();
				rhs[je] = one().to_cplx();
				for jr in 0..je {
					rhs[jr] =
						&bcoefr * &B[(jr, je)] - A[(jr, je)].mul_real(&acoef);
				}
			}
			let dmin = safmin
				.fmax(ulp * &acoefa * anorm)
				.fmax(ulp * &bcoefa * bnorm);
			let mut j = je;
			while j > 0 {
				j -= 1;
				let b0 = B[(j, j)].copy();
				solve_shifted_1x1(
					dmin.copy(),
					acoef.copy(),
					A[(j, j)].copy(),
					b0,
					&mut rhs[j],
					bcoefr.copy(),
				);
				let ref creala = rhs[j].mul_real(&acoef);
				let ref crealb = &bcoefr * &rhs[j];
				for jr in 0..j {
					rhs[jr] =
						&rhs[jr] - creala * &A[(jr, j)] + crealb * &B[(jr, j)];
				}
			}
			let (mut tmp, _) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
			let mut tmp = tmp.as_mat_mut().col_mut(0);
			linalg::matmul::matmul(
				tmp.rb_mut(),
				Accum::Replace,
				u.rb().get(.., ..je + 1),
				rhs.rb().get(..je + 1),
				one().to_cplx(),
				par.reborrow(),
			);
			let mut u = u.rb_mut().col_mut(je);
			u.copy_from(&tmp);
			let ref scale = u.norm_l2().recip();
			zip!(u).for_each(|unzip!(u)| {
				*u = u.mul_real(scale);
			});
		}
	}
}

fn qz_to_gevd_real_scratch<T: RealField>(n: usize, _: ParShape) -> StackReq {
	StackReq::all_of(&[
		temp_mat_scratch::<T>(n, 1),
		temp_mat_scratch::<T>(n, 1),
		temp_mat_scratch::<T>(n, 2),
		temp_mat_scratch::<T>(2, 2),
		temp_mat_scratch::<T>(2, 2),
		StackReq::all_of(&[
			temp_mat_scratch::<T>(2, 2),
			temp_mat_scratch::<T>(n, 1),
			linalg::lu::full_pivoting::factor::lu_in_place_scratch::<
				usize,
				Complex<T>,
			>(2, 2, ParShape::Seq, Default::default()),
			linalg::lu::full_pivoting::solve::solve_in_place_scratch::<
				usize,
				Complex<T>,
			>(2, 1, ParShape::Seq),
		]),
		temp_mat_scratch::<T>(n, 2),
	])
}

fn qz_to_gevd_cplx_scratch<T: ComplexField>(n: usize, _: ParShape) -> StackReq {
	StackReq::all_of(&[temp_mat_scratch::<T>(n, 1); 4])
}

/// computes the layout of the workspace required to compute a matrix pair's
/// generalized eigendecomposition
pub fn gevd_scratch<T: ComplexField>(
	dim: usize,
	left: ComputeEigenvectors,
	right: ComputeEigenvectors,
	par: ParShape,
	params: Spec<GevdParams, T>,
) -> StackReq {
	let _ = (left, right);
	let n = dim;
	let block_size =
		linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n, n);
	StackReq::any_of(&[
		linalg::temp_mat_scratch::<T>(block_size, n).and(
			linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(n, n, block_size, par, default())
				.or(linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(n, block_size, n)),
		),
		gen_hessenberg::generalized_hessenberg_scratch::<T>(n, auto!(T)),
		if const { T::IS_REAL } {
			qz_real::hessenberg_to_qz_scratch::<T::Real>(n, par, params.schur)
		} else {
			qz_cplx::hessenberg_to_qz_scratch::<T>(n, par, params.schur)
		},
		if const { T::IS_REAL } {
			qz_to_gevd_real_scratch::<T::Real>(n, par)
		} else {
			qz_to_gevd_cplx_scratch::<T>(n, par)
		},
	])
}
/// computes the real matrix pair $(A, B)$'s eigendecomposition
///
/// the eigenvalues are stored in $S$, the left eigenvectors in $U_L$, and the
/// right eigenvectors in $U_R$
///
/// the values in $A$ and $B$ after this function is called are unspecified.
#[track_caller]
pub fn gevd_real<T: RealField>(
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	S_re: DiagMut<'_, T>,
	S_im: DiagMut<'_, T>,
	beta: DiagMut<'_, T>,
	U_left: Option<MatMut<'_, T>>,
	U_right: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<GevdParams, T>,
) -> Result<(), GevdError> {
	compute_gevd_generic(
		A,
		B,
		S_re.column_vector_mut(),
		S_im.column_vector_mut(),
		beta.column_vector_mut(),
		U_left,
		U_right,
		par,
		stack,
		params.config,
		qz_real::hessenberg_to_qz,
		qz_to_gevd_real,
	)
}
/// computes the complex matrix pair $(A, B)$'s eigendecomposition
///
/// the eigenvalues are stored in $S$, the left eigenvectors in $U_L$, and the
/// right eigenvectors in $U_R$
///
/// the values in $A$ and $B$ after this function is called are unspecified.
#[track_caller]
pub fn gevd_cplx<T: ComplexField>(
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	S: DiagMut<'_, T>,
	beta: DiagMut<'_, T>,
	U_left: Option<MatMut<'_, T>>,
	U_right: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<GevdParams, T>,
) -> Result<(), GevdError> {
	compute_gevd_generic(
		A,
		B,
		S.column_vector_mut(),
		ColMut::from_slice_mut(&mut []),
		beta.column_vector_mut(),
		U_left,
		U_right,
		par,
		stack,
		params.config,
		|A: MatMut<'_, T>,
		 B: MatMut<'_, T>,
		 Q: Option<MatMut<'_, T>>,
		 Z: Option<MatMut<'_, T>>,
		 alphar: ColMut<'_, T>,
		 _: ColMut<'_, T>,
		 beta: ColMut<'_, T>,
		 eigenvectors: ComputeEigenvectors,
		 par: Par,
		 params: GeneralizedSchurParams,
		 stack: &mut MemStack| {
			qz_cplx::hessenberg_to_qz(
				A,
				B,
				Q,
				Z,
				alphar,
				beta,
				eigenvectors,
				par,
				params,
				stack,
			)
		},
		qz_to_gevd_cplx,
	)
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils;
	use dyn_stack::MemBuffer;
	use equator::assert;

	/// faer#217: a zero shift with a zero right hand side must not produce `0 *
	/// inf`. The protected branch replaces the denominator with `smin`, so the
	/// result is zero.
	#[test]
	fn shifted_1x1_zero_shift_zero_rhs_stays_finite() {
		let smin = 1e-30_f64;
		// ca*A - w*d0 == 0
		for (ca, a, w, d0) in [
			(1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64),
			(1.0, 0.0, 3.0, 0.0),
			(2.0, 1.0, 1.0, 2.0),
		] {
			let mut b = 0.0_f64;
			solve_shifted_1x1(smin, ca, a, d0, &mut b, w);
			core::assert!(b.is_finite(), "ca={ca} a={a} w={w} d0={d0}: {b}");
			core::assert_eq!(b, 0.0);
		}
		// a nonzero right hand side in the same branch stays finite as well
		let mut b = 1.0_f64;
		solve_shifted_1x1(smin, 1.0, 1.0, 1.0, &mut b, 1.0);
		core::assert!(b.is_finite(), "{b}");
		core::assert!(b > 0.0, "{b}");
		// a tiny, representable denominator: dividing by it *and* by `smin` used to
		// overflow to infinity. `1.0 + 1e-16` was meant to be that denominator, but it
		// rounds to `1.0` in f64, so it exercised the exact-zero branch instead.
		let mut b = 1.0_f64;
		solve_shifted_1x1(smin, 1.0, 1e-300, 1.0, &mut b, 0.0);
		core::assert!(b.is_finite(), "{b}");
	}

	/// The complex helper has the same protected branch, for a real right hand
	/// side and for a complex one (two columns).
	#[test]
	fn complex_shifted_1x1_zero_shift_zero_rhs_stays_finite() {
		use crate::mat::Mat;
		let smin = 1e-30_f64;
		let a = Mat::<f64>::from_fn(1, 1, |_, _| 1.0);
		// real right hand side: ca*1 - wr*1 == 0, wi == 0
		let mut b = Mat::<f64>::zeros(2, 1);
		solve_complex_shifted_1x1(
			smin,
			1.0,
			a.as_ref(),
			1.0,
			b.as_mut(),
			1.0,
			0.0,
		);
		core::assert!(b.is_all_finite(), "{b:?}");
		core::assert_eq!(b[(0, 0)], 0.0);
		// two columns: the complex case, CR == CI == 0
		let mut b = Mat::<f64>::zeros(2, 2);
		solve_complex_shifted_1x1(
			smin,
			1.0,
			a.as_ref(),
			1.0,
			b.as_mut(),
			1.0,
			0.0,
		);
		core::assert!(b.is_all_finite(), "{b:?}");
		// nonzero right hand side in the same branch
		let mut b = Mat::<f64>::from_fn(2, 2, |i, j| (i + j + 1) as f64);
		solve_complex_shifted_1x1(
			smin,
			1.0,
			a.as_ref(),
			1.0,
			b.as_mut(),
			1.0,
			0.0,
		);
		core::assert!(b.is_all_finite(), "{b:?}");
	}

	/// faer#220: an exactly singular shifted 2x2 block, with a nonzero right
	/// hand side and the norm of `C` above `smin`. The second pivot is replaced
	/// by `smin`, so the solution is finite and nonzero.
	#[test]
	fn complex_shifted_2x2_singular_block_stays_finite() {
		use crate::mat::Mat;
		let smin = 1e-30_f64;
		let mut mem = MemBuffer::new(dyn_stack::StackReq::new::<f64>(256));
		let stack = MemStack::new(&mut mem);
		// real shift: `C = A` with `A` of rank one
		let a = Mat::<f64>::from_fn(2, 2, |_, _| 1.0);
		let mut b = Mat::<f64>::from_fn(2, 1, |i, _| (i + 1) as f64);
		solve_complex_shifted_2x2(
			smin,
			1.0,
			a.as_ref(),
			1.0,
			1.0,
			b.as_mut(),
			0.0,
			0.0,
			stack,
		);
		core::assert!(b.is_all_finite(), "{b:?}");
		core::assert!(b.norm_max() > 0.0, "{b:?}");
		// complex shift: `A = [[2, 1], [-1, 2]]` shifted by its own eigenvalue
		// `2 + i`, in both orientations the callers pass
		let a =
			Mat::<f64>::from_fn(2, 2, |i, j| [[2.0, 1.0], [-1.0, 2.0]][i][j]);
		for a in [a.as_ref(), a.transpose()] {
			let mut b =
				Mat::<f64>::from_fn(2, 2, |i, j| (i + 2 * j + 1) as f64);
			solve_complex_shifted_2x2(
				smin,
				1.0,
				a,
				1.0,
				1.0,
				b.as_mut(),
				2.0,
				1.0,
				stack,
			);
			core::assert!(b.is_all_finite(), "{b:?}");
			core::assert!(b.norm_max() > 0.0, "{b:?}");
		}
	}

	/// A shifted 2x2 block whose largest entry is below `smin` is replaced by
	/// `smin * I`: the solution is the right hand side divided by `smin`, not
	/// divided again by the block.
	#[test]
	fn complex_shifted_2x2_tiny_block_is_a_scaling() {
		use crate::mat::Mat;
		let smin = 1e-20_f64;
		let mut mem = MemBuffer::new(dyn_stack::StackReq::new::<f64>(256));
		let stack = MemStack::new(&mut mem);
		let a = Mat::<f64>::from_fn(2, 2, |i, j| {
			1e-30 * [[2.0, 1.0], [-1.0, 2.0]][i][j]
		});
		for nw in [1, 2] {
			let b0 = Mat::<f64>::from_fn(2, nw, |i, j| (i + 2 * j + 1) as f64);
			let mut b = b0.clone();
			solve_complex_shifted_2x2(
				smin,
				1.0,
				a.as_ref(),
				1.0,
				1.0,
				b.as_mut(),
				0.0,
				0.0,
				stack,
			);
			for j in 0..nw {
				for i in 0..2 {
					core::assert_eq!(b[(i, j)], b0[(i, j)] * smin.recip());
				}
			}
		}
	}

	fn check_complex_shifted_2x2_large_rhs<T: RealField>() {
		let eps = eps::<T>();
		let smin = &eps * &eps;
		let big = min_positive::<T>().recip() / from_f64::<T>(16.0);
		let mut mem = MemBuffer::new(dyn_stack::StackReq::new::<T>(256));
		for tiny in [false, true] {
			let c = if tiny { &smin * &eps } else { eps.copy() };
			let a = Mat::<T>::from_fn(2, 2, |i, j| {
				&c * from_f64::<T>([[1.0, 5.0], [2.0, 3.0]][i][j])
			});
			for a in [a.as_ref(), a.transpose()] {
				for nw in [1, 2] {
					for j in [0, 1] {
						let mut b = Mat::<T>::from_fn(3, nw, |_, _| big.copy());
						let scale = solve_complex_shifted_2x2(
							smin.copy(),
							one(),
							a,
							one(),
							one(),
							b.as_mut().subrows_mut(j, 2),
							zero(),
							if nw == 2 { c.copy() } else { zero() },
							MemStack::new(&mut mem),
						);
						core::assert!(
							b.is_all_finite(),
							"tiny={tiny}, nw={nw}: {b:?}"
						);
						core::assert!(
							scale > zero() && scale < one(),
							"{scale:?}"
						);
						rescale_gevd_rhs(b.as_mut(), j, 2, scale);
						// The untouched row represents the rest of the eigenvector.
						// C*x must equal that row after both scales are propagated.
						let anchor = if j == 0 { 2 } else { 0 };
						let unit = if tiny { &smin } else { &c };
						let target = &b[(anchor, 0)] / unit;
						core::assert!(target > zero(), "{b:?}");
						for i in 0..2 {
							let mut rr = -&target;
							let mut ri = if nw == 2 {
								-&b[(anchor, 1)] / unit
							} else {
								zero()
							};
							for k in 0..2 {
								let ar = if tiny {
									if i == k { one() } else { zero() }
								} else {
									&a[(i, k)] / &c
								};
								let xr = &b[(j + k, 0)];
								let xi = if nw == 2 {
									b[(j + k, 1)].copy()
								} else {
									zero()
								};
								rr += &ar * xr;
								ri += &ar * &xi;
								if !tiny && nw == 2 && i == k {
									rr += &xi;
									ri -= xr;
								}
							}
							let residual = rr.abs().fmax(ri.abs())
								/ target.fmax(one::<T>());
							core::assert!(
								residual < &eps * from_f64::<T>(128.0),
								"residual={residual:?}"
							);
						}
					}
				}
			}
		}
	}

	#[test]
	fn complex_shifted_2x2_large_rhs_f32() {
		check_complex_shifted_2x2_large_rhs::<f32>();
	}

	#[test]
	fn complex_shifted_2x2_large_rhs_f64() {
		check_complex_shifted_2x2_large_rhs::<f64>();
	}
	#[test]
	fn test_lishen_() {
		let approx_eq = utils::approx::CwiseMat(
			utils::approx::ApproxEq::<f64>::eps() * 128.0,
		);
		let A = &[
			[
				0.0000000067116330731078674,
				-0.0000000024189842394753674,
				0.00000000032505193899740187,
				0.0,
				0.0,
				0.0,
			],
			[
				-0.0000000024189842394753674,
				0.000000011866543705551487,
				-0.00000001582083742901503,
				0.0,
				0.0,
				0.0,
			],
			[
				0.00000000032505193899740187,
				-0.00000001582083742901503,
				0.00000002523087227632826,
				0.0,
				0.0,
				0.0,
			],
			[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 1.0f64],
		];
		let B = &[
			[
				0.00016038376001067668,
				-0.00003674235041182116,
				-0.0000029397921270058582,
				-1.0,
				-0.0,
				-0.0,
			],
			[
				-0.00003674235041182116,
				0.0002038454538304322,
				-0.00021213416901900504,
				-0.0,
				-1.0,
				-0.0,
			],
			[
				-0.0000029397921270058582,
				-0.00021213416901900504,
				0.0003954706173886255,
				-0.0,
				-0.0,
				-1.0,
			],
			[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0, 0.0, 0.0f64],
		];
		let A = MatRef::from_row_major_array(A);
		let B = MatRef::from_row_major_array(B);
		let n = A.nrows();
		{
			let mut H = A.to_owned();
			let mut T = B.to_owned();
			let mut alpha_re = Diag::<f64>::zeros(n);
			let mut alpha_im = Diag::<f64>::zeros(n);
			let mut beta = Diag::<f64>::zeros(n);
			let mut UL = Mat::<f64>::identity(n, n);
			let mut UR = Mat::<f64>::identity(n, n);
			gevd_real(
				H.as_mut(),
				T.as_mut(),
				alpha_re.rb_mut(),
				alpha_im.rb_mut(),
				beta.rb_mut(),
				Some(UL.as_mut()),
				Some(UR.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(gevd_scratch::<f64>(
					n,
					ComputeEigenvectors::Yes,
					ComputeEigenvectors::Yes,
					ParShape::Seq,
					default(),
				))),
				default(),
			)
			.unwrap();
			{
				let mut i = 0;
				while i < n {
					if alpha_im[i] == 0.0 {
						let u = UR.col(i);
						let a = alpha_re[i];
						let b = beta[i];
						assert!((b / a) * & A * u ~ B * u);
						i += 1;
					} else {
						let ur = UR.col(i);
						let ui = UR.col(i + 1);
						let ar = alpha_re[i];
						let ai = alpha_im[i];
						let b = beta[i];
						assert!(b * & A * ur ~ B * (ar * ur - ai * ui));
						assert!(b * & A * ui ~ B * (ar * ui + ai * ur));
						i += 2;
					}
				}
			}
			{
				let mut i = 0;
				while i < n {
					if alpha_im[i] == 0.0 {
						let u = UL.col(i);
						let a = alpha_re[i];
						let b = beta[i];
						assert!(b / a * u.adjoint() * & A ~ u.adjoint() * B);
						i += 1;
					} else {
						let ur = UL.col(i);
						let ui = UL.col(i + 1);
						let ar = alpha_re[i];
						let ai = alpha_im[i];
						let b = beta[i];
						assert!(
							b * ur.adjoint() * & A ~ (ar * ur - ai * ui).adjoint() * B
						);
						assert!(
							b * ui.adjoint() * & A ~ (ar * ui + ai * ur).adjoint() * B
						);
						i += 2;
					}
				}
			}
		}
	}
	#[test]
	fn test_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);
		let approx_eq = utils::approx::CwiseMat(
			utils::approx::ApproxEq::<f64>::eps() * 128.0,
		);
		let n = 20;
		let A = &CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);
		let B = &CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);
		{
			let mut H = A.to_owned();
			let mut T = B.to_owned();
			let mut alpha_re = Diag::<c64>::zeros(n);
			let mut beta = Diag::<c64>::zeros(n);
			let mut UL = Mat::<c64>::identity(n, n);
			let mut UR = Mat::<c64>::identity(n, n);
			gevd_cplx(
				H.as_mut(),
				T.as_mut(),
				alpha_re.rb_mut(),
				beta.rb_mut(),
				Some(UL.as_mut()),
				Some(UR.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(gevd_scratch::<c64>(
					n,
					ComputeEigenvectors::Yes,
					ComputeEigenvectors::Yes,
					ParShape::Seq,
					default(),
				))),
				default(),
			)
			.unwrap();
			{
				let mut i = 0;
				while i < n {
					let u = UR.col(i);
					let a = alpha_re[i];
					let b = beta[i];
					assert!(Scale(b / a) * A * u ~ B * u);
					i += 1;
				}
			}
			{
				let mut i = 0;
				while i < n {
					let u = UL.col(i);
					let a = alpha_re[i];
					let b = beta[i];
					assert!(Scale(b / a) * u.adjoint() * A ~ u.adjoint() * B);
					i += 1;
				}
			}
		}
	}
	/// faer#234: the left-vector substitution of `qz_to_gevd_real`, checked
	/// directly on a pencil already in generalized Schur form (so `Q = I` and
	/// no QZ runs) against a plain complex forward substitution, rather than
	/// through a residual. The pencil has a real eigenvalue first, so the
	/// complex pairs start at `je > 0` and their substitution runs on the
	/// `rhs` window that begins at `je` (offset `j - je`; the right vectors use
	/// a window from row 0, offset `j`). Its three pairs `2 + 0.05 k +- i` are
	/// coupled, so the working vector grows between blocks.
	///
	/// The window/offset pair itself is not observable here or anywhere else:
	/// it is read only when the 2x2 solve returns `scale != 1`, which
	/// `shifted_2x2_scale` explains is out of reach today.
	/// Replacing it by the right vectors' form, or changing only the offset or
	/// only the window, leaves every GEVD test green (faer#234). This test pins
	/// the values the path produces, so a refactor that changes them fails.
	///
	/// Each column pair must equal, up to one complex factor, the vector `y`
	/// with `y^T A = lambda y^T B` for the eigenvalue with positive imaginary
	/// part (the convention `gevd_real` documents).
	fn check_left_vectors_of_schur_pencil<T: RealField + Copy + Into<f64>>(
		tol: f64,
	) {
		use crate::mat::Mat;
		use num_complex::Complex as C;
		let n = 9;
		let delta = 0.05;
		// blocks: [0], [1, 2], [3, 4], [5, 6], [7], [8]
		let starts = [0usize, 1, 3, 5, 7, 8];
		let mut a = Mat::<f64>::zeros(n, n);
		let mut b = Mat::<f64>::zeros(n, n);
		a[(0, 0)] = 0.5;
		for (k, &o) in [1usize, 3, 5].iter().enumerate() {
			let x = 2.0 + k as f64 * delta;
			a[(o, o)] = x;
			a[(o, o + 1)] = 1.0;
			a[(o + 1, o)] = -1.0;
			a[(o + 1, o + 1)] = x;
			if o + 2 < 7 {
				a[(o + 1, o + 3)] = 0.75;
			}
		}
		a[(7, 7)] = 3.0;
		a[(8, 8)] = -1.0;
		// the 2x2 blocks of B are multiples of I (diagonal, as the Schur form
		// requires, and equal so that the block's eigenvalue is `(x +- i) / b`)
		for (bi, &i) in starts.iter().enumerate() {
			let ie = starts.get(bi + 1).copied().unwrap_or(n);
			for r in i..ie {
				b[(r, r)] = 1.0 + 0.125 * bi as f64;
			}
			for j in ie..n {
				for r in i..ie {
					if a[(r, j)] == 0.0 {
						a[(r, j)] = 0.1 * ((r + 2 * j) % 5) as f64 - 0.2;
					}
					b[(r, j)] = 0.05 * ((3 * r + j) % 7) as f64 - 0.15;
				}
			}
		}
		let a_t = Mat::<T>::from_fn(n, n, |i, j| from_f64::<T>(a[(i, j)]));
		let b_t = Mat::<T>::from_fn(n, n, |i, j| from_f64::<T>(b[(i, j)]));
		let mut u = Mat::<T>::identity(n, n);
		let mut mem =
			MemBuffer::new(qz_to_gevd_real_scratch::<T>(n, ParShape::Seq));
		qz_to_gevd_real(
			a_t.as_ref(),
			b_t.as_ref(),
			Some(u.as_mut()),
			None,
			Par::Seq,
			MemStack::new(&mut mem),
		);
		let u = Mat::<f64>::from_fn(n, n, |i, j| u[(i, j)].into());
		assert!(u.is_all_finite());

		let block = |i: usize| {
			let bi = starts.iter().rposition(|&s| s <= i).unwrap();
			let ie = starts.get(bi + 1).copied().unwrap_or(n);
			(starts[bi], ie - starts[bi])
		};
		let mut je = 0;
		while je < n {
			let (_, nw) = block(je);
			// the eigenvalue with positive imaginary part, from the block
			let lambda = if nw == 1 {
				C::new(a[(je, je)] / b[(je, je)], 0.0)
			} else {
				C::new(a[(je, je)], a[(je, je + 1)].abs()) / b[(je, je)]
			};
			let m = |i: usize, j: usize| {
				C::new(a[(i, j)], 0.0) - lambda * b[(i, j)]
			};
			// y^T M = 0, y zero above the block
			let mut y = vec![C::new(0.0, 0.0); n];
			if nw == 1 {
				y[je] = C::new(1.0, 0.0);
			} else {
				// left null vector of the singular 2x2 block
				y[je] = C::new(1.0, 0.0);
				y[je + 1] = -m(je, je) / m(je + 1, je);
			}
			let mut j = je + nw;
			while j < n {
				let (_, na) = block(j);
				// y_blk^T M_blk = -sum_{i < j} y_i M_{i, blk}
				let rhs: Vec<C<f64>> = (0..na)
					.map(|c| {
						-(0..j).map(|i| y[i] * m(i, j + c)).sum::<C<f64>>()
					})
					.collect();
				if na == 1 {
					y[j] = rhs[0] / m(j, j);
				} else {
					// solve [y0 y1] [[m00 m01] [m10 m11]] = [r0 r1]
					let (m00, m01, m10, m11) =
						(m(j, j), m(j, j + 1), m(j + 1, j), m(j + 1, j + 1));
					let det = m00 * m11 - m01 * m10;
					y[j] = (rhs[0] * m11 - rhs[1] * m10) / det;
					y[j + 1] = (rhs[1] * m00 - rhs[0] * m01) / det;
				}
				j += na;
			}
			let got: Vec<C<f64>> = (0..n)
				.map(|i| {
					C::new(
						u[(i, je)],
						if nw == 2 { u[(i, je + 1)] } else { 0.0 },
					)
				})
				.collect();
			// sine of the complex angle between `got` and `y`, from the part of
			// `got` orthogonal to `y` (not `1 - cos^2`, which cancels below
			// `sqrt(eps)`)
			let ny: f64 = y.iter().map(|z| z.norm_sqr()).sum::<f64>();
			let ng: f64 = got.iter().map(|z| z.norm_sqr()).sum::<f64>();
			let c = (0..n).map(|i| y[i].conj() * got[i]).sum::<C<f64>>() / ny;
			let sin =
				((0..n).map(|i| (got[i] - y[i] * c).norm_sqr()).sum::<f64>()
					/ ng)
					.sqrt();
			assert!(sin < tol, "left vector at {je}: sin = {sin:e}");
			// zero above the block
			for (i, z) in got.iter().enumerate().take(je) {
				assert!(*z == C::new(0.0, 0.0), "left vector at {je}: row {i}");
			}
			je += nw;
		}
	}

	#[test]
	fn left_vectors_of_schur_pencil_f32() {
		check_left_vectors_of_schur_pencil::<f32>(1e-5);
	}

	#[test]
	fn left_vectors_of_schur_pencil_f64() {
		check_left_vectors_of_schur_pencil::<f64>(1e-13);
	}
}

// tests by @sjoelund
#[cfg(test)]
mod t {
	use faer::dyn_stack::{MemBuffer, MemStack};
	use faer::linalg::evd::ComputeEigenvectors as CE;
	use faer::prelude::*;
	use faer::{Par, Spec, linalg};

	fn mk(n: usize, seed: u64) -> Mat<f64> {
		let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
		let mut next = move || {
			s = s
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
		};
		Mat::from_fn(n, n, |i, j| next() + if i == j { n as f64 } else { 0.0 })
	}

	/// `gevd_real`, with the right-eigenvector flag as a parameter.
	fn gevd(a: &Mat<f64>, b: &Mat<f64>, vectors: bool) -> Vec<(f64, f64)> {
		let n = a.nrows();
		let (mut am, mut bm) = (a.clone(), b.clone());
		let mut sre = faer::diag::Diag::<f64>::zeros(n);
		let mut sim = faer::diag::Diag::<f64>::zeros(n);
		let mut bet = faer::diag::Diag::<f64>::zeros(n);
		let mut u = Mat::<f64>::zeros(n, n);
		let ce = if vectors { CE::Yes } else { CE::No };
		let mut mem = MemBuffer::new(linalg::gevd::gevd_scratch::<f64>(
			n,
			CE::No,
			ce,
			ParShape::Seq,
			Spec::default(),
		));
		linalg::gevd::gevd_real(
			am.as_mut(),
			bm.as_mut(),
			sre.as_mut(),
			sim.as_mut(),
			bet.as_mut(),
			None,
			vectors.then(|| u.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			Spec::default(),
		)
		.unwrap();
		(0..n).map(|k| (sre[k] / bet[k], sim[k] / bet[k])).collect()
	}

	/// |det(A - lambda B)|, zero exactly at an eigenvalue.
	fn residual(a: &Mat<f64>, b: &Mat<f64>, lam: (f64, f64)) -> f64 {
		let n = a.nrows();
		let mut m: Vec<(f64, f64)> = (0..n * n)
			.map(|k| {
				let (i, j) = (k % n, k / n);
				(a[(i, j)] - lam.0 * b[(i, j)], -lam.1 * b[(i, j)])
			})
			.collect();
		let mut det = (1.0f64, 0.0f64);
		for col in 0..n {
			let p = (col..n)
				.max_by(|&x, &y| {
					let f = |r: usize| m[r + col * n].0.hypot(m[r + col * n].1);
					f(x).partial_cmp(&f(y)).unwrap()
				})
				.unwrap();
			if p != col {
				for j in 0..n {
					m.swap(col + j * n, p + j * n);
				}
				det = (-det.0, -det.1);
			}
			let d = m[col + col * n];
			det = (det.0 * d.0 - det.1 * d.1, det.0 * d.1 + det.1 * d.0);
			let dn = d.0 * d.0 + d.1 * d.1;
			if dn == 0.0 {
				continue;
			}
			for r in col + 1..n {
				let x = m[r + col * n];
				let f = (
					(x.0 * d.0 + x.1 * d.1) / dn,
					(x.1 * d.0 - x.0 * d.1) / dn,
				);
				for j in col..n {
					let y = m[col + j * n];
					m[r + j * n].0 -= f.0 * y.0 - f.1 * y.1;
					m[r + j * n].1 -= f.0 * y.1 + f.1 * y.0;
				}
			}
		}
		det.0.hypot(det.1)
	}

	/// ISSUE 1 (documentation): `gevd_real` writes a complex conjugate pair's
	/// eigenvalue into the FIRST slot only. The second slot holds the 2x2
	/// block's other diagonal entry, which is not an eigenvalue -- reading it
	/// gives a value whose `det(A - lambda B)` is O(1e-2) rather than O(1e-16).
	///
	/// faer's own `Solvers` layer never reads it: `solvers.rs::real_to_cplx`
	/// builds `S[j + 1]` from `S_re[j]`, not `S_re[j + 1]`. Nothing on
	/// `gevd_real` says so, and the same applies to
	/// `qz_real::hessenberg_to_qz`.
	#[test]
	fn eigenvalues_only_gives_wrong_complex_pairs() {
		let (mut bad_novec, mut bad_vec) = (0, 0);
		for n in [3usize, 4, 5, 6, 8, 10, 12] {
			for seed in 0..40u64 {
				let a = mk(n, seed * 7919 + n as u64);
				let b = mk(n, seed * 7919 + n as u64 + 5000);
				let scale = (0..n * n)
					.map(|k| a[(k % n, k / n)].abs())
					.fold(1.0f64, f64::max);
				let tol = 1e-9 * scale.powi(n as i32);
				for (lams, vectors) in
					[(gevd(&a, &b, false), false), (gevd(&a, &b, true), true)]
				{
					for &l in &lams {
						if l.1 == 0.0 {
							continue;
						}
						if residual(&a, &b, l) > tol {
							if vectors {
								bad_vec += 1
							} else {
								bad_novec += 1
							}
						}
					}
				}
			}
		}
		assert_eq!(bad_novec, 0, "eigenvalues-only path is wrong");
		assert_eq!(bad_vec, 0, "eigenvalues-only path is wrong");
	}

	/// BUG 2: `gevd_scratch` under-allocates, so `gevd_real` panics inside
	/// `temp_mat_zeroed`. n = 2 with eigenvectors requested.
	#[test]
	fn gevd_scratch_is_too_small_for_n2() {
		let mut panics = 0;
		for seed in 0..40u64 {
			let a = mk(2, seed * 7919 + 2);
			let b = mk(2, seed * 7919 + 2 + 5000);
			if std::panic::catch_unwind(|| gevd(&a, &b, true)).is_err() {
				panics += 1;
			}
		}
		assert_eq!(panics, 0, "gevd_scratch under-allocates");
	}
	/// `gevd_real` writes into caller-provided output buffers, so a reused buffer
	/// must not leak into the eigenvectors through the entries it leaves unwritten.
	#[test]
	fn reused_eigenvector_buffers() {
		fn check<T: faer::traits::RealField>(
			a: Mat<T>,
			b: Mat<T>,
			prefills: [T; 2],
			tol: T,
		) {
			for prefill in prefills {
				for (want_left, want_right) in
					[(false, true), (true, false), (true, true)]
				{
					let mut am = a.clone();
					let mut bm = b.clone();
					let mut sre = faer::diag::Diag::<T>::zeros(2);
					let mut sim = faer::diag::Diag::<T>::zeros(2);
					let mut bet = faer::diag::Diag::<T>::zeros(2);
					let mut ul = Mat::<T>::full(2, 2, prefill.clone());
					let mut ur = Mat::<T>::full(2, 2, prefill.clone());
					let ce = |v: bool| {
						if v { CE::Yes } else { CE::No }
					};
					linalg::gevd::gevd_real(
						am.as_mut(),
						bm.as_mut(),
						sre.as_mut(),
						sim.as_mut(),
						bet.as_mut(),
						want_left.then(|| ul.as_mut()),
						want_right.then(|| ur.as_mut()),
						Par::Seq,
						MemStack::new(&mut MemBuffer::new(
							linalg::gevd::gevd_scratch::<T>(
								2,
								ce(want_left),
								ce(want_right),
								ParShape::Seq,
								Spec::default(),
							),
						)),
						Spec::default(),
					)
					.unwrap();
					// the residual alone would also accept all-zero eigenvectors, and
					// `norm_max` skips NaN entries, so require real vectors and finite sums
					let nonzero = |v: &Mat<T>| {
						for j in 0..v.ncols() {
							assert!(v.col(j).norm_l2() > tol);
						}
					};
					if want_right {
						assert!(ur.is_all_finite());
						nonzero(&ur);
						let res =
							(&a * &ur) * bet.as_ref() - &ur * sre.as_ref();
						assert!(res.is_all_finite());
						assert!(res.norm_max() < tol);
					}
					if want_left {
						assert!(ul.is_all_finite());
						nonzero(&ul);
						let res = bet.as_ref() * (ul.adjoint() * &a)
							- sre.as_ref() * (ul.adjoint() * &b);
						assert!(res.is_all_finite());
						assert!(res.norm_max() < tol);
					}
				}
			}
		}
		check(
			Mat::<f32>::from_fn(2, 2, |i, j| [[2.0, 1.0], [0.0, 3.0]][i][j]),
			Mat::<f32>::identity(2, 2),
			[0.0, 2.0],
			1e-5,
		);
		check(
			Mat::<f64>::from_fn(2, 2, |i, j| [[2.0, 1.0], [0.0, 3.0]][i][j]),
			Mat::<f64>::identity(2, 2),
			[0.0, 2.0],
			1e-12,
		);
	}
}
