//! low level implementation of the various $LU$ decompositions
use crate::internal_prelude::*;
pub mod full_pivoting;
pub mod partial_pivoting;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod lapack;

/// The rank-truncation threshold `max(abs_tol, rel_tol * first_pivot)` shared by both $LU$s, or
/// `None` when no tolerance is set or the threshold is not finite (a non-finite threshold would
/// reject every later pivot, so it switches the tolerance off rather than truncating the rank).
pub(crate) fn rank_threshold<T: ComplexField>(
	abs_tol: Option<f64>,
	rel_tol: Option<f64>,
	first_pivot: &T::Real,
) -> Option<T::Real> {
	if abs_tol.is_none() && rel_tol.is_none() {
		return None;
	}
	let abs = from_f64::<T::Real>(abs_tol.unwrap_or(0.0));
	let rel = from_f64::<T::Real>(rel_tol.unwrap_or(0.0)).mul_real(first_pivot);
	let threshold = if rel > abs { rel } else { abs };
	if threshold.is_finite() { Some(threshold) } else { None }
}

/// Whether the rank-truncation rule rejects a pivot of modulus `pivot`: `pivot <= threshold`. A
/// NaN modulus is never rejected, so a NaN cannot end the factorization as a small pivot.
#[inline]
pub(crate) fn rank_rejects<T: ComplexField>(pivot: &T::Real, threshold: &Option<T::Real>) -> bool {
	match threshold {
		Some(threshold) => *pivot <= *threshold,
		None => false,
	}
}
