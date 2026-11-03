//! the unitary factor $Q$ of a packed $LQ$ factorization, and its orthonormal
//! complement (the null space of the factored matrix).
//!
//! $Q^\top$ is the $QR$ basis of the transposed view, so both are the $QR$
//! start-block routine applied to transposed views: a block of rows of $Q$ is the
//! matching block of columns of that basis.
use crate::assert;
use crate::internal_prelude::*;
/// the workspace required by [`unitary`] for an `out` of shape
/// `out_nrows x ncols`.
pub fn unitary_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	out_nrows: usize,
	block_size: usize,
	par: ParShape,
) -> StackReq {
	let _ = nrows;
	let _ = par;
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(ncols, block_size, out_nrows)
}
/// builds the unitary factor $Q$ of the packed factorization in `packed`.
///
/// `out` is $\min(m, n) \times n$ for the thin $Q$ or $n \times n$ for the full
/// $Q$. With `positive`, row $j$ of $Q$ is scaled by $d_j / |d_j|$, where $d_j$
/// is the diagonal of the triangular factor, matching the
/// [`copy_l`](crate::linalg::lq::no_pivoting::factor::copy_l) phase so that the
/// product $LQ$ is unchanged and the diagonal of $L$ is real and nonnegative.
#[track_caller]
pub fn unitary<T: ComplexField>(
	out: MatMut<'_, T>,
	packed: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	positive: bool,
	par: Par,
	stack: &mut MemStack,
) {
	let (m, n) = packed.shape();
	let k = Ord::min(m, n);
	let q = out.nrows();
	let mut out = out;
	assert!(all(out.ncols() == n, q == k || q == n));
	linalg::qr::no_pivoting::unitary::apply_to_start_block(
		out.rb_mut().transpose_mut(),
		0,
		q,
		packed.transpose(),
		Q_coeff,
		par,
		stack,
	);
	if positive {
		for j in 0..k {
			let phase = linalg::positive_phase(packed[(j, j)].clone());
			let row = out.rb_mut().row_mut(j);
			z!(row).for_each(|unzip!(x)| *x *= &phase);
		}
	}
}
/// the workspace required by [`null`].
pub fn null_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	block_size: usize,
	par: ParShape,
) -> StackReq {
	let k = Ord::min(nrows, ncols);
	let _ = par;
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(ncols, block_size, ncols - k)
}
/// builds an orthonormal basis of the null space of the factored matrix: the rows
/// $\min(m, n) + 1, \dots, n$ of the full $Q$, so that `out` is
/// $(n - \min(m, n)) \times n$ with orthonormal rows and $A \, out^H = 0$.
#[track_caller]
pub fn null<T: ComplexField>(
	out: MatMut<'_, T>,
	packed: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let (m, n) = packed.shape();
	let k = Ord::min(m, n);
	assert!(all(out.nrows() == n - k, out.ncols() == n));
	linalg::qr::no_pivoting::unitary::apply_to_start_block(
		out.transpose_mut(),
		k,
		n - k,
		packed.transpose(),
		Q_coeff,
		par,
		stack,
	);
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::linalg::lq::no_pivoting::factor;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, c64};
	use dyn_stack::MemBuffer;
	/// factors `a` and returns the packed form, its coefficients and the reported
	/// rank.
	fn packed<T: ComplexField>(
		a: MatRef<'_, T>,
		block_size: usize,
	) -> (Mat<T>, Mat<T>, usize) {
		let (m, n) = a.shape();
		let k = Ord::min(m, n);
		let mut packed = a.cloned();
		let mut coeff = Mat::full(block_size, k, infinity());
		let rank = factor::lq_in_place(
			packed.as_mut(),
			coeff.as_mut(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(
				factor::lq_in_place_scratch::<T>(
					m,
					n,
					block_size,
					ParShape::Seq,
					default(),
				),
			)),
			default(),
		)
		.rank;
		(packed, coeff, rank)
	}
	fn check(a: MatRef<'_, c64>, block_size: usize) {
		let (m, n) = a.shape();
		let k = Ord::min(m, n);
		let (packed, coeff, rank) = packed(a, block_size);
		let approx_eq = CwiseMat(ApproxEq::eps() * (Ord::max(m, n) as f64));
		for positive in [false, true] {
			let mut l = Mat::<c64>::zeros(m, k);
			factor::copy_l(l.as_mut(), packed.as_ref(), positive);
			if positive {
				// the phase is exact; the residue is the rounding of $d / |d|$.
				for j in 0..k {
					let d = l[(j, j)];
					assert!(
						d.re >= 0.0 && d.im.abs() <= 1e-14 * d.abs().max(1.0)
					);
				}
			}
			let mut q = Mat::<c64>::zeros(k, n);
			unitary(
				q.as_mut(),
				packed.as_ref(),
				coeff.as_ref(),
				positive,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(unitary_scratch::<c64>(
					m,
					n,
					k,
					block_size,
					ParShape::Seq,
				))),
			);
			assert!((&l * &q) ~ a);
			assert!((&q * q.adjoint()) ~ Mat::<c64>::identity(k, k));
			let mut q_full = Mat::<c64>::zeros(n, n);
			unitary(
				q_full.as_mut(),
				packed.as_ref(),
				coeff.as_ref(),
				positive,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(unitary_scratch::<c64>(
					m,
					n,
					n,
					block_size,
					ParShape::Seq,
				))),
			);
			assert!(q_full.get(..k, ..) ~ q.as_ref());
			assert!((&q_full * q_full.adjoint()) ~ Mat::<c64>::identity(n, n));
			let mut complement = Mat::<c64>::zeros(n - k, n);
			null(
				complement.as_mut(),
				packed.as_ref(),
				coeff.as_ref(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(null_scratch::<c64>(
					m,
					n,
					block_size,
					ParShape::Seq,
				))),
			);
			assert!(q_full.get(k.., ..) ~ complement.as_ref());
			assert!((&complement * complement.adjoint()) ~ Mat::<c64>::identity(n - k, n - k));
			// $A N^H = L (Q N^H) = 0$ for every rank, because the null block is
			// orthogonal to the rows of the thin $Q$.
			assert!((&a.cloned() * complement.adjoint()) ~ Mat::<c64>::zeros(m, n - k));
		}
		assert!(rank <= k);
	}
	#[test]
	fn test_unitary_and_null() {
		let rng = &mut StdRng::seed_from_u64(1);
		for (m, n) in [(7, 4), (4, 7), (5, 5), (0, 3), (3, 0), (0, 0), (64, 96)]
		{
			let a = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			for block_size in [1, Ord::max(Ord::min(Ord::min(m, n), 8), 1)] {
				check(a.as_ref(), block_size);
			}
		}
	}
	#[test]
	fn test_rank_deficient() {
		// the trailing rows of $A$ are exactly zero, so the trailing diagonal
		// entries of $L$ are exactly zero and the rank is known.
		let rng = &mut StdRng::seed_from_u64(2);
		for (m, n, r) in [(6, 4, 2), (4, 6, 3), (5, 5, 1)] {
			let mut a = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			for i in r..m {
				for j in 0..n {
					a[(i, j)] = c64::new(0.0, 0.0);
				}
			}
			let (_, _, rank) = packed(a.as_ref(), Ord::min(m, n));
			assert!(rank == r);
			check(a.as_ref(), Ord::max(Ord::min(Ord::min(m, n), 4), 1));
		}
	}
}
