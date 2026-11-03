//! the unitary factor $Q$ of a packed $QR$ factorization, and its orthonormal
//! complement (the left null space of the factored matrix).
//!
//! The output conventions (the full $Q$ being the $m \times m$ unitary, the
//! complement being its trailing columns, and the positive diagonal gauge) follow
//! MatrixAlgebraKit.jl `src/implementations/qr.jl` and `src/common/gauge.jl` (MIT,
//! Copyright (c) QuantumKitHub contributors). They are the same conventions as
//! tenferro-linalg's `QrGauge::PositiveDiagonal`, applied to the factors here.
use crate::assert;
use crate::internal_prelude::*;
/// starts `out` as the identity block at rows `offset..offset + q` and applies
/// the $Q$ sequence to it from the left.
///
/// Shared with the LQ module, which passes its transposed view as both the basis
/// and the target.
#[track_caller]
pub(crate) fn apply_to_start_block<T: ComplexField>(
	out: MatMut<'_, T>,
	offset: usize,
	q: usize,
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let mut out = out;
	out.rb_mut().fill(zero());
	for j in 0..q {
		out[(offset + j, j)] = one();
	}
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
		Q_basis, Q_coeff, Conj::No, out, par, stack,
	);
}
/// the workspace required by [`unitary`] for an `out` of shape `nrows x out_ncols`.
pub fn unitary_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	out_ncols: usize,
	block_size: usize,
	par: ParShape,
) -> StackReq {
	let _ = ncols;
	let _ = par;
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(nrows, block_size, out_ncols)
}
/// builds the unitary factor $Q$ of the packed factorization in `Q_basis`.
///
/// `Q_basis` is the packed factorization or its first $\min(m, n)$ columns.
///
/// `out` is $m \times \min(m, n)$ for the thin $Q$ or $m \times m$ for the full
/// $Q$. With `positive`, column $j$ of $Q$ is scaled by $d_j / |d_j|$, where
/// $d_j$ is the diagonal of $R$, matching the
/// [`copy_r`](crate::linalg::qr::no_pivoting::factor::copy_r) phase so that the
/// product $QR$ is unchanged and the diagonal of $R$ is real and nonnegative.
#[track_caller]
pub fn unitary<T: ComplexField>(
	out: MatMut<'_, T>,
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	positive: bool,
	par: Par,
	stack: &mut MemStack,
) {
	let (m, n) = Q_basis.shape();
	let k = Ord::min(m, n);
	let q = out.ncols();
	let mut out = out;
	assert!(all(out.nrows() == m, q == k || q == m));
	// the sequence is read from the first `k` columns of the packed matrix, as
	// `qr_in_place` leaves it, so a caller that has not truncated `R` away passes
	// the whole matrix.
	let basis = Q_basis.get(.., ..k);
	apply_to_start_block(out.rb_mut(), 0, q, basis, Q_coeff, par, stack);
	if positive {
		for j in 0..k {
			let phase = linalg::positive_phase(Q_basis[(j, j)].clone());
			let col = out.rb_mut().col_mut(j);
			z!(col).for_each(|unzip!(x)| *x *= &phase);
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
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(nrows, block_size, nrows - k)
}
/// builds an orthonormal basis of the left null space of the factored matrix: the
/// columns $\min(m, n) + 1, \dots, m$ of the full $Q$, so that `out` is
/// $m \times (m - \min(m, n))$ with orthonormal columns and $A^H out = 0$.
#[track_caller]
pub fn null<T: ComplexField>(
	out: MatMut<'_, T>,
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let (m, n) = Q_basis.shape();
	let k = Ord::min(m, n);
	assert!(all(out.nrows() == m, out.ncols() == m - k));
	let basis = Q_basis.get(.., ..k);
	apply_to_start_block(out, k, m - k, basis, Q_coeff, par, stack);
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::linalg::qr::no_pivoting::factor;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, c64};
	use dyn_stack::MemBuffer;
	fn packed<T: ComplexField>(
		a: MatRef<'_, T>,
		block_size: usize,
	) -> (Mat<T>, Mat<T>, usize) {
		let (m, n) = a.shape();
		let k = Ord::min(m, n);
		let mut packed = a.cloned();
		let mut coeff = Mat::full(block_size, k, infinity());
		let rank = factor::qr_in_place(
			packed.as_mut(),
			coeff.as_mut(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(
				factor::qr_in_place_scratch::<T>(
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
		let basis = packed.as_ref().get(.., ..k);
		for positive in [false, true] {
			let mut r = Mat::<c64>::zeros(k, n);
			factor::copy_r(r.as_mut(), packed.as_ref(), positive);
			if positive {
				// the phase is exact; the residue is the rounding of $d / |d|$.
				for j in 0..k {
					let d = r[(j, j)];
					assert!(d.re >= 0.0 && d.im.abs() <= 1e-14 * d.abs().max(1.0));
				}
			}
			let mut q = Mat::<c64>::zeros(m, k);
			unitary(
				q.as_mut(),
				basis,
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
			assert!((&q * &r) ~ a);
			assert!((q.adjoint() * &q) ~ Mat::<c64>::identity(k, k));
			// the whole packed matrix is accepted too: the sequence is read from its
			// first `k` columns, so a caller that keeps `R` in place need not slice.
			let mut q_full = Mat::<c64>::zeros(m, m);
			unitary(
				q_full.as_mut(),
				packed.as_ref(),
				coeff.as_ref(),
				positive,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(unitary_scratch::<c64>(
					m,
					n,
					m,
					block_size,
					ParShape::Seq,
				))),
			);
			assert!(q_full.get(.., ..k) ~ q.as_ref());
			assert!((q_full.adjoint() * &q_full) ~ Mat::<c64>::identity(m, m));
			let mut complement = Mat::<c64>::zeros(m, m - k);
			null(
				complement.as_mut(),
				basis,
				coeff.as_ref(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(null_scratch::<c64>(
					m,
					n,
					block_size,
					ParShape::Seq,
				))),
			);
			assert!(q_full.get(.., k..) ~ complement.as_ref());
			assert!((complement.adjoint() * &complement) ~ Mat::<c64>::identity(m - k, m - k));
			// $A^H N = R^H (Q^H N) = 0$ for every rank.
			assert!((a.cloned().adjoint() * &complement) ~ Mat::<c64>::zeros(n, m - k));
		}
		assert!(rank <= k);
	}
	#[test]
	fn test_unitary_and_null() {
		let rng = &mut StdRng::seed_from_u64(3);
		for (m, n) in [(7, 4), (4, 7), (5, 5), (0, 3), (3, 0), (0, 0), (96, 64)]
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
		let rng = &mut StdRng::seed_from_u64(4);
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
