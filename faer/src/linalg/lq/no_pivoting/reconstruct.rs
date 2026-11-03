//! rebuilding $A = LQ$ from a packed $LQ$ factorization.
use crate::assert;
use crate::internal_prelude::*;
pub fn reconstruct_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	block_size: usize,
	par: ParShape,
) -> StackReq {
	_ = par;
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(ncols, block_size, nrows)
}
#[track_caller]
pub fn reconstruct<T: ComplexField>(
	out: MatMut<'_, T>,
	packed: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let (m, n) = packed.shape();
	let k = Ord::min(m, n);
	assert!(all(out.nrows() == m, out.ncols() == n));
	let mut out = out;
	out.fill(zero());
	out.rb_mut().get_mut(.., ..k).copy_from_triangular_lower(packed.get(.., ..k));
	// $LQ = (Q^\top L^\top)^\top$: the transposed output is the $QR$ basis
	// applied to $L^\top$.
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
		packed.transpose(),
		Q_coeff,
		Conj::No,
		out.rb_mut().transpose_mut(),
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
	#[test]
	fn test_reconstruct() {
		let rng = &mut StdRng::seed_from_u64(0);
		for (m, n) in [(100, 50), (50, 100), (0, 4), (4, 0)] {
			let A = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			let k = Ord::min(m, n);
			let block_size = Ord::min(
				factor::recommended_block_size::<c64>(m, n),
				Ord::max(k, 1),
			);
			let mut lq = A.clone();
			let mut coeff = Mat::zeros(block_size, k);
			factor::lq_in_place(
				lq.as_mut(),
				coeff.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					factor::lq_in_place_scratch::<c64>(
						m,
						n,
						block_size,
						ParShape::Seq,
						default(),
					),
				)),
				default(),
			);
			let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));
			let mut rec = Mat::zeros(m, n);
			reconstruct(
				rec.as_mut(),
				lq.as_ref(),
				coeff.as_ref(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					reconstruct_scratch::<c64>(m, n, block_size, ParShape::Seq),
				)),
			);
			assert!(rec ~ A);
		}
	}
}
