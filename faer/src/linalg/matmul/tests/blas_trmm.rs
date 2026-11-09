// Reference BLAS Level-3 test program DBLAT3 and its S/C/Z siblings, subroutine
// DCHK3 (DTRMM), distributed in OpenMathLib/OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80 under test/{d,s,c,z}blat3.f, and the
// CBLAS wrapper in ctest/. Source URL:
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat3.f
// Original test program: Copyright (c) 1989-2012 Jack Dongarra, Iain Duff,
// Jeremy Du Croz, Sven Hammarling (Argonne National Laboratory, AERE Harwell,
// Numerical Algorithms Group Ltd). Netlib BLAS test-program terms:
// faer/COPYING.BLAS.NETLIB. This Rust adaptation is new code under the crate's
// MIT licence.
// Material changes: DCHK3's parameter sweep (N = 0,1,2,3,7,31,63; alpha
// 0.0,0.7,1.0; LDA = N+1 padding) is cut to a bounded shape set with exact
// alpha, and the rogue value (-1.0D10) becomes NaN. Strict-triangle block
// structures, which BLAS has no DIAG value for, are included because faer
// exposes them.

/// the triangular operand: exact integers inside the stored triangle, NaN
/// everywhere the block structure does not access (including a unit or strict
/// diagonal)
fn operand(n: usize, structure: BlockStructure) -> Mat<Scalar> {
	Mat::from_fn(n, n, |i, j| {
		if accessed(structure, i, j) {
			integer(i, j)
		} else {
			nan()
		}
	})
}

const STRUCTURES: [BlockStructure; 6] = [
	BlockStructure::TriangularLower,
	BlockStructure::StrictTriangularLower,
	BlockStructure::UnitTriangularLower,
	BlockStructure::TriangularUpper,
	BlockStructure::StrictTriangularUpper,
	BlockStructure::UnitTriangularUpper,
];

#[test]
fn trmm_matches_an_independent_product() {
	for structure in STRUCTURES {
		for trans in 0..3 {
			for (m, n) in
				[(0, 0), (1, 1), (2, 1), (3, 2), (4, 4), (5, 3), (8, 2)]
			{
				for left in [true, false] {
					let order = if left { m } else { n };
					let a = operand(order, structure);
					let a_before = a.clone();
					let b =
						Mat::<Scalar>::from_fn(m, n, |i, j| integer(i + 1, j));
					// op(A) for trans = N, T, C
					let (a_view, a_structure, a_conj) = match trans {
						0 => (a.as_ref(), structure, Conj::No),
						1 => (
							a.as_ref().transpose(),
							structure.transpose(),
							Conj::No,
						),
						_ => (
							a.as_ref().transpose(),
							structure.transpose(),
							Conj::Yes,
						),
					};
					let (
						lhs,
						lhs_structure,
						conj_lhs,
						rhs,
						rhs_structure,
						conj_rhs,
					) = if left {
						(
							a_view,
							a_structure,
							a_conj,
							b.as_ref(),
							BlockStructure::Rectangular,
							Conj::No,
						)
					} else {
						(
							b.as_ref(),
							BlockStructure::Rectangular,
							Conj::No,
							a_view,
							a_structure,
							a_conj,
						)
					};
					for alpha in [z(0.0), z(1.0), z(2.0), z(0.5)] {
						let mut dst = Mat::<Scalar>::full(m, n, rogue());
						matmul_with_conj(
							&mut dst,
							BlockStructure::Rectangular,
							Accum::Replace,
							lhs,
							lhs_structure,
							conj_lhs,
							rhs,
							rhs_structure,
							conj_rhs,
							alpha,
							Par::Seq,
						);
						let want = blended(
							Accum::Replace,
							Mat::<Scalar>::zeros(m, n).as_ref(),
							BlockStructure::Rectangular,
							lhs,
							lhs_structure,
							conj_lhs,
							rhs,
							rhs_structure,
							conj_rhs,
							alpha,
						);
						crate::assert!(
							exact(dst.as_ref(), want.as_ref()),
							"trmm structure={structure:?} trans={trans} m={m} \
							 n={n} left={left} alpha={:?}\ngot={:?}\nwant={:?}",
							real(&alpha),
							dst.as_ref(),
							want.as_ref()
						);
					}
					crate::assert!(
						exact(a.as_ref(), a_before.as_ref()),
						"trmm modified its triangular operand: \
						 structure={structure:?} trans={trans} left={left}"
					);
				}
			}
		}
	}
}

#[test]
fn trmm_reads_and_writes_nothing_outside_its_operands() {
	use BlockStructure::*;
	for (structure, left) in [
		(TriangularLower, true),
		(UnitTriangularUpper, true),
		(StrictTriangularLower, false),
		(TriangularUpper, false),
	] {
		let (m, n) = (5, 4);
		let order = if left { m } else { n };
		// the operand lives at (1, 1) inside a rogue-filled parent, so a read
		// or write outside the logical matrix shows up as well
		let mut parent = Mat::<Scalar>::full(order + 2, order + 2, rogue());
		parent
			.as_mut()
			.submatrix_mut(1, 1, order, order)
			.copy_from(&operand(order, structure));
		let before = parent.clone();
		let b = Mat::<Scalar>::from_fn(m, n, |i, j| integer(i, j));
		let mut dst = Mat::<Scalar>::zeros(m, n);
		{
			let a = parent.as_ref().submatrix(1, 1, order, order);
			if left {
				matmul_with_conj(
					&mut dst,
					Rectangular,
					Accum::Replace,
					a,
					structure,
					Conj::No,
					b.as_ref(),
					Rectangular,
					Conj::No,
					one(),
					Par::Seq,
				);
			} else {
				matmul_with_conj(
					&mut dst,
					Rectangular,
					Accum::Replace,
					b.as_ref(),
					Rectangular,
					Conj::No,
					a,
					structure,
					Conj::No,
					one(),
					Par::Seq,
				);
			}
		}
		crate::assert!(
			untouched_outside(
				before.as_ref(),
				parent.as_ref(),
				1,
				1,
				order,
				order
			),
			"trmm touched memory outside its operand: structure={structure:?} \
			 left={left}"
		);
	}
}
