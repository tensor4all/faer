// Reference BLAS Level-2 DCHK4 (DGER) and DCHK5 (DSYR) in test/dblat2.f, and
// Level-3 DCHK4 (DSYRK) in test/dblat3.f, with their S/C/Z siblings and CBLAS
// wrappers, distributed in OpenMathLib/OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Source URLs:
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat2.f
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat3.f
// Original test programs: Copyright (c) 1989-2012 Jack Dongarra, Iain Duff,
// Jeremy Du Croz, Sven Hammarling. Netlib BLAS test-program terms:
// faer/COPYING.BLAS.NETLIB. This Rust adaptation is new code under the crate's
// MIT licence.
// Material changes: the random DMAKE/DBEG fixtures become exact integers, the
// dimension sweep is bounded, and the rogue value (-1.0D10) becomes NaN. faer
// has no `ger`/`syr`/`syrk` entry points: the rank-1 update is `matmul` with
// K = 1, and the symmetric rank-k update is `matmul_with_conj` with a
// triangular destination, so those two calls carry the source checks. SYR2K has
// no counterpart (it is two rank-k terms, i.e. two calls).

#[test]
fn rank_one_update_matches_the_source_contract() {
	// DGER: A = alpha * x * y^T + beta * A over the whole (m, n) rectangle,
	// through the K = 1 path of `matmul`
	for (m, n) in [(0, 0), (1, 1), (2, 3), (3, 2), (5, 4), (8, 8)] {
		for beta in [Accum::Replace, Accum::Add] {
			for alpha in [z(0.0), z(1.0), z(-2.0), z(0.5)] {
				let x = Mat::<Scalar>::from_fn(m, 1, |i, _| integer(i + 1, 0));
				let y = Mat::<Scalar>::from_fn(1, n, |_, j| integer(0, j + 2));
				let a = Mat::<Scalar>::from_fn(m, n, |i, j| integer(i + 3, j));
				// column major: the rank_update branch
				let mut dst = a.clone();
				matmul(&mut dst, beta, x.as_ref(), y.as_ref(), alpha, Par::Seq);
				check_blend(
					"ger",
					&a,
					&dst,
					BlockStructure::Rectangular,
					x.as_ref(),
					BlockStructure::Rectangular,
					Conj::No,
					y.as_ref(),
					BlockStructure::Rectangular,
					Conj::No,
					beta,
					alpha,
				);
				// row major destination: the transposed rank_update branch
				let mut storage = a.transpose().to_owned();
				matmul(
					storage.transpose_mut(),
					beta,
					x.as_ref(),
					y.as_ref(),
					alpha,
					Par::Seq,
				);
				check_blend(
					"ger row major",
					&a,
					&storage.transpose().to_owned(),
					BlockStructure::Rectangular,
					x.as_ref(),
					BlockStructure::Rectangular,
					Conj::No,
					y.as_ref(),
					BlockStructure::Rectangular,
					Conj::No,
					beta,
					alpha,
				);
			}
		}
	}
}

#[test]
fn symmetric_rank_k_update_writes_only_the_stored_triangle() {
	// DSYRK/HERK: the referenced triangle of C is alpha * A * op(A)^T +
	// beta * C, and the other triangle must not be read or written (the source
	// test's LDERES 'SY' check)
	for structure in [
		BlockStructure::TriangularLower,
		BlockStructure::TriangularUpper,
	] {
		for trans in 0..3 {
			for (n, k) in [(0, 0), (1, 1), (2, 3), (3, 2), (4, 4), (5, 3)] {
				let (ma, na) = if trans == 0 { (n, k) } else { (k, n) };
				let a =
					Mat::<Scalar>::from_fn(ma, na, |i, j| integer(i, j + 1));
				let a_before = a.clone();
				let (a_view, a_structure, a_conj, rhs, rhs_structure, conj_rhs) =
					match trans {
						0 => (
							a.as_ref(),
							BlockStructure::Rectangular,
							Conj::No,
							a.as_ref().transpose(),
							BlockStructure::Rectangular,
							Conj::Yes,
						),
						1 => (
							a.as_ref().transpose(),
							BlockStructure::Rectangular,
							Conj::No,
							a.as_ref(),
							BlockStructure::Rectangular,
							Conj::No,
						),
						_ => (
							a.as_ref().transpose(),
							BlockStructure::Rectangular,
							Conj::Yes,
							a.as_ref(),
							BlockStructure::Rectangular,
							Conj::No,
						),
					};
				for beta in [Accum::Replace, Accum::Add] {
					for alpha in [z(0.0), z(1.0), z(-1.0), z(0.5)] {
						// the unaccessed triangle holds the source's rogue
						// value
						let c = Mat::<Scalar>::from_fn(n, n, |i, j| {
							if accessed(structure, i, j) {
								integer(i + 2, j + 5)
							} else {
								nan()
							}
						});
						let mut dst = c.clone();
						matmul_with_conj(
							&mut dst,
							structure,
							beta,
							a_view,
							a_structure,
							a_conj,
							rhs,
							rhs_structure,
							conj_rhs,
							alpha,
							Par::Seq,
						);
						let want = blended(
							beta,
							c.as_ref(),
							structure,
							a_view,
							a_structure,
							a_conj,
							rhs,
							rhs_structure,
							conj_rhs,
							alpha,
						);
						crate::assert!(
							exact(dst.as_ref(), want.as_ref()),
							"rank-k structure={structure:?} trans={trans} \
							 n={n} k={k} beta={beta:?} \
							 alpha={:?}\ngot={:?}\nwant={:?}",
							real(&alpha),
							dst.as_ref(),
							want.as_ref()
						);
						crate::assert!(
							exact(a.as_ref(), a_before.as_ref()),
							"rank-k modified A: structure={structure:?} \
							 trans={trans}"
						);
					}
				}
			}
		}
	}
}

fn check_blend(
	name: &str,
	dst_before: &Mat<Scalar>,
	dst: &Mat<Scalar>,
	dst_structure: BlockStructure,
	lhs: MatRef<'_, Scalar>,
	lhs_structure: BlockStructure,
	conj_lhs: Conj,
	rhs: MatRef<'_, Scalar>,
	rhs_structure: BlockStructure,
	conj_rhs: Conj,
	beta: Accum,
	alpha: Scalar,
) {
	let alpha_dbg = real(&alpha);
	let want = blended(
		beta,
		dst_before.as_ref(),
		dst_structure,
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
		"{name} beta={beta:?} alpha={:?}\ngot={:?}\nwant={:?}",
		alpha_dbg,
		dst.as_ref(),
		want.as_ref()
	);
}
