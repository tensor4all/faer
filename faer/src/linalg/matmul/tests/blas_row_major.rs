// The CBLAS layer's row-major wrappers for the products this fork already
// ports: ctest/c_{s,d,c,z}blas3.c, distributed in OpenMathLib/OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Source URLs:
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/ctest/c_dblas3.c
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat3.f
// Original credits, from the file headers: the CBLAS wrappers were
// "Written by D.P. Manley, Digital Equipment Corporation", prefixed "C_" to the
// BLAS routines, and "Modified by T. H. Do, 2/19/98, SGI/CRAY Research"; the
// numerical checks they run are the 1989 level-3 program of Jack Dongarra, Iain
// Duff, Jeremy Du Croz and Sven Hammarling. Netlib BLAS test-program terms:
// faer/COPYING.BLAS.NETLIB. This Rust adaptation is new code under the crate's
// MIT licence.
//
// What the wrapper adds to the checks already ported: `CblasRowMajor` copies the
// arguments into row-major buffers, calls the row-major entry, and copies the
// result back (`ctest/c_dblas3.c:26-69` for DGEMM), so the kernel sees a
// destination whose *row* stride is the leading dimension and whose column
// stride is 1 (the other way round from a column-major `Mat`). faer has no copy
// layer -- a transposed view has that layout -- so the adaptation is the view,
// and what these tests check is that the kernels reach the reference result with
// a *destination* in that layout. The row-major *operand* view is not new: the
// level-3 port reaches it through the `'T'`/`'C'` arms of `viewed`
// (`blas_gemm.rs`). Row-major *destinations* are not wholly new either -- the
// rank-1 update (`blas_rank_update.rs`) and the right-hand sides of the TRSM port
// (`blas_trsm_cases.rs`) already use one -- but no product reached the multiply
// kernels with one before this file.

/// the buffer whose transposed view is `m` in row-major storage: an
/// `ncols x nrows` column-major copy of `m^T`
fn row_major_buffer(m: MatRef<'_, Scalar>) -> Mat<Scalar> {
	m.transpose().to_owned()
}

#[test]
fn gemm_writes_a_row_major_destination() {
	// DGEMM through `CblasRowMajor`: the destination (and, as the wrapper does,
	// both operands) stored row-major. The shape set is the one the level-3 port
	// already uses; what this adds is the destination, since that port's own
	// destinations are all column-major; `beta` runs over both `Accum` values
	// because one of them reads the destination.
	for (m, n, k) in [
		(0, 0, 0),
		(0, 2, 3),
		(2, 0, 3),
		(2, 3, 0),
		(1, 1, 1),
		(2, 3, 2),
		(3, 2, 5),
		(4, 4, 4),
		(7, 3, 3),
	] {
		let a = Mat::<Scalar>::from_fn(m, k, |i, j| integer(i, j + 1));
		let b = Mat::<Scalar>::from_fn(k, n, |i, j| integer(i + 2, j));
		let c = Mat::<Scalar>::from_fn(m, n, |i, j| integer(i + 3, j));
		let a_rm = row_major_buffer(a.as_ref());
		let b_rm = row_major_buffer(b.as_ref());
		// the buffers the kernel is given, before the call: the reference product
		// is computed from these, and each is compared against itself afterwards,
		// so a write into an operand is caught after the call that made it (the
		// logical matrices above are not the operands)
		let a_rm_before = a_rm.clone();
		let b_rm_before = b_rm.clone();
		// the source's `LDC = n + 1`: the logical destination is the first `n`
		// columns of a row-major parent one column wider, filled with the rogue
		// value, so a write into the padding shows up as well
		let pad = n + 1;
		let mut c_parent = Mat::<Scalar>::full(pad, m, rogue());
		for i in 0..m {
			for j in 0..n {
				c_parent[(j, i)] = c[(i, j)].copy();
			}
		}
		let c_parent_before = c_parent.clone();
		for beta in [Accum::Replace, Accum::Add] {
			for alpha in [z(0.0), z(1.0), z(-2.0), z(0.5)] {
				let mut parent = c_parent_before.clone();
				matmul(
					parent.as_mut().transpose_mut().submatrix_mut(0, 0, m, n),
					beta,
					a_rm.as_ref().transpose(),
					b_rm.as_ref().transpose(),
					alpha,
					Par::Seq,
				);
				let got = parent.transpose().submatrix(0, 0, m, n).to_owned();
				check_blend(
					"gemm row major",
					&c,
					&got,
					BlockStructure::Rectangular,
					a_rm_before.as_ref().transpose(),
					BlockStructure::Rectangular,
					Conj::No,
					b_rm_before.as_ref().transpose(),
					BlockStructure::Rectangular,
					Conj::No,
					beta,
					alpha,
				);
				crate::assert!(
					untouched_outside(
						c_parent_before.transpose(),
						parent.transpose(),
						0,
						0,
						m,
						n
					),
					"gemm row major wrote into the padded column: m={m} \
					 n={n} k={k} beta={beta:?} alpha={:?}",
					real(&alpha)
				);
				crate::assert!(
					exact(a_rm.as_ref(), a_rm_before.as_ref())
						&& exact(b_rm.as_ref(), b_rm_before.as_ref()),
					"gemm row major modified an operand: m={m} n={n} k={k} \
					 beta={beta:?} alpha={:?}",
					real(&alpha)
				);
			}
		}
	}
}

#[test]
fn trmm_writes_a_row_major_destination() {
	// DTRMM through `CblasRowMajor`: the triangular operand is square and is
	// passed in its own storage, read as a transposed view for the transposed
	// arms exactly as the column-major port reads it, so what this test changes
	// is the destination, which is row-major. `Accum::Replace` is the whole
	// contract rather than a choice: BLAS `?trmm` has no `beta` term (it is
	// `B := alpha * op(A) * B`), and the column-major port runs `Replace` too.
	for structure in STRUCTURES {
		for trans in 0..3 {
			for (m, n) in [(0, 0), (1, 1), (2, 3), (3, 2), (4, 4), (5, 3)] {
				for left in [true, false] {
					let order = if left { m } else { n };
					let a = operand(order, structure);
					let a_before = a.clone();
					let b =
						Mat::<Scalar>::from_fn(m, n, |i, j| integer(i + 1, j));
					let b_before = b.clone();
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
					let (lhs, lhs_structure, conj_lhs, rhs, rhs_structure) =
						if left {
							(
								a_view,
								a_structure,
								a_conj,
								b_before.as_ref(),
								BlockStructure::Rectangular,
							)
						} else {
							(
								b_before.as_ref(),
								BlockStructure::Rectangular,
								Conj::No,
								a_view,
								a_structure,
							)
						};
					let conj_rhs = if left { Conj::No } else { a_conj };
					for alpha in [z(0.0), z(1.0), z(2.0), z(0.5)] {
						let empty = Mat::<Scalar>::zeros(m, n);
						let mut dst = row_major_buffer(empty.as_ref());
						matmul_with_conj(
							dst.as_mut().transpose_mut(),
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
							empty.as_ref(),
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
							exact(dst.as_ref().transpose(), want.as_ref()),
							"trmm row major structure={structure:?} \
							 trans={trans} m={m} n={n} left={left} \
							 alpha={:?}\ngot={:?}\nwant={:?}",
							real(&alpha),
							dst.as_ref().transpose(),
							want.as_ref()
						);
					}
					crate::assert!(
						exact(a.as_ref(), a_before.as_ref())
							&& exact(b.as_ref(), b_before.as_ref()),
						"trmm row major modified an operand: \
						 structure={structure:?} trans={trans} left={left}"
					);
				}
			}
		}
	}
}

#[test]
fn rank_k_update_writes_a_row_major_triangle() {
	// DSYRK/DHERK through `CblasRowMajor`: the stored triangle sits in a
	// row-major square, so the triangular-destination branch runs against a
	// destination whose row stride is the leading dimension rather than 1; the
	// other triangle still holds the source's rogue value and must stay
	// untouched.
	for structure in [
		BlockStructure::TriangularLower,
		BlockStructure::TriangularUpper,
	] {
		// the four arms are the source's `?syrk`/`?herk` with 'N' and 'T'
		for trans in 0..4 {
			for (n, k) in [(0, 0), (1, 1), (2, 3), (3, 2), (4, 4), (5, 3)] {
				let (ma, na) =
					if trans == 0 || trans == 3 { (n, k) } else { (k, n) };
				let a = Mat::<Scalar>::from_fn(ma, na, |i, j| integer(i, j + 1));
				let a_before = a.clone();
				let (a_view, a_conj, rhs, conj_rhs) = match trans {
					0 => (a.as_ref(), Conj::No, a.as_ref().transpose(), Conj::No),
					1 => (a.as_ref().transpose(), Conj::No, a.as_ref(), Conj::No),
					2 => {
						(a.as_ref().transpose(), Conj::Yes, a.as_ref(), Conj::No)
					},
					_ => (a.as_ref(), Conj::No, a.as_ref().transpose(), Conj::Yes),
				};
				for beta in [Accum::Replace, Accum::Add] {
					for alpha in [z(0.0), z(1.0), z(-1.0), z(0.5)] {
						let c = Mat::<Scalar>::from_fn(n, n, |i, j| {
							if accessed(structure, i, j) {
								integer(i + 2, j + 5)
							} else {
								nan()
							}
						});
						let mut dst = row_major_buffer(c.as_ref());
						matmul_with_conj(
							dst.as_mut().transpose_mut(),
							structure,
							beta,
							a_view,
							BlockStructure::Rectangular,
							a_conj,
							rhs,
							BlockStructure::Rectangular,
							conj_rhs,
							alpha,
							Par::Seq,
						);
						let want = blended(
							beta,
							c.as_ref(),
							structure,
							a_view,
							BlockStructure::Rectangular,
							a_conj,
							rhs,
							BlockStructure::Rectangular,
							conj_rhs,
							alpha,
						);
						crate::assert!(
							exact(dst.as_ref().transpose(), want.as_ref()),
							"rank-k row major structure={structure:?} \
							 trans={trans} n={n} k={k} beta={beta:?} \
							 alpha={:?}\ngot={:?}\nwant={:?}",
							real(&alpha),
							dst.as_ref().transpose(),
							want.as_ref()
						);
						crate::assert!(
							exact(a.as_ref(), a_before.as_ref()),
							"rank-k row major modified A: \
							 structure={structure:?} trans={trans}"
						);
					}
				}
			}
		}
	}
}
