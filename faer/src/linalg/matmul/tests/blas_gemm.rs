// Reference BLAS Level-3 DCHK1 (DGEMM) in test/dblat3.f and Level-2 DCHK1
// (DGEMV) in test/dblat2.f, with their S/C/Z siblings and CBLAS wrappers,
// distributed in OpenMathLib/OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Source URLs:
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat3.f
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat2.f
// Original test programs: Level 3 DCHK1 is Copyright (c) 1989 Jack Dongarra,
// Iain Duff, Jeremy Du Croz, Sven Hammarling (`dblat3.f:403-407`, dated
// 8-February-1989); Level 2 DCHK1 is Copyright (c) 1987 Richard Hanson and
// Jeremy Du Croz (`dblat2.f:439-441`, dated 10-August-1987). Netlib BLAS
// test-program terms: faer/COPYING.BLAS.NETLIB. This Rust adaptation is new code
// under the crate's MIT licence.
// Material changes: the random DMAKE/DBEG fixtures become exact integers, the
// dimension sweep is bounded, and the rogue value (-1.0D10) becomes NaN. faer
// has no separate `gemm`/`gemv` entry points: `matmul` is the whole product,
// and its `ncols == 1` / `nrows == 1` routes are the matvec ones, so those two
// calls carry the source checks. The source's `TRANS` letters become operand
// transposition and `Conj`, since a real transpose and a conjugate transpose
// are the same call in faer and differ only for the complex types.
//
// Two deliberate deviations from the reference routines are asserted here,
// because they are what faer's generic `matmul` promises rather than what
// `?gemv`/`?gemm` promise:
// - `Accum::Replace` is the BLAS `beta = 0` case, and it does not read the
//   destination, so the sentinel in the destination cannot leak into the
//   result. `Accum::Add` does read it, and the test asserts the sentinel comes
//   back, so the two arms cannot be confused.
// - an empty reduction is `beta * dst` in faer (`dst` is zeroed for
//   `Accum::Replace`), whereas `?gemv` returns before touching `y` at all when
//   `m == 0` or `n == 0`. The source test asserts `y` is unchanged there
//   (`LDE(YS, YY, LY)` on its `NULL` path); faer's `matmul` contract is the
//   algebraic one and this file asserts that instead. Recorded in
//   docs/blas-test-inventory.md.

#[test]
fn gemm_matches_the_source_dimension_sweep() {
	// DCHK1 GEMM: `C = alpha * op(A) * op(B) + beta * C` over the (m, n, k)
	// sweep. The Fortran driver's `test/dblat3.dat` lists its dimensions as
	// `0 1 2 3 7 31 63`, so its `NULL` path for `m <= 0 or n <= 0` and the empty
	// `k = 0` reduction are reached there; the CBLAS driver's `ctest/din3` starts
	// at 1. `alpha = 0` and `beta = 0` (in its `Accum::Replace` form) are in both
	// drivers' `ALF`/`BET` lists.
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
		// the logical matrices, and the storage the source would hand to
		// `?gemm` for each `TRANS`: the same logical `A` is stored as `m x k`
		// for 'N' and as `k x m` for 'T'/'C'
		let a = Mat::<Scalar>::from_fn(m, k, |i, j| integer(i, j + 1));
		let a_t = a.transpose().to_owned();
		let b = Mat::<Scalar>::from_fn(k, n, |i, j| integer(i + 2, j));
		let b_t = b.transpose().to_owned();
		let c = Mat::<Scalar>::from_fn(m, n, |i, j| integer(i + 4, j + 3));
		// 0 = 'N', 1 = 'T', 2 = 'C' (the source's ICH = 'NTC')
		for (ta, tb) in [
			(0, 0),
			(1, 0),
			(0, 1),
			(1, 1),
			(2, 0),
			(0, 2),
			(2, 2),
		] {
			let (lhs, lhs_conj) = viewed(&a, &a_t, ta);
			let (rhs, rhs_conj) = viewed(&b, &b_t, tb);
			for beta in [Accum::Replace, Accum::Add] {
				for alpha in [z(0.0), z(1.0), z(-2.0), z(0.5)] {
					let mut dst = c.clone();
					// `matmul_with_conj` rather than `matmul`: the source's
					// 'C' arm conjugates, and `matmul` derives the
					// conjugation from the operand's type
					matmul_with_conj(
						&mut dst,
						BlockStructure::Rectangular,
						beta,
						lhs,
						BlockStructure::Rectangular,
						lhs_conj,
						rhs,
						BlockStructure::Rectangular,
						rhs_conj,
						alpha,
						Par::Seq,
					);
					check_blend(
						"gemm",
						&c,
						&dst,
						BlockStructure::Rectangular,
						lhs,
						BlockStructure::Rectangular,
						lhs_conj,
						rhs,
						BlockStructure::Rectangular,
						rhs_conj,
						beta,
						alpha,
					);
				}
			}
		}
		crate::assert!(
			exact(a.as_ref(), a_t.transpose())
				&& exact(b.as_ref(), b_t.transpose()),
			"gemm modified an operand: m={m} n={n} k={k}"
		);
	}
}

#[test]
fn gemm_reads_the_destination_only_when_it_accumulates() {
	// the source's `beta = 0` half: `C` need not be set on entry, so a
	// `Replace` call must produce the product from a destination that holds a
	// value no correct result can contain. `Accum::Add` must read it, which is
	// what makes the sentinel able to tell the two arms apart.
	for (m, n, k) in [(1, 1, 1), (2, 3, 2), (3, 2, 4), (2, 2, 0)] {
		let a = Mat::<Scalar>::from_fn(m, k, |i, j| integer(i, j + 1));
		let b = Mat::<Scalar>::from_fn(k, n, |i, j| integer(i + 2, j));
		for alpha in [z(1.0), z(-2.0), z(0.0)] {
			let poisoned = Mat::<Scalar>::from_fn(m, n, |_, _| nan());
			let mut replaced = poisoned.clone();
			matmul(
				&mut replaced,
				Accum::Replace,
				a.as_ref(),
				b.as_ref(),
				alpha,
				Par::Seq,
			);
			for i in 0..m {
				for j in 0..n {
					// a NaN that reached the result means the destination was
					// read; the product of this fixture is never NaN
					crate::assert!(
						!real(&replaced[(i, j)]).is_nan()
							&& !imag(&replaced[(i, j)]).is_nan(),
						"gemm read the replaced destination: m={m} n={n} \
						 k={k} alpha={:?} at ({i}, {j})",
						real(&alpha)
					);
				}
			}
			let mut added = poisoned.clone();
			matmul(
				&mut added,
				Accum::Add,
				a.as_ref(),
				b.as_ref(),
				alpha,
				Par::Seq,
			);
			for i in 0..m {
				for j in 0..n {
					// the accumulate arm does read it, so the sentinel survives
					crate::assert!(
						real(&added[(i, j)]).is_nan(),
						"gemm did not accumulate into the destination: \
						 m={m} n={n} k={k} alpha={:?} at ({i}, {j})",
						real(&alpha)
					);
				}
			}
		}
	}
}

#[test]
fn gemv_matches_the_source_dimension_sweep() {
	// DCHK1 GEMV: `y = alpha * op(A) * x + beta * y`, through `matmul`'s
	// `ncols == 1` route, and its transpose (`nrows == 1`) sibling.
	// `test/dblat2.dat` sweeps `n = 0, 1, 2, 3, 5, 9, 63` with `m` just below
	// and just above `n`, so the two empty directions are real cases there.
	// The CBLAS `ctest/din2` starts at 1 and does not reach them.
	for (m, k) in [(0, 0), (0, 3), (3, 0), (1, 1), (3, 2), (2, 3), (5, 4)] {
		let a = Mat::<Scalar>::from_fn(m, k, |i, j| integer(i, j + 1));
		let a_before = a.clone();
		// y (m x 1) = A (m x k) * x (k x 1)
		let x = Mat::<Scalar>::from_fn(k, 1, |i, _| integer(i + 2, 5));
		let y = Mat::<Scalar>::from_fn(m, 1, |i, _| integer(i + 4, 3));
		for beta in [Accum::Replace, Accum::Add] {
			for alpha in [z(0.0), z(1.0), z(-2.0), z(0.5)] {
				let mut dst = y.clone();
				matmul(&mut dst, beta, a.as_ref(), x.as_ref(), alpha, Par::Seq);
				check_blend(
					"gemv",
					&y,
					&dst,
					BlockStructure::Rectangular,
					a.as_ref(),
					BlockStructure::Rectangular,
					Conj::No,
					x.as_ref(),
					BlockStructure::Rectangular,
					Conj::No,
					beta,
					alpha,
				);
			}
		}
		// y^T (1 x k) = x^T (1 x m) * A (m x k): the other matvec route, whose
		// left operand is a transposed column (`try_as_row_major` succeeds)
		let xt = Mat::<Scalar>::from_fn(m, 1, |i, _| integer(i + 1, 2));
		let yt = Mat::<Scalar>::from_fn(1, k, |_, j| integer(j + 3, 4));
		for beta in [Accum::Replace, Accum::Add] {
			for alpha in [z(0.0), z(1.0), z(-2.0), z(0.5)] {
				let mut dst = yt.clone();
				matmul(
					&mut dst,
					beta,
					xt.as_ref().transpose(),
					a.as_ref(),
					alpha,
					Par::Seq,
				);
				check_blend(
					"gemv transposed",
					&yt,
					&dst,
					BlockStructure::Rectangular,
					xt.as_ref().transpose(),
					BlockStructure::Rectangular,
					Conj::No,
					a.as_ref(),
					BlockStructure::Rectangular,
					Conj::No,
					beta,
					alpha,
				);
			}
		}
		crate::assert!(
			exact(a.as_ref(), a_before.as_ref()),
			"gemv modified its matrix: m={m} k={k}"
		);
	}
}

#[test]
fn gemv_zeroes_the_destination_of_an_empty_reduction() {
	// faer's contract for the empty reduction: `y = beta * y` with no products,
	// so `Accum::Replace` zeroes the destination and `Accum::Add` returns it
	// unchanged. The source test's `NULL` path asserts the stronger,
	// BLAS-specific "y is untouched even for `beta != 1`"; the difference is
	// recorded in docs/blas-test-inventory.md. The empty `m`/`n` dimensions
	// themselves are in `gemv_matches_the_source_dimension_sweep`.
	let a = Mat::<Scalar>::from_fn(3, 0, |_, _| zero());
	let x = Mat::<Scalar>::from_fn(0, 1, |_, _| zero());
	let y = Mat::<Scalar>::from_fn(3, 1, |i, _| integer(i + 1, 2));
	for beta in [Accum::Replace, Accum::Add] {
		let mut dst = y.clone();
		matmul(&mut dst, beta, a.as_ref(), x.as_ref(), z(1.0), Par::Seq);
		if matches!(beta, Accum::Add) {
			crate::assert!(
				exact(dst.as_ref(), y.as_ref()),
				"gemv changed the destination of a zero-length reduction"
			);
		} else {
			// the algebraic empty sum, not the source's untouched `y`
			crate::assert!(
				(0..3).all(|i| real(&dst[(i, 0)]) == 0.0
					&& imag(&dst[(i, 0)]) == 0.0),
				"gemv did not replace with a zero-length reduction"
			);
		}
	}
}

/// `a` read as the source's `TRANS` letter says: `0` = 'N', `1` = 'T'
/// (transposed, not conjugated), `2` = 'C' (transposed and conjugated). The
/// untransposed and transposed storages are passed in so that both arms read
/// the same logical matrix.
fn viewed<'a>(
	a: &'a Mat<Scalar>,
	a_t: &'a Mat<Scalar>,
	trans: u8,
) -> (MatRef<'a, Scalar>, Conj) {
	match trans {
		0 => (a.as_ref(), Conj::No),
		1 => (a_t.as_ref().transpose(), Conj::No),
		_ => (a_t.as_ref().transpose(), Conj::Yes),
	}
}
