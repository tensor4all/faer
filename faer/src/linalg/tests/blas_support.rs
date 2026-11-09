// Shared oracles for the OpenBLAS BLAS-level test ports. Inventory and source
// identities: docs/blas-test-inventory.md.
//
// The reference BLAS test programs generate the parts of an operand that a
// routine must not access as a rogue value (`DMAKE`, `ROGUE = -1.0D10`) and
// compare the result against an independently accumulated product (`DMMCH`).
// These helpers keep that structure. The fixture values themselves are new
// exact-integer sequences -- complex-valued for the complex types, so that a
// conjugation is observable -- and the unreferenced half holds NaN instead of
// the rogue value, so that a masked read still poisons the result.
// Netlib BLAS test-program credit: see faer/COPYING.BLAS.NETLIB.

/// the block structure of a matmul operand; aliased so that an includer can
/// import `BlockStructure` itself
type Structure = crate::linalg::matmul::triangular::BlockStructure;

/// whether the block structure says entry `(i, j)` is accessed at all
fn accessed(s: Structure, i: usize, j: usize) -> bool {
	use crate::linalg::matmul::triangular::BlockStructure::*;
	match s {
		Rectangular => true,
		TriangularLower => i >= j,
		StrictTriangularLower => i > j,
		UnitTriangularLower => i > j,
		TriangularUpper => i <= j,
		StrictTriangularUpper => i < j,
		UnitTriangularUpper => i < j,
	}
}

/// entry `(i, j)` as the block structure says it is read: an unaccessed entry
/// is zero, and an implicit unit diagonal is one
fn entry(a: MatRef<'_, Scalar>, s: Structure, i: usize, j: usize) -> Scalar {
	use crate::linalg::matmul::triangular::BlockStructure::*;
	if matches!(s, UnitTriangularLower | UnitTriangularUpper) && i == j {
		return one();
	}
	if accessed(s, i, j) {
		a[(i, j)].copy()
	} else {
		zero()
	}
}

fn conjugated(x: Scalar, conj: Conj) -> Scalar {
	if matches!(conj, Conj::Yes) {
		x.conj()
	} else {
		x
	}
}

/// `lhs * rhs` with both operands read as their block structure says
fn product(
	lhs: MatRef<'_, Scalar>,
	lhs_structure: Structure,
	conj_lhs: Conj,
	rhs: MatRef<'_, Scalar>,
	rhs_structure: Structure,
	conj_rhs: Conj,
) -> Mat<Scalar> {
	assert_eq!(lhs.ncols(), rhs.nrows());
	Mat::from_fn(lhs.nrows(), rhs.ncols(), |i, j| {
		let mut acc = zero();
		for p in 0..lhs.ncols() {
			acc = acc
				+ conjugated(entry(lhs, lhs_structure, i, p), conj_lhs)
					* conjugated(entry(rhs, rhs_structure, p, j), conj_rhs);
		}
		acc
	})
}

/// `beta * dst + alpha * lhs * rhs`, with only the entries the destination
/// structure says are computed written (the rest keep `dst`'s value)
fn blended(
	beta: Accum,
	dst: MatRef<'_, Scalar>,
	dst_structure: Structure,
	lhs: MatRef<'_, Scalar>,
	lhs_structure: Structure,
	conj_lhs: Conj,
	rhs: MatRef<'_, Scalar>,
	rhs_structure: Structure,
	conj_rhs: Conj,
	alpha: Scalar,
) -> Mat<Scalar> {
	let prod =
		product(lhs, lhs_structure, conj_lhs, rhs, rhs_structure, conj_rhs);
	assert_eq!((dst.nrows(), dst.ncols()), prod.shape());
	Mat::from_fn(dst.nrows(), dst.ncols(), |i, j| {
		if !accessed(dst_structure, i, j) {
			return dst[(i, j)].copy();
		}
		let add = alpha * prod[(i, j)].copy();
		if matches!(beta, Accum::Add) {
			dst[(i, j)].copy() + add
		} else {
			add
		}
	})
}

/// equal componentwise, counting a NaN component as equal to itself (the
/// reference test's `LDE`, so that an untouched sentinel compares to itself)
fn same_number(x: Scalar, y: Scalar) -> bool {
	let same = |a: Real, b: Real| a == b || a.is_nan() && b.is_nan();
	same(real(&x), real(&y)) && same(imag(&x), imag(&y))
}

/// elementwise equality that counts a NaN sentinel as equal to itself; the
/// fixtures are integers or exact binary fractions, so no tolerance is needed
fn exact(a: MatRef<'_, Scalar>, b: MatRef<'_, Scalar>) -> bool {
	a.shape() == b.shape()
		&& (0..a.nrows()).all(|i| {
			(0..a.ncols())
				.all(|j| same_number(a[(i, j)].copy(), b[(i, j)].copy()))
		})
}

/// every entry outside `(r0..r0 + rows, c0..c0 + cols)` is unchanged (the
/// source test's leading-dimension padding: `LDA = n + 1`, filled with the
/// rogue value)
fn untouched_outside(
	before: MatRef<'_, Scalar>,
	after: MatRef<'_, Scalar>,
	r0: usize,
	c0: usize,
	rows: usize,
	cols: usize,
) -> bool {
	before.shape() == after.shape()
		&& (0..before.nrows()).all(|i| {
			(0..before.ncols()).all(|j| {
				if i >= r0 && i < r0 + rows && j >= c0 && j < c0 + cols {
					return true;
				}
				same_number(before[(i, j)].copy(), after[(i, j)].copy())
			})
		})
}

/// the source's `ROGUE`, for memory a routine must not touch at all
fn rogue() -> Scalar {
	from_f64(-1.0e10)
}

fn nan() -> Scalar {
	from_f64(f64::NAN)
}

/// the exact fixture sequence: components in `[-2, 2]`, both parts nonzero for
/// the complex types so that a conjugation changes the result
fn integer(i: usize, j: usize) -> Scalar {
	scalar(
		((i * 7 + j * 3) % 5) as Real - 2.0,
		((i * 3 + j * 5) % 5) as Real - 2.0,
	)
}
