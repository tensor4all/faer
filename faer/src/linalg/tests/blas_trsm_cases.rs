// Reference BLAS Level-3 test program DBLAT3 (DCHK3, DTRSM) and Level-2
// DCHK3 (DTRSV) in test/dblat2.f, with their S/C/Z siblings and CBLAS wrappers,
// distributed in OpenMathLib/OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Source URLs:
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat3.f
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat2.f
// Original test programs: Copyright (c) 1989-2012 Jack Dongarra, Iain Duff,
// Jeremy Du Croz, Sven Hammarling. Netlib BLAS test-program terms:
// faer/COPYING.BLAS.NETLIB. This Rust adaptation is new code under the crate's
// MIT licence.
// Material changes: the random DMAKE/DBEG fixtures become exact integers with a
// unit or a +/-1/+/-2 diagonal, the dimension sweep is bounded, and the rogue
// value (-1.0D10) that the source stores in an unused triangle and a unit
// diagonal becomes NaN. The source solves and then multiplies back to check the
// residual; here the exact reference solution is recovered bit for bit, and the
// in-place right-hand side is a submatrix of a rogue-filled parent, so the
// source's LDA padding also covers writes.

/// `op(A)[(i, j)]` with `op` = N (0), T (1) or C (2), reading only the stored
/// triangle and an implicit unit diagonal
fn op_entry(
	a: MatRef<'_, Scalar>,
	structure: BlockStructure,
	trans: usize,
	i: usize,
	j: usize,
) -> Scalar {
	match trans {
		0 => entry(a, structure, i, j),
		1 => entry(a, structure, j, i),
		_ => entry(a, structure, j, i).conj(),
	}
}

/// the triangular operand: exact integers off the diagonal, an exact
/// `1, -1, 2, -2` diagonal, and NaN where the structure does not access
fn operand(order: usize, structure: BlockStructure) -> Mat<Scalar> {
	Mat::from_fn(order, order, |i, j| {
		if !accessed(structure, i, j) {
			nan()
		} else if i == j {
			from_f64([1.0, -1.0, 2.0, -2.0][i % 4])
		} else {
			integer(i + 1, j)
		}
	})
}

fn solve(
	a: MatRef<'_, Scalar>,
	structure: BlockStructure,
	conj: Conj,
	rhs: MatMut<'_, Scalar>,
	par: Par,
) {
	let unit = matches!(
		structure,
		BlockStructure::UnitTriangularLower
			| BlockStructure::UnitTriangularUpper
	);
	match (structure.is_lower(), unit) {
		(true, false) => {
			ts::solve_lower_triangular_in_place_with_conj(a, conj, rhs, par)
		},
		(true, true) => ts::solve_unit_lower_triangular_in_place_with_conj(
			a, conj, rhs, par,
		),
		(false, false) => {
			ts::solve_upper_triangular_in_place_with_conj(a, conj, rhs, par)
		},
		(false, true) => ts::solve_unit_upper_triangular_in_place_with_conj(
			a, conj, rhs, par,
		),
	}
}

/// the right-hand side that the exact solution `x` produces in `op(A) X = B`
/// (`left`) or `X op(A) = B`
fn rhs_from_solution(
	a: MatRef<'_, Scalar>,
	structure: BlockStructure,
	trans: usize,
	left: bool,
	x: MatRef<'_, Scalar>,
	alpha: Scalar,
) -> Mat<Scalar> {
	let order = if left { x.nrows() } else { x.ncols() };
	Mat::<Scalar>::from_fn(x.nrows(), x.ncols(), |i, j| {
		let mut acc = zero::<Scalar>();
		for k in 0..order {
			acc = acc
				+ if left {
					op_entry(a, structure, trans, i, k) * x[(k, j)].copy()
				} else {
					x[(i, k)].copy() * op_entry(a, structure, trans, k, j)
				};
		}
		alpha.copy() * acc
	})
}

/// the system `op(A) X = B` (`left`) or `X op(A) = B`, solved and compared with
/// `alpha * x`
fn check_solve(
	rows: usize,
	cols: usize,
	structure: BlockStructure,
	trans: usize,
	left: bool,
	alpha: Scalar,
) {
	let order = if left { rows } else { cols };
	let a = operand(order, structure);
	let a_before = a.clone();
	let x = Mat::<Scalar>::from_fn(rows, cols, |i, j| integer(i + j, j + 2));
	let mut b =
		rhs_from_solution(a.as_ref(), structure, trans, left, x.as_ref(), alpha);
	// the system matrix that `X op(A) = B` turns into is the plain transpose
	let (a_view, a_structure, conj) = match (left, trans) {
		(true, 0) | (false, 1) => (a.as_ref(), structure, Conj::No),
		(true, 1) => (a.as_ref().transpose(), structure.transpose(), Conj::No),
		(true, _) => (a.as_ref().transpose(), structure.transpose(), Conj::Yes),
		(false, 0) => (a.as_ref().transpose(), structure.transpose(), Conj::No),
		(false, _) => (a.as_ref(), structure, Conj::Yes),
	};
	if left {
		solve(a_view, a_structure, conj, b.as_mut(), Par::Seq);
	} else {
		solve(
			a_view,
			a_structure,
			conj,
			b.as_mut().transpose_mut(),
			Par::Seq,
		);
	}
	let want =
		Mat::<Scalar>::from_fn(rows, cols, |i, j| alpha * x[(i, j)].copy());
	let ctx = format!(
		"trsm structure={structure:?} trans={trans} left={left} rows={rows} \
		 cols={cols}"
	);
	crate::assert!(
		exact(b.as_ref(), want.as_ref()),
		"{ctx}\ngot={:?}\nwant={:?}",
		b.as_ref(),
		want.as_ref()
	);
	crate::assert!(exact(a.as_ref(), a_before.as_ref()), "{ctx}: A changed");
}

#[test]
fn trsm_recovers_the_exact_solution_with_an_implicit_diagonal() {
	use BlockStructure::*;
	for structure in [
		TriangularLower,
		TriangularUpper,
		UnitTriangularLower,
		UnitTriangularUpper,
	] {
		for trans in 0..3 {
			for left in [true, false] {
				for (rows, cols) in
					[(0, 0), (1, 1), (2, 1), (3, 2), (4, 4), (5, 3), (8, 2)]
				{
					for alpha in [z(1.0), z(2.0)] {
						check_solve(rows, cols, structure, trans, left, alpha);
					}
				}
			}
		}
	}
}

#[test]
fn trsm_does_not_touch_memory_outside_its_operand() {
	use BlockStructure::*;
	for structure in [TriangularLower, UnitTriangularUpper] {
		for (left, order) in [(true, 4usize), (false, 3usize)] {
			let (rows, cols) = if left { (order, 2) } else { (2, order) };
			// both operands live at (1, 1) inside their own rogue-filled parent,
			// so a read or write outside the logical matrix shows up as well
			let mut a_parent = Mat::<Scalar>::full(order + 2, order + 2, rogue());
			a_parent
				.as_mut()
				.submatrix_mut(1, 1, order, order)
				.copy_from(&operand(order, structure));
			let a_before = a_parent.clone();
			// a known system: `x` is the solution its right-hand side produces,
			// built from the coefficient matrix as it was stored before the call
			let x =
				Mat::<Scalar>::from_fn(rows, cols, |i, j| integer(i + j, j + 2));
			let rhs = rhs_from_solution(
				a_before.as_ref().submatrix(1, 1, order, order),
				structure,
				0,
				left,
				x.as_ref(),
				one(),
			);
			let mut parent = Mat::<Scalar>::full(rows + 2, cols + 2, rogue());
			parent
				.as_mut()
				.submatrix_mut(1, 1, rows, cols)
				.copy_from(&rhs);
			let before = parent.clone();
			{
				let a = a_parent.as_ref().submatrix(1, 1, order, order);
				let b = parent.as_mut().submatrix_mut(1, 1, rows, cols);
				if left {
					solve(a, structure, Conj::No, b, Par::Seq);
				} else {
					solve(
						a.transpose(),
						structure.transpose(),
						Conj::No,
						b.transpose_mut(),
						Par::Seq,
					);
				}
			}
			let ctx = format!(
				"trsm structure={structure:?} left={left} order={order}"
			);
			let got = parent.as_ref().submatrix(1, 1, rows, cols);
			crate::assert!(
				exact(got, x.as_ref()),
				"{ctx}\ngot={:?}\nwant={:?}",
				got,
				x.as_ref()
			);
			crate::assert!(
				untouched_outside(
					before.as_ref(),
					parent.as_ref(),
					1,
					1,
					rows,
					cols
				),
				"{ctx}: the solve wrote outside its right-hand side"
			);
			// the whole parent, not only its padding: the stored operand has to
			// come back unchanged (NaN compares to NaN)
			crate::assert!(
				exact(a_before.as_ref(), a_parent.as_ref()),
				"{ctx}: the triangular operand or its parent was modified"
			);
		}
	}
}

/// A conjugating solve of an explicit diagonal: `conj(A) X = B` with a
/// purely imaginary pivot. A real diagonal (what [`operand`] stores) cannot
/// tell `1/a` from `1/conj(a)`, so the one-element base case at
/// `triangular_solve.rs` could lose its conjugation unnoticed. `2i` is exact
/// under conjugation and its reciprocal (`-i/2`) is exact in binary, so the
/// comparison stays exact. The real suites return immediately: `conj` is the
/// identity there and the two expressions coincide by definition.
#[test]
fn trsv_conjugates_an_imaginary_diagonal_exactly() {
	use BlockStructure::*;
	let pivot = scalar(0.0, 2.0);
	if pivot == zero() {
		return;
	}
	for structure in [TriangularLower, TriangularUpper] {
		for (conj, want) in [(Conj::No, scalar(0.0, -0.5)), (Conj::Yes, scalar(0.0, 0.5))] {
			let mut a = Mat::<Scalar>::full(1, 1, zero());
			a[(0, 0)] = pivot;
			let a_before = a.clone();
			let mut b = Mat::<Scalar>::full(1, 1, one());
			solve(a.as_ref(), structure, conj, b.as_mut(), Par::Seq);
			let ctx = format!("1x1 structure={structure:?} conj={conj:?}");
			crate::assert!(
				same_number(b[(0, 0)].copy(), want),
				"{ctx}: got {:?} want {:?}",
				b[(0, 0)],
				want
			);
			crate::assert!(
				exact(a_before.as_ref(), a.as_ref()),
				"{ctx}: the operand was modified"
			);
		}
	}
}
