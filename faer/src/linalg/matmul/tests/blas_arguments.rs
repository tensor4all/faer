// The OpenBLAS argument-error checks: `DCHKE` in test/dblat2.f (Level 2) and
// test/dblat3.f (Level 3), with the XERBLA shim they depend on,
// test/xerbla_test.c, and ctest/c_xerbla.c with ctest/cblas_test.h, distributed
// in OpenMathLib/OpenBLAS 791b02a1a00c996b66a894f6df93421c53b62d80. Source
// URLs:
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat2.f
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat3.f
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/ctest/cblas_test.h
// Original credits, from the file headers: Level 2 `DCHKE` was written on
// 10-August-1987 by Richard Hanson (Sandia National Labs) and Jeremy Du Croz
// (NAG Central Office); Level 3 `DCHKE` on 8-February-1989 by Jack Dongarra
// (Argonne National Laboratory), Iain Duff (AERE Harwell), Jeremy Du Croz and
// Sven Hammarling (Numerical Algorithms Group Ltd); `ctest/cblas_test.h` says
// "Written by Keita Teranishi", and `ctest/c_xerbla.c` carries no notice.
// Netlib BLAS test-program terms: faer/COPYING.BLAS.NETLIB. This Rust
// adaptation is new code under the crate's MIT licence.
//
// Material change: reference BLAS reports a bad argument by calling XERBLA with
// the argument's position, and the source test intercepts that call and
// compares the index. faer has no error return and no argument index; it
// rejects a bad shape with a panic that names the violated precondition, so
// each check below asserts the panic and its message instead of the index. What
// is ported is the rejection, not the reporting mechanism.
//
// The message is part of the assertion: a bare `should_panic` would also pass
// on an unrelated panic inside the call.
//
// `DCHKE` calls each routine with one argument deliberately inconsistent and
// relies on the routine rejecting it before the one-element arrays it passes
// are touched. faer cannot be exercised that way once its check is removed, so
// the shapes here are picked so that the unguarded call reads *less* than the
// arrays hold: the non-vacuity runs remove a check and must stay in bounds.
// That is a constraint on running the mutant, not a property of `DCHKE`.
//
// `matmul_with_conj`'s structure check (`dst_nrows == dst_ncols` for a
// non-dense destination) has no arm here: the unguarded kernel is not provably
// in-bounds for a non-square triangular destination, so removing that check is
// not a safe mutation to run.

#[test]
#[should_panic(expected = "lhs_ncols == rhs_nrows")]
fn matmul_panics_on_an_incompatible_inner_dimension() {
	let mut dst = Mat::<Scalar>::from_fn(2, 2, |_, _| one());
	let lhs = Mat::<Scalar>::from_fn(2, 2, |_, _| one());
	let rhs = Mat::<Scalar>::from_fn(3, 2, |_, _| one());
	matmul(&mut dst, Accum::Replace, lhs.as_ref(), rhs.as_ref(), one(), Par::Seq);
}

#[test]
#[should_panic(expected = "self.nrows().unbound() == nrows.unbound()")]
fn inner_prod_panics_on_operands_of_different_length() {
	let lhs = Col::<Scalar>::from_fn(3, |_| one());
	let rhs = Col::<Scalar>::from_fn(4, |_| one());
	let _ = inner_prod(
		lhs.as_ref().transpose(),
		Conj::No,
		rhs.as_ref(),
		Conj::No,
	);
}

#[test]
#[should_panic(expected = "rhs.nrows() == triangular_lower.ncols()")]
fn triangular_solve_panics_on_a_mismatched_right_hand_side() {
	let lower = Mat::<Scalar>::from_fn(3, 3, |i, j| {
		if i == j { one() } else { zero() }
	});
	let mut rhs = Mat::<Scalar>::from_fn(2, 1, |_, _| one());
	crate::linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
		lower.as_ref(),
		Conj::No,
		rhs.as_mut(),
		Par::Seq,
	);
}
