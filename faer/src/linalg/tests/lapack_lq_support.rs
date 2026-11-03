// QRT01/02/03 properties read for the transposed factorization: a native scalar
// elementary-Householder product replaces ORGLQ/UNGLQ as the oracle, and the
// deterministic fixture replaces LATMS/LATMR, as in the sibling QR family.
// Adapted LAPACK LIN numerical properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

// Independent elementary-reflector product of the transposed factorization: the
// reflector of step `r` acts on the ambient index range `r..n` and its essential
// part is stored along row `r` of the packed matrix. faer stores `1 / tau` on
// each block diagonal of the coefficient matrix. No production Q-application,
// reconstruction or null-space formation defines this oracle.
fn scalar_lq(
	packed: MatRef<'_, Scalar>,
	coeff: MatRef<'_, Scalar>,
	q_rows: usize,
) -> Mat<Scalar> {
	let n = packed.ncols();
	let k = coeff.ncols();
	let bs = coeff.nrows();
	// $Q^\top$ in the ambient $n$-space, as the product of the reflectors.
	let mut q = Mat::<Scalar>::identity(n, n);
	for r in (0..k).rev() {
		let tau = real(&coeff[(r % bs, r)]);
		let inv = 1.0 / tau;
		if inv == 0.0 {
			continue;
		}
		for j in 0..n {
			let mut dot = q[(r, j)];
			for i in r + 1..n {
				dot += conj(&packed[(r, i)]) * q[(i, j)];
			}
			dot *= inv;
			q[(r, j)] -= dot;
			for i in r + 1..n {
				q[(i, j)] -= packed[(r, i)] * dot;
			}
		}
	}
	// the first `q_rows` rows of $Q$ are the first `q_rows` columns of $Q^\top$.
	Mat::from_fn(q_rows, n, |i, j| q[(j, i)])
}
// $Q$ has orthonormal rows, so its deviation from an isometry is measured on
// $Q Q^H$ (the sibling family's `unitary_ratio` measures $Q^H Q$, which is the
// same only when $Q$ is square).
fn row_unitary_ratio(q: MatRef<'_, Scalar>, eps: Real) -> Real {
	let gram = Mat::from_fn(q.nrows(), q.nrows(), |i, j| {
		let mut v = z(if i == j { 1.0 } else { 0.0 });
		for k in 0..q.ncols() {
			v -= q[(i, k)] * conj(&q[(j, k)]);
		}
		v
	});
	norm_one(gram.as_ref()) / (q.ncols().max(1) as Real * eps)
}
fn lq_fixture(m: usize, n: usize, kind: usize) -> Mat<Scalar> {
	if kind == 0 {
		return Mat::zeros(m, n);
	}
	if kind == 4 {
		return Mat::from_fn(m, n, |i, j| phase(i) * (j + 1) as Real);
	}
	let scale = match kind {
		6 => Real::MAX.sqrt() / 16.0,
		7 => Real::MIN_POSITIVE.sqrt() * 16.0,
		_ => 1.0,
	};
	let q = unitary(m);
	let v = unitary(n);
	let d = Mat::from_fn(m, n, |i, j| {
		if i == j {
			z(scale
				* match kind {
					1 => 1.0,
					2 => 1.0 + i as Real / m.min(n).max(1) as Real,
					3 => Real::EPSILON
						.sqrt()
						.powf(i as Real / m.min(n).max(1) as Real),
					5 => {
						if i + 1 == m.min(n) {
							0.0
						} else {
							1.0
						}
					},
					_ => 1.0 + i as Real / m.min(n).max(1) as Real,
				})
		} else {
			z(0.0)
		}
	});
	product(
		product(q.as_ref(), d.as_ref()).as_ref(),
		adjoint(v.as_ref()).as_ref(),
	)
}
#[test]
fn lapack_lq_reflector_checkers() {
	// one reflector on the ambient index range $0..2$, essential in row 0.
	let packed =
		Mat::from_fn(2, 2, |i, j| if i == 0 && j > 0 { phase(0) } else { guard() });
	let coeff = Mat::full(1, 1, z(1.0));
	let q = scalar_lq(packed.as_ref(), coeff.as_ref(), 1);
	crate::assert!(row_unitary_ratio(q.as_ref(), lin_eps()) < 30.0);
	let bad = Mat::full(1, 1, z(2.0));
	crate::assert!(
		row_unitary_ratio(
			scalar_lq(packed.as_ref(), bad.as_ref(), 1).as_ref(),
			lin_eps()
		) > 30.0
	);
}

// The linked LAPACK of the opt-in cross-check (the same selection the examples
// use): the `-src` crates only carry the link directives.
#[cfg(any(openblas, mkl, blis))]
#[allow(unused_extern_crates)]
extern crate lapack_sys;
#[cfg(blis)]
#[allow(unused_extern_crates)]
extern crate blis_src;
// BLIS provides no `?gelqf`/`?orglq`, so the BLIS configuration takes LAPACK from
// the netlib `lapack-src` its dev-dependency declares (as `examples/bench.rs`
// does).
#[cfg(blis)]
#[allow(unused_extern_crates)]
extern crate lapack_src;
#[cfg(mkl)]
#[allow(unused_extern_crates)]
extern crate intel_mkl_src;
#[cfg(openblas)]
#[allow(unused_extern_crates)]
extern crate openblas_src;

// LAPACK entry points used by the opt-in cross-check. The `as _` casts take the
// pointer type from each routine's own signature, so the four arms compile for
// every scalar instantiation and only the one matching `Scalar` runs.
#[cfg(any(openblas, mkl, blis))]
unsafe fn gelqf(
	m: &i32,
	n: &i32,
	a: *mut Scalar,
	lda: &i32,
	tau: *mut Scalar,
	work: *mut Scalar,
	lwork: &i32,
	info: *mut i32,
) {
	use lapack_sys as la;
	unsafe {
		if <Scalar as ComplexField>::IS_NATIVE_F32 {
			la::sgelqf_(m, n, a as _, lda, tau as _, work as _, lwork, info);
		} else if <Scalar as ComplexField>::IS_NATIVE_F64 {
			la::dgelqf_(m, n, a as _, lda, tau as _, work as _, lwork, info);
		} else if <Scalar as ComplexField>::IS_NATIVE_C32 {
			la::cgelqf_(m, n, a as _, lda, tau as _, work as _, lwork, info);
		} else {
			la::zgelqf_(m, n, a as _, lda, tau as _, work as _, lwork, info);
		}
	}
}
#[cfg(any(openblas, mkl, blis))]
unsafe fn orglq(
	m: &i32,
	n: &i32,
	k: &i32,
	a: *mut Scalar,
	lda: &i32,
	tau: *mut Scalar,
	work: *mut Scalar,
	lwork: &i32,
	info: *mut i32,
) {
	use lapack_sys as la;
	unsafe {
		if <Scalar as ComplexField>::IS_NATIVE_F32 {
			la::sorglq_(m, n, k, a as _, lda, tau as _, work as _, lwork, info);
		} else if <Scalar as ComplexField>::IS_NATIVE_F64 {
			la::dorglq_(m, n, k, a as _, lda, tau as _, work as _, lwork, info);
		} else if <Scalar as ComplexField>::IS_NATIVE_C32 {
			la::cunglq_(m, n, k, a as _, lda, tau as _, work as _, lwork, info);
		} else {
			la::zunglq_(m, n, k, a as _, lda, tau as _, work as _, lwork, info);
		}
	}
}
