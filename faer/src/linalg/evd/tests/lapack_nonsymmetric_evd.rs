// CHKHS eigenvector / GET22 properties adapted from LAPACK 3.12.0,
// OpenBLAS 791b02a1a00c996b66a894f6df93421c53b62d80.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD. Euclidean complex moduli and faer's unit-2-norm
// normalization replace GET22's component/max-entry normalization convention.
type C = num_complex::Complex<Real>;
fn cx(x: Scalar) -> C {
	C::new(real(&x), imag(&x))
}
fn decoded(x: MatRef<'_, Scalar>, wi: &[Real], i: usize, j: usize) -> C {
	if !<Scalar as ComplexField>::IS_REAL {
		cx(x[(i, j)])
	} else if wi[j] > 0.0 {
		C::new(real(&x[(i, j)]), real(&x[(i, j + 1)]))
	} else if wi[j] < 0.0 {
		C::new(real(&x[(i, j - 1)]), -real(&x[(i, j)]))
	} else {
		C::new(real(&x[(i, j)]), 0.0)
	}
}
fn eigenvector_ratios(
	a: MatRef<'_, Scalar>,
	x: MatRef<'_, Scalar>,
	wr: &[Real],
	wi: &[Real],
	left: bool,
) -> (Real, Real) {
	let n = a.nrows();
	if wr.iter().chain(wi).any(|v| !v.is_finite()) || !max_entry(x).is_finite()
	{
		return (Real::INFINITY, Real::INFINITY);
	}
	if <Scalar as ComplexField>::IS_REAL {
		for j in 0..n {
			if (wi[j] > 0.0 && (j + 1 == n || wi[j + 1] != -wi[j]))
				|| (wi[j] < 0.0 && (j == 0 || wi[j - 1] != -wi[j]))
			{
				return (Real::INFINITY, Real::INFINITY);
			}
		}
	}
	let scale = max_entry(a);
	if !scale.is_finite() {
		return (Real::INFINITY, Real::INFINITY);
	}
	let scale = if scale == 0.0 { 1.0 } else { scale };
	let entry = |i: usize, j: usize| {
		if left {
			cx(a[(j, i)]).conj() / scale
		} else {
			cx(a[(i, j)]) / scale
		}
	};
	let mut an = 0.0 as Real;
	for j in 0..n {
		let mut col = 0.0;
		for i in 0..n {
			col += entry(i, j).norm();
		}
		an = an.max(col);
	}
	let mut err = 0.0 as Real;
	let mut xn = 0.0 as Real;
	let mut normerr = 0.0 as Real;
	for j in 0..n {
		let mut col = 0.0;
		let mut xcol = 0.0;
		let mut two = 0.0 as Real;
		let lambda = if left {
			C::new(wr[j], -wi[j])
		} else {
			C::new(wr[j], wi[j])
		} / scale;
		for i in 0..n {
			let v = decoded(x, wi, i, j);
			let mut r = -v * lambda;
			xcol += v.norm();
			two = two.hypot(v.norm());
			for k in 0..n {
				r += entry(i, k) * decoded(x, wi, k, j);
			}
			col += r.norm();
		}
		if !col.is_finite() || !two.is_finite() {
			return (Real::INFINITY, Real::INFINITY);
		}
		err = err.max(col);
		xn = xn.max(xcol);
		normerr = normerr.max((two - 1.0).abs());
	}
	let residual = if an == 0.0 {
		if err == 0.0 { 0.0 } else { Real::INFINITY }
	} else {
		err / an / xn.max(Real::EPSILON) / Real::EPSILON
	};
	(residual, normerr / (n.max(1) as Real * Real::EPSILON))
}
/// The threshold of the GET22 residual ratio at size `n` (#112, maintainer decision 2026-10-29):
/// `20 * max(1, sqrt(n / 16))`.
///
/// CHKHS holds the ratio to 20 only at `n <= 16` (LAPACK 3.12.0 `nep.in`: N = 0, 1, 2, 3, 5, 10,
/// 16), and GET22's ratio has no dimension factor, so 20 carried to n = 63..129 also fails
/// LAPACK's own `dgeev` (17.5 to 34.4 on these fixtures). At `n <= 16` this is LAPACK's 20. The
/// square root grows like an accumulated rounding error and still catches a residual that grows
/// with `n`: the clamp defect fixed in #121 gave `(2n - 1) / 2`, 62.5 at n = 63 against 39.7 here,
/// where a linear `20 * n / 16` (78.75) would have let it pass.
fn get22_threshold(n: usize) -> Real {
	20.0 * (n as Real / 16.0).sqrt().max(1.0)
}
#[test]
fn lapack_nonsymmetric_evd_threshold_keeps_a_residual_growing_with_n_visible() {
	crate::assert!(get22_threshold(0) == 20.0 && get22_threshold(16) == 20.0);
	// the #121 clamp defect at n = 63 fails, LAPACK's dgeev on the same fixtures passes
	let t63 = get22_threshold(63);
	crate::assert!(62.5 > t63 && 28.6 < t63);
	crate::assert!(34.4 < get22_threshold(129));
	// a linear scaling would not have caught it
	crate::assert!(62.5 < 20.0 * 63.0 / 16.0);
}
// Each output mode is an independently rounded spectrum, and the CHKHS
// budget holds each one to 20 epsilons of the exact spectrum (scaled by
// `||A||_1`). Two such spectra can then differ by the sum of their budgets,
// so the mode-vs-mode comparison uses the composed budget (#78). The residual
// and unit-norm checks keep their own threshold of 20.
const MODE_AGREEMENT_BUDGET: Real = 2.0 * 20.0;
#[cfg(feature = "rayon")]
#[test]
fn lapack_nonsymmetric_evd_mode_agreement_budget_composes() {
	let eps = Real::EPSILON;
	// 1 + 15 eps and 1 - 15 eps are both within 20 of the exact 1, and their
	// gap of 30 is within the composed budget but not within 20.
	let above = [(1.0 + 15.0 * eps, 0.0)];
	let below = [(1.0 - 15.0 * eps, 0.0)];
	crate::assert!(spectrum_matches_scaled(&above, &[(1.0, 0.0)], 1.0, 20.0));
	crate::assert!(spectrum_matches_scaled(&below, &[(1.0, 0.0)], 1.0, 20.0));
	crate::assert!(!spectrum_matches_scaled(&above, &below, 1.0, 20.0));
	crate::assert!(spectrum_matches_scaled(
		&above,
		&below,
		1.0,
		MODE_AGREEMENT_BUDGET
	));
	// a gap beyond both budgets combined is still rejected
	crate::assert!(!spectrum_matches_scaled(
		&[(1.0 + 25.0 * eps, 0.0)],
		&[(1.0 - 25.0 * eps, 0.0)],
		1.0,
		MODE_AGREEMENT_BUDGET
	));
}
#[test]
fn lapack_nonsymmetric_evd_checkers_require_pair_signs() {
	let a = Mat::from_fn(2, 2, |i, j| {
		z(if i == j {
			0.0
		} else if i < j {
			-1.0
		} else {
			1.0
		})
	});
	let r = (0.5 as Real).sqrt();
	let mut x = if <Scalar as ComplexField>::IS_REAL {
		Mat::from_fn(2, 2, |i, j| {
			z(if i == j {
				if i == 0 { r } else { -r }
			} else {
				0.0
			})
		})
	} else {
		Mat::from_fn(2, 2, |i, j| {
			if i == 0 {
				z(r)
			} else {
				scalar(0.0, if j == 0 { -r } else { r })
			}
		})
	};
	let (res, norm) = eigenvector_ratios(
		a.as_ref(),
		x.as_ref(),
		&[0.0, 0.0],
		&[1.0, -1.0],
		false,
	);
	crate::assert!(res < 20.0 && norm < 20.0);
	x[(1, 1)] = -x[(1, 1)];
	if !<Scalar as ComplexField>::IS_REAL {
		x[(1, 0)] = -x[(1, 0)];
	}
	crate::assert!(
		eigenvector_ratios(
			a.as_ref(),
			x.as_ref(),
			&[0.0, 0.0],
			&[1.0, -1.0],
			false
		)
		.0 > 20.0
	);
}
fn full_evd_case(
	run: &mut Run,
	a: MatRef<'_, Scalar>,
	layout: usize,
	mut par: Par,
	threshold: usize,
	kind: usize,
) {
	let n = a.nrows();
	let mut ap = guarded(n, n, layout);
	let mut lp = guarded(n, n, layout);
	let mut rp = guarded(n, n, layout);
	interior_mut(lp.as_mut(), layout).fill(z(1.0));
	interior_mut(rp.as_mut(), layout).fill(z(1.0));
	let mut wr = Col::full(n, Real::NAN);
	let mut wi = Col::full(n, Real::NAN);
	let mut p = EvdParams { ..auto!(Scalar) };
	p.evd_from_schur.recursion_threshold = threshold;
	let p = Spec::new(p);
	let mut modes: [Option<Vec<(Real, Real)>>; 2] = [None, None];
	for (left, right) in
		[(true, true), (false, true), (true, false), (false, false)]
	{
		let mut mem = MemBuffer::new(evd_scratch::<Scalar>(
			n,
			if left {
				ComputeEigenvectors::Yes
			} else {
				ComputeEigenvectors::No
			},
			if right {
				ComputeEigenvectors::Yes
			} else {
				ComputeEigenvectors::No
			},
			par.shape(),
			p,
		));
		for pass in 0..2 {
			let input = scaled(a, if pass == 0 { 1.0 } else { 0.5 });
			interior_mut(ap.as_mut(), layout).copy_from(&input);
			let ctx = format!(
				"EVD kind={kind} n={n} layout={layout} left={left} right={right} threshold={threshold} pass={pass} par={par:?}"
			);
			wr.as_mut().fill(Real::NAN);
			wi.as_mut().fill(Real::NAN);
			if run.call(&ctx, || {
				evd_call(
					interior(ap.as_ref(), layout),
					wr.as_mut(),
					wi.as_mut(),
					if left {
						Some(interior_mut(lp.as_mut(), layout))
					} else {
						None
					},
					if right {
						Some(interior_mut(rp.as_mut(), layout))
					} else {
						None
					},
					par.reborrow(),
					MemStack::new(&mut mem),
					p,
				)
			}) {
				let wr: Vec<_> = wr.iter().copied().collect();
				let wi: Vec<_> = wi.iter().copied().collect();
				run.require(
					&ctx,
					"nonfinite eigenvalues",
					wr.iter().chain(&wi).all(|v| v.is_finite()),
				);
				let values: Vec<_> =
					wr.iter().copied().zip(wi.iter().copied()).collect();
				if [0, 1, 2, 4, 6, 7].contains(&kind) {
					if let Some(ref before) = modes[pass] {
						run.require(
							&ctx,
							"eigenvalue modes disagree",
							spectrum_matches_scaled(
								&values,
								before,
								norm_one(input.as_ref()),
								MODE_AGREEMENT_BUDGET,
							),
						);
					} else {
						modes[pass] = Some(values);
					}
				}
				for (want, x, is_left) in [
					(left, interior(lp.as_ref(), layout), true),
					(right, interior(rp.as_ref(), layout), false),
				] {
					if want {
						let (res, norm) = eigenvector_ratios(
							input.as_ref(),
							x,
							&wr,
							&wi,
							is_left,
						);
						run.ratio(
							&ctx,
							if is_left {
								"left residual"
							} else {
								"right residual"
							},
							res,
							get22_threshold(input.nrows()),
						);
						run.ratio(&ctx, "vector unit 2-norm", norm, 20.0);
					}
				}
			}
			run.require(
				&ctx,
				"immutable input changed",
				same_bits(input.as_ref(), interior(ap.as_ref(), layout)),
			);
			run.require(
				&ctx,
				"guard overwritten",
				guards_ok(ap.as_ref())
					&& guards_ok(lp.as_ref())
					&& guards_ok(rp.as_ref()),
			);
		}
	}
}
fn triangular_schur(
	n: usize,
	kind: usize,
) -> (Mat<Scalar>, Vec<Real>, Vec<Real>) {
	let mut t = hessenberg_input(n, kind);
	if !<Scalar as ComplexField>::IS_REAL {
		for i in 0..n {
			for j in 0..i {
				t[(i, j)] = z(0.0);
			}
			if kind == 4 {
				t[(i, i)] = scalar(1.0, if i % 2 == 0 { 1.0 } else { -1.0 });
			}
		}
	} else if kind == 3 {
		for j in 0..n {
			for i in 0..n {
				t[(i, j)] = z(if i == j || j == i + 1 { 1.0 } else { 0.0 });
			}
		}
	}
	let mut wr = vec![0.0; n];
	let mut wi = vec![0.0; n];
	for i in 0..n {
		wr[i] = real(&t[(i, i)]);
		wi[i] = imag(&t[(i, i)]);
	}
	if <Scalar as ComplexField>::IS_REAL && kind == 4 {
		for i in (0..n.saturating_sub(1)).step_by(2) {
			wi[i] = 1.0;
			wi[i + 1] = -1.0;
		}
	}
	(t, wr, wi)
}
fn from_schur_case(
	run: &mut Run,
	t: MatRef<'_, Scalar>,
	wr: &[Real],
	wi: &[Real],
	threshold: usize,
	kind: usize,
) {
	let n = t.nrows();
	let q = unitary(n);
	let a = product(
		product(q.as_ref(), t).as_ref(),
		adjoint(q.as_ref()).as_ref(),
	);
	let p = Spec::new(EvdFromSchurParams {
		recursion_threshold: threshold,
		..auto!(Scalar)
	});
	let mut x = Mat::full(n, n, guard());
	let mut mem = MemBuffer::new(eigenvectors_from_schur_scratch::<Scalar>(n));
	for want_q in [false, true] {
		let ctx = format!(
			"Schur vectors kind={kind} n={n} Q={want_q} threshold={threshold}"
		);
		if run.call(&ctx, || {
			vectors_call(
				t,
				if want_q { Some(q.as_ref()) } else { None },
				x.as_mut(),
				Par::Seq,
				MemStack::new(&mut mem),
				p,
			)
		}) {
			let (res, norm) = eigenvector_ratios(
				if want_q { a.as_ref() } else { t },
				x.as_ref(),
				wr,
				wi,
				false,
			);
			run.ratio(&ctx, "right residual", res, get22_threshold(x.nrows()));
			run.ratio(&ctx, "unit 2-norm", norm, 20.0);
		}
	}
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_nonsymmetric_evd_cases() {
	let mut run = Run::new("nonsymmetric_evd");
	for n in [0, 1, 2, 3, 5, 10, 16, 63, 64, 65] {
		for kind in 0..9 {
			let a = general_case(n, kind);
			for threshold in [4, 64] {
				full_evd_case(
					&mut run,
					a.as_ref(),
					0,
					Par::Seq,
					threshold,
					kind,
				);
			}
		}
	}
	for n in [3, 16] {
		for kind in [4, 8] {
			let a = general_case(n, kind);
			for layout in 1..4 {
				full_evd_case(&mut run, a.as_ref(), layout, Par::Seq, 4, kind);
			}
		}
	}
	for n in [0, 1, 2, 3, 5, 10, 16, 63, 64, 65] {
		for kind in [0, 1, 2, 3, 4, 5] {
			let (t, wr, wi) = triangular_schur(n, kind);
			for threshold in [4, 64] {
				from_schur_case(
					&mut run,
					t.as_ref(),
					&wr,
					&wi,
					threshold,
					kind,
				);
			}
		}
	}
	run.finish(10 * 9 * 2 * 8 + 2 * 2 * 3 * 8 + 10 * 6 * 2 * 2);
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_nonsymmetric_evd_parallel() {
	let mut run = Run::new("nonsymmetric_evd_parallel");
	let a = general_case(129, 4);
	full_evd_case(&mut run, a.as_ref(), 0, Par::Seq, 64, 4);
	for threads in [2, 4] {
		crate::team::with_team(threads, |par| {
			full_evd_case(
				&mut run,
				a.as_ref(),
				0,
				par,
				64,
				4,
			)
		});
	}
	run.finish(24);
}

#[test]
fn lapack_nonsymmetric_evd_checkers_require_left_adjoint() {
	let beta = scalar(
		1.0,
		if <Scalar as ComplexField>::IS_REAL {
			0.0
		} else {
			1.0
		},
	);
	let a = Mat::from_fn(2, 2, |i, j| {
		if i == j {
			z((i + 2) as Real)
		} else if i < j {
			beta
		} else {
			z(0.0)
		}
	});
	let scale = (1.0 + magnitude(beta) * magnitude(beta)).sqrt();
	let x = Mat::from_fn(2, 2, |i, j| match (i, j) {
		(0, 0) => z(1.0) / scale,
		(1, 0) => -conj(&beta) / scale,
		(1, 1) => z(1.0),
		_ => z(0.0),
	});
	let r = eigenvector_ratios(
		a.as_ref(),
		x.as_ref(),
		&[2.0, 3.0],
		&[0.0, 0.0],
		true,
	);
	crate::assert!(r.0 < 20.0 && r.1 < 20.0);
	crate::assert!(
		eigenvector_ratios(
			a.as_ref(),
			x.as_ref(),
			&[2.0, 3.0],
			&[0.0, 0.0],
			false
		)
		.0 > 20.0
	);
	if !<Scalar as ComplexField>::IS_REAL {
		let a = Mat::full(1, 1, scalar(2.0, 1.0));
		let x = Mat::<Scalar>::identity(1, 1);
		assert_eq!(
			eigenvector_ratios(a.as_ref(), x.as_ref(), &[2.0], &[1.0], true).0,
			0.0
		);
	}
}
/// #112: the back substitution clamped a vanishing shift to `eps * sum |t_ij|`, which grows with
/// `n`, so on a Jordan block (kind 3) and a tightly clustered triangle (kind 5) the GET22 residual
/// ratio grew like `n` (62.5 at n = 63, where LAPACK's `dgeev` gives 0.5 on the same Jordan block).
/// The clamp is now relative to the eigenvalue and the largest entry. Run with the regular tests,
/// at the sizes of the opt-in family that failed, both eigenvector sides, CHKHS's threshold.
#[test]
fn lapack_nonsymmetric_evd_clustered_residuals_do_not_grow_with_n() {
	for n in [63, 65] {
		for kind in [3, 5] {
			let a = general_case(n, kind);
			let mut l = Mat::<Scalar>::zeros(n, n);
			let mut r = Mat::<Scalar>::zeros(n, n);
			let mut wr = Col::<Real>::zeros(n);
			let mut wi = Col::<Real>::zeros(n);
			let p = Spec::new(EvdParams { ..auto!(Scalar) });
			let mut mem = MemBuffer::new(evd_scratch::<Scalar>(
				n,
				ComputeEigenvectors::Yes,
				ComputeEigenvectors::Yes,
				Par::Seq.shape(),
				p,
			));
			evd_call(
				a.as_ref(),
				wr.as_mut(),
				wi.as_mut(),
				Some(l.as_mut()),
				Some(r.as_mut()),
				Par::Seq,
				MemStack::new(&mut mem),
				p,
			)
			.unwrap();
			let wr: Vec<_> = wr.iter().copied().collect();
			let wi: Vec<_> = wi.iter().copied().collect();
			for (x, left) in [(l.as_ref(), true), (r.as_ref(), false)] {
				let (res, _) = eigenvector_ratios(a.as_ref(), x, &wr, &wi, left);
				crate::assert!(res < 20.0, "kind {kind}, n {n}, left {left}: residual ratio {res}");
			}
		}
	}
}
/// A strongly graded matrix `D A D^-1`, `D = diag(10^linspace(-6, 6, n))` (#145). Its Schur form
/// spans many orders of magnitude, so after `scaled_into_range` the 2 x 2 blocks of the
/// complex pairs have entries far below one. The back substitution compared the determinant of
/// such a block with the entry-sized clamp, replaced the determinant of every block and returned
/// a wrong vector for each complex pair: a GET22 ratio of `5.9e8` here at `f64` n = 16. On the `f64` n = 64 Schur form,
/// LAPACK's `dtrevc3` gives `2.2e-4` and the fixed back substitution `3.1e-4`.
///
/// The second matrix also fails upstream: two complex pairs `+-1e-5 i` and `+-2e-5 i` coupled
/// by `1e8`, so the shifted block's determinant (about `1e-10`) is below an entry-sized clamp
/// although its entries are not. Before the fix: `1.5e12` (`f64`); `dgeev` gives `9.6e-14`.
#[test]
fn lapack_nonsymmetric_evd_graded_pairs_keep_their_residual() {
	let coupled = Mat::from_fn(4, 4, |i, j| {
		scalar(
			match (i, j) {
				(0, 1) => 1e-5,
				(1, 0) => -1e-5,
				(2, 3) => 2e-5,
				(3, 2) => -2e-5,
				(0 | 1, 2 | 3) => 1e8,
				_ => 0.0,
			},
			0.0,
		)
	});
	let graded = |n: usize| {
		let d = |i: usize| (10.0 as Real).powf(-6.0 + 12.0 * i as Real / (n - 1) as Real);
		Mat::from_fn(n, n, |i, j| sample(i, j) * scalar(d(i) / d(j), 0.0))
	};
	for a in [coupled, graded(16), graded(64), graded(128)] {
		let n = a.nrows();
		let mut l = Mat::<Scalar>::zeros(n, n);
		let mut r = Mat::<Scalar>::zeros(n, n);
		let mut wr = Col::<Real>::zeros(n);
		let mut wi = Col::<Real>::zeros(n);
		let p = Spec::new(EvdParams { ..auto!(Scalar) });
		let mut mem = MemBuffer::new(evd_scratch::<Scalar>(
			n,
			ComputeEigenvectors::Yes,
			ComputeEigenvectors::Yes,
			Par::Seq.shape(),
			p,
		));
		evd_call(
			a.as_ref(),
			wr.as_mut(),
			wi.as_mut(),
			Some(l.as_mut()),
			Some(r.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			p,
		)
		.unwrap();
		let wr: Vec<_> = wr.iter().copied().collect();
		let wi: Vec<_> = wi.iter().copied().collect();
		for (x, left) in [(l.as_ref(), true), (r.as_ref(), false)] {
			let (res, _) = eigenvector_ratios(a.as_ref(), x, &wr, &wi, left);
			crate::assert!(
				res < get22_threshold(n),
				"n {n}, left {left}: residual ratio {res}"
			);
		}
	}
}
/// A standardized real Schur form whose upper 2 x 2 block, shifted by the lower pair's
/// eigenvalue `i q`, has a small determinant (about `-2^-29`) next to large entries (`2^19`).
/// Raising only the determinant to a floor while keeping the block's adjugate gives a residual of
/// about `1e14`; `?LALN2` instead raises the second pivot of its elimination and solves that
/// perturbed system (#145 review). Real types only: the complex adapters take a triangular form.
#[test]
fn lapack_nonsymmetric_evd_pair_solve_with_a_small_determinant() {
	if !<Scalar as ComplexField>::IS_REAL {
		return;
	}
	let m = (2.0 as Real).powi(19);
	let q = 1.0 + (2.0 as Real).powi(-30);
	let t = Mat::from_fn(4, 4, |i, j| {
		scalar(
			match (i, j) {
				(0, 1) | (0, 2) => m,
				(1, 0) => -1.0 / m,
				(1, 3) => -1.0,
				(2, 3) => q,
				(3, 2) => -q,
				_ => 0.0,
			},
			0.0,
		)
	});
	let mut x = Mat::<Scalar>::zeros(4, 4);
	let p = Spec::new(EvdFromSchurParams { ..auto!(Scalar) });
	let mut mem = MemBuffer::new(eigenvectors_from_schur_scratch::<Scalar>(4));
	vectors_call(t.as_ref(), None, x.as_mut(), Par::Seq, MemStack::new(&mut mem), p).unwrap();
	let wr = [0.0 as Real; 4];
	let wi = [1.0, -1.0, q, -q];
	let (res, _) = eigenvector_ratios(t.as_ref(), x.as_ref(), &wr, &wi, false);
	crate::assert!(res < get22_threshold(4), "residual ratio {res}");
}
/// A 2 x 2 block `[[0, t], [-t, 0]]` above a real eigenvalue 0, with `t` below the square root of
/// the smallest positive number. Inverting the shifted block through `t^2` underflowed and gave
/// a nonfinite eigenvector (`NoConvergence`); `?LALN2`'s elimination never forms `t^2` (#145
/// review). Real types only, as above.
#[test]
fn lapack_nonsymmetric_evd_real_shift_block_below_the_square_root_of_min_positive() {
	if !<Scalar as ComplexField>::IS_REAL {
		return;
	}
	let t = if core::mem::size_of::<Real>() == 8 {
		(2.0 as Real).powi(-600)
	} else {
		(2.0 as Real).powi(-80)
	};
	let a = Mat::from_fn(3, 3, |i, j| {
		scalar(
			match (i, j) {
				(0, 1) => t,
				(1, 0) => -t,
				(0, 2) => 1.0,
				_ => 0.0,
			},
			0.0,
		)
	});
	let mut x = Mat::<Scalar>::zeros(3, 3);
	let p = Spec::new(EvdFromSchurParams { ..auto!(Scalar) });
	let mut mem = MemBuffer::new(eigenvectors_from_schur_scratch::<Scalar>(3));
	vectors_call(a.as_ref(), None, x.as_mut(), Par::Seq, MemStack::new(&mut mem), p).unwrap();
	let (res, _) = eigenvector_ratios(a.as_ref(), x.as_ref(), &[0.0; 3], &[t, -t, 0.0], false);
	crate::assert!(res < get22_threshold(3), "residual ratio {res}");
}
