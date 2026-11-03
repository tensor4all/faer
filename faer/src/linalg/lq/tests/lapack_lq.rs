// LQT01/02/03 (the transposed QRT01/02/03): reconstruction, orthonormality of the
// compact and full Q, the null-space block, and the positive diagonal gauge.
// Adapted LAPACK LIN numerical properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace
// LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

use crate::linalg::lq::no_pivoting::{factor, reconstruct, unitary};
#[allow(clippy::too_many_arguments)]
fn lq_case(
	run: &mut Run,
	m: usize,
	n: usize,
	kind: usize,
	layout: usize,
	bs: usize,
	mut par: Par,
) {
	let a = lq_fixture(m, n, kind);
	let k = m.min(n);
	let mut ap = guarded(m, n, layout);
	let mut out = guarded(m, n, layout);
	let mut l_out = guarded(m, k, layout);
	let mut l_full_out = guarded(m, n, layout);
	let mut q_out = guarded(k, n, layout);
	let mut q_full_out = guarded(n, n, layout);
	let mut null_out = guarded(n - k, n, layout);
	let mut coeff = Mat::full(bs, k, guard());
	let params = Spec::new(factor::LqParams { ..auto!(Scalar) });
	let mut mem = MemBuffer::new(factor::lq_in_place_scratch::<Scalar>(
		m,
		n,
		bs,
		par.shape(),
		params,
	));
	let mut rm = MemBuffer::new(reconstruct::reconstruct_scratch::<Scalar>(
		m,
		n,
		bs,
		par.shape(),
	));
	let mut qm = MemBuffer::new(unitary::unitary_scratch::<Scalar>(
		m,
		n,
		k,
		bs,
		par.shape(),
	));
	let mut qfm = MemBuffer::new(unitary::unitary_scratch::<Scalar>(
		m,
		n,
		n,
		bs,
		par.shape(),
	));
	let mut nm = MemBuffer::new(unitary::null_scratch::<Scalar>(
		m,
		n,
		bs,
		par.shape(),
	));
	let dim = Ord::max(m, n);
	for pass in 0..2 {
		let input = scaled(a.as_ref(), if pass == 0 { 1.0 } else { 0.5 });
		interior_mut(ap.as_mut(), layout).copy_from(&input);
		let ctx = format!(
			"LQ m={m} n={n} kind={kind} layout={layout} block={bs} pass={pass} par={par:?}"
		);
		let mut rank = None;
		if run.call(&ctx, || {
			rank = Some(
				factor::lq_in_place(
					interior_mut(ap.as_mut(), layout),
					coeff.as_mut(),
					par.reborrow(),
					MemStack::new(&mut mem),
					params,
				)
				.rank,
			);
			Ok::<_, ()>(())
		}) {
			run.require(
				&ctx,
				"invalid reported rank",
				rank.unwrap() <= k && (kind != 0 || rank.unwrap() == 0),
			);
			let packed = interior(ap.as_ref(), layout);
			let before = ap.clone();
			let cb = coeff.clone();
			// the independent oracle: $L$ from the lower triangle, $Q$ from the
			// elementary reflectors.
			let q = scalar_lq(packed, coeff.as_ref(), k);
			let l = Mat::from_fn(m, k, |i, j| {
				if i >= j {
					packed[(i, j)]
				} else {
					z(0.0)
				}
			});
			run.ratio(
				&ctx,
				"L-Q AP",
				lin_difference(
					product(l.as_ref(), q.as_ref()).as_ref(),
					input.as_ref(),
					dim,
				),
				30.0,
			);
			run.ratio(
				&ctx,
				"Q unitary",
				row_unitary_ratio(q.as_ref(), lin_eps()),
				30.0,
			);
			if run.aux(&ctx, || {
				reconstruct::reconstruct(
					interior_mut(out.as_mut(), layout),
					packed,
					coeff.as_ref(),
					par.reborrow(),
					MemStack::new(&mut rm),
				);
				Ok::<_, ()>(())
			}) {
				run.ratio(
					&ctx,
					"API reconstruction",
					lin_difference(
						interior(out.as_ref(), layout),
						input.as_ref(),
						dim,
					),
					30.0,
				);
			}
			for positive in [false, true] {
				if run.aux(&ctx, || {
					factor::copy_l(
						interior_mut(l_out.as_mut(), layout),
						packed,
						positive,
					);
					unitary::unitary(
						interior_mut(q_out.as_mut(), layout),
						packed,
						coeff.as_ref(),
						positive,
						par.reborrow(),
						MemStack::new(&mut qm),
					);
					Ok::<_, ()>(())
				}) {
					// the API factors are the oracle's, up to the block apply's
					// arithmetic order.
					// the oracle is the raw factorization, so it is compared with the
					// ungauged API factor only.
					if !positive {
						run.ratio(
							&ctx,
							"API unitary",
							lin_difference(
								interior(q_out.as_ref(), layout),
								q.as_ref(),
								dim,
							),
							30.0,
						);
					}
					run.ratio(
						&ctx,
						"API unitary orthonormal",
						row_unitary_ratio(interior(q_out.as_ref(), layout), lin_eps()),
						30.0,
					);
					let l_api = interior(l_out.as_ref(), layout);
					run.ratio(
						&ctx,
						"API L-Q",
						lin_difference(
							product(l_api, interior(q_out.as_ref(), layout))
								.as_ref(),
							input.as_ref(),
							dim,
						),
						30.0,
					);
					if positive {
						// the phase is exact, so the residue is the rounding of
						// $d / |d|$ against the entry's own magnitude; a zero entry
						// keeps the raw sign and is accepted as zero.
						let mut worst = 0.0 as Real;
						for j in 0..k {
							let d = l_api[(j, j)].clone();
							let abs = real(&d).hypot(imag(&d));
							if !(real(&d) >= 0.0) {
								worst = Real::INFINITY;
							} else if abs > 0.0 {
								worst = worst.max(imag(&d).abs() / abs);
							}
						}
						run.ratio(
							&ctx,
							"positive gauge diagonal",
							worst / lin_eps(),
							30.0,
						);
					}
				}
			}
			// the full $L$ is the compact one padded with zero columns.
			if run.aux(&ctx, || {
				factor::copy_l(
					interior_mut(l_full_out.as_mut(), layout),
					packed,
					false,
				);
				Ok::<_, ()>(())
			}) && n > k
			{
				let l_full = interior(l_full_out.as_ref(), layout);
				run.ratio(
					&ctx,
					"full L continues the compact L",
					lin_difference(l_full.subcols(0, k), l.as_ref(), dim),
					30.0,
				);
				run.require(
					&ctx,
					"full L trailing columns",
					max_entry(l_full.subcols(k, n - k)) == 0.0,
				);
			}
			// the full Q continues the compact one and its trailing rows are the
			// null block, which annihilates the input.
			if run.aux(&ctx, || {
				unitary::unitary(
					interior_mut(q_full_out.as_mut(), layout),
					packed,
					coeff.as_ref(),
					false,
					par.reborrow(),
					MemStack::new(&mut qfm),
				);
				unitary::null(
					interior_mut(null_out.as_mut(), layout),
					packed,
					coeff.as_ref(),
					par.reborrow(),
					MemStack::new(&mut nm),
				);
				Ok::<_, ()>(())
			}) {
				let q_full = interior(q_full_out.as_ref(), layout);
				let complement = interior(null_out.as_ref(), layout);
				run.ratio(&ctx, "full Q unitary", row_unitary_ratio(q_full, lin_eps()), 30.0);
				run.ratio(
					&ctx,
					"full Q continues the compact Q",
					lin_difference(q_full.subrows(0, k), q.as_ref(), dim),
					30.0,
				);
				run.ratio(
					&ctx,
					"null block",
					lin_difference(complement, q_full.subrows(k, n - k), dim),
					30.0,
				);
				let residual =
					product(input.as_ref(), adjoint(complement).as_ref());
				let scale = max_entry(input.as_ref()) * Ord::max(k, 1) as Real;
				run.ratio(
					&ctx,
					"A N^H",
					if scale == 0.0 {
						if max_entry(residual.as_ref()) == 0.0 {
							0.0
						} else {
							Real::INFINITY
						}
					} else {
						norm_one(residual.as_ref())
							/ (scale * lin_eps() * dim.max(1) as Real)
					},
					30.0,
				);
			}
			run.require(
				&ctx,
				"factors changed",
				same_bits(ap.as_ref(), before.as_ref())
					&& same_bits(coeff.as_ref(), cb.as_ref()),
			);
		}
		run.require(
			&ctx,
			"guards overwritten",
			guards_ok(ap.as_ref())
				&& guards_ok(out.as_ref())
				&& guards_ok(l_out.as_ref())
				&& guards_ok(q_out.as_ref())
				&& guards_ok(q_full_out.as_ref())
				&& guards_ok(l_full_out.as_ref())
				&& guards_ok(null_out.as_ref()),
		);
	}
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_lq_cases() {
	let mut run = Run::new("lq");
	for (m, n) in [
		(0, 0),
		(0, 3),
		(3, 0),
		(1, 1),
		(2, 2),
		(3, 5),
		(5, 3),
		(10, 10),
		(50, 50),
		(47, 47),
		(48, 48),
		(49, 49),
		(3, 3),
		(4, 4),
		(5, 5),
		(63, 63),
		(64, 64),
		(65, 65),
	] {
		for kind in 0..8 {
			for bs in [1, 4] {
				lq_case(
					&mut run,
					m,
					n,
					kind,
					(m + kind) % 4,
					bs,
					Par::Seq,
				);
			}
		}
	}
	run.finish(18 * 8 * 2 * 2);
}
/// Cross-check against LAPACK's own `?gelqf` and `?orglq`/`?unglq` on the same
/// fixture. Both sides are gauged (real nonnegative triangular diagonal) so the
/// factors are unique and comparable entry by entry, not only through a residual.
/// Opt-in for the same reason the linked examples are: it needs a LAPACK.
#[cfg(any(openblas, mkl, blis))]
#[test]
#[ignore = "opt-in LAPACK cross-check (#31): needs a linked LAPACK (--cfg openblas)"]
fn lapack_lq_reference() {
	for (m, n) in [(4, 4), (7, 4), (4, 7), (17, 33), (33, 17), (33, 33)] {
		let k = m.min(n);
		let a = lq_fixture(m, n, 2);
		let (m_i, n_i, k_i, lda) = (m as i32, n as i32, k as i32, a.col_stride() as i32);
		let mut info = 0;
		// the workspace query of `?gelqf`
		let mut tau = vec![z(0.0); k];
		let mut query = z(0.0);
		let mut packed = a.clone();
		unsafe {
			gelqf(
				&m_i, &n_i, packed.as_ptr_mut(), &lda, tau.as_mut_ptr(),
				&mut query, &(-1), &mut info,
			);
		}
		assert_eq!(info, 0, "LAPACK workspace query failed");
		let lwork = real(&query) as i32;
		let mut work = vec![z(0.0); Ord::max(lwork as usize, 1)];
		unsafe {
			gelqf(
				&m_i, &n_i, packed.as_ptr_mut(), &lda, tau.as_mut_ptr(),
				work.as_mut_ptr(), &lwork, &mut info,
			);
		}
		assert_eq!(info, 0, "LAPACK gelqf failed");
		let l_lapack = Mat::from_fn(m, k, |i, j| {
			if i >= j {
				packed[(i, j)]
			} else {
				z(0.0)
			}
		});
		// `?orglq` builds the first `k` rows of `Q` from the reflectors, in a
		// `k x n` array with leading dimension `k`.
		let mut q_lapack = Mat::from_fn(k, n, |i, j| packed[(i, j)]);
		let lda_q = q_lapack.col_stride() as i32;
		unsafe {
			orglq(
				&k_i, &n_i, &k_i, q_lapack.as_ptr_mut(), &lda_q,
				tau.as_mut_ptr(), work.as_mut_ptr(), &lwork, &mut info,
			);
		}
		assert_eq!(info, 0, "LAPACK orglq failed");
		// the same gauge on both sides: $L$'s column $j$ by $\overline{d_j}/|d_j|$
		// and $Q$'s row $j$ by $d_j/|d_j|$.
		let gauge = |l: &mut Mat<Scalar>, q: &mut Mat<Scalar>| {
			for j in 0..k {
				let d = l[(j, j)].clone();
				let abs = real(&d).hypot(imag(&d));
				let s = if abs == 0.0 {
					z(1.0)
				} else {
					scalar(real(&d) / abs, imag(&d) / abs)
				};
				for i in j..m {
					l[(i, j)] = l[(i, j)].clone() * conj(&s);
				}
				for i in 0..n {
					q[(j, i)] = q[(j, i)].clone() * s.clone();
				}
			}
		};
		let mut l_lapack = l_lapack;
		let mut q_lapack = q_lapack;
		gauge(&mut l_lapack, &mut q_lapack);

		// faer, with the same gauge through the public entry points.
		let params = Spec::new(factor::LqParams { ..auto!(Scalar) });
		let block_size =
			Ord::max(factor::recommended_block_size::<Scalar>(m, n).min(k.max(1)), 1);
		let mut packed_faer = a.clone();
		let mut coeff = Mat::<Scalar>::zeros(block_size, k);
		factor::lq_in_place(
			packed_faer.as_mut(),
			coeff.as_mut(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(factor::lq_in_place_scratch::<Scalar>(
				m,
				n,
				block_size,
				ParShape::Seq,
				params,
			))),
			params,
		);
		let mut l_faer = Mat::<Scalar>::zeros(m, k);
		factor::copy_l(l_faer.as_mut(), packed_faer.as_ref(), true);
		let mut q_faer = Mat::<Scalar>::zeros(k, n);
		unitary::unitary(
			q_faer.as_mut(),
			packed_faer.as_ref(),
			coeff.as_ref(),
			true,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(unitary::unitary_scratch::<Scalar>(
				m,
				n,
				k,
				block_size,
				ParShape::Seq,
			))),
		);
		let dim = Ord::max(m, n);
		crate::assert!(
			lin_difference(l_faer.as_ref(), l_lapack.as_ref(), dim) < 30.0,
			"LAPACK LQ m={m} n={n}: L differs from ?gelqf"
		);
		crate::assert!(
			lin_difference(q_faer.as_ref(), q_lapack.as_ref(), dim) < 30.0,
			"LAPACK LQ m={m} n={n}: Q differs from ?orglq"
		);
	}
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_lq_parallel() {
	let mut run = Run::new("lq_parallel");
	// a tall and a wide shape: the blocked panel is the row block of the packed
	// form, which is the column block for the tall case and the whole matrix for
	// the wide one.
	for (m, n) in [(257, 257), (257, 129), (129, 257)] {
		lq_case(&mut run, m, n, 2, 0, 8, Par::Seq);
		for threads in [2, 4] {
			crate::team::with_team(threads, |par| {
				lq_case(&mut run, m, n, 2, threads / 2, 8, par)
			});
		}
	}
	run.finish(18);
}
