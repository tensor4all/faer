//! Public contract of the packed LQ factorization and its factor extraction:
//! exact scratch queries, every output mode, reused output buffers, views, and
//! `Par::Seq` against a two-thread team.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lq::no_pivoting::{factor, reconstruct, unitary};
use faer::{Auto, Mat, Par, ParShape, c32, c64};
use faer::traits::ComplexField;
mod common;
use common::*;

fn sample_mat<T: ComplexField>(m: usize, n: usize, seed: f64) -> Mat<T> {
	Mat::from_fn(m, n, |i, j| {
		T::from_f64(sample(i, j) + seed * ((i + 2 * j) as f64).sin())
	})
}

/// Factors `a` and returns the packed form and its coefficients.
fn pack<T: ComplexField>(
	a: MatRef<'_, T>,
	block_size: usize,
	par: Par<'_>,
) -> (Mat<T>, Mat<T>) {
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let params = <factor::LqParams as Auto<T>>::auto().into();
	let mut mem = MemBuffer::new(factor::lq_in_place_scratch::<T>(
		m,
		n,
		block_size,
		par.shape(),
		params,
	));
	let mut packed = a.to_owned();
	let mut coeff = Mat::<T>::full(block_size, k, T::from_f64(f64::INFINITY));
	factor::lq_in_place(
		packed.as_mut(),
		coeff.as_mut(),
		par,
		MemStack::new(&mut mem),
		params,
	);
	(packed, coeff)
}

/// Every output mode at the exact scratch size, twice into the same buffers.
fn contract<T: ComplexField>(m: usize, n: usize) {
	let k = Ord::min(m, n);
	let a = sample_mat::<T>(m, n, 0.0);
	let block_size = Ord::max(factor::recommended_block_size::<T>(m, n).min(Ord::max(k, 1)), 1);
	let (packed, coeff) = pack(a.as_ref(), block_size, Par::Seq);

	// reused output buffers, one pass of garbage then the real call
	let mut l = Mat::<T>::full(m, k, T::from_f64(-7.5));
	let mut l_full = Mat::<T>::full(m, n, T::from_f64(-7.5));
	let mut q = Mat::<T>::full(k, n, T::from_f64(-7.5));
	let mut q_full = Mat::<T>::full(n, n, T::from_f64(-7.5));
	let mut null = Mat::<T>::full(n - k, n, T::from_f64(-7.5));
	let mut rebuilt = Mat::<T>::full(m, n, T::from_f64(-7.5));
	let mut q_mem = MemBuffer::new(unitary::unitary_scratch::<T>(
		m,
		n,
		k,
		block_size,
		ParShape::Seq,
	));
	let mut full_mem = MemBuffer::new(unitary::unitary_scratch::<T>(
		m,
		n,
		n,
		block_size,
		ParShape::Seq,
	));
	let mut null_mem = MemBuffer::new(unitary::null_scratch::<T>(
		m,
		n,
		block_size,
		ParShape::Seq,
	));
	let mut rebuild_mem = MemBuffer::new(reconstruct::reconstruct_scratch::<T>(
		m,
		n,
		block_size,
		ParShape::Seq,
	));
	factor::copy_l(l.as_mut(), packed.as_ref(), true);
	factor::copy_l(l_full.as_mut(), packed.as_ref(), true);
	unitary::unitary(
		q.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		true,
		Par::Seq,
		MemStack::new(&mut q_mem),
	);
	unitary::unitary(
		q_full.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		true,
		Par::Seq,
		MemStack::new(&mut full_mem),
	);
	unitary::null(
		null.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		Par::Seq,
		MemStack::new(&mut null_mem),
	);
	reconstruct::reconstruct(
		rebuilt.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		Par::Seq,
		MemStack::new(&mut rebuild_mem),
	);

	// $A = LQ$ with the positive gauge, $Q$ with orthonormal rows, and the full
	// $Q$ continuing the compact one.
	close((&l * &q).as_ref(), a.as_ref());
	close((&l * q_full.as_ref().get(..k, ..)).as_ref(), a.as_ref());
	close(rebuilt.as_ref(), a.as_ref());
	close(
		(&q * q.adjoint()).as_ref(),
		Mat::<T>::identity(k, k).as_ref(),
	);
	close(
		(&q_full * q_full.adjoint()).as_ref(),
		Mat::<T>::identity(n, n).as_ref(),
	);
	close(q_full.as_ref().get(..k, ..), q.as_ref());
	close(q_full.as_ref().get(k.., ..), null.as_ref());
	// the full $L$ is the compact one padded with zero columns.
	close(l_full.as_ref().get(.., ..k), l.as_ref());
	assert!(l_full.get(.., k..).ncols() == 0 || l_full.get(.., k..).norm_l2() == zero());
	// the null block annihilates $A$.
	let residual = &a * null.adjoint();
	assert!(residual.norm_l2() <= tolerance::<T>(n.max(m)) * a.norm_l2());
	// the gauge makes the triangular diagonal real and nonnegative.
	for j in 0..k {
		let d = l[(j, j)].clone();
		assert!(d.real() >= zero());
		assert!(
			{
			let bound = eps::<T::Real>()
				* from_f64::<T::Real>(4.0)
				* d.abs().fmax(one::<T::Real>());
			d.imag().abs() <= bound
		}
		);
	}
}

/// Views: a non-unit-stride packed form and non-unit-stride outputs.
fn views<T: ComplexField>(m: usize, n: usize) {
	let k = Ord::min(m, n);
	let a = sample_mat::<T>(m, n, 0.25);
	// column-major buffer with one guard row, so the interior has a stride > m.
	let mut wide = Mat::<T>::full(m + 1, n, T::from_f64(3.25));
	wide.as_mut().get_mut(..m, ..).copy_from(a.as_ref());
	let mut q_wide = Mat::<T>::full(n + 1, n + 3, T::from_f64(-2.5));
	let mut coeff = Mat::<T>::full(
		Ord::max(factor::recommended_block_size::<T>(m, n).min(Ord::max(k, 1)), 1),
		k,
		T::from_f64(f64::INFINITY),
	);
	let block_size = coeff.nrows();
	let params = <factor::LqParams as Auto<T>>::auto().into();
	let mut mem = MemBuffer::new(factor::lq_in_place_scratch::<T>(
		m,
		n,
		block_size,
		ParShape::Seq,
		params,
	));
	factor::lq_in_place(
		wide.as_mut().get_mut(..m, ..),
		coeff.as_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
		params,
	);
	let mut q_mem = MemBuffer::new(unitary::unitary_scratch::<T>(
		m,
		n,
		k,
		block_size,
		ParShape::Seq,
	));
	unitary::unitary(
		q_wide.as_mut().submatrix_mut(1, 2, k, n),
		wide.as_ref().get(..m, ..),
		coeff.as_ref(),
		false,
		Par::Seq,
		MemStack::new(&mut q_mem),
	);
	// the guard row of the packed buffer is untouched, and the strided factor is
	// the one the plain factorization gives.
	let plain = sample_mat::<T>(m, n, 0.25);
	let (packed_plain, coeff_plain) = pack(plain.as_ref(), block_size, Par::Seq);
	let mut q = Mat::<T>::zeros(k, n);
	unitary::unitary(
		q.as_mut(),
		packed_plain.as_ref(),
		coeff_plain.as_ref(),
		false,
		Par::Seq,
		MemStack::new(&mut q_mem),
	);
	close(q_wide.as_ref().submatrix(1, 2, k, n), q.as_ref());
	assert_eq!(wide[(m, 0)], T::from_f64(3.25));
}

#[cfg(feature = "threadteam")]
fn team_agrees<T: ComplexField>(m: usize, n: usize) {
	let k = Ord::min(m, n);
	// `Q` is compared entry by entry against `200 n eps`, which holds only for a
	// well-conditioned matrix: `Q` moves by about `cond * eps` under any change of
	// summation order. The shift brings the condition number of these fixtures to
	// at most 21; the unshifted ones (1e9-1e11) are checked against the
	// perturbation bound in `team_agrees_ill_conditioned` instead.
	let mut a = sample_mat::<T>(m, n, 0.5);
	for i in 0..k {
		a[(i, i)] = a[(i, i)].clone() + T::from_f64(16.0);
	}
	let block_size = Ord::max(
		factor::recommended_block_size::<T>(m, n).min(Ord::max(k, 1)) / 2,
		1,
	);
	let (packed_seq, coeff_seq) = pack(a.as_ref(), block_size, Par::Seq);
	let mut q_seq = Mat::<T>::zeros(k, n);
	let mut q_mem = MemBuffer::new(unitary::unitary_scratch::<T>(
		m,
		n,
		k,
		block_size,
		ParShape::Seq,
	));
	unitary::unitary(
		q_seq.as_mut(),
		packed_seq.as_ref(),
		coeff_seq.as_ref(),
		false,
		Par::Seq,
		MemStack::new(&mut q_mem),
	);
	faer::team::with_team(2, |mut par: Par<'_>| {
		let (packed_team, coeff_team) = pack(a.as_ref(), block_size, par.reborrow());
		let mut q_team = Mat::<T>::zeros(k, n);
		let mut mem = MemBuffer::new(unitary::unitary_scratch::<T>(
			m,
			n,
			k,
			block_size,
			par.shape(),
		));
		unitary::unitary(
			q_team.as_mut(),
			packed_team.as_ref(),
			coeff_team.as_ref(),
			false,
			par,
			MemStack::new(&mut mem),
		);
		close(q_team.as_ref(), q_seq.as_ref());
	});
}

/// The unshifted `sample_mat` fixtures, whose condition numbers are 1e9-1e11
/// (257x129: 1.4e9). Here the two runs need not agree to `200 n eps`: on
/// `main` (`c7470bd`) column-major c64 QR of the same matrix already differs by
/// 3.4e-7 between a 2-thread team and `Par::Seq`, and both reconstruct it. What
/// holds is checked instead:
///
/// - each run is a backward-stable factorization: `|A - L Q|_F <= 1e-12 |A|_F`
///   (measured: below 1e-15), and `Q` has orthonormal rows to the same level;
/// - the two `Q` agree within the first-order perturbation bound of the `Q`
///   factor. With the positive diagonal gauge, `Q` is a function of the first
///   `k` rows `G` of `A` (for `m <= n`, all of `A`; for `m > n`, `Q = L1^-1 G`
///   with `G` square), and `|dQ|_F <= sqrt(2) |dG|_F / sigma_min(G)` to first
///   order (the QR bound applied to `G^H`). A computed `Q^` is within its
///   orthogonality defect `o = |Q^ Q^H - I|_F` of an exactly unitary `Q~` that
///   factors `A + dA` with `|dA|_F <= |A - L Q^|_F + |A|_F o`. So
///   `|Q^_1 - Q^_2|_F <= sqrt(2) (|dA_1|_F + |dA_2|_F) / sigma_min(G) + o_1 + o_2`,
///   every term measured in the run itself. The expansion holds while that
///   first-order term `eps1` is small against 1, and the terms it drops are a
///   relative `O(eps1)` of it; the test requires `eps1 < 0.1` and allows a
///   factor 2 for them, which is the only constant in the bound. Measured at
///   this revision: residuals 7e-16 to 1e-15 of `|A|_F`, `eps1` up to 7.7e-3
///   (257x257), and the c64 runs differ by 3.0e-7 (257x129) and 4.4e-8
///   (257x257) against bounds of 6.3e-3 and 1.5e-2; the f64 runs agree exactly.
#[cfg(feature = "threadteam")]
fn team_agrees_ill_conditioned<T: ComplexField<Real = f64>>(m: usize, n: usize) {
	let k = Ord::min(m, n);
	let a = sample_mat::<T>(m, n, 0.5);
	let block_size = Ord::max(
		factor::recommended_block_size::<T>(m, n).min(Ord::max(k, 1)) / 2,
		1,
	);
	// `L` and `Q` in the positive gauge, and the run's backward error and
	// orthogonality defect
	let factors = |mut par: Par<'_>| {
		let (packed, coeff) = pack(a.as_ref(), block_size, par.reborrow());
		let mut l = Mat::<T>::zeros(m, k);
		factor::copy_l(l.as_mut(), packed.as_ref(), true);
		let mut q = Mat::<T>::zeros(k, n);
		let mut mem = MemBuffer::new(unitary::unitary_scratch::<T>(
			m,
			n,
			k,
			block_size,
			par.shape(),
		));
		unitary::unitary(
			q.as_mut(),
			packed.as_ref(),
			coeff.as_ref(),
			true,
			par,
			MemStack::new(&mut mem),
		);
		let residual = (&a - &l * &q).norm_l2();
		let defect = (&q * q.adjoint() - Mat::<T>::identity(k, k)).norm_l2();
		(q, residual, defect)
	};
	let a_norm = a.norm_l2();
	let (q_seq, r_seq, o_seq) = factors(Par::Seq);
	let (q_team, r_team, o_team) = faer::team::with_team(2, |par| factors(par));
	let tag = format!("{m}x{n}");
	for (residual, defect, run) in [(r_seq, o_seq, "seq"), (r_team, o_team, "team")] {
		assert!(residual <= 1e-12 * a_norm, "{tag} {run}: |A - LQ| = {residual:e}");
		assert!(defect <= 1e-12, "{tag} {run}: |QQ^H - I| = {defect:e}");
	}
	let sigma = a.as_ref().subrows(0, k).singular_values(Par::Seq).unwrap();
	let sigma_min = sigma.iter().cloned().fold(f64::INFINITY, f64::min);
	let backward = (r_seq + a_norm * o_seq) + (r_team + a_norm * o_team);
	let first_order = 2.0f64.sqrt() * backward / sigma_min + o_seq + o_team;
	assert!(first_order < 0.1, "{tag}: the linearization does not hold ({first_order:e})");
	let diff = (&q_team - &q_seq).norm_l2();
	assert!(diff <= 2.0 * first_order, "{tag}: |Q_team - Q_seq| = {diff:e}, bound {:e}", 2.0 * first_order);
}

#[test]
fn lq_contract() {
	for (m, n) in [(0, 0), (0, 3), (3, 0), (1, 1), (8, 8), (8, 3), (3, 8), (100, 33)] {
		contract::<f64>(m, n);
		contract::<f32>(m, n);
		contract::<c64>(m, n);
		contract::<c32>(m, n);
	}
}

#[test]
fn lq_views() {
	for (m, n) in [(8, 3), (3, 8), (8, 8)] {
		views::<f64>(m, n);
		views::<c64>(m, n);
	}
}

#[cfg(feature = "threadteam")]
#[test]
fn lq_parallel_matches_sequential() {
	for (m, n) in [(257, 129), (129, 257), (257, 257)] {
		team_agrees::<f64>(m, n);
		team_agrees::<c64>(m, n);
	}
}

#[cfg(feature = "threadteam")]
#[test]
fn lq_parallel_ill_conditioned_within_perturbation_bound() {
	for (m, n) in [(257, 129), (129, 257), (257, 257)] {
		team_agrees_ill_conditioned::<f64>(m, n);
		team_agrees_ill_conditioned::<c64>(m, n);
	}
}
