//! tensor4all/faer#370: `gevd_real` on a pencil `(s A, s B)` scaled far from
//! one. The eigenvalues `alpha / beta` and the eigenvector directions of `(s A,
//! s B)` are those of `(A, B)`, so every `s` is compared with `s = 1`: the
//! eigenvectors must be finite and span the same eigenspaces.
//!
//! The pencils are built as `U (T, W) V^T` with orthogonal `U`, `V`, so the QZ
//! runs on a dense pencil, and they hold:
//!
//! - `one_pair`: one complex pair and two real eigenvalues, so the pair's
//!   vectors meet 1 by 1 blocks shifted by a complex eigenvalue;
//! - `two_pairs`: two distinct complex pairs, so each pair's vectors meet the
//!   other pair's 2 by 2 block shifted by a complex eigenvalue;
//! - `repeated_pair`: the faer#220 pencil `diag(R, R, 3)`, `B = I`, where the
//!   shifted 2 by 2 block is exactly singular and its second pivot is replaced
//!   by `smin` (LAPACK `dlaln2`).
//!
//! tensor4all/faer#375 adds the eigenvalue-only call below the range where the
//! QZ used to underflow to `alpha = beta = 0`, the power-of-two exactness of
//! the prescaling, the pairs whose scaling back would underflow or overflow,
//! and `c64` and `f32`.
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{
	GevdError, GevdParams, gevd_cplx, gevd_real, gevd_scratch,
};
use faer::{Mat, Par, ParShape, Spec, c64};

/// A fixed pseudo-random value in `[-1, 1)`.
fn sample(i: usize, j: usize, seed: u64) -> f64 {
	let mut s = (i as u64 + 1)
		.wrapping_mul(0x9e37_79b9_7f4a_7c15)
		.wrapping_add((j as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9))
		.wrapping_add(seed.wrapping_mul(0x94d0_49bb_1331_11eb));
	s ^= s >> 31;
	s = s.wrapping_mul(0xd6e8_feb8_6659_fd93);
	s ^= s >> 32;
	((s >> 11) as f64 / (1u64 << 52) as f64) - 1.0
}

fn orthogonal(n: usize, seed: u64) -> Mat<f64> {
	Mat::<f64>::from_fn(n, n, |i, j| sample(i, j, seed))
		.qr(Par::Seq)
		.compute_Q(Par::Seq)
}

/// `(U T V^T, U W V^T)` for a quasi-triangular `T` whose 2 by 2 diagonal blocks
/// start at `blocks`, and an upper triangular `W` with a diagonal near one.
fn pencil(diag: &[[f64; 4]], seed: u64) -> (Mat<f64>, Mat<f64>) {
	let n: usize = diag.iter().map(|d| if d[2] == 0.0 { 1 } else { 2 }).sum();
	let mut t = Mat::<f64>::zeros(n, n);
	let mut w = Mat::<f64>::zeros(n, n);
	for i in 0..n {
		for j in i..n {
			t[(i, j)] = 0.5 * sample(i, j, seed + 1);
			w[(i, j)] = 0.5 * sample(i, j, seed + 2);
		}
		w[(i, i)] = 1.0 + 0.25 * sample(i, i, seed + 3);
	}
	let mut k = 0;
	for d in diag {
		if d[2] == 0.0 {
			t[(k, k)] = d[0];
			k += 1;
		} else {
			// [[a, b], [c, d]] with b * c < 0 has a complex pair
			t[(k, k)] = d[0];
			t[(k, k + 1)] = d[1];
			t[(k + 1, k)] = d[2];
			t[(k + 1, k + 1)] = d[3];
			w[(k, k + 1)] = 0.0;
			k += 2;
		}
	}
	let u = orthogonal(n, seed + 4);
	let v = orthogonal(n, seed + 5);
	(&u * &t * v.transpose(), &u * &w * v.transpose())
}

fn one_pair() -> (Mat<f64>, Mat<f64>) {
	pencil(
		&[
			[1.0, 2.0, -0.5, 1.0],
			[3.0, 0.0, 0.0, 0.0],
			[-2.0, 0.0, 0.0, 0.0],
		],
		10,
	)
}

fn two_pairs() -> (Mat<f64>, Mat<f64>) {
	pencil(&[[1.0, 2.0, -0.5, 1.0], [-2.0, 1.0, -3.0, -1.5]], 20)
}

fn repeated_pair() -> (Mat<f64>, Mat<f64>) {
	let mut a = Mat::<f64>::zeros(5, 5);
	for o in [0usize, 2] {
		a[(o, o)] = 2.0;
		a[(o, o + 1)] = 1.0;
		a[(o + 1, o)] = -1.0;
		a[(o + 1, o + 1)] = 2.0;
	}
	a[(4, 4)] = 3.0;
	(a, Mat::<f64>::identity(5, 5))
}

const PENCILS: [(&str, fn() -> (Mat<f64>, Mat<f64>)); 3] = [
	("one_pair", one_pair),
	("two_pairs", two_pairs),
	("repeated_pair", repeated_pair),
];

struct Gevd {
	/// `alpha / beta`
	lambda: Vec<c64>,
	left: Mat<c64>,
	right: Mat<c64>,
}

/// Columns `j`, `j + 1` of a real eigenvector matrix hold the real and
/// imaginary parts of the vector for `lambda[j]` (`im > 0`); `lambda[j + 1]`
/// gets the conjugate.
fn complex_vectors(u: &Mat<f64>, lambda: &[c64]) -> Mat<c64> {
	let n = u.nrows();
	let mut out = Mat::<c64>::zeros(n, n);
	let mut j = 0;
	while j < n {
		// a NaN eigenvalue or an unpaired last column reads as real, so the
		// column is still compared rather than skipped
		if !(lambda[j].im > 0.0 && j + 1 < n) {
			for i in 0..n {
				out[(i, j)] = c64::new(u[(i, j)], 0.0);
			}
			j += 1;
		} else {
			for i in 0..n {
				out[(i, j)] = c64::new(u[(i, j)], u[(i, j + 1)]);
				out[(i, j + 1)] = c64::new(u[(i, j)], -u[(i, j + 1)]);
			}
			j += 2;
		}
	}
	out
}

type Outputs = (Gevd, Mat<f64>, Mat<f64>);

fn gevd(a: &Mat<f64>, b: &Mat<f64>, s: f64) -> Result<Outputs, GevdError> {
	let n = a.nrows();
	let mut a = Mat::<f64>::from_fn(n, n, |i, j| s * a[(i, j)]);
	let mut b = Mat::<f64>::from_fn(n, n, |i, j| s * b[(i, j)]);
	let mut s_re = Diag::<f64>::zeros(n);
	let mut s_im = Diag::<f64>::zeros(n);
	let mut beta = Diag::<f64>::zeros(n);
	let mut u_left = Mat::<f64>::zeros(n, n);
	let mut u_right = Mat::<f64>::zeros(n, n);
	let mut mem = MemBuffer::new(gevd_scratch::<f64>(
		n,
		ComputeEigenvectors::Yes,
		ComputeEigenvectors::Yes,
		ParShape::Seq,
		Spec::<GevdParams, f64>::default(),
	));
	gevd_real(
		a.as_mut(),
		b.as_mut(),
		s_re.as_mut(),
		s_im.as_mut(),
		beta.as_mut(),
		Some(u_left.as_mut()),
		Some(u_right.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Spec::<GevdParams, f64>::default(),
	)?;
	let lambda: Vec<c64> = (0..n)
		.map(|j| {
			c64::new(s_re.column_vector()[j], s_im.column_vector()[j])
				/ beta.column_vector()[j]
		})
		.collect();
	let left = complex_vectors(&u_left, &lambda);
	let right = complex_vectors(&u_right, &lambda);
	Ok((
		Gevd {
			lambda,
			left,
			right,
		},
		u_left,
		u_right,
	))
}

/// Distance of `x` from the span of the columns of `basis`, relative to `|x|`.
fn sin_to_span(x: &[c64], basis: &[Vec<c64>]) -> f64 {
	let dot = |u: &[c64], v: &[c64]| -> c64 {
		u.iter().zip(v).map(|(u, v)| u.conj() * v).sum()
	};
	let norm = |u: &[c64]| dot(u, u).re.sqrt();
	// modified Gram-Schmidt, twice
	let mut q: Vec<Vec<c64>> = Vec::new();
	for b in basis {
		let mut v = b.clone();
		for _ in 0..2 {
			for e in &q {
				let c = dot(e, &v);
				for (v, e) in v.iter_mut().zip(e) {
					*v -= c * e;
				}
			}
		}
		let nv = norm(&v);
		if nv > 1e-8 * norm(b) {
			q.push(v.iter().map(|v| v / nv).collect());
		}
	}
	let mut r = x.to_vec();
	for _ in 0..2 {
		for e in &q {
			let c = dot(e, &r);
			for (r, e) in r.iter_mut().zip(e) {
				*r -= c * e;
			}
		}
	}
	norm(&r) / norm(x)
}

/// The largest eigenvalue distance and eigenvector angle between `(s A, s B)`
/// and `(A, B)`. Each eigenvalue of `scaled` is matched with the reference
/// eigenvalues within `cluster` of it. Each eigenvector is measured against the
/// span of the reference eigenvectors of its cluster (two vectors for the
/// repeated pair), and each reference eigenvector against the span of the
/// scaled eigenvectors of its cluster, so a scaled result that loses a direction
/// of a repeated eigenvalue's eigenspace fails too. A NaN distance counts as
/// infinite.
struct Diff {
	lambda: f64,
	left: f64,
	right: f64,
}

/// `sin_to_span` of column `j` of `x` against the columns `ks` of `basis`;
/// infinite when `ks` is empty or the distance is NaN.
fn column_to_span(
	x: &Mat<c64>,
	j: usize,
	basis: &Mat<c64>,
	ks: &[usize],
) -> f64 {
	let n = x.nrows();
	if ks.is_empty() {
		return f64::INFINITY;
	}
	let col: Vec<c64> = (0..n).map(|i| x[(i, j)]).collect();
	let basis: Vec<Vec<c64>> = ks
		.iter()
		.map(|&k| (0..n).map(|i| basis[(i, k)]).collect())
		.collect();
	let sin = sin_to_span(&col, &basis);
	if sin.is_nan() { f64::INFINITY } else { sin }
}

fn compare(reference: &Gevd, scaled: &Gevd) -> Diff {
	let n = reference.lambda.len();
	let cluster = 1e-6;
	let near = |x: c64, y: c64| (x - y).norm() <= cluster * (1.0 + x.norm());
	let mut d = Diff {
		lambda: 0.0,
		left: 0.0,
		right: 0.0,
	};
	for j in 0..n {
		let best = (0..n)
			.map(|k| {
				(reference.lambda[k] - scaled.lambda[j]).norm()
					/ reference.lambda[k].norm()
			})
			.fold(f64::INFINITY, f64::min);
		d.lambda = d.lambda.max(best);
	}
	// scaled against reference, then reference against scaled
	for (from, to) in [(scaled, reference), (reference, scaled)] {
		for j in 0..n {
			let ks: Vec<usize> = (0..n)
				.filter(|&k| near(from.lambda[j], to.lambda[k]))
				.collect();
			d.left = d.left.max(column_to_span(&from.left, j, &to.left, &ks));
			d.right =
				d.right.max(column_to_span(&from.right, j, &to.right, &ks));
		}
	}
	d
}

/// Eigenvalue and eigenvector agreement tolerance. Scaling by a power of ten is
/// inexact, so `(s A, s B)` is `(A, B)` with each entry perturbed by up to
/// `eps` relative, and the QZ adds an `O(n eps)` backward error; both move the
/// eigenvalues and eigenvectors by `eps` times their condition numbers. These
/// pencils are well conditioned.
///
/// Two scans are quoted here and their numbers must not be mixed. **With the
/// fix** (this code, the scan below over every `s = 10^k` at which the scaled
/// pencil is computed correctly) the largest difference is `4.1e-15`, and
/// `3.7e-15` on a second host, so `1e-13` (about `450 eps`) leaves a margin of
/// 25. **Before the fix** (the unscaled complex division, the failure this test
/// guards) the same scan's largest passing difference is `1.06e-14` for
/// `two_pairs` at `1e-155`, where this tolerance leaves a margin of 9.5, not 25;
/// that near-miss is why the failure went unnoticed, and the losses just above
/// it are what the test does catch: `9.0e-13` at `1e-156` and `1.2e-10` at
/// `1e-157`.
const TOL: f64 = 1e-13;

fn check(s: f64) {
	for (name, make) in PENCILS {
		let (a, b) = make();
		let (reference, ..) = gevd(&a, &b, 1.0).unwrap();
		let (scaled, u_left, u_right) = gevd(&a, &b, s).unwrap();
		assert!(
			u_left.is_all_finite() && u_right.is_all_finite(),
			"{name}, s = {s:e}: non-finite eigenvectors\nleft = {u_left:?}\nright = {u_right:?}"
		);
		let d = compare(&reference, &scaled);
		println!(
			"{name}, s = {s:e}: eigenvalue {:e}, left sin {:e}, right sin {:e}",
			d.lambda, d.left, d.right,
		);
		assert!(
			d.lambda <= TOL && d.left <= TOL && d.right <= TOL,
			"{name}, s = {s:e}: eigenvalue {:e}, left sin {:e}, right sin {:e} > {TOL:e}",
			d.lambda,
			d.left,
			d.right,
		);
	}
}

#[test]
fn scaled_pencil_small() {
	check(1e-150);
}

#[test]
fn scaled_pencil_control() {
	check(1.0);
}

#[test]
fn scaled_pencil_large() {
	check(1e160);
}

/// Far beyond the faer#370 range on both sides.
#[test]
fn scaled_pencil_extremes() {
	check(1e-250);
	check(1e300);
}

/// The largest relative distance `|l - r| / |r|` between matched eigenvalues,
/// minimized over all one-to-one matchings of `computed` with `reference`
/// (a bottleneck assignment, by exhaustive search: the pencils here have at
/// most five eigenvalues). unlike matching each computed eigenvalue with its
/// nearest reference, this counts multiplicities: a spectrum that repeats one
/// eigenvalue in place of another, or misses one, is far from the reference.
/// infinite when the lengths differ or a distance is NaN.
fn spectrum_distance(computed: &[c64], reference: &[c64]) -> f64 {
	fn go(
		computed: &[c64],
		reference: &[c64],
		used: &mut Vec<bool>,
		worst: f64,
		best: &mut f64,
	) {
		let Some((l, rest)) = computed.split_first() else {
			*best = best.min(worst);
			return;
		};
		for (k, r) in reference.iter().enumerate() {
			if used[k] {
				continue;
			}
			let d = (l - r).norm() / r.norm();
			let d = if d.is_nan() { f64::INFINITY } else { d };
			let w = worst.max(d);
			if w >= *best {
				continue;
			}
			used[k] = true;
			go(rest, reference, used, w, best);
			used[k] = false;
		}
	}
	if computed.len() != reference.len() {
		return f64::INFINITY;
	}
	let mut best = f64::INFINITY;
	go(
		computed,
		reference,
		&mut vec![false; reference.len()],
		0.0,
		&mut best,
	);
	best
}

/// [`spectrum_distance`] itself: a permutation of the reference matches, and
/// replacing one eigenvalue by another (computed or reference side) does not,
/// although every eigenvalue still has an exact nearest neighbour.
#[test]
fn spectrum_distance_counts_multiplicities() {
	let (a, b) = one_pair();
	let v = gevd_values(&a, &b, 1.0, false);
	let lambda: Vec<c64> =
		v.alpha.iter().zip(&v.beta).map(|(a, &b)| a / b).collect();
	let mut permuted = lambda.clone();
	permuted.reverse();
	assert!(spectrum_distance(&permuted, &lambda) == 0.0);
	let mut repeated = lambda.clone();
	repeated[0] = lambda[lambda.len() - 1];
	for (computed, reference) in [(&repeated, &lambda), (&lambda, &repeated)] {
		let d = spectrum_distance(computed, reference);
		assert!(
			d > 1e-2,
			"{computed:?} matched {reference:?} to {d:e}, one eigenvalue is missing"
		);
	}
}

/// tensor4all/faer#375: below about `1e-280` (`one_pair`) and `1.3e-293`
/// (`two_pairs`) the QZ underflowed and returned `alpha = beta = 0` for every
/// eigenvalue, with `Ok` when no eigenvectors were requested. the pencil is now
/// prescaled, so the eigenvectors and the eigenvalues agree with `s = 1`.
#[test]
fn scaled_pencil_tiny() {
	check(1e-290);
	check(1e-301);
	check(1e-305);
}

/// The eigenvalue-only call of [`scaled_pencil_tiny`], where the defect
/// returned `Ok`: no `alpha = beta = 0`, every eigenvalue finite, and the
/// spectrum within `TOL` (relative) of the one at `s = 1`, matched one to one
/// with multiplicities ([`spectrum_distance`]).
#[test]
fn scaled_pencil_tiny_eigenvalues_only() {
	for (name, make) in PENCILS {
		let (a, b) = make();
		let reference = gevd_values(&a, &b, 1.0, false);
		assert!(reference.ok);
		let lambda_ref: Vec<c64> = reference
			.alpha
			.iter()
			.zip(&reference.beta)
			.map(|(a, &b)| a / b)
			.collect();
		for s in [1e-290, 1e-301, 1e-305] {
			let v = gevd_values(&a, &b, s, false);
			assert!(v.ok, "{name}, s = {s:e}: Err");
			assert!(
				!has_zero_pair(&v),
				"{name}, s = {s:e}: alpha = beta = 0 returned with Ok\nalpha = {:?}\nbeta = {:?}",
				v.alpha,
				v.beta
			);
			let lambda: Vec<c64> =
				v.alpha.iter().zip(&v.beta).map(|(a, &b)| a / b).collect();
			for l in &lambda {
				assert!(
					l.re.is_finite() && l.im.is_finite(),
					"{name}, s = {s:e}: eigenvalue {l:?}"
				);
			}
			let err = spectrum_distance(&lambda, &lambda_ref);
			assert!(
				err <= TOL,
				"{name}, s = {s:e}: spectrum {lambda:?} against {lambda_ref:?}, relative error {err:e} > {TOL:e}"
			);
		}
	}
}

/// The prescaling factor is a power of two, so two pencils that differ by a
/// power of two and are both prescaled get `alpha` and `beta` that differ by
/// exactly that power.
#[test]
fn scaled_pencil_power_of_two_is_exact() {
	let k = 2f64.powi(100);
	for big in [false, true] {
		for (name, make) in PENCILS {
			let (a, b) = make();
			let s = if big {
				2f64.powi(800)
			} else {
				2f64.powi(-1000)
			};
			let lo = gevd_values(&a, &b, s, false);
			let hi = gevd_values(&a, &b, s * k, false);
			assert!(lo.ok && hi.ok);
			for j in 0..lo.beta.len() {
				assert!(
					lo.alpha[j] * k == hi.alpha[j]
						&& lo.beta[j] * k == hi.beta[j],
					"{name}, s = {s:e}, j = {j}: ({:?}, {:e}) * 2^100 != ({:?}, {:e})",
					lo.alpha[j],
					lo.beta[j],
					hi.alpha[j],
					hi.beta[j],
				);
			}
		}
	}
}

/// Every eigenvalue of `v` as its pair `(alpha, beta)` divided by
/// `max(|alpha|, |beta|)`; the pair must be finite and not both zero.
fn finite_pairs(v: &Values, what: &str) -> Vec<(c64, c64)> {
	v.alpha
		.iter()
		.zip(&v.beta)
		.map(|(a, &b)| {
			assert!(
				a.re.is_finite() && a.im.is_finite() && b.is_finite(),
				"{what}: non-finite pair ({a:?}, {b:e})"
			);
			assert!(
				!(*a == c64::new(0.0, 0.0) && b == 0.0),
				"{what}: alpha = beta = 0 returned with Ok"
			);
			let m = a.norm().max(b.abs());
			(a / m, c64::new(b / m, 0.0))
		})
		.collect()
}

/// Chordal distance between two eigenvalues given as normalized pairs, which
/// also compares infinite eigenvalues (`beta = 0`).
fn chordal(x: (c64, c64), y: (c64, c64)) -> f64 {
	let n = |p: (c64, c64)| (p.0.norm_sqr() + p.1.norm_sqr()).sqrt();
	(x.0 * y.1 - y.0 * x.1).norm() / (n(x) * n(y))
}

/// A pencil with subnormal entries whose second eigenvalue has `alpha` and
/// `beta` below the smallest subnormal at the input's scale. scaling the QZ's
/// values back would underflow both to zero (an indeterminate eigenvalue), so
/// the pair is scaled by a common power of two instead and keeps the ratio of
/// the same pencil times `2^1000` (exact), which is computed without
/// prescaling. found by a random search over pencils with entries between
/// `1e-324` and `1e-300`.
#[test]
fn scaled_pencil_underflow_keeps_the_ratio() {
	let a = Mat::<f64>::from_fn(2, 2, |i, j| {
		[[0.0, -0.0], [-1e-323, -6.4e-323]][i][j]
	});
	let b = Mat::<f64>::from_fn(2, 2, |i, j| {
		[
			[1.713968517e-315, 4e-323],
			[3.4855627839772177e-303, -5.326281543e-314],
		][i][j]
	});
	let normal = gevd_values(&a, &b, 2f64.powi(1000), false);
	assert!(normal.ok);
	let reference = finite_pairs(&normal, "s = 2^1000");
	for vectors in [false, true] {
		let v = gevd_values(&a, &b, 1.0, vectors);
		let what = format!("vectors = {vectors}");
		assert!(v.ok, "{what}: Err");
		for (l, r) in finite_pairs(&v, &what).into_iter().zip(&reference) {
			let d = chordal(l, *r);
			assert!(
				d <= TOL,
				"{what}: pair {l:?}, reference {r:?}, chordal distance {d:e}"
			);
		}
	}
}

/// tensor4all/faer review of #378: `A = B = [[m, m], [-m, m]]`, `m = 1.3e308`,
/// is finite and regular with the double eigenvalue `1`. it is prescaled down,
/// and scaling the QZ's `alpha` and `beta` back overflowed both to `inf`, so
/// `alpha / beta` was NaN with `Ok`. the pair is now scaled by a common power
/// of two and stays finite.
#[test]
fn scaled_pencil_overflow_keeps_the_ratio() {
	let m = 1.3e308;
	let a = Mat::<f64>::from_fn(2, 2, |i, j| [[m, m], [-m, m]][i][j]);
	for vectors in [false, true] {
		let v = gevd_values(&a, &a, 1.0, vectors);
		let what = format!("vectors = {vectors}");
		assert!(v.ok, "{what}: Err");
		let one = (c64::new(1.0, 0.0), c64::new(1.0, 0.0));
		for l in finite_pairs(&v, &what) {
			let d = chordal(l, one);
			assert!(
				d <= TOL,
				"{what}: pair {l:?}, expected eigenvalue 1, chordal distance {d:e}"
			);
		}
	}
}

/// Review of #378: a one-by-one pencil whose `A` and `B` are prescaled in
/// opposite directions. `A = [[2^-1074]]` (the smallest subnormal) is scaled
/// up and `B = [[1e300]]` down, so the two inverse factors are `2^-616` and
/// `2^538`; their ratio, `2^-1154`, is not representable. a fallback that formed
/// it returned `alpha = 0` for the nonzero `alpha`. the pencil's own pair
/// `(2^-1074, 1e300)` is representable, so `(alpha, beta)` must be a nonzero
/// multiple of it (`beta >= 0`, as the QZ normalizes it, makes the multiple
/// negative for a negative `B`): `alpha / a = beta / b`, nonzero and finite. the mirrored
/// pencil (an eigenvalue of order `1e300 / 2^-1074`, not representable as a
/// number) is checked the same way.
#[test]
fn scaled_pencil_opposite_scales_keep_the_pair() {
	let tiny = f64::from_bits(1);
	for (a, b) in [(tiny, 1e300), (1e300, tiny), (-tiny, 1e300), (1e300, -tiny)]
	{
		let am = Mat::<f64>::from_fn(1, 1, |_, _| a);
		let bm = Mat::<f64>::from_fn(1, 1, |_, _| b);
		for vectors in [false, true] {
			let v = gevd_values(&am, &bm, 1.0, vectors);
			let what = format!("A = {a:e}, B = {b:e}, vectors = {vectors}");
			assert!(v.ok, "{what}: Err");
			let (alpha, beta) = (v.alpha[0], v.beta[0]);
			assert!(
				alpha.im == 0.0 && alpha.re.is_finite() && beta.is_finite(),
				"{what}: ({alpha:?}, {beta:e})"
			);
			assert!(
				alpha.re != 0.0 && beta != 0.0,
				"{what}: a nonzero component came back as zero: ({alpha:?}, {beta:e})"
			);
			// both quotients are the common factor, `+-` a power of two of moderate size
			let (qa, qb) = (alpha.re / a, beta / b);
			assert!(
				qa != 0.0 && (qa - qb).abs() <= TOL * qa.abs(),
				"{what}: ({alpha:?}, {beta:e}) is not a nonzero multiple of (a, b): {qa:e} against {qb:e}"
			);
		}
	}
}

/// `gevd_cplx` (`c64`) and `gevd_real` (`f32`) share the prescaling; their
/// eigenvalues at a tiny `s` must agree with `s = 1`.
#[test]
fn scaled_pencil_tiny_c64_f32() {
	use faer::linalg::gevd::gevd_cplx;
	let (a, b) = one_pair();
	let n = a.nrows();
	let lambda_c64 = |s: f64| -> (bool, Vec<c64>, Vec<c64>) {
		// a complex pencil: `(A + i A^T, B)` has non-real, unpaired eigenvalues
		let mut am = Mat::<c64>::from_fn(n, n, |i, j| {
			c64::new(s * a[(i, j)], s * a[(j, i)])
		});
		let mut bm =
			Mat::<c64>::from_fn(n, n, |i, j| c64::new(s * b[(i, j)], 0.0));
		let mut al = Diag::<c64>::zeros(n);
		let mut be = Diag::<c64>::zeros(n);
		let mut mem = MemBuffer::new(gevd_scratch::<c64>(
			n,
			ComputeEigenvectors::No,
			ComputeEigenvectors::No,
			ParShape::Seq,
			Spec::<GevdParams, c64>::default(),
		));
		let ok = gevd_cplx(
			am.as_mut(),
			bm.as_mut(),
			al.as_mut(),
			be.as_mut(),
			None,
			None,
			Par::Seq,
			MemStack::new(&mut mem),
			Spec::<GevdParams, c64>::default(),
		)
		.is_ok();
		let al: Vec<c64> = (0..n).map(|j| al.column_vector()[j]).collect();
		let be: Vec<c64> = (0..n).map(|j| be.column_vector()[j]).collect();
		(ok, al, be)
	};
	let lambda_f32 = |s: f32| -> (bool, Vec<c64>, Vec<c64>) {
		let mut am = Mat::<f32>::from_fn(n, n, |i, j| s * a[(i, j)] as f32);
		let mut bm = Mat::<f32>::from_fn(n, n, |i, j| s * b[(i, j)] as f32);
		let mut re = Diag::<f32>::zeros(n);
		let mut im = Diag::<f32>::zeros(n);
		let mut be = Diag::<f32>::zeros(n);
		let mut mem = MemBuffer::new(gevd_scratch::<f32>(
			n,
			ComputeEigenvectors::No,
			ComputeEigenvectors::No,
			ParShape::Seq,
			Spec::<GevdParams, f32>::default(),
		));
		let ok = gevd_real(
			am.as_mut(),
			bm.as_mut(),
			re.as_mut(),
			im.as_mut(),
			be.as_mut(),
			None,
			None,
			Par::Seq,
			MemStack::new(&mut mem),
			Spec::<GevdParams, f32>::default(),
		)
		.is_ok();
		let al: Vec<c64> = (0..n)
			.map(|j| {
				c64::new(
					re.column_vector()[j] as f64,
					im.column_vector()[j] as f64,
				)
			})
			.collect();
		let be: Vec<c64> = (0..n)
			.map(|j| c64::new(be.column_vector()[j] as f64, 0.0))
			.collect();
		(ok, al, be)
	};
	// `alpha / beta` without squaring a tiny `beta` (num-complex's division
	// forms `|beta|^2`)
	let ratio = |al: &c64, be: &c64| {
		let m = al.norm().max(be.norm());
		(al / m) / (be / m)
	};
	let compare = |name: &str,
	               s: f64,
	               tol: f64,
	               r: (bool, Vec<c64>, Vec<c64>),
	               v: (bool, Vec<c64>, Vec<c64>)| {
		assert!(r.0 && v.0, "{name}, s = {s:e}: Err");
		for (al, be) in v.1.iter().zip(&v.2) {
			assert!(
				!(*al == c64::new(0.0, 0.0) && *be == c64::new(0.0, 0.0)),
				"{name}, s = {s:e}: alpha = beta = 0 returned with Ok"
			);
		}
		let lambda: Vec<c64> =
			v.1.iter().zip(&v.2).map(|(a, b)| ratio(a, b)).collect();
		let lambda_ref: Vec<c64> =
			r.1.iter().zip(&r.2).map(|(a, b)| ratio(a, b)).collect();
		let err = spectrum_distance(&lambda, &lambda_ref);
		assert!(
			err <= tol,
			"{name}, s = {s:e}: spectrum {lambda:?} against {lambda_ref:?}, relative error {err:e} > {tol:e}"
		);
	};
	for s in [1e-290, 1e-305] {
		compare("c64", s, TOL, lambda_c64(1.0), lambda_c64(s));
	}
	// `f32`: `TOL` scaled by the ratio of the epsilons
	let tol32 = TOL * (f32::EPSILON as f64 / f64::EPSILON);
	for s in [1e-30f32, 1e-36] {
		compare("f32", s as f64, tol32, lambda_f32(1.0), lambda_f32(s));
	}
}

/// `alpha` (real and imaginary parts) and `beta` of `(s A, s B)`, with or
/// without the eigenvectors, and whether the call returned `Ok`.
struct Values {
	alpha: Vec<c64>,
	beta: Vec<f64>,
	ok: bool,
}

fn gevd_values(a: &Mat<f64>, b: &Mat<f64>, s: f64, vectors: bool) -> Values {
	let n = a.nrows();
	let mut a = Mat::<f64>::from_fn(n, n, |i, j| s * a[(i, j)]);
	let mut b = Mat::<f64>::from_fn(n, n, |i, j| s * b[(i, j)]);
	let mut s_re = Diag::<f64>::zeros(n);
	let mut s_im = Diag::<f64>::zeros(n);
	let mut beta = Diag::<f64>::zeros(n);
	let mut u_left = Mat::<f64>::zeros(n, n);
	let mut u_right = Mat::<f64>::zeros(n, n);
	let ce = if vectors {
		ComputeEigenvectors::Yes
	} else {
		ComputeEigenvectors::No
	};
	let mut mem = MemBuffer::new(gevd_scratch::<f64>(
		n,
		ce,
		ce,
		ParShape::Seq,
		Spec::<GevdParams, f64>::default(),
	));
	let ok = gevd_real(
		a.as_mut(),
		b.as_mut(),
		s_re.as_mut(),
		s_im.as_mut(),
		beta.as_mut(),
		vectors.then(|| u_left.as_mut()),
		vectors.then(|| u_right.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Spec::<GevdParams, f64>::default(),
	)
	.is_ok();
	Values {
		alpha: (0..n)
			.map(|j| c64::new(s_re.column_vector()[j], s_im.column_vector()[j]))
			.collect(),
		beta: (0..n).map(|j| beta.column_vector()[j]).collect(),
		ok,
	}
}

/// `true` when some eigenvalue came out as `alpha = beta = 0`.
fn has_zero_pair(v: &Values) -> bool {
	v.alpha
		.iter()
		.zip(&v.beta)
		.any(|(a, &b)| *a == c64::new(0.0, 0.0) && b == 0.0)
}

/// tensor4all/faer#375: the lower end of the QZ's range. For each pencil and
/// for the eigenvalue-only and eigenvector calls, scans `s = 10^(k/10)` and
/// bisects (in `log10 s`) every change of "some `alpha = beta = 0`" and of "the
/// eigenvalues agree with `s = 1` to `TOL`". Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn scan_zero_eigenvalue_boundary() {
	for (name, make) in PENCILS {
		let (a, b) = make();
		let reference = gevd_values(&a, &b, 1.0, false);
		let lambda_ref: Vec<c64> = reference
			.alpha
			.iter()
			.zip(&reference.beta)
			.map(|(a, &b)| a / b)
			.collect();
		for vectors in [false, true] {
			let state = |e: f64| {
				let v = gevd_values(&a, &b, 10f64.powf(e), vectors);
				let zero = has_zero_pair(&v);
				let accurate = v.ok
					&& v.alpha.iter().zip(&v.beta).all(|(a, &b)| {
						let l = a / b;
						lambda_ref
							.iter()
							.map(|r| (r - l).norm() / r.norm())
							.fold(f64::INFINITY, f64::min)
							<= TOL
					});
				(zero, v.ok, accurate)
			};
			let mut prev: Option<(f64, (bool, bool, bool))> = None;
			for k in -3240..=-2500 {
				let e = k as f64 / 10.0;
				let st = state(e);
				if let Some((e0, st0)) = prev {
					if st != st0 {
						// bisect the first component that changed
						let (mut lo, mut hi) = (e0, e);
						for _ in 0..50 {
							let mid = 0.5 * (lo + hi);
							if state(mid) == st0 {
								lo = mid;
							} else {
								hi = mid;
							}
						}
						println!(
							"{name}, vectors = {vectors}: (zero, ok, accurate) = {st0:?} at s = {:.6e}, {st:?} at s = {:.6e}",
							10f64.powf(lo),
							10f64.powf(hi),
						);
					}
				}
				prev = Some((e, st));
			}
			println!(
				"{name}, vectors = {vectors}: at s = 1e-250 {:?}",
				state(-250.0)
			);
		}
	}
}

/// Prints, for each pencil and each `s = 10^k`, whether the eigenvectors are
/// finite and how far they are from `s = 1`. Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn scan_scaled_pencil() {
	for (name, make) in PENCILS {
		let (a, b) = make();
		let (reference, ..) = gevd(&a, &b, 1.0).unwrap();
		let mut ok = Vec::new();
		let mut worst = (0.0f64, 0i32);
		for k in -320..=308 {
			let s = format!("1e{k}").parse::<f64>().unwrap();
			if s == 0.0 || !s.is_finite() {
				continue;
			}
			// the scaled pencil must be representable with full precision:
			// every nonzero entry normal and none overflowing
			let normal = |m: &Mat<f64>| {
				(0..m.nrows()).all(|i| {
					(0..m.ncols()).all(|j| {
						let x = s * m[(i, j)];
						x.is_finite()
							&& (m[(i, j)] == 0.0
								|| x.abs() >= f64::MIN_POSITIVE)
					})
				})
			};
			if !normal(&a) || !normal(&b) {
				continue;
			}
			let Ok((scaled, ul, ur)) = gevd(&a, &b, s) else {
				println!("  1e{k}: Err");
				ok.push((k, false, false, f64::NAN));
				continue;
			};
			let finite = ul.is_all_finite() && ur.is_all_finite();
			let d = compare(&reference, &scaled);
			let m = d.lambda.max(d.left).max(d.right);
			let good = finite && m <= TOL;
			if good && m > worst.0 {
				worst = (m, k);
			}
			ok.push((k, good, finite, m));
		}
		// maximal runs of good k
		let mut runs = Vec::new();
		let mut start = None;
		for (i, &(k, good, ..)) in ok.iter().enumerate() {
			match (good, start) {
				(true, None) => start = Some(k),
				(false, Some(s0)) => {
					runs.push((s0, ok[i - 1].0));
					start = None;
				},
				_ => {},
			}
		}
		if let Some(s0) = start {
			runs.push((s0, ok.last().unwrap().0));
		}
		// bisect each change of finiteness between neighbouring powers of ten
		for w in ok.windows(2) {
			let ((k0, _, f0, _), (k1, _, f1, _)) = (w[0], w[1]);
			if k1 == k0 + 1 && f0 != f1 {
				let finite_at = |e: f64| {
					let s = 10f64.powf(e);
					gevd(&a, &b, s).is_ok_and(|(_, ul, ur)| {
						ul.is_all_finite() && ur.is_all_finite()
					})
				};
				let (mut lo, mut hi) = (k0 as f64, k1 as f64);
				for _ in 0..40 {
					let mid = 0.5 * (lo + hi);
					if finite_at(mid) == f0 {
						lo = mid;
					} else {
						hi = mid;
					}
				}
				println!(
					"{name}: finite = {f0} at s = {:e}, {f1} at s = {:e}",
					10f64.powf(lo),
					10f64.powf(hi)
				);
			}
		}
		println!(
			"{name}: good exponent ranges {runs:?}, worst good diff {:e} at 1e{}",
			worst.0, worst.1
		);
		for &(k, good, finite, m) in &ok {
			if !good {
				println!("  1e{k}: finite = {finite}, diff = {m:e}");
			}
		}
	}
}

/// `gevd_cplx` (left + right) on `(s A, s B)` taken as a complex pencil, in the
/// same form as `gevd`: the eigenvectors are already complex columns.
fn gevd_c64(
	a: &Mat<f64>,
	b: &Mat<f64>,
	s: f64,
) -> Result<(Gevd, Mat<c64>, Mat<c64>), GevdError> {
	let n = a.nrows();
	let mut a = Mat::<c64>::from_fn(n, n, |i, j| c64::new(s * a[(i, j)], 0.0));
	let mut b = Mat::<c64>::from_fn(n, n, |i, j| c64::new(s * b[(i, j)], 0.0));
	let mut alpha = Diag::<c64>::zeros(n);
	let mut beta = Diag::<c64>::zeros(n);
	let mut u_left = Mat::<c64>::zeros(n, n);
	let mut u_right = Mat::<c64>::zeros(n, n);
	let mut mem = MemBuffer::new(gevd_scratch::<c64>(
		n,
		ComputeEigenvectors::Yes,
		ComputeEigenvectors::Yes,
		ParShape::Seq,
		Spec::<GevdParams, c64>::default(),
	));
	gevd_cplx(
		a.as_mut(),
		b.as_mut(),
		alpha.as_mut(),
		beta.as_mut(),
		Some(u_left.as_mut()),
		Some(u_right.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Spec::<GevdParams, c64>::default(),
	)?;
	// `alpha / beta` with both scaled by `max(|alpha|, |beta|)` first: the
	// plain complex quotient squares `|beta|`, which overflows at `s = 1e154`
	// (the defect of tensor4all/faer#370, in the oracle instead of the code)
	let lambda: Vec<c64> = (0..n)
		.map(|j| {
			let (x, y) = (alpha.column_vector()[j], beta.column_vector()[j]);
			let m = x.norm().max(y.norm());
			(x / m) / (y / m)
		})
		.collect();
	Ok((
		Gevd {
			lambda,
			left: u_left.clone(),
			right: u_right.clone(),
		},
		u_left,
		u_right,
	))
}

/// The scales tensor4all/faer#370 names: the two ends of the old range
/// (`1e-146`, `1e154`), one step outside each (`1e-150`, `1e160`), and the
/// control `s = 1`. With `faer/src/linalg/gevd/mod.rs` from before
/// tensor4all/faer#374 (`7167896`), `gevd_real` returned `Ok` with NaN left and
/// right eigenvectors for `two_pairs` at `1e154` and `1e160` (16 NaN entries
/// each) and for `repeated_pair` at `1e-150` and `1e160` (20 each);
/// `one_pair`, whose pair meets only 1 by 1 blocks, was finite at all five.
/// `gevd_cplx` passed at all five before and after.
const ISSUE_SCALES: [f64; 5] = [1e-150, 1e-146, 1.0, 1e154, 1e160];

fn nan_count<T: Copy>(m: &Mat<T>, is_nan: impl Fn(T) -> bool) -> usize {
	(0..m.nrows())
		.map(|i| (0..m.ncols()).filter(|&j| is_nan(m[(i, j)])).count())
		.sum()
}

/// Runs every pencil at every entry of `scales` through `run`, prints one
/// row per case (finite, NaN count of left + right, differences against
/// `s = 1`), and only then asserts, so a failing run still prints the whole
/// table.
fn check_issue_scales<U>(
	kind: &str,
	scales: &[f64],
	run: impl Fn(&Mat<f64>, &Mat<f64>, f64) -> Result<(Gevd, U, U), GevdError>,
	nans: impl Fn(&U) -> usize,
) {
	let mut failures = Vec::new();
	println!(
		"{kind}: pencil, s, finite, NaN entries, eigenvalue, left sin, right sin"
	);
	for (name, make) in PENCILS {
		let (a, b) = make();
		let (reference, ..) = run(&a, &b, 1.0).unwrap();
		for &s in scales {
			let row = match run(&a, &b, s) {
				Err(e) => {
					failures.push(format!("{name}, s = {s:e}: {e:?}"));
					format!("{name}, {s:e}, Err({e:?})")
				},
				Ok((scaled, ul, ur)) => {
					let nan = nans(&ul) + nans(&ur);
					let d = compare(&reference, &scaled);
					let finite = nan == 0
						&& scaled.left.is_all_finite()
						&& scaled.right.is_all_finite();
					if !(finite
						&& d.lambda <= TOL && d.left <= TOL && d.right <= TOL)
					{
						failures.push(format!(
							"{name}, s = {s:e}: finite = {finite}, NaN = {nan}, \
							 eigenvalue {:e}, left sin {:e}, right sin {:e} \
							 (tolerance {TOL:e})",
							d.lambda, d.left, d.right
						));
					}
					format!(
						"{name}, {s:e}, {finite}, {nan}, {:e}, {:e}, {:e}",
						d.lambda, d.left, d.right
					)
				},
			};
			println!("{kind}: {row}");
		}
	}
	assert!(failures.is_empty(), "{kind}:\n{}", failures.join("\n"));
}

/// tensor4all/faer#370 at the issue's scales, `f64` (`gevd_real`): the complex
/// pairs' eigenvectors go through `solve_complex_shifted_2x2`, the path #374
/// changed.
#[test]
fn issue_scales_f64() {
	check_issue_scales("f64", &ISSUE_SCALES, gevd, |u: &Mat<f64>| {
		nan_count(u, f64::is_nan)
	});
}

/// tensor4all/faer#370 at the issue's scales, `c64` (`gevd_cplx` on the same
/// real pencils): its back substitution does not use the real 2 by 2 solve, and
/// this asserts what #374 only measured.
#[test]
fn issue_scales_c64() {
	check_issue_scales("c64", &ISSUE_SCALES, gevd_c64, |u: &Mat<c64>| {
		nan_count(u, |z: c64| z.re.is_nan() || z.im.is_nan())
	});
}

/// The ends of the range the `gevd_real` and `gevd_cplx` rustdoc guarantees
/// (every entry of `(s A, s B)` normal and finite): `1e-305`, the smallest
/// power of ten at which all three pencils are normal, and `1e307`, the
/// largest at which none overflows.
const RANGE_ENDS: [f64; 2] = [1e-305, 1e307];

/// Past the upper end the entries overflow to infinity and both functions
/// return `Err(NoConvergence)`, the documented outcome, never `Ok`.
const OVERFLOWING: f64 = 1e308;

#[test]
fn documented_range_ends_f64() {
	check_issue_scales("f64", &RANGE_ENDS, gevd, |u: &Mat<f64>| {
		nan_count(u, f64::is_nan)
	});
	for (name, make) in PENCILS {
		let (a, b) = make();
		let r = gevd(&a, &b, OVERFLOWING);
		assert!(
			matches!(r, Err(GevdError::NoConvergence)),
			"f64 {name}, s = {OVERFLOWING:e}: expected Err(NoConvergence), got Ok"
		);
	}
}

#[test]
fn documented_range_ends_c64() {
	check_issue_scales("c64", &RANGE_ENDS, gevd_c64, |u: &Mat<c64>| {
		nan_count(u, |z: c64| z.re.is_nan() || z.im.is_nan())
	});
	for (name, make) in PENCILS {
		let (a, b) = make();
		let r = gevd_c64(&a, &b, OVERFLOWING);
		assert!(
			matches!(r, Err(GevdError::NoConvergence)),
			"c64 {name}, s = {OVERFLOWING:e}: expected Err(NoConvergence), got Ok"
		);
	}
}
