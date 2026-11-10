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
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{GevdError, GevdParams, gevd_real, gevd_scratch};
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

/// Far beyond the old range on both sides, where the pencil is still computed
/// correctly by the QZ.
#[test]
fn scaled_pencil_extremes() {
	check(1e-250);
	check(1e300);
}

/// Below about `1e-280` (for `one_pair`) the QZ itself underflows and returns
/// `alpha = beta = 0`, so the eigenvectors cannot be formed. That is reported
/// as an error rather than as NaN eigenvectors with `Ok`.
#[test]
fn scaled_pencil_below_range_is_an_error() {
	let (a, b) = one_pair();
	let r = gevd(&a, &b, 1e-290);
	assert!(
		matches!(r, Err(GevdError::NoConvergence)),
		"expected Err(NoConvergence), got {:?}",
		r.map(|(g, ul, ur)| (g.lambda, ul, ur))
	);
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
