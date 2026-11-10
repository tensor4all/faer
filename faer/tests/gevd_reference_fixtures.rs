//! faer#234: differential test of `gevd_real` against Reference LAPACK
//! (`dggev`, `dgges`, `dtgsen`) on fixtures generated once, offline, by
//! `tests/fixtures/gevd_reference/generate.jl` (Julia 1.12.5, OpenBLAS 0.3.29,
//! LAPACK 3.12.0; each fixture's header records what wrote it).
//!
//! The comparison is against the conditioning of the pencil, not elementwise.
//! With `eps` the unit roundoff and `|(A, B)| = hypot(|A|_1, |B|_1)`, each
//! quantity is divided by its first-order error bound and the ratio must stay
//! below 20, the threshold of Reference LAPACK's `TESTING/dgg.in`:
//!
//! - eigenvalue: chordal distance to the matched reference eigenvalue over
//!   `eps |(A, B)| / RCONDE`, RCONDE being the reference's reciprocal condition
//!   number of that eigenvalue (`dtgsna`'s definition). When RCONDE is zero the
//!   bound is vacuous and only the distance is reported;
//! - residual: `|beta A x - alpha B x|_2 / ((|beta| |A|_F + |alpha| |B|_F)
//!   |x|_2)` over `eps`, for every right and left vector;
//! - vector: the sine of the largest angle between the real span of our vector
//!   (`{re x, im x}` for a complex pair, one 2-D subspace rather than two
//!   vectors) and the reference deflating subspace of its eigenvalue's cluster,
//!   over `eps |(A, B)| / Dif`, `Dif = min(Difu, Difl)` of that cluster. A
//!   vector of a repeated or nearly repeated eigenvalue is not determined, its
//!   cluster's subspace is. The left side uses the left deflating subspace,
//!   which holds `gevd_real`'s left vectors (`y^T A = lambda y^T B`) and
//!   LAPACK's (`y^H A = lambda y^H B`) alike.
//!
//! For a cluster the reference finds semisimple, our vectors must also span it:
//! the smallest singular value of the normalized real basis exceeds `sqrt(eps)`.
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{GevdParams, gevd_real, gevd_scratch};
use faer::{Mat, MatRef, Par, ParShape, Spec, c64};

const THRESH: f64 = 20.0;

struct Cluster {
	dif: f64,
	semisimple: bool,
	right: Mat<f64>,
	left: Mat<f64>,
}

struct Fixture {
	name: String,
	a: Mat<f64>,
	b: Mat<f64>,
	alpha: Vec<c64>,
	beta: Vec<f64>,
	/// `dggev`'s vectors, real storage
	vr: Mat<f64>,
	vl: Mat<f64>,
	rconde: Vec<f64>,
	cluster_of: Vec<usize>,
	clusters: Vec<Cluster>,
}

struct Tokens<'a>(core::iter::Peekable<core::str::SplitWhitespace<'a>>);

impl Tokens<'_> {
	fn word(&mut self) -> &str {
		self.0.next().expect("truncated fixture")
	}

	fn expect(&mut self, key: &str) {
		let w = self.word();
		assert!(w == key, "expected `{key}`, found `{w}`");
	}

	fn num<T: core::str::FromStr>(&mut self) -> T
	where
		T::Err: core::fmt::Debug,
	{
		self.word().parse().unwrap()
	}

	fn vec<T: core::str::FromStr>(&mut self, key: &str) -> Vec<T>
	where
		T::Err: core::fmt::Debug,
	{
		self.expect(key);
		let n: usize = self.num();
		(0..n).map(|_| self.num()).collect()
	}

	fn mat(&mut self, key: &str) -> Mat<f64> {
		self.expect(key);
		let (m, n): (usize, usize) = (self.num(), self.num());
		let v: Vec<f64> = (0..m * n).map(|_| self.num()).collect();
		Mat::from_fn(m, n, |i, j| v[i * n + j])
	}
}

fn parse(text: &str) -> Fixture {
	let body: String = text
		.lines()
		.filter(|l| !l.starts_with('#'))
		.collect::<Vec<_>>()
		.join("\n");
	let mut t = Tokens(body.split_whitespace().peekable());
	t.expect("name");
	let name = t.word().to_owned();
	t.expect("n");
	let _n: usize = t.num();
	let a = t.mat("a");
	let b = t.mat("b");
	let re: Vec<f64> = t.vec("alpha_re");
	let im: Vec<f64> = t.vec("alpha_im");
	let beta = t.vec("beta");
	let vr = t.mat("vr");
	let vl = t.mat("vl");
	let rconde = t.vec("rconde");
	let cluster_of = t.vec("cluster_of");
	t.expect("clusters");
	let count: usize = t.num();
	let clusters = (0..count)
		.map(|c| {
			t.expect("cluster");
			assert!(t.num::<usize>() == c);
			t.expect("dim");
			let _dim: usize = t.num();
			t.expect("dif");
			let dif = t.num();
			t.expect("semisimple");
			let semisimple = t.num::<u8>() == 1;
			let right = t.mat("right_basis");
			let left = t.mat("left_basis");
			Cluster {
				dif,
				semisimple,
				right,
				left,
			}
		})
		.collect();
	assert!(t.0.peek().is_none(), "trailing data in fixture");
	Fixture {
		name,
		a,
		b,
		alpha: re.iter().zip(&im).map(|(&r, &i)| c64::new(r, i)).collect(),
		beta,
		vr,
		vl,
		rconde,
		cluster_of,
		clusters,
	}
}

struct Ours {
	alpha: Vec<c64>,
	beta: Vec<f64>,
	left: Mat<f64>,
	right: Mat<f64>,
}

fn ours(a: &Mat<f64>, b: &Mat<f64>) -> Ours {
	let n = a.nrows();
	let mut a = a.clone();
	let mut b = b.clone();
	let mut s_re = Diag::<f64>::zeros(n);
	let mut s_im = Diag::<f64>::zeros(n);
	let mut beta = Diag::<f64>::zeros(n);
	let mut left = Mat::<f64>::zeros(n, n);
	let mut right = Mat::<f64>::zeros(n, n);
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
		Some(left.as_mut()),
		Some(right.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Spec::<GevdParams, f64>::default(),
	)
	.unwrap();
	Ours {
		alpha: (0..n)
			.map(|j| c64::new(s_re.column_vector()[j], s_im.column_vector()[j]))
			.collect(),
		beta: (0..n).map(|j| beta.column_vector()[j]).collect(),
		left,
		right,
	}
}

fn chordal(a0: c64, b0: f64, a1: c64, b1: f64) -> f64 {
	(a0 * b1 - a1 * b0).norm() / ((a0.norm().hypot(b0)) * (a1.norm().hypot(b1)))
}

fn norm_1(m: MatRef<'_, f64>) -> f64 {
	(0..m.ncols())
		.map(|j| (0..m.nrows()).map(|i| m[(i, j)].abs()).sum::<f64>())
		.fold(0.0, f64::max)
}

/// The real span of the vector for eigenvalue `j`: one column for a real
/// eigenvalue, the real and imaginary parts for a pair.
fn real_span(u: MatRef<'_, f64>, j: usize, pair: bool) -> Mat<f64> {
	let m = if pair { 2 } else { 1 };
	Mat::from_fn(u.nrows(), m, |i, k| u[(i, j + k)])
}

/// Orthonormal basis of `span(u)`, Gram-Schmidt twice.
fn orthonormal(u: MatRef<'_, f64>) -> Mat<f64> {
	let n = u.nrows();
	let mut q = u.to_owned();
	for k in 0..q.ncols() {
		for _ in 0..2 {
			for l in 0..k {
				let d: f64 = (0..n).map(|i| q[(i, l)] * q[(i, k)]).sum();
				for i in 0..n {
					q[(i, k)] -= d * q[(i, l)];
				}
			}
		}
		let nrm = (0..n).map(|i| q[(i, k)] * q[(i, k)]).sum::<f64>().sqrt();
		for i in 0..n {
			q[(i, k)] /= nrm;
		}
	}
	q
}

/// `sin` of the largest principal angle between `span(u)` (one or two
/// columns) and `span(w)`, `w` orthonormal.
fn sin_angle(u: MatRef<'_, f64>, w: MatRef<'_, f64>) -> f64 {
	let q = orthonormal(u);
	let p = &q - w * (w.transpose() * &q);
	let g = p.transpose() * &p;
	// largest eigenvalue of the 1x1 or 2x2 Gram matrix
	let lmax = if g.nrows() == 1 {
		g[(0, 0)]
	} else {
		let (x, y, z) = (g[(0, 0)], g[(0, 1)], g[(1, 1)]);
		0.5 * (x + z) + (0.25 * (x - z) * (x - z) + y * y).sqrt()
	};
	lmax.max(0.0).sqrt()
}

/// `|beta M x - alpha N x|_2 / ((|beta| |M|_F + |alpha| |N|_F) |x|_2)` for the
/// complex vector stored in `span` (`re + i im`).
fn residual(
	m: MatRef<'_, f64>,
	nm: MatRef<'_, f64>,
	span: MatRef<'_, f64>,
	alpha: c64,
	beta: f64,
) -> f64 {
	let n = m.nrows();
	let x = |i: usize| {
		c64::new(
			span[(i, 0)],
			if span.ncols() == 2 { span[(i, 1)] } else { 0.0 },
		)
	};
	let mut r = 0.0f64;
	for i in 0..n {
		let mut mx = c64::new(0.0, 0.0);
		let mut nx = c64::new(0.0, 0.0);
		for k in 0..n {
			mx += x(k) * m[(i, k)];
			nx += x(k) * nm[(i, k)];
		}
		r += (mx * beta - nx * alpha).norm_sqr();
	}
	let xn = (0..n).map(|i| x(i).norm_sqr()).sum::<f64>().sqrt();
	r.sqrt() / ((beta.abs() * m.norm_l2() + alpha.norm() * nm.norm_l2()) * xn)
}

#[derive(Default)]
struct Worst {
	eigenvalue: f64,
	residual: f64,
	right: f64,
	left: f64,
	/// largest chordal distance where RCONDE = 0 leaves no bound
	unbounded_eigenvalue: Option<f64>,
	/// in a defective cluster, the largest angle (sine) from our vector to the
	/// nearest of LAPACK's vectors of that cluster. Reported, not asserted:
	/// the first-order bounds above do not apply to a defective eigenvalue,
	/// whose vector moves by about `eps^(1/m)` for a Jordan block of size `m`
	defective_vector: Option<f64>,
	/// smallest `sigma_min` of our normalized basis over the semisimple
	/// clusters holding more than one vector
	rank: Option<f64>,
	failures: Vec<String>,
}

fn check(fixture: &str) -> Worst {
	let f = parse(fixture);
	let n = f.a.nrows();
	let eps = f64::EPSILON;
	let abnorm = norm_1(f.a.as_ref()).hypot(norm_1(f.b.as_ref()));
	let o = ours(&f.a, &f.b);
	assert!(o.left.is_all_finite() && o.right.is_all_finite());

	// match our eigenvalues to the reference's, nearest first
	let mut d: Vec<(f64, usize, usize)> = (0..n)
		.flat_map(|i| (0..n).map(move |j| (i, j)))
		.map(|(i, j)| {
			(chordal(o.alpha[i], o.beta[i], f.alpha[j], f.beta[j]), i, j)
		})
		.collect();
	d.sort_by(|x, y| x.0.total_cmp(&y.0));
	let mut to_ref = vec![usize::MAX; n];
	let mut used = vec![false; n];
	for (_, i, j) in d {
		if to_ref[i] == usize::MAX && !used[j] {
			to_ref[i] = j;
			used[j] = true;
		}
	}

	let mut w = Worst::default();
	// the match has to preserve conjugate pairs: the `rconde == 0` branch below
	// records a distance but has no bound to compare it against, so a corrupted
	// member of a pair could consume an unbounded reference entry in silence.
	// compare eigenvalues, not `alpha`: `dggev` writes a pair as conjugate
	// `alpha` with two different `beta`, so only the quotients are conjugate
	// (and only to the 17 digits the fixture was written at)
	let eig_ours: Vec<c64> = (0..n).map(|i| o.alpha[i] / o.beta[i]).collect();
	let eig_ref: Vec<c64> = (0..n).map(|j| f.alpha[j] / f.beta[j]).collect();
	let near_conj =
		|x: c64, y: c64| (x - y.conj()).norm() <= 1e-12 * (1.0 + x.norm());
	for i in 0..n {
		if eig_ours[i].im == 0.0 {
			continue;
		}
		let j = to_ref[i];
		let mut partner = false;
		for l in 0..n {
			if l == i || !near_conj(eig_ours[l], eig_ours[i]) {
				continue;
			}
			partner = true;
			let k = to_ref[l];
			if !near_conj(eig_ref[k], eig_ref[j]) {
				w.failures.push(format!(
					"{}: conjugate pair ({i}, {l}) ({:?}, {:?}) is matched onto reference ({j}, {k}) ({:?}, {:?}), which is not a pair",
					f.name, eig_ours[i], eig_ours[l], eig_ref[j], eig_ref[k]
				));
			}
		}
		if !partner {
			w.failures.push(format!(
				"{}: eigenvalue {i} ({:?}) is complex but has no conjugate partner",
				f.name, eig_ours[i]
			));
		}
	}

	for i in 0..n {
		let j = to_ref[i];
		let dist = chordal(o.alpha[i], o.beta[i], f.alpha[j], f.beta[j]);
		if f.rconde[j] == 0.0 {
			let u = w.unbounded_eigenvalue.get_or_insert(0.0);
			*u = u.max(dist);
		} else {
			let ratio = dist / (eps * abnorm / f.rconde[j]);
			w.eigenvalue = w.eigenvalue.max(ratio);
			if !(ratio < THRESH) {
				w.failures.push(format!(
					"{}: eigenvalue {i} ({:?}): chordal {dist:.3e}, rconde {:.3e}, ratio {ratio:.1}",
					f.name,
					o.alpha[i] / o.beta[i],
					f.rconde[j]
				));
			}
		}
	}

	let mut i = 0;
	// per cluster, the real spans of our right and of our left vectors
	let mut basis: Vec<[Vec<Mat<f64>>; 2]> =
		vec![[Vec::new(), Vec::new()]; f.clusters.len()];
	while i < n {
		let pair = o.alpha[i].im != 0.0;
		assert!(
			!pair || o.alpha[i].im > 0.0,
			"pair at {i} starts with im < 0"
		);
		let c = f.cluster_of[to_ref[i]];
		let cl = &f.clusters[c];
		let bound = eps * abnorm / cl.dif;
		for (s, side, u, wb, m, nm) in [
			(0, "right", &o.right, &cl.right, f.a.as_ref(), f.b.as_ref()),
			(
				1,
				"left",
				&o.left,
				&cl.left,
				f.a.transpose(),
				f.b.transpose(),
			),
		] {
			let span = real_span(u.as_ref(), i, pair);
			let r = residual(m, nm, span.as_ref(), o.alpha[i], o.beta[i]) / eps;
			w.residual = w.residual.max(r);
			if !(r < THRESH) {
				w.failures.push(format!(
					"{}: {side} vector {i}: residual ratio {r:.1}",
					f.name
				));
			}
			let ratio = if cl.dif < 0.0 {
				0.0
			} else {
				sin_angle(span.as_ref(), wb.as_ref()) / bound
			};
			if s == 0 {
				w.right = w.right.max(ratio);
			} else {
				w.left = w.left.max(ratio);
			}
			if !cl.semisimple {
				let v = if s == 0 { &f.vr } else { &f.vl };
				let nearest = (0..n)
					.filter(|&k| {
						f.cluster_of[k] == c
							&& (f.alpha[k].im == 0.0
								|| (f.alpha[k].im > 0.0
									&& k + 1 < n && f.alpha[k + 1].im < 0.0))
					})
					.map(|k| {
						let r = real_span(v.as_ref(), k, f.alpha[k].im != 0.0);
						let q = orthonormal(r.as_ref());
						sin_angle(span.as_ref(), q.as_ref())
					})
					.fold(f64::INFINITY, f64::min);
				let e = w.defective_vector.get_or_insert(0.0);
				*e = e.max(nearest);
			}
			basis[c][s].push(span);
			if !(ratio < THRESH) {
				w.failures.push(format!(
					"{}: {side} vector {i} (cluster {c}): sin angle / bound {ratio:.1}, dif {:.3e}",
					f.name, cl.dif
				));
			}
		}
		i += if pair { 2 } else { 1 };
	}
	// a semisimple cluster holding more than one vector must be spanned, not
	// just touched: two copies of one vector would pass the angle check
	for (c, cl) in f.clusters.iter().enumerate() {
		if !cl.semisimple || cl.right.ncols() <= 2 {
			continue;
		}
		for (s, side) in [(0, "right"), (1, "left")] {
			let cols: Vec<_> = basis[c][s]
				.iter()
				.flat_map(|v| (0..v.ncols()).map(move |k| v.col(k).to_owned()))
				.collect();
			let m = Mat::from_fn(n, cols.len(), |r, k| {
				cols[k][r] / cols[k].norm_l2()
			});
			let smin = m
				.singular_values(Par::Seq)
				.unwrap()
				.into_iter()
				.fold(f64::INFINITY, f64::min);
			let entry = w.rank.get_or_insert(f64::INFINITY);
			*entry = entry.min(smin);
			if !(smin > eps.sqrt()) {
				w.failures.push(format!(
					"{}: cluster {c}: {side} vectors do not span it, sigma_min {smin:.3e}",
					f.name
				));
			}
		}
	}
	w
}

fn report(name: &str, w: &Worst) {
	println!(
		"{name}: max ratios: eigenvalue {:.2}, residual {:.2}, right angle {:.2}, left angle {:.2}{}{}{}",
		w.eigenvalue,
		w.residual,
		w.right,
		w.left,
		match w.unbounded_eigenvalue {
			Some(d) =>
				format!(", unbounded (rconde = 0) eigenvalue chordal {d:.2e}"),
			None => String::new(),
		},
		match w.defective_vector {
			Some(d) => format!(
				", defective: sin angle to LAPACK's nearest vector {d:.2e}"
			),
			None => String::new(),
		},
		match w.rank {
			Some(s) => format!(", semisimple basis sigma_min {s:.2e}"),
			None => String::new(),
		},
	);
	for msg in &w.failures {
		println!("  FAIL {msg}");
	}
}

macro_rules! fixture_test {
	($name:ident) => {
		#[test]
		fn $name() {
			let w = check(include_str!(concat!(
				"fixtures/gevd_reference/",
				stringify!($name),
				".txt"
			)));
			report(stringify!($name), &w);
			assert!(w.failures.is_empty(), "{} failure(s)", w.failures.len());
		}
	};
}

fixture_test!(semisimple_schur);
fixture_test!(semisimple_transformed);
fixture_test!(defective_schur);
fixture_test!(near_defective_transformed);
fixture_test!(defective_chain22_schur);
fixture_test!(defective_transformed);
fixture_test!(random12);
