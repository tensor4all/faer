//! Public contract of the rank truncation of the column-pivoted QR (#303):
//! `max_rank`/`abs_tol`/`rel_tol` in [`ColPivQrParams`] and `rank` in [`ColPivQrInfo`],
//! mirroring the dense LUs (#294).
//!
//! - options that never fire change no bit, and the default reports `rank = min(m, n)`;
//! - an exactly rank-`r` matrix reports `rank = r`, square, tall and wide, for
//!   f32/f64/c32/c64;
//! - `max_rank = r` stops after the untruncated run's first `r` steps, whose entries and
//!   permutation it reproduces bit for bit;
//! - `rel_tol` is measured against the first diagonal entry, not the previous one;
//! - a NaN or an infinity never turns into a truncated rank.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::qr::col_pivoting::factor::{
	self as cpqr, ColPivQrInfo, ColPivQrParams,
};
use faer::traits::ComplexField;
use faer::{Auto, Mat, MatRef, Par, c32, c64};

trait Entry: ComplexField + Copy {
	fn pair(re: f64, im: f64) -> Self;
	fn modulus(&self) -> f64;
	fn eps() -> f64;
}
impl Entry for f64 {
	fn pair(re: f64, _: f64) -> Self {
		re
	}
	fn modulus(&self) -> f64 {
		self.abs()
	}
	fn eps() -> f64 {
		f64::EPSILON
	}
}
impl Entry for f32 {
	fn pair(re: f64, _: f64) -> Self {
		re as f32
	}
	fn modulus(&self) -> f64 {
		self.abs() as f64
	}
	fn eps() -> f64 {
		f32::EPSILON as f64
	}
}
impl Entry for c64 {
	fn pair(re: f64, im: f64) -> Self {
		c64::new(re, im)
	}
	fn modulus(&self) -> f64 {
		self.norm()
	}
	fn eps() -> f64 {
		f64::EPSILON
	}
}
impl Entry for c32 {
	fn pair(re: f64, im: f64) -> Self {
		c32::new(re as f32, im as f32)
	}
	fn modulus(&self) -> f64 {
		self.norm() as f64
	}
	fn eps() -> f64 {
		f32::EPSILON as f64
	}
}

/// Deterministic entries in [-1, 1) from an integer generator (no libm, so no platform drift).
fn lcg_mat(m: usize, n: usize, seed: u64) -> Mat<f64> {
	let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
	Mat::from_fn(m, n, |_, _| {
		s = s
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
	})
}
fn sample<T: Entry>(m: usize, n: usize, seed: u64) -> Mat<T> {
	let re = lcg_mat(m, n, seed);
	let im = lcg_mat(m, n, seed + 1000);
	Mat::from_fn(m, n, |i, j| T::pair(re[(i, j)], im[(i, j)]))
}
/// An exactly rank-`r` product `X Y` of two random factors.
fn rank_r<T: Entry>(m: usize, n: usize, r: usize, seed: u64) -> Mat<T> {
	let x = sample::<T>(m, r, seed);
	let y = sample::<T>(r, n, seed + 7);
	&x * &y
}

struct Qr<T> {
	factor: Mat<T>,
	house: Mat<T>,
	col: Vec<usize>,
	info: ColPivQrInfo,
}
fn factor<T: Entry>(
	a: MatRef<'_, T>,
	par: Par<'_>,
	params: ColPivQrParams,
) -> Qr<T> {
	let (m, n) = a.shape();
	let size = Ord::min(m, n);
	let bs = 15;
	let mut qr = a.to_owned();
	let mut house = Mat::<T>::zeros(bs, size);
	let (mut p, mut pi) = (vec![0usize; n], vec![0usize; n]);
	let mut mem = MemBuffer::new(cpqr::qr_in_place_scratch::<usize, T>(
		m,
		n,
		bs,
		par.shape(),
		params.into(),
	));
	let (info, _) = cpqr::qr_in_place(
		qr.as_mut(),
		house.as_mut(),
		&mut p,
		&mut pi,
		par,
		MemStack::new(&mut mem),
		params.into(),
	);
	Qr {
		factor: qr,
		house,
		col: p,
		info,
	}
}
fn params<T: Entry>(
	max_rank: Option<usize>,
	abs_tol: Option<f64>,
	rel_tol: Option<f64>,
) -> ColPivQrParams {
	ColPivQrParams {
		max_rank,
		abs_tol,
		rel_tol,
		..<ColPivQrParams as Auto<T>>::auto()
	}
}

fn fnv(h: &mut u64, x: u64) {
	for b in x.to_le_bytes() {
		*h ^= b as u64;
		*h = h.wrapping_mul(0x100_0000_01b3);
	}
}
fn hash_f64(h: &mut u64, a: MatRef<'_, f64>) {
	for j in 0..a.ncols() {
		for i in 0..a.nrows() {
			fnv(h, a[(i, j)].to_bits());
		}
	}
}
fn hash_c64(h: &mut u64, a: MatRef<'_, c64>) {
	for j in 0..a.ncols() {
		for i in 0..a.nrows() {
			fnv(h, a[(i, j)].re.to_bits());
			fnv(h, a[(i, j)].im.to_bits());
		}
	}
}

const SHAPES: [(usize, usize); 6] = [
	(37, 37),
	(64, 48),
	(48, 64),
	(300, 17),
	(17, 300),
	(129, 129),
];

// ---- (b) options that never fire change no bit ------------------------------------------------

/// A `max_rank` of `min(m, n)` or more, and a zero tolerance on a matrix of full numerical
/// rank, are the default call: the factor, the `tau` panel, the permutation and the
/// transposition count stay bit-identical, and the rank is `min(m, n)`.
#[test]
fn inactive_options_change_no_bit() {
	for &(m, n) in &SHAPES {
		let a = sample::<c64>(m, n, 11);
		let size = Ord::min(m, n);
		for par_on_team in [false, true] {
			let run = |par: Par<'_>| {
				let mut par = par;
				let base = factor(
					a.as_ref(),
					par.reborrow(),
					params::<c64>(None, None, None),
				);
				for (mr, at, rt) in [
					(Some(usize::MAX), None, None),
					(Some(size), Some(0.0), Some(0.0)),
				] {
					let cut = factor(
						a.as_ref(),
						par.reborrow(),
						params::<c64>(mr, at, rt),
					);
					assert!(
						cut.factor == base.factor
							&& cut.house == base.house
							&& cut.col == base.col,
						"{m}x{n} {mr:?}"
					);
					assert_eq!(
						cut.info.transposition_count,
						base.info.transposition_count
					);
					assert_eq!(cut.info.rank, size, "{m}x{n} {mr:?}");
				}
				assert_eq!(base.info.rank, size);
			};
			if par_on_team {
				faer::team::with_team(4, run);
			} else {
				run(Par::Seq);
			}
		}
	}
}

// ---- (a) numerical rank ----------------------------------------------------------------------

fn check_rank<T: Entry>(rel_tol: f64) {
	for &(m, n, r) in &[
		(40usize, 40usize, 7usize),
		(60, 25, 9),
		(25, 60, 9),
		(33, 33, 1),
	] {
		let a = rank_r::<T>(m, n, r, (m * n + r) as u64);
		let q = factor(
			a.as_ref(),
			Par::Seq,
			params::<T>(None, None, Some(rel_tol)),
		);
		assert_eq!(q.info.rank, r, "{m}x{n} rank {r}");
		// the documented rule, checked on the returned factor: the first rejected diagonal
		// entry is at `rank`, and every accepted one is strictly above the threshold
		let pivot0 = q.factor[(0, 0)].modulus();
		let threshold = (rel_tol * pivot0).max(0.0);
		for k in 0..r {
			assert!(
				q.factor[(k, k)].modulus() > threshold,
				"{m}x{n}: accepted diagonal {k} = {}",
				q.factor[(k, k)].modulus()
			);
		}
		if r < Ord::min(m, n) {
			assert!(
				q.factor[(r, r)].modulus() <= threshold,
				"{m}x{n}: rejected diagonal {r} = {}",
				q.factor[(r, r)].modulus()
			);
		}
		// the same answer on a team, where the trailing updates are split
		let q = faer::team::with_team(4, |par| {
			factor(a.as_ref(), par, params::<T>(None, None, Some(rel_tol)))
		});
		assert_eq!(q.info.rank, r, "{m}x{n} rank {r} on a team");
	}
}
#[test]
fn numerical_rank_is_returned() {
	check_rank::<f64>(1e-10);
	check_rank::<c64>(1e-10);
	check_rank::<f32>(1e-3);
	check_rank::<c32>(1e-3);
}

// ---- (c) max_rank ----------------------------------------------------------------------------

/// The truncated run performs the untruncated run's first `r` steps with the same
/// arithmetic, so its permutation and every entry of `R[..r, :]` (the rows those steps
/// finalize) agree bit for bit; its own `rank` is `r`, and the columns past `r` are not part
/// of the factorization.
fn check_max_rank<T: Entry>() {
	for &(m, n, r) in &[
		(60usize, 45usize, 10usize),
		(45, 60, 13),
		(50, 50, 17),
		(200, 40, 33),
	] {
		let a = sample::<T>(m, n, (m + 3 * n) as u64);
		let whole = factor(a.as_ref(), Par::Seq, params::<T>(None, None, None));
		for par_on_team in [false, true] {
			let cut = if par_on_team {
				faer::team::with_team(4, |par| {
					factor(a.as_ref(), par, params::<T>(Some(r), None, None))
				})
			} else {
				factor(a.as_ref(), Par::Seq, params::<T>(Some(r), None, None))
			};
			assert_eq!(cut.info.rank, r, "{m}x{n} max_rank {r}");
			assert_eq!(
				cut.col[..r],
				whole.col[..r],
				"{m}x{n}: permutation prefix"
			);
			// the steps past `r` keep swapping the columns they do not factor, so an entry
			// is matched by the original column it holds
			let mut whole_inv = vec![0usize; n];
			for (i, &c) in whole.col.iter().enumerate() {
				whole_inv[c] = i;
			}
			for j in 0..n {
				let wj = whole_inv[cut.col[j]];
				for i in 0..r {
					assert!(
						cut.factor[(i, j)] == whole.factor[(i, wj)],
						"{m}x{n}: entry ({i}, {j}) of the truncated factor: {:?} vs {:?}",
						cut.factor[(i, j)],
						whole.factor[(i, wj)]
					);
				}
			}
		}
	}
	// `max_rank = 0` factors nothing: the permutation is the identity, the rank is zero and
	// the upper triangle is the input up to the scaling round trip
	let a = sample::<T>(7, 5, 1);
	let q = factor(a.as_ref(), Par::Seq, params::<T>(Some(0), None, None));
	assert_eq!(q.info.rank, 0);
	assert_eq!(q.info.transposition_count, 0);
	assert_eq!(q.col, (0..5).collect::<Vec<_>>());
	for j in 0..5 {
		for i in 0..=j {
			let d = (q.factor[(i, j)] - a[(i, j)]).modulus();
			assert!(d <= T::eps() * (1.0 + a[(i, j)].modulus()), "({i}, {j})");
		}
	}
}
#[test]
fn max_rank_stops_after_the_untruncated_steps() {
	check_max_rank::<f64>();
	check_max_rank::<c64>();
	check_max_rank::<f32>();
	check_max_rank::<c32>();
}

// ---- (d) rel_tol is relative to the first diagonal entry --------------------------------------

/// An upper triangular matrix whose diagonal is `1, 1e-3 (6e-4 real), 1e-6`: column pivoting
/// meets it in that order, so the diagonal of `R` is that sequence. `rel_tol = 1e-4` rejects
/// `1e-6` because it is below `1e-4 * |R[0, 0]|`, although it is above `1e-4 * |R[1, 1]|`
/// (the previous entry): the "relative to the previous entry" rule would accept it and
/// report rank 3.
fn graded<T: Entry>(m: usize, n: usize) -> Mat<T> {
	let mut a = Mat::<T>::zeros(m, n);
	a[(0, 0)] = T::pair(1.0, 0.0);
	// modulus 1e-3 for complex types, 6e-4 for real ones (only the real part is kept)
	a[(1, 1)] = T::pair(-6e-4, 8e-4);
	a[(2, 2)] = T::pair(-1e-6, 0.0);
	a
}
fn check_first_diagonal_rule<T: Entry>() {
	for (m, n) in [(5usize, 4usize), (4, 6), (3, 3)] {
		let a = graded::<T>(m, n);
		for (abs_tol, rel_tol, want) in [
			(None, Some(1e-4), 2),
			(None, Some(1e-7), 3),
			(Some(1e-8), None, 3),
			(Some(1e-5), None, 2),
			(Some(1e-8), Some(1e-4), 2),
			(Some(1e-2), Some(1e-9), 1),
			(Some(2.0), None, 0),
		] {
			let q = factor(
				a.as_ref(),
				Par::Seq,
				params::<T>(None, abs_tol, rel_tol),
			);
			assert_eq!(
				q.info.rank, want,
				"{m}x{n} abs {abs_tol:?} rel {rel_tol:?}"
			);
			// `max_rank` and a tolerance: the smaller wins
			let q = factor(
				a.as_ref(),
				Par::Seq,
				params::<T>(Some(1), abs_tol, rel_tol),
			);
			assert_eq!(q.info.rank, Ord::min(want, 1));
		}
	}
	// an all-zero matrix has rank 0 under any tolerance
	let z = Mat::<T>::zeros(4, 3);
	assert_eq!(
		factor(z.as_ref(), Par::Seq, params::<T>(None, Some(0.0), None))
			.info
			.rank,
		0
	);
	assert_eq!(
		factor(z.as_ref(), Par::Seq, params::<T>(None, None, Some(0.0)))
			.info
			.rank,
		0
	);
}
#[test]
fn rel_tol_is_relative_to_the_first_diagonal() {
	check_first_diagonal_rule::<f64>();
	check_first_diagonal_rule::<c64>();
	check_first_diagonal_rule::<f32>();
	check_first_diagonal_rule::<c32>();
}

// ---- NaN / infinity --------------------------------------------------------------------------

/// A rank-1 matrix with one non-finite entry: every tolerance would truncate it to rank 1 if
/// the non-finite entry were invisible. The documented choice is that a non-finite input
/// switches the tolerances off, so the rank is `min(m, n)` capped by `max_rank`.
#[test]
fn non_finite_input_is_never_a_truncated_rank() {
	for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
		for (m, n) in [(6usize, 5usize), (5, 6)] {
			let mut a = rank_r::<f64>(m, n, 1, 5);
			a[(2, 3)] = bad;
			for (max_rank, want) in [(None, Ord::min(m, n)), (Some(3), 3)] {
				let q = factor(
					a.as_ref(),
					Par::Seq,
					params::<f64>(max_rank, Some(1e-12), Some(1e-8)),
				);
				assert_eq!(q.info.rank, want, "{bad} {m}x{n}");
				assert!(
					!q.factor.is_all_finite(),
					"{bad} {m}x{n}: the non-finite entry is in the factor"
				);
			}
		}
	}
	// a tolerance that is not finite is ignored rather than rejecting every entry
	let a = sample::<f64>(5, 5, 9);
	assert_eq!(
		factor(
			a.as_ref(),
			Par::Seq,
			params::<f64>(None, Some(f64::INFINITY), None)
		)
		.info
		.rank,
		5
	);
}
