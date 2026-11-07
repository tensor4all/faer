//! Public contract of the rank truncation of the partial- and full-pivoting LU (#294):
//! `max_rank`/`abs_tol`/`rel_tol` in the params and `rank` in the info.
//!
//! - with every option `None` the factorization is the one before the options existed, bit for
//!   bit (a golden digest recorded from `1d36fbb`, and the explicit no-op options);
//! - an exactly rank-`r` matrix reports `rank = r`, square, tall and wide, for f32/f64/c32/c64;
//! - `max_rank` stops on the untruncated run's first pivots and leaves the documented trailing
//!   block (the Schur complement for full pivoting, the permuted input for partial pivoting);
//! - `rel_tol` is measured against the first pivot, not the previous one;
//! - a NaN or an infinity never turns into a truncated rank.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::full_pivoting::factor::{self as flu, FullPivLuParams};
use faer::linalg::lu::partial_pivoting::factor::{self as plu, PartialPivLuParams};
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
		s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

struct Flu<T> {
	lu: Mat<T>,
	row: Vec<usize>,
	col: Vec<usize>,
	info: flu::FullPivLuInfo,
}
fn full<T: Entry>(a: MatRef<'_, T>, par: Par<'_>, params: FullPivLuParams) -> Flu<T> {
	let (m, n) = a.shape();
	let mut lu = a.to_owned();
	let (mut p, mut pi, mut q, mut qi) = (vec![0usize; m], vec![0usize; m], vec![0usize; n], vec![0usize; n]);
	let mut mem = MemBuffer::new(flu::lu_in_place_scratch::<usize, T>(m, n, par.shape(), params.into()));
	let (info, _, _) =
		flu::lu_in_place(lu.as_mut(), &mut p, &mut pi, &mut q, &mut qi, par, MemStack::new(&mut mem), params.into());
	Flu { lu, row: p, col: q, info }
}
struct Plu<T> {
	lu: Mat<T>,
	row: Vec<usize>,
	info: plu::PartialPivLuInfo,
}
fn partial<T: Entry>(a: MatRef<'_, T>, par: Par<'_>, params: PartialPivLuParams) -> Plu<T> {
	let (m, n) = a.shape();
	let mut lu = a.to_owned();
	let (mut p, mut pi) = (vec![0usize; m], vec![0usize; m]);
	let mut mem = MemBuffer::new(plu::lu_in_place_scratch::<usize, T>(m, n, par.shape(), params.into()));
	let (info, _) = plu::lu_in_place(lu.as_mut(), &mut p, &mut pi, par, MemStack::new(&mut mem), params.into());
	Plu { lu, row: p, info }
}
fn fparams<T: Entry>(max_rank: Option<usize>, abs_tol: Option<f64>, rel_tol: Option<f64>) -> FullPivLuParams {
	FullPivLuParams { max_rank, abs_tol, rel_tol, ..<FullPivLuParams as Auto<T>>::auto() }
}
fn pparams<T: Entry>(max_rank: Option<usize>, abs_tol: Option<f64>, rel_tol: Option<f64>) -> PartialPivLuParams {
	PartialPivLuParams { max_rank, abs_tol, rel_tol, ..<PartialPivLuParams as Auto<T>>::auto() }
}

/// `max |(P A Q^T)[i, j] - (L[:, :r] U[:r, :])[i, j]|` over every entry outside the trailing
/// `(m - r) x (n - r)` block, and over the trailing block against `trailing(i, j)`.
fn residual<T: Entry>(
	a: MatRef<'_, T>,
	lu: MatRef<'_, T>,
	row: &[usize],
	col: Option<&[usize]>,
	r: usize,
	trailing: impl Fn(usize, usize) -> T,
) -> f64 {
	let (m, n) = a.shape();
	let mut worst: f64 = 0.0;
	for j in 0..n {
		for i in 0..m {
			let mut acc = T::pair(0.0, 0.0);
			for k in 0..Ord::min(r, Ord::min(i + 1, j + 1)) {
				let l = if k == i { T::pair(1.0, 0.0) } else { lu[(i, k)] };
				acc = acc + l * lu[(k, j)];
			}
			let aij = a[(row[i], col.map_or(j, |c| c[j]))];
			let want = if i >= r && j >= r { aij - trailing(i, j) } else { aij };
			worst = worst.max((want - acc).modulus());
		}
	}
	worst
}

fn inverse(perm: &[usize]) -> Vec<usize> {
	let mut inv = vec![0; perm.len()];
	for (i, &p) in perm.iter().enumerate() {
		inv[p] = i;
	}
	inv
}

// ---- (b) no option set: the factorization before the options existed -------------------------

fn fnv(h: &mut u64, x: u64) {
	for b in x.to_le_bytes() {
		*h ^= b as u64;
		*h = h.wrapping_mul(0x100_0000_01b3);
	}
}
const SHAPES: [(usize, usize); 6] = [(37, 37), (64, 48), (48, 64), (300, 17), (17, 300), (129, 129)];
fn hash_mat_f64(h: &mut u64, a: MatRef<'_, f64>) {
	for j in 0..a.ncols() {
		for i in 0..a.nrows() {
			fnv(h, a[(i, j)].to_bits());
		}
	}
}
fn hash_mat_c64(h: &mut u64, a: MatRef<'_, c64>) {
	for j in 0..a.ncols() {
		for i in 0..a.nrows() {
			fnv(h, a[(i, j)].re.to_bits());
			fnv(h, a[(i, j)].im.to_bits());
		}
	}
}
/// Digest of the default-parameter factorizations of `SHAPES` (f64 then c64 per shape): the
/// transposition count, the permutation(s) and every bit of the packed factor.
fn digest(par: &mut Par<'_>, full_pivoting: bool) -> u64 {
	let mut h = 0xcbf2_9ce4_8422_2325u64;
	for (idx, &(m, n)) in SHAPES.iter().enumerate() {
		let a = sample::<f64>(m, n, idx as u64);
		let ac = sample::<c64>(m, n, idx as u64);
		if full_pivoting {
			let f = full(a.as_ref(), par.reborrow(), fparams::<f64>(None, None, None));
			fnv(&mut h, f.info.transposition_count as u64);
			f.row.iter().chain(f.col.iter()).for_each(|&x| fnv(&mut h, x as u64));
			hash_mat_f64(&mut h, f.lu.as_ref());
			let f = full(ac.as_ref(), par.reborrow(), fparams::<c64>(None, None, None));
			fnv(&mut h, f.info.transposition_count as u64);
			f.row.iter().chain(f.col.iter()).for_each(|&x| fnv(&mut h, x as u64));
			hash_mat_c64(&mut h, f.lu.as_ref());
		} else {
			let f = partial(a.as_ref(), par.reborrow(), pparams::<f64>(None, None, None));
			fnv(&mut h, f.info.transposition_count as u64);
			f.row.iter().for_each(|&x| fnv(&mut h, x as u64));
			hash_mat_f64(&mut h, f.lu.as_ref());
			let f = partial(ac.as_ref(), par.reborrow(), pparams::<c64>(None, None, None));
			fnv(&mut h, f.info.transposition_count as u64);
			f.row.iter().for_each(|&x| fnv(&mut h, x as u64));
			hash_mat_c64(&mut h, f.lu.as_ref());
		}
	}
	h
}

/// Every option `None` (the default) reproduces the factorization of `1d36fbb`, the commit before
/// the options existed, bit for bit, sequential and on a team of 4. The digests were recorded on
/// that commit by the same `digest` on aarch64 (NEON); the SIMD kernels and their rounding differ
/// between architectures, so the golden is only compared there. The `rank` is `min(m, n)`.
#[test]
fn no_option_is_the_previous_factorization() {
	let mut seq = Par::Seq;
	let (p_seq, f_seq) = (digest(&mut seq, false), digest(&mut seq, true));
	let (p_team, f_team) =
		faer::team::with_team(4, |mut par| (digest(&mut par, false), digest(&mut par, true)));
	#[cfg(target_arch = "aarch64")]
	{
		assert_eq!(p_seq, GOLDEN_PLU_SEQ, "partial pivoting, Par::Seq");
		assert_eq!(p_team, GOLDEN_PLU_TEAM4, "partial pivoting, team of 4");
		assert_eq!(f_seq, GOLDEN_FLU_SEQ, "full pivoting, Par::Seq");
		assert_eq!(f_team, GOLDEN_FLU_TEAM4, "full pivoting, team of 4");
	}
	let _ = (p_seq, p_team, f_seq, f_team);
	for &(m, n) in &SHAPES {
		let a = sample::<f64>(m, n, 3);
		assert_eq!(full(a.as_ref(), Par::Seq, fparams::<f64>(None, None, None)).info.rank, Ord::min(m, n));
		assert_eq!(partial(a.as_ref(), Par::Seq, pparams::<f64>(None, None, None)).info.rank, Ord::min(m, n));
	}
	// an exactly zero matrix stops the full pivoting factorization early, as it always did; with
	// no option set the rank stays `min(m, n)`, so the old output is unchanged
	let z = Mat::<f64>::zeros(6, 4);
	assert_eq!(full(z.as_ref(), Par::Seq, fparams::<f64>(None, None, None)).info.rank, 4);
	assert_eq!(partial(z.as_ref(), Par::Seq, pparams::<f64>(None, None, None)).info.rank, 4);
}
#[cfg(target_arch = "aarch64")]
const GOLDEN_PLU_SEQ: u64 = 0x8bc4432078e1dd43;
#[cfg(target_arch = "aarch64")]
const GOLDEN_PLU_TEAM4: u64 = 0x28ef0fbc45f618c8;
#[cfg(target_arch = "aarch64")]
const GOLDEN_FLU_SEQ: u64 = 0xb8c2a6445097522c;
#[cfg(target_arch = "aarch64")]
const GOLDEN_FLU_TEAM4: u64 = 0xb8c2a6445097522c;

/// On every architecture: options that never fire (a `max_rank` of `min(m, n)` or more, a zero
/// tolerance on a full-rank matrix) leave the factor, the permutations and the transposition
/// count bit-identical to the default call.
#[test]
fn inactive_options_change_no_bit() {
	for &(m, n) in &SHAPES {
		let a = sample::<c64>(m, n, 11);
		for par_on_team in [false, true] {
			let run = |par: Par<'_>| {
				let mut par = par;
				let base_f = full(a.as_ref(), par.reborrow(), fparams::<c64>(None, None, None));
				let base_p = partial(a.as_ref(), par.reborrow(), pparams::<c64>(None, None, None));
				for (mr, at, rt) in [(Some(usize::MAX), None, None), (Some(Ord::min(m, n)), Some(0.0), Some(0.0))] {
					let f = full(a.as_ref(), par.reborrow(), fparams::<c64>(mr, at, rt));
					assert!(f.lu == base_f.lu && f.row == base_f.row && f.col == base_f.col, "{m}x{n} {mr:?}");
					assert_eq!(f.info.transposition_count, base_f.info.transposition_count);
					assert_eq!(f.info.rank, Ord::min(m, n));
					let p = partial(a.as_ref(), par.reborrow(), pparams::<c64>(mr, at, rt));
					assert!(p.lu == base_p.lu && p.row == base_p.row, "{m}x{n} {mr:?}");
					assert_eq!(p.info.transposition_count, base_p.info.transposition_count);
					assert_eq!(p.info.rank, Ord::min(m, n));
				}
			};
			if par_on_team {
				faer::team::with_team(4, run);
			} else {
				run(Par::Seq);
			}
		}
	}
}

// ---- (a) numerical rank -----------------------------------------------------------------------

fn check_rank<T: Entry>(rel_tol: f64) {
	for &(m, n, r) in &[(40usize, 40usize, 7usize), (60, 25, 9), (25, 60, 9), (33, 33, 1)] {
		let a = rank_r::<T>(m, n, r, (m * n + r) as u64);
		let tol = 50.0 * T::eps() * (Ord::max(m, n) as f64) * r as f64;
		let f = full(a.as_ref(), Par::Seq, fparams::<T>(None, None, Some(rel_tol)));
		assert_eq!(f.info.rank, r, "full pivoting {m}x{n} rank {r}");
		// the trailing block is the Schur complement, so the identity holds with it subtracted,
		// and the stop bounds it entrywise by sqrt(2) |rejected pivot| <= sqrt(2) rel_tol |pivot_0|
		let pivot0 = f.lu[(0, 0)].modulus();
		let s = |i: usize, j: usize| f.lu[(i, j)];
		assert!(residual(a.as_ref(), f.lu.as_ref(), &f.row, Some(&f.col), r, s) <= tol * pivot0);
		for j in r..n {
			for i in r..m {
				assert!(f.lu[(i, j)].modulus() <= 2f64.sqrt() * rel_tol * pivot0);
			}
		}
		let p = faer::team::with_team(4, |par| partial(a.as_ref(), par, pparams::<T>(None, None, Some(rel_tol))));
		assert_eq!(p.info.rank, r, "partial pivoting {m}x{n} rank {r}");
		let zero = |_: usize, _: usize| T::pair(0.0, 0.0);
		// `L[:, :r] U[:r, :]` reproduces the row-permuted input up to the rejected Schur
		// complement, which is rounding for an exactly rank-`r` input
		assert!(residual(a.as_ref(), p.lu.as_ref(), &p.row, None, r, zero) <= tol * pivot0 * 10.0);
	}
}
#[test]
fn numerical_rank_is_returned() {
	check_rank::<f64>(1e-10);
	check_rank::<c64>(1e-10);
	check_rank::<f32>(1e-3);
	check_rank::<c32>(1e-3);
}

// ---- (c) max_rank ------------------------------------------------------------------------------

fn check_max_rank<T: Entry>() {
	for &(m, n, r) in &[(60usize, 45usize, 10usize), (45, 60, 13), (50, 50, 17), (200, 40, 33)] {
		let a = sample::<T>(m, n, (m + 3 * n) as u64);
		let tol = 100.0 * T::eps() * (Ord::max(m, n) as f64);
		// full pivoting: the truncated run takes the untruncated run's first `r` steps, with the
		// same arithmetic, so its factors agree bit for bit, and the trailing block is the Schur
		// complement of those steps
		let whole = full(a.as_ref(), Par::Seq, fparams::<T>(None, None, None));
		let cut = full(a.as_ref(), Par::Seq, fparams::<T>(Some(r), None, None));
		assert_eq!(cut.info.rank, r);
		assert_eq!(cut.row[..r], whole.row[..r], "full pivoting rows {m}x{n}");
		assert_eq!(cut.col[..r], whole.col[..r], "full pivoting columns {m}x{n}");
		// the later steps of the untruncated run swap the rows of `L[r.., :r]` and the columns of
		// `U[:r, r..]`, so the entries are matched by their original row and column
		let (wrow, wcol) = (inverse(&whole.row), inverse(&whole.col));
		for j in 0..n {
			for i in 0..m {
				if i < r || j < r {
					let w = whole.lu[(wrow[cut.row[i]], wcol[cut.col[j]])];
					assert!(cut.lu[(i, j)] == w, "full pivoting factor ({i}, {j}) of {m}x{n}");
				}
			}
		}
		let s = |i: usize, j: usize| cut.lu[(i, j)];
		assert!(residual(a.as_ref(), cut.lu.as_ref(), &cut.row, Some(&cut.col), r, s) <= tol);

		// partial pivoting: the panel recursion of `r` columns splits its blocks differently
		// from the `min(m, n)` one, so the rows it selects agree exactly and the factor
		// entries to rounding; the trailing block is the row-permuted input, untouched
		let whole = partial(a.as_ref(), Par::Seq, pparams::<T>(None, None, None));
		for par_on_team in [false, true] {
			let cut = if par_on_team {
				faer::team::with_team(4, |par| partial(a.as_ref(), par, pparams::<T>(Some(r), None, None)))
			} else {
				partial(a.as_ref(), Par::Seq, pparams::<T>(Some(r), None, None))
			};
			assert_eq!(cut.info.rank, r);
			assert_eq!(cut.row[..r], whole.row[..r], "partial pivoting rows {m}x{n}");
			let wrow = inverse(&whole.row);
			for j in 0..n {
				for i in 0..m {
					if i < r || j < r {
						let w = whole.lu[(wrow[cut.row[i]], j)];
						assert!((cut.lu[(i, j)] - w).modulus() <= tol, "({i}, {j}) of {m}x{n}");
					} else {
						assert!(cut.lu[(i, j)] == a[(cut.row[i], j)], "trailing ({i}, {j}) of {m}x{n}");
					}
				}
			}
			// outside the trailing block `P A = L_r U_r` up to rounding (the trailing block, where
			// the difference is the Schur complement, was compared with the input above)
			let lr_ur = residual(a.as_ref(), cut.lu.as_ref(), &cut.row, None, r, |i, j| {
				let mut acc = T::pair(0.0, 0.0);
				for k in 0..r {
					acc = acc + cut.lu[(i, k)] * cut.lu[(k, j)];
				}
				a[(cut.row[i], j)] - acc
			});
			assert!(lr_ur <= tol);
		}
	}
	// `max_rank = 0` touches nothing but the rank
	let a = sample::<T>(7, 5, 1);
	assert_eq!(full(a.as_ref(), Par::Seq, fparams::<T>(Some(0), None, None)).lu, a);
	let p = partial(a.as_ref(), Par::Seq, pparams::<T>(Some(0), None, None));
	assert!(p.lu == a && p.info.rank == 0);
}
#[test]
fn max_rank_stops_on_the_untruncated_pivots() {
	check_max_rank::<f64>();
	check_max_rank::<c64>();
	check_max_rank::<f32>();
	check_max_rank::<c32>();
}

// ---- (d) rel_tol is relative to the first pivot -------------------------------------------------

/// A matrix whose pivots are exactly 1, 1e-3 (6e-4 for real types), 1e-6 (one nonzero per row and column, scattered),
/// in column order so partial pivoting meets them in that order too. 1e-6 is ten orders of
/// magnitude above the f64 machine precision; `rel_tol = 1e-4` rejects it because it is below
/// `1e-4 * |pivot_0|`, although it is above `1e-4 * |pivot_1|` (the previous pivot): the
/// "relative to the previous pivot" rule would accept it and return rank 3.
fn graded<T: Entry>(m: usize, n: usize) -> Mat<T> {
	let mut a = Mat::<T>::zeros(m, n);
	a[(m - 1, 0)] = T::pair(1.0, 0.0);
	// modulus 1e-3 for complex types, 6e-4 for real ones (only the real part is kept)
	a[(1, 1)] = T::pair(-6e-4, 8e-4);
	a[(0, 2)] = T::pair(-1e-6, 0.0);
	a
}
fn check_first_pivot_rule<T: Entry>() {
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
			let f = full(a.as_ref(), Par::Seq, fparams::<T>(None, abs_tol, rel_tol));
			assert_eq!(f.info.rank, want, "full {m}x{n} abs {abs_tol:?} rel {rel_tol:?}");
			let p = partial(a.as_ref(), Par::Seq, pparams::<T>(None, abs_tol, rel_tol));
			assert_eq!(p.info.rank, want, "partial {m}x{n} abs {abs_tol:?} rel {rel_tol:?}");
			// `max_rank` and a tolerance: the smaller wins
			let f = full(a.as_ref(), Par::Seq, fparams::<T>(Some(1), abs_tol, rel_tol));
			assert_eq!(f.info.rank, Ord::min(want, 1));
			let p = partial(a.as_ref(), Par::Seq, pparams::<T>(Some(1), abs_tol, rel_tol));
			assert_eq!(p.info.rank, Ord::min(want, 1));
		}
	}
	// an all-zero matrix has rank 0 under any tolerance
	let z = Mat::<T>::zeros(4, 3);
	assert_eq!(full(z.as_ref(), Par::Seq, fparams::<T>(None, Some(0.0), None)).info.rank, 0);
	assert_eq!(partial(z.as_ref(), Par::Seq, pparams::<T>(None, None, Some(0.0))).info.rank, 0);
}
#[test]
fn rel_tol_is_relative_to_the_first_pivot() {
	check_first_pivot_rule::<f64>();
	check_first_pivot_rule::<c64>();
	check_first_pivot_rule::<f32>();
	check_first_pivot_rule::<c32>();
}

// ---- NaN / infinity ------------------------------------------------------------------------------

/// A rank-1 matrix with one non-finite entry: every tolerance would truncate it to rank 1 if the
/// non-finite entry were invisible. The documented choice is that a non-finite input switches the
/// tolerances off, so the rank is `min(m, n)` (capped by `max_rank`) and the NaN reaches the
/// factor.
#[test]
fn non_finite_input_is_never_a_truncated_rank() {
	for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
		for (m, n) in [(6usize, 5usize), (5, 6)] {
			let mut a = rank_r::<f64>(m, n, 1, 5);
			a[(2, 3)] = bad;
			for (max_rank, want) in [(None, Ord::min(m, n)), (Some(3), 3)] {
				let f = full(a.as_ref(), Par::Seq, fparams::<f64>(max_rank, Some(1e-12), Some(1e-8)));
				assert_eq!(f.info.rank, want, "full {bad} {m}x{n}");
				assert!(!f.lu.is_all_finite(), "full {bad} {m}x{n}: the non-finite entry is in the factor");
				let p = partial(a.as_ref(), Par::Seq, pparams::<f64>(max_rank, Some(1e-12), Some(1e-8)));
				assert_eq!(p.info.rank, want, "partial {bad} {m}x{n}");
				assert!(!p.lu.is_all_finite(), "partial {bad} {m}x{n}: the non-finite entry is in the factor");
			}
		}
	}
	// a tolerance that is not finite (an infinite `abs_tol`) is ignored too, rather than rejecting
	// every pivot
	let a = sample::<f64>(5, 5, 9);
	assert_eq!(full(a.as_ref(), Par::Seq, fparams::<f64>(None, Some(f64::INFINITY), None)).info.rank, 5);
	assert_eq!(partial(a.as_ref(), Par::Seq, pparams::<f64>(None, Some(f64::INFINITY), None)).info.rank, 5);
}
