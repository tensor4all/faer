//! Public contract of `faer::linalg::householder::upgrade_householder_factor` when the square
//! factor spans several target blocks (tensor4all/faer#265).
//!
//! The call takes an `n x n` factor whose diagonal holds the reflectors' `tau` (the denominators)
//! and whose `prev_block_size`-wide diagonal blocks already hold their Gram coefficients, and fills
//! in the rest of each `block_size`-wide diagonal block: for `i < j` in the same target block,
//! `T[i, j] = (V^H V)[i, j]`, where `V` is the unit lower trapezoidal basis (its diagonal is an
//! implicit one and its strict upper part is not read). The diagonal, the lower triangle and every
//! entry between two target blocks are left as the caller passed them.
//!
//! Before the fix, the multiple-block branch split the factor at `block_count / 2`, a number of
//! blocks used as a column offset, so a child could be cut through a block: the 9-column,
//! `block_size = 2` call of the issue left two coefficients zero and wrote one between blocks.

use faer::linalg::householder::upgrade_householder_factor;
use faer::{Mat, Par, c32, c64};

trait Scalar:
	faer_traits::ComplexField + Copy + PartialEq + core::fmt::Debug
{
	const EPS: f64;
	fn new(re: f64, im: f64) -> Self;
	fn widen(self) -> c64;
}

impl Scalar for f32 {
	const EPS: f64 = f32::EPSILON as f64;
	fn new(re: f64, _: f64) -> Self {
		re as f32
	}
	fn widen(self) -> c64 {
		c64::new(self as f64, 0.0)
	}
}

impl Scalar for f64 {
	const EPS: f64 = f64::EPSILON;
	fn new(re: f64, _: f64) -> Self {
		re
	}
	fn widen(self) -> c64 {
		c64::new(self, 0.0)
	}
}

impl Scalar for c32 {
	const EPS: f64 = f32::EPSILON as f64;
	fn new(re: f64, im: f64) -> Self {
		c32::new(re as f32, im as f32)
	}
	fn widen(self) -> c64 {
		c64::new(self.re as f64, self.im as f64)
	}
}

impl Scalar for c64 {
	const EPS: f64 = f64::EPSILON;
	fn new(re: f64, im: f64) -> Self {
		c64::new(re, im)
	}
	fn widen(self) -> c64 {
		self
	}
}

/// deterministic values in `[-1, 1)`
fn val(i: usize, j: usize, salt: u64) -> f64 {
	let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
		^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
		^ salt.wrapping_mul(0x1656_67B1_9E37_79F9);
	x ^= x >> 29;
	x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
	x ^= x >> 32;
	(x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// stored in every factor entry the call must not write
const SENTINEL: (f64, f64) = (-7.5, 2.25);

/// the basis: its strict lower part is data, its diagonal and strict upper part are garbage the
/// call must not read (as in a QR, where they hold `R`)
fn basis<T: Scalar>(m: usize, n: usize) -> Mat<T> {
	Mat::from_fn(m, n, |i, j| {
		if i > j {
			T::new(val(i, j, 1), val(i, j, 2))
		} else {
			T::new(1.0e3 + val(i, j, 3), -2.0e3)
		}
	})
}

/// `(V^H V)[i, j]` of the unit lower basis, in `c64` from the stored values, and the bound
/// `sum_k |V[k, i]| |V[k, j]|` the rounding error of that product scales with
fn gram<T: Scalar>(v: &Mat<T>, i: usize, j: usize) -> (c64, f64) {
	let unit = |k: usize, c: usize| -> c64 {
		if k == c {
			c64::new(1.0, 0.0)
		} else if k < c {
			c64::new(0.0, 0.0)
		} else {
			v[(k, c)].widen()
		}
	};
	let mut sum = c64::new(0.0, 0.0);
	let mut bound = 0.0;
	for k in 0..v.nrows() {
		let (a, b) = (unit(k, i), unit(k, j));
		sum += a.conj() * b;
		bound += a.norm() * b.norm();
	}
	(sum, bound)
}

fn tau<T: Scalar>(j: usize) -> T {
	T::new(0.75 + 0.01 * j as f64, 0.03 * j as f64)
}

/// The factor as a blocked QR hands it over: `tau` on the diagonal, the Gram coefficients inside
/// each `prev`-wide diagonal block, and a sentinel everywhere else.
fn input_factor<T: Scalar>(v: &Mat<T>, prev: usize) -> Mat<T> {
	let n = v.ncols();
	Mat::from_fn(n, n, |i, j| {
		if i == j {
			tau::<T>(j)
		} else if i < j && i / prev == j / prev {
			let g = gram(v, i, j).0;
			T::new(g.re, g.im)
		} else {
			T::new(SENTINEL.0, SENTINEL.1)
		}
	})
}

/// Checks every entry of the upgraded factor separately: the within-block Gram coefficients
/// against the `c64` product within the rounding bound, and the denominators, the lower triangle
/// and the between-block entries for being exactly what the caller passed.
fn check<T: Scalar>(label: &str, v: &Mat<T>, factor: &Mat<T>, block: usize) {
	let n = v.ncols();
	let m = v.nrows();
	let sentinel = T::new(SENTINEL.0, SENTINEL.1);
	let mut wrong = Vec::new();
	for j in 0..n {
		for i in 0..n {
			let got = factor[(i, j)];
			if i == j {
				if got != tau::<T>(j) {
					wrong.push(format!(
						"denominator ({i},{j}): expected {:?}, got {got:?}",
						tau::<T>(j)
					));
				}
			} else if i > j || i / block != j / block {
				if got != sentinel {
					wrong.push(format!(
						"outside the blocks ({i},{j}): expected untouched, got {got:?}"
					));
				}
			} else {
				let (want, bound) = gram(v, i, j);
				let tol = 4.0 * (m as f64) * T::EPS * bound;
				let err = (got.widen() - want).norm();
				if !(err <= tol) {
					wrong.push(format!("coefficient ({i},{j}): expected {want}, got {got:?} (|err| {err:.3e} > {tol:.3e})"));
				}
			}
		}
	}
	assert!(
		wrong.is_empty(),
		"{label} [{}]: {} wrong entries:\n{}",
		core::any::type_name::<T>(),
		wrong.len(),
		wrong.join("\n")
	);
}

fn run<T: Scalar>(m: usize, n: usize, block: usize, prev: usize, par: Par) {
	let v = basis::<T>(m, n);
	let mut factor = input_factor(&v, prev);
	upgrade_householder_factor(factor.as_mut(), v.as_ref(), block, prev, par);
	check(
		&format!("m={m} n={n} block={block} prev={prev}"),
		&v,
		&factor,
		block,
	);
}

/// `(m, n, block_size, prev_block_size)`, sequential
const SEQ_CASES: &[(usize, usize, usize, usize)] = &[
	// the issue's shape: four 2-wide blocks and a 1-wide tail
	(9, 9, 2, 1),
	(40, 9, 2, 1),
	// two full blocks, no tail
	(20, 16, 8, 1),
	// three blocks (4, 4, 3): an odd count, previous blocks dividing the target
	(11, 11, 4, 2),
	(30, 11, 4, 2),
	// five blocks (8 x 4 + 5) over 4-wide previous blocks
	(60, 37, 8, 4),
	// previous blocks of 8 take the branch that keeps their coefficients and fills the rest
	(90, 70, 16, 8),
	// already at the target width: nothing to do
	(24, 24, 8, 8),
	// a single target block whose width is not a multiple of the previous one
	(10, 6, 6, 4),
	(10, 5, 6, 4),
];

fn seq_all<T: Scalar>() {
	for &(m, n, block, prev) in SEQ_CASES {
		run::<T>(m, n, block, prev, Par::Seq);
	}
}

#[test]
fn several_blocks_f32() {
	seq_all::<f32>();
}

#[test]
fn several_blocks_f64() {
	seq_all::<f64>();
}

#[test]
fn several_blocks_c32() {
	seq_all::<c32>();
}

#[test]
fn several_blocks_c64() {
	seq_all::<c64>();
}

/// The reproduction in tensor4all/faer#265 as filed: square unit-lower basis with 0.1 below the
/// diagonal, zeros between blocks.
#[test]
fn issue_265_reproduction() {
	let n = 9;
	let b = 2;
	let basis = Mat::<f64>::from_fn(n, n, |i, j| {
		if i == j {
			1.0
		} else if i > j {
			0.1
		} else {
			0.0
		}
	});
	let mut factor = Mat::<f64>::zeros(n, n);
	for j in 0..n {
		factor[(j, j)] = (1.0 + 0.01 * (n - j - 1) as f64) / 2.0;
	}
	upgrade_householder_factor(factor.as_mut(), basis.as_ref(), b, 1, Par::Seq);
	let mut wrong = Vec::new();
	for j in 0..n {
		for i in 0..j {
			let expected = if i / b == j / b {
				0.1 + 0.01 * (n - j - 1) as f64
			} else {
				0.0
			};
			if (factor[(i, j)] - expected).abs() > 1e-12 {
				wrong.push(format!(
					"({i},{j}): expected {expected}, got {}",
					factor[(i, j)]
				));
			}
		}
	}
	assert!(
		wrong.is_empty(),
		"independent block Gram coefficients differ:\n{}",
		wrong.join("\n")
	);
}

/// Several target blocks need `block_size` to be a multiple of `prev_block_size`; the call rejects
/// any other pair instead of guessing a partition.
#[test]
#[should_panic]
fn several_blocks_reject_a_previous_size_that_does_not_divide() {
	let v = basis::<f64>(12, 12);
	let mut factor = input_factor(&v, 4);
	upgrade_householder_factor(factor.as_mut(), v.as_ref(), 6, 4, Par::Seq);
}

/// The same contract on a team. The wide cases (`dispatch`) are deep enough for the call to open
/// team regions, asserted from the dispatch counters, so the recursion's halves run on a split of
/// the team; the narrow ones may stay under the work floor (whether they do is the cost model's
/// decision, not part of this contract) and only check the coefficients on a team token.
/// One test, so nothing else in this binary dispatches a team while the counters are read.
#[cfg(feature = "threadteam")]
#[test]
fn several_blocks_on_a_team() {
	use faer::utils::team::stats;
	// `(m, n, block_size, prev_block_size, dispatch)`
	const TEAM_CASES: &[(usize, usize, usize, usize, bool)] = &[
		(4096, 9, 2, 1, false),
		(4096, 11, 4, 2, false),
		(4096, 16, 8, 1, false),
		(4096, 37, 8, 4, true),
		(4096, 70, 16, 8, true),
	];
	fn all<T: Scalar>(mut par: Par, width: usize) {
		for &(m, n, block, prev, dispatch) in TEAM_CASES {
			let _ = stats::take();
			run::<T>(m, n, block, prev, par.reborrow());
			let evidence = stats::take();
			assert!(
				!dispatch
					|| evidence.run.iter().skip(2).sum::<usize>()
						+ evidence.split.iter().skip(2).sum::<usize>()
						> 0,
				"width={width} m={m} n={n} block={block} prev={prev} [{}]: no team regions: {evidence:?}",
				core::any::type_name::<T>()
			);
		}
	}
	for width in [2, 4] {
		faer::team::with_team(width, |mut par| {
			stats::enable(true);
			all::<f32>(par.reborrow(), width);
			all::<f64>(par.reborrow(), width);
			all::<c32>(par.reborrow(), width);
			all::<c64>(par.reborrow(), width);
			stats::enable(false);
		});
	}
}
