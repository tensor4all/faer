//! Independent definitions and congruences, not fixtures ported from PfaPack.
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::pfaffian::{PfaffianError, pfaffian, pfaffian_scratch};
use faer::{Mat, MatRef, Par, c32, c64};
use faer_traits::ComplexField;
use faer_traits::ext::*;
use faer_traits::math_utils::{from_f64, one, zero};

fn compute<T: ComplexField>(
	a: MatRef<'_, T>,
	par: Par<'_>,
) -> Result<T, PfaffianError> {
	let mut memory = MemBuffer::new(pfaffian_scratch::<T>(a.nrows()));
	pfaffian(a, par, MemStack::new(&mut memory))
}

fn sample(i: usize, j: usize, seed: u64) -> f64 {
	let mut x = (i as u64 * 1_000_003 + j as u64)
		.wrapping_add(seed.wrapping_mul(0x9E3779B97F4A7C15));
	x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
	x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
	x ^= x >> 31;
	((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.25
}

fn random_skew<T: ComplexField>(
	n: usize,
	seed: u64,
	entry: impl Fn(f64, f64) -> T,
) -> Mat<T> {
	Mat::from_fn(n, n, |i, j| {
		if i == j {
			return zero();
		}
		let (p, q) = (i.max(j), i.min(j));
		let x = entry(sample(p, q, seed), sample(p, q, seed + 71));
		if i > j { x } else { -x }
	})
}

fn close<T: ComplexField>(got: T, want: T, tolerance: f64) {
	assert!(
		got.is_finite() && want.is_finite(),
		"nonfinite oracle comparison"
	);
	let error = (&got - &want).abs();
	let bound = from_f64::<T::Real>(tolerance) * (got.abs() + want.abs());
	assert!(
		error <= bound,
		"got={got:?}, want={want:?}, error={error:?}, bound={bound:?}"
	);
}

#[test]
#[should_panic(expected = "nonfinite oracle comparison")]
fn oracle_comparison_rejects_infinity() {
	close(f64::INFINITY, 1.0, 2e-11);
}

// Expansion along the first row: the combinatorial definition, with no
// elimination.
fn recursive<T: ComplexField>(a: MatRef<'_, T>, remaining: &[usize]) -> T {
	if remaining.is_empty() {
		return one();
	}
	let mut value = zero::<T>();
	for j in 1..remaining.len() {
		let rest: Vec<_> = remaining
			.iter()
			.enumerate()
			.filter_map(|(k, &i)| (k != 0 && k != j).then_some(i))
			.collect();
		let term = &a[(remaining[0], remaining[j])] * recursive(a, &rest);
		value = if j % 2 == 1 {
			value + term
		} else {
			value - term
		};
	}
	value
}

fn oracle_cases<T: ComplexField>(
	entry: impl Copy + Fn(f64, f64) -> T,
	tolerance: f64,
) {
	for n in [0, 2, 4, 6, 8] {
		for seed in [1, 17, 171, 981] {
			let a = random_skew(n, seed, entry);
			let got = compute(a.as_ref(), Par::Seq).unwrap();
			close(
				got,
				recursive(a.as_ref(), &(0..n).collect::<Vec<_>>()),
				tolerance,
			);
		}
	}
	for n in [2, 8, 16, 30, 32, 34] {
		let a = random_skew(n, 93, entry);
		let pf = compute(a.as_ref(), Par::Seq).unwrap();
		close(&pf * &pf, a.determinant(Par::Seq), tolerance * n as f64);
	}
}

#[test]
fn real_and_complex_definition_and_determinant() {
	oracle_cases(|r, _| r, 2e-11);
	oracle_cases(|r, _| r as f32, 2e-4);
	oracle_cases(c64::new, 2e-11);
	oracle_cases(|r, i| c32::new(r as f32, i as f32), 2e-4);
}

#[test]
fn shapes_singularity_and_negative_pfaffian() {
	let mut empty = MemBuffer::new(StackReq::EMPTY);
	for (m, n, error) in [
		(2, 4, PfaffianError::NonSquare),
		(3, 2, PfaffianError::NonSquare),
		(3, 3, PfaffianError::OddDimension),
	] {
		let a = Mat::<f64>::zeros(m, n);
		assert_eq!(
			pfaffian(a.as_ref(), Par::Seq, MemStack::new(&mut empty)),
			Err(error)
		);
	}
	assert_eq!(compute(Mat::<f64>::zeros(0, 0).as_ref(), Par::Seq), Ok(1.0));
	assert_eq!(
		compute(Mat::<f64>::zeros(8, 8).as_ref(), Par::Seq),
		Err(PfaffianError::ZeroPivot { index: 0 })
	);
	let a = faer::mat![[0.0, -3.0], [3.0, 0.0]];
	assert_eq!(compute(a.as_ref(), Par::Seq), Ok(-3.0));
	let mut singular = Mat::<f64>::zeros(4, 4);
	singular[(1, 0)] = 2.0;
	assert_eq!(
		compute(singular.as_ref(), Par::Seq),
		Err(PfaffianError::ZeroPivot { index: 2 })
	);
}

#[test]
fn lower_triangle_strides_and_input_preservation() {
	let a = random_skew(34, 177, c64::new);
	let before = a.clone();
	let want = compute(a.as_ref(), Par::Seq).unwrap();
	// A transposed backing allocation gives row-major input with a negative row
	// stride.
	let mut backing =
		Mat::<c64>::from_fn(68, 68, |_, _| c64::new(f64::NAN, f64::NAN));
	let mut view = backing
		.as_mut()
		.transpose_mut()
		.reverse_rows_mut()
		.submatrix_mut(3, 7, 34, 34);
	for j in 0..34 {
		for i in j + 1..34 {
			view[(i, j)] = a[(i, j)];
		}
	}
	let got = compute(view.as_ref(), Par::Seq).unwrap();
	assert_eq!(got, want);
	for j in 0..34 {
		for i in 0..34 {
			assert_eq!(a[(i, j)], before[(i, j)]);
		}
	}
}

// Pf(L D L^T) = det(L) Pf(D) = product(d); the generator never calls
// elimination.
fn known_congruence(n: usize, complex: bool) -> (Mat<c64>, c64) {
	let l = Mat::<c64>::from_fn(n, n, |i, j| {
		if i == j {
			c64::new(1.0, 0.0)
		} else if i > j {
			c64::new(
				sample(i, j, 44),
				if complex { sample(i, j, 49) } else { 0.0 },
			)
		} else {
			c64::new(0.0, 0.0)
		}
	});
	let mut expected = c64::new(1.0, 0.0);
	let mut a = Mat::<c64>::zeros(n, n);
	for k in (0..n).step_by(2) {
		let d = if k % 4 == 0 { -1.0 } else { 2.0 };
		expected *= d;
		for j in 0..n {
			for i in 0..n {
				a[(i, j)] +=
					d * (l[(i, k)] * l[(j, k + 1)] - l[(i, k + 1)] * l[(j, k)]);
			}
		}
	}
	// Move the largest first-pair pivot far into the panel/trailing block.
	// This forces a nonadjacent swap after pending updates have accumulated.
	let mut perm: Vec<_> = (0..n).collect();
	perm.swap(3, n - 1);
	let a = Mat::from_fn(n, n, |i, j| a[(perm[i], perm[j])]);
	(a, -expected)
}

#[test]
fn known_pfaffian_across_panel_and_simd_tail_boundaries() {
	for n in [6, 30, 32, 34, 62, 64, 66, 128, 130] {
		for complex in [false, true] {
			let (a, expected) = known_congruence(n, complex);
			close(compute(a.as_ref(), Par::Seq).unwrap(), expected, 2e-11);
		}
	}
	// Many equal pivot magnitudes exercise tie-breaking as the tail length
	// changes.
	for n in [6, 10, 34, 66] {
		let a = Mat::<f64>::from_fn(n, n, |i, j| {
			if i < j {
				1.0
			} else if i > j {
				-1.0
			} else {
				0.0
			}
		});
		assert_eq!(compute(a.as_ref(), Par::Seq), Ok(1.0));
	}
}

#[test]
fn finite_range_and_nonfinite_failures() {
	for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
		let mut a = Mat::<f64>::zeros(4, 4);
		a[(3, 2)] = value;
		assert_eq!(
			compute(a.as_ref(), Par::Seq),
			Err(PfaffianError::NumericalFailure)
		);
	}
	let mut nonfinite = Mat::<c64>::zeros(4, 4);
	nonfinite[(3, 2)] = c64::new(0.0, f64::NAN);
	assert_eq!(
		compute(nonfinite.as_ref(), Par::Seq),
		Err(PfaffianError::NumericalFailure)
	);
	assert_eq!(
		compute(Mat::<c64>::zeros(4, 4).as_ref(), Par::Seq),
		Err(PfaffianError::ZeroPivot { index: 0 })
	);
	let mut overflow = Mat::<f64>::zeros(4, 4);
	for j in 0..4 {
		for i in j + 1..4 {
			overflow[(i, j)] = 1e308;
		}
	}
	overflow[(3, 1)] = -1e308;
	assert_eq!(
		compute(overflow.as_ref(), Par::Seq),
		Err(PfaffianError::NumericalFailure)
	);
	let tiny = f64::from_bits(1);
	let mut a = Mat::<f64>::zeros(2, 2);
	a[(1, 0)] = tiny;
	assert_eq!(compute(a.as_ref(), Par::Seq), Ok(-tiny));
	let mut a = Mat::<f64>::zeros(4, 4);
	for (first, second, expected) in [
		(1e-200, 1e200, Some(1.0)),
		(tiny, 1e308, Some(tiny * 1e308)),
		(1e-200, 1e-200, None),
		(1e200, 1e200, None),
	] {
		a[(1, 0)] = -first;
		a[(3, 2)] = -second;
		match expected {
			Some(value) => {
				close(compute(a.as_ref(), Par::Seq).unwrap(), value, 3e-16)
			},
			None => assert_eq!(
				compute(a.as_ref(), Par::Seq),
				Err(PfaffianError::NumericalFailure)
			),
		}
	}
}

#[test]
fn runtime_selected_lane() {
	struct Lane;
	impl pulp::WithSimd for Lane {
		type Output = (&'static str, usize, usize, bool);

		fn with_simd<S: pulp::Simd>(self, _: S) -> Self::Output {
			(
				core::any::type_name::<S>(),
				core::mem::size_of::<<f64 as ComplexField>::SimdVec<S>>() / 8,
				core::mem::size_of::<
					<faer_traits::ComplexImpl<f64> as ComplexField>::SimdVec<S>,
				>() / 16,
				S::IS_SCALAR,
			)
		}
	}
	let (lane, real, complex, scalar) = pulp::Arch::new().dispatch(Lane);
	eprintln!(
		"Pfaffian runtime lane: {lane}; f64 lanes={real}; c64 lanes={complex}"
	);
	assert_eq!(real, if scalar { complex } else { 2 * complex });
}

#[test]
#[should_panic]
fn insufficient_scratch_panics() {
	let a = Mat::<f64>::zeros(2, 2);
	let mut memory = MemBuffer::new(StackReq::EMPTY);
	let _ = pfaffian(a.as_ref(), Par::Seq, MemStack::new(&mut memory));
}

#[test]
fn generic_extended_precision_real_and_complex() {
	use faer_traits::{cx128, fx128};
	let a = random_skew(8, 128, |r, _| from_f64::<fx128>(r));
	close(
		compute(a.as_ref(), Par::Seq).unwrap(),
		recursive(a.as_ref(), &(0..8).collect::<Vec<_>>()),
		1e-28,
	);
	let a = random_skew(8, 128, |r, i| cx128::new(from_f64(r), from_f64(i)));
	close(
		compute(a.as_ref(), Par::Seq).unwrap(),
		recursive(a.as_ref(), &(0..8).collect::<Vec<_>>()),
		1e-28,
	);
}

#[cfg(feature = "threadteam")]
#[test]
fn team_copy_checks_nonleading_columns() {
	let (mut a, expected) = known_congruence(128, true);
	faer::team::with_team(8, |mut par| {
		close(
			compute(a.as_ref(), par.reborrow()).unwrap(),
			expected,
			2e-11,
		);
		// The last worker's lower triangle must be validated even when the
		// first column would otherwise give an immediate zero pivot.
		a.fill(c64::new(0.0, 0.0));
		a[(127, 126)] = c64::new(0.0, f64::NAN);
		assert_eq!(
			compute(a.as_ref(), par),
			Err(PfaffianError::NumericalFailure)
		);
	});
}

#[cfg(feature = "threadteam")]
#[test]
fn team_agreement_and_actual_column_dispatch() {
	fn check<T: ComplexField>(a: Mat<T>) {
		use faer::utils::team::stats;
		let want = compute(a.as_ref(), Par::Seq).unwrap();
		for width in [1, 3, 8, 32] {
			faer::team::with_team(width, |par| {
				stats::enable(true);
				let _ = stats::take();
				let got = compute(a.as_ref(), par).unwrap();
				let evidence = stats::take();
				stats::enable(false);
				assert_eq!(got, want, "width={width}");
				if width > 1 {
					assert!(
						evidence.run.iter().skip(2).sum::<usize>() > 0,
						"no team column updates: {evidence:?}"
					);
				}
			});
		}
	}
	check(random_skew(258, 37, |r, _| r));
	check(random_skew(258, 37, |r, _| r as f32));
	check(random_skew(258, 37, c64::new));
	check(random_skew(258, 37, |r, i| c32::new(r as f32, i as f32)));
}
