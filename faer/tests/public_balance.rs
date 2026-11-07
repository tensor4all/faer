mod common;
use common::*;
use faer::linalg::evd::*;

fn check<T: ComplexField>(make: fn(f64, f64) -> T) {
	for n in SIZES {
		let original = Mat::from_fn(n, n, |i, j| {
			let exponent = 4 * ((j % 5) as i32 - (i % 5) as i32);
			make(sample(i, j), sample(j, i))
				.mul_pow2(from_f64::<T::Real>(2.0f64.powi(exponent)))
		});
		let mut b = original.clone();
		let (mut perm, mut scale) = (vec![0; n], vec![zero(); n]);
		let mut mem = MemBuffer::new(balance_scratch());
		let active = balance_in_place(
			b.as_mut(),
			&mut perm,
			&mut scale,
			Par::Seq,
			MemStack::new(&mut mem),
		)
		.unwrap();
		assert_eq!(active, 0..if n == 1 { 0 } else { n });
		for i in 0..n {
			let mut r = zero::<T::Real>();
			let mut c = zero::<T::Real>();
			for j in 0..n {
				assert_eq!(
					b[(i, j)]
						.mul_pow2(scale[i].copy())
						.mul_pow2(scale[j].recip()),
					original[(perm[i], perm[j])]
				);
				r = r.hypot(&b[(i, j)].abs());
				c = c.hypot(&b[(j, i)].abs());
			}
			if c > zero() && r > zero() {
				let current = &c + &r;
				for f in [0.5, 2.0] {
					let changed = &c * from_f64::<T::Real>(f)
						+ &r * from_f64::<T::Real>(1.0 / f);
					assert!(changed >= from_f64::<T::Real>(0.95) * &current);
				}
			}
		}
		// Every vector is back-transformed by exactly P D, for arbitrary column
		// count.
		let mut x = Mat::from_fn(n, 3, |i, j| make(sample(i, j), sample(j, i)));
		let before = x.clone();
		let mut mem = MemBuffer::new(balance_eigenvectors_scratch::<T>(n));
		balance_eigenvectors_in_place(
			x.as_mut(),
			&perm,
			&scale,
			Par::Seq,
			MemStack::new(&mut mem),
		);
		for i in 0..n {
			for j in 0..3 {
				assert_eq!(
					x[(perm[i], j)],
					before[(i, j)].mul_pow2(scale[i].copy())
				);
			}
		}
	}
}
#[test]
fn exact_similarity_balance_criterion_and_backtransform() {
	check::<f64>(|r, _| r);
	check::<f32>(|r, _| r as f32);
	check::<c64>(c64::new);
	check::<c32>(|r, i| c32::new(r as f32, i as f32));
}
#[test]
fn permutation_isolates_and_preserves_eigenvalues() {
	for n in SIZES {
		// A triangular matrix scrambled by a permutation; every eigenvalue is
		// isolatable.
		let perm: Vec<_> = (0..n).rev().collect();
		let original = Mat::from_fn(n, n, |i, j| {
			let (i, j) = (perm[i], perm[j]);
			if i == j {
				(i + 1) as f64
			} else if i < j {
				sample(i, j)
			} else {
				0.0
			}
		});
		let mut b = original.clone();
		let (mut p, mut s) = (vec![0; n], vec![0.0; n]);
		let mut mem = MemBuffer::new(balance_scratch());
		let active = balance_in_place(
			b.as_mut(),
			&mut p,
			&mut s,
			Par::Seq,
			MemStack::new(&mut mem),
		)
		.unwrap();
		assert!(active.is_empty());
		for j in 0..n {
			for i in 0..n {
				assert_eq!(b[(i, j)], original[(p[i], p[j])]);
				if i > j {
					assert_eq!(b[(i, j)], 0.0);
				}
			}
		}
		let mut spectrum: Vec<_> = (0..n).map(|i| b[(i, i)]).collect();
		spectrum.sort_by(|a, b| a.partial_cmp(b).unwrap());
		assert_eq!(spectrum, (1..=n).map(|i| i as f64).collect::<Vec<_>>());
	}
}
#[test]
fn eigenvalues_and_right_eigenvectors_survive_balancing() {
	for n in SIZES {
		let a = Mat::from_fn(n, n, |i, j| {
			c64::new(sample(i, j), sample(j, i + 1))
				* 2.0f64.powi(4 * ((j % 4) as i32 - (i % 4) as i32))
		});
		let mut b = a.clone();
		let (mut p, mut s) = (vec![0; n], vec![0.0; n]);
		let mut mem = MemBuffer::new(balance_scratch());
		balance_in_place(
			b.as_mut(),
			&mut p,
			&mut s,
			Par::Seq,
			MemStack::new(&mut mem),
		)
		.unwrap();
		let mut values = faer::diag::Diag::zeros(n);
		let mut x = Mat::zeros(n, n);
		let mut mem = MemBuffer::new(evd_scratch::<c64>(
			n,
			ComputeEigenvectors::No,
			ComputeEigenvectors::Yes,
			ParShape::Seq,
			Default::default(),
		));
		evd_cplx(
			b.as_ref(),
			values.as_mut(),
			None,
			Some(x.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default(),
		)
		.unwrap();
		let mut mem = MemBuffer::new(balance_eigenvectors_scratch::<c64>(n));
		balance_eigenvectors_in_place(
			x.as_mut(),
			&p,
			&s,
			Par::Seq,
			MemStack::new(&mut mem),
		);
		close((&a * &x).as_ref(), (&x * values.as_ref()).as_ref());
	}
}
#[test]
fn nonfinite_does_not_mutate_and_extreme_input_terminates() {
	let mut a = faer::mat![[1.0f64, f64::INFINITY], [2.0, 3.0]];
	let before = a.clone();
	let (mut p, mut s) = ([9; 2], [3.0; 2]);
	let mut mem = MemBuffer::new(balance_scratch());
	assert_eq!(
		balance_in_place(
			a.as_mut(),
			&mut p,
			&mut s,
			Par::Seq,
			MemStack::new(&mut mem)
		),
		Err(BalanceError::NonFiniteInput)
	);
	assert_eq!(a, before);
	assert_eq!(p, [9; 2]);
	assert_eq!(s, [3.0; 2]);
	let mut a = faer::mat![[1.0, 1e300], [1e-300, 2.0]];
	balance_in_place(
		a.as_mut(),
		&mut p,
		&mut s,
		Par::Seq,
		MemStack::new(&mut mem),
	)
	.unwrap();
	assert!(a.as_ref().is_all_finite());
	// LAPACK 3.5+ includes the diagonal in its norm test, avoiding
	// overbalancing when the matrix is already dominated by that diagonal.
	let mut a = faer::mat![[1e30, 1e10], [1e-10, 2e30]];
	let before = a.clone();
	balance_in_place(
		a.as_mut(),
		&mut p,
		&mut s,
		Par::Seq,
		MemStack::new(&mut mem),
	)
	.unwrap();
	assert_eq!(a, before);
	assert_eq!(s, [1.0, 1.0]);
}

/// tensor4all/faer#304: `xGEBAL`'s four jobs are selectable, and each one runs only the
/// steps it names. `balance_in_place` stays `Both`.
fn run_job(
	a: &Mat<f64>,
	job: BalanceJob,
) -> (Mat<f64>, Vec<usize>, Vec<f64>, core::ops::Range<usize>) {
	let n = a.nrows();
	let mut b = a.clone();
	let (mut perm, mut scale) = (vec![0usize; n], vec![0.0f64; n]);
	let mut mem = MemBuffer::new(balance_scratch());
	let active = balance_in_place_with(
		b.as_mut(),
		&mut perm,
		&mut scale,
		job,
		Par::Seq,
		MemStack::new(&mut mem),
	)
	.unwrap();
	(b, perm, scale, active)
}

#[test]
fn jobs_select_the_permutation_and_the_scaling() {
	// an upper triangular matrix: the permutation can isolate both ends, and the
	// entries are graded enough that the scaling step accepts a factor
	let a = faer::mat![[1.0, 1024.0, 0.0], [0.0, 2.0, 512.0], [0.0, 0.0, 4.0],];
	let n = a.nrows();

	// `N`: nothing is touched, and the metadata is the identity
	let (b_none, perm_none, scale_none, active_none) =
		run_job(&a, BalanceJob::None);
	assert_eq!(b_none, a);
	assert_eq!(perm_none, (0..n).collect::<Vec<_>>());
	assert_eq!(scale_none, vec![1.0; n]);
	assert_eq!(active_none, 0..n);

	// `P`: the permutation alone, so every scale is one and the entries are the input's
	let (b_perm, perm_perm, scale_perm, active_perm) =
		run_job(&a, BalanceJob::Permute);
	assert_eq!(scale_perm, vec![1.0; n]);
	assert!(active_perm.start > 0 || active_perm.end < n);
	for i in 0..n {
		for j in 0..n {
			assert_eq!(b_perm[(i, j)], a[(perm_perm[i], perm_perm[j])]);
		}
	}

	// `S`: the scaling alone, so the permutation is the identity and the active range is
	// the whole matrix
	let (b_scale, perm_scale, scale_scale, active_scale) =
		run_job(&a, BalanceJob::Scale);
	assert_eq!(perm_scale, (0..n).collect::<Vec<_>>());
	assert_eq!(active_scale, 0..n);
	assert!(scale_scale.iter().any(|s| *s != 1.0));
	for i in 0..n {
		for j in 0..n {
			assert_eq!(
				b_scale[(i, j)],
				a[(i, j)] * scale_scale[j] / scale_scale[i]
			);
		}
	}

	// `B` is the wrapper's behaviour, bit for bit
	let (b_both, perm_both, scale_both, active_both) =
		run_job(&a, BalanceJob::Both);
	let mut b_wrapper = a.clone();
	let (mut perm_wrapper, mut scale_wrapper) =
		(vec![0usize; n], vec![0.0f64; n]);
	let mut mem = MemBuffer::new(balance_scratch());
	let active_wrapper = balance_in_place(
		b_wrapper.as_mut(),
		&mut perm_wrapper,
		&mut scale_wrapper,
		Par::Seq,
		MemStack::new(&mut mem),
	)
	.unwrap();
	assert_eq!(b_both, b_wrapper);
	assert_eq!(active_both, active_wrapper);
	assert_eq!(perm_both, perm_wrapper);
	assert_eq!(scale_both, scale_wrapper);
	// and it is not the same call as `S` on a matrix the permutation isolates
	assert_ne!(b_both, b_scale);
}
