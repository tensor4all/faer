mod common;
use common::*;
use faer::MatMut;
use faer::linalg::gevd;
fn scaled_close<T: ComplexField>(
	got: MatRef<'_, T>,
	expected: MatRef<'_, T>,
	scale: &T::Real,
) {
	let bound =
		tolerance::<T>(got.nrows()) * expected.norm_l2().fmax(one::<T::Real>());
	for j in 0..got.ncols() {
		for i in 0..got.nrows() {
			assert!(
				(&got[(i, j)].real() / scale - expected[(i, j)].real()).abs()
					<= bound
			);
			assert!(
				(&got[(i, j)].imag() / scale - expected[(i, j)].imag()).abs()
					<= bound
			);
		}
	}
}
fn triangular<T: ComplexField>(
	value: fn(f64, f64) -> T,
	scales: &[f64],
	mut reorder: impl FnMut(
		MatMut<'_, T>,
		MatMut<'_, T>,
		Option<MatMut<'_, T>>,
		Option<MatMut<'_, T>>,
		usize,
		usize,
		&mut MemStack,
	) -> Result<usize, gevd::SchurReorderError>,
) {
	for &n in &[0, 1, 2, 3, 8, 33, 100] {
		for infinite in [false, true] {
			for down in [false, true] {
				let s0 = Mat::<T>::from_fn(n, n, |i, j| {
					if i == j {
						value((i + 1) as f64, 0.2 * (i + 1) as f64)
					} else if i < j {
						value(0.03 * sample(i, j), 0.02 * sample(j, i))
					} else {
						zero()
					}
				});
				let t0 = Mat::<T>::from_fn(n, n, |i, j| {
					if i == j {
						from_f64(if infinite && i + 1 == n { 0.0 } else { 1.0 })
					} else if i < j {
						value(0.02 * sample(i, j), 0.01 * sample(j, i))
					} else {
						zero()
					}
				});
				for &sc in if n == 2 { scales } else { &[1.0] } {
					let scale = from_f64::<T::Real>(sc);
					let mut s = Mat::<T>::from_fn(n, n, |i, j| {
						s0[(i, j)].mul_real(&scale)
					});
					let mut t = Mat::<T>::from_fn(n, n, |i, j| {
						t0[(i, j)].mul_real(&scale)
					});
					let mut q = Mat::<T>::identity(n, n);
					let mut z = q.clone();
					let mut mem =
						MemBuffer::new(gevd::reorder_schur_scratch::<T>(n));
					let last = n.saturating_sub(1);
					let (from, to) = if down { (0, last) } else { (last, 0) };
					let pos = reorder(
						s.as_mut(),
						t.as_mut(),
						Some(q.as_mut()),
						Some(z.as_mut()),
						from,
						to,
						MemStack::new(&mut mem),
					)
					.unwrap();
					assert_eq!(pos, to);
					unitary(q.as_ref());
					unitary(z.as_ref());
					scaled_close(
						(&q * &s * z.adjoint()).as_ref(),
						s0.as_ref(),
						&scale,
					);
					scaled_close(
						(&q * &t * z.adjoint()).as_ref(),
						t0.as_ref(),
						&scale,
					);
					for j in 0..n {
						for i in j + 1..n {
							assert_eq!(s[(i, j)], zero());
							assert_eq!(t[(i, j)], zero());
						}
					}
					for i in 0..n {
						let old = if down {
							if i == last { 0 } else { i + 1 }
						} else {
							if i == 0 { last } else { i - 1 }
						};
						let a = value((old + 1) as f64, 0.2 * (old + 1) as f64);
						let b = from_f64::<T>(if infinite && old == last {
							0.0
						} else {
							1.0
						});
						let imag = if T::IS_REAL {
							zero()
						} else {
							(-one::<T>()).sqrt()
						};
						let sn = from_real::<T>(&(&s[(i, i)].real() / &scale))
							+ imag.mul_real(&(&s[(i, i)].imag() / &scale));
						let tn = from_real::<T>(&(&t[(i, i)].real() / &scale))
							+ imag.mul_real(&(&t[(i, i)].imag() / &scale));
						assert!(
							(&sn * &b - &a * &tn).abs()
								<= tolerance::<T>(n)
									* a.abs().fmax(one::<T::Real>())
						);
						assert_eq!(
							t[(i, i)] == zero(),
							infinite && old == last
						);
					}
				}
			}
		}
	}
}
#[test]
fn f32_triangular() {
	triangular::<f32>(
		|r, _| r as f32,
		&[1., 1e-30, 1e30],
		|s, t, q, z, i, j, stack| {
			gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
		},
	);
}
#[test]
fn f64_triangular() {
	triangular::<f64>(
		|r, _| r,
		&[1., 1e-300, 1e300],
		|s, t, q, z, i, j, stack| {
			gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
		},
	);
}
#[test]
fn c32_triangular() {
	triangular::<c32>(
		|r, i| c32::new(r as f32, i as f32),
		&[1., 1e-30, 1e30],
		|s, t, q, z, i, j, stack| {
			gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
		},
	);
}
#[test]
fn c64_triangular() {
	triangular::<c64>(
		c64::new,
		&[1., 1e-300, 1e300],
		|s, t, q, z, i, j, stack| {
			gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
		},
	);
}

fn polynomial<T: RealField>(s: MatRef<'_, T>, t: MatRef<'_, T>) -> [T; 3] {
	let mut scale = zero::<T>();
	for matrix in [s, t] {
		for j in 0..s.ncols() {
			for i in 0..s.nrows() {
				scale = scale.fmax(matrix[(i, j)].abs());
			}
		}
	}
	let s = Mat::<T>::from_fn(s.nrows(), s.ncols(), |i, j| &s[(i, j)] / &scale);
	let t = Mat::<T>::from_fn(t.nrows(), t.ncols(), |i, j| &t[(i, j)] / &scale);
	if s.nrows() == 1 {
		return [zero(), -t[(0, 0)].copy(), s[(0, 0)].copy()];
	}
	[
		&t[(0, 0)] * &t[(1, 1)] - &t[(0, 1)] * &t[(1, 0)],
		-(&s[(0, 0)] * &t[(1, 1)] + &s[(1, 1)] * &t[(0, 0)]
			- &s[(0, 1)] * &t[(1, 0)]
			- &s[(1, 0)] * &t[(0, 1)]),
		&s[(0, 0)] * &s[(1, 1)] - &s[(0, 1)] * &s[(1, 0)],
	]
}
fn real_blocks<T: RealField>(scales: &[f64]) {
	for sizes in [&[2, 1][..], &[1, 2][..], &[2, 2][..], &[2, 1, 2][..]] {
		for down in [false, true] {
			for &scale in scales {
				let scale = from_f64::<T>(scale);
				let n: usize = sizes.iter().sum();
				let mut s = Mat::<T>::from_fn(n, n, |i, j| {
					if i < j {
						from_f64(0.02 * sample(i, j))
					} else {
						zero()
					}
				});
				let mut t = Mat::<T>::from_fn(n, n, |i, j| {
					if i < j {
						from_f64(0.01 * sample(j, i))
					} else {
						zero()
					}
				});
				let mut offsets = Vec::new();
				let mut start = 0;
				for (index, &size) in sizes.iter().enumerate() {
					offsets.push(start);
					if size == 1 {
						s[(start, start)] = from_f64(5.0);
						t[(start, start)] = zero();
					} else {
						s[(start, start)] = from_f64(0.1 * (index + 1) as f64);
						s[(start + 1, start + 1)] = s[(start, start)].copy();
						s[(start, start + 1)] = from_f64(1.0);
						s[(start + 1, start)] = from_f64(-1.0);
						t[(start, start)] = from_f64(1.2);
						t[(start + 1, start + 1)] = from_f64(0.7);
					}
					start += size;
				}
				for j in 0..n {
					for i in 0..n {
						s[(i, j)] = &s[(i, j)] * &scale;
						t[(i, j)] = &t[(i, j)] * &scale;
					}
				}
				let old_s = s.clone();
				let old_t = t.clone();
				let mut q = Mat::identity(n, n);
				let mut z = q.clone();
				let mut order: Vec<_> = (0..sizes.len()).collect();
				let (from, to, expected) = if down {
					let first = order.remove(0);
					order.push(first);
					(0, *offsets.last().unwrap(), n - sizes[0])
				} else {
					let last = order.pop().unwrap();
					order.insert(0, last);
					(*offsets.last().unwrap(), 0, 0)
				};
				let mut mem =
					MemBuffer::new(gevd::reorder_schur_scratch::<T>(n));
				// The second row also names the whole conjugate source/destination.
				let from = from
					+ usize::from(
						sizes[if down { 0 } else { sizes.len() - 1 }] == 2,
					);
				let to = to
					+ usize::from(
						sizes[if down { sizes.len() - 1 } else { 0 }] == 2,
					);
				assert_eq!(
					gevd::reorder_schur_real(
						s.as_mut(),
						t.as_mut(),
						Some(q.as_mut()),
						Some(z.as_mut()),
						from,
						to,
						Par::Seq,
						MemStack::new(&mut mem)
					)
					.unwrap(),
					expected
				);
				let normalized_s =
					Mat::<T>::from_fn(n, n, |i, j| &old_s[(i, j)] / &scale);
				let normalized_t =
					Mat::<T>::from_fn(n, n, |i, j| &old_t[(i, j)] / &scale);
				scaled_close(
					(&q * &s * z.adjoint()).as_ref(),
					normalized_s.as_ref(),
					&scale,
				);
				scaled_close(
					(&q * &t * z.adjoint()).as_ref(),
					normalized_t.as_ref(),
					&scale,
				);
				unitary(q.as_ref());
				unitary(z.as_ref());
				let mut row = 0;
				for index in order {
					let k = sizes[index];
					let original = offsets[index];
					let a = polynomial(
						old_s.submatrix(original, original, k, k),
						old_t.submatrix(original, original, k, k),
					);
					let b = polynomial(
						s.submatrix(row, row, k, k),
						t.submatrix(row, row, k, k),
					);
					let sa = a.iter().fold(zero::<T>(), |v, x| v.fmax(x.abs()));
					let sb = b.iter().fold(zero::<T>(), |v, x| v.fmax(x.abs()));
					for i in 0..3 {
						for j in 0..3 {
							let difference = (&a[i] / &sa) * (&b[j] / &sb)
								- (&a[j] / &sa) * (&b[i] / &sb);
							assert!(difference.abs() <= tolerance::<T>(n));
						}
					}
					if k == 2 {
						assert_ne!(s[(row + 1, row)], zero());
					} else {
						assert_eq!(t[(row, row)], zero());
					}
					row += k;
				}
				for j in 0..n {
					for i in j + 1..n {
						assert_eq!(t[(i, j)], zero());
						if i > j + 1 {
							assert_eq!(s[(i, j)], zero());
						}
					}
				}
			}
		}
	}
}
#[test]
fn f32_blocks() {
	real_blocks::<f32>(&[1.0, 1e-30, 1e30]);
}
#[test]
fn f64_blocks() {
	real_blocks::<f64>(&[1.0, 1e-300, 1e300]);
}

fn failure_and_modes<T: ComplexField>(
	mut reorder: impl FnMut(
		MatMut<'_, T>,
		MatMut<'_, T>,
		Option<MatMut<'_, T>>,
		Option<MatMut<'_, T>>,
		usize,
		usize,
		&mut MemStack,
	) -> Result<usize, gevd::SchurReorderError>,
) {
	// One regular swap succeeds before the nonregular block rejects the next.
	let source_s = Mat::<T>::from_fn(3, 3, |i, j| {
		if i == j { from_f64(i as f64) } else { zero() }
	});
	let source_t =
		Mat::<T>::from_fn(
			3,
			3,
			|i, j| if i == j && i > 0 { one() } else { zero() },
		);
	let mut expected_s = source_s.clone();
	let mut expected_t = source_t.clone();
	let mut expected_q = Mat::<T>::identity(3, 3);
	let mut expected_z = expected_q.clone();
	let mut mem = MemBuffer::new(gevd::reorder_schur_scratch::<T>(3));
	assert_eq!(
		reorder(
			expected_s.as_mut(),
			expected_t.as_mut(),
			Some(expected_q.as_mut()),
			Some(expected_z.as_mut()),
			2,
			1,
			MemStack::new(&mut mem)
		)
		.unwrap(),
		1
	);
	let mut s = source_s.clone();
	let mut t = source_t.clone();
	let mut q = Mat::<T>::identity(3, 3);
	let mut z = q.clone();
	let error = reorder(
		s.as_mut(),
		t.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		2,
		0,
		MemStack::new(&mut mem),
	)
	.unwrap_err();
	assert_eq!(error.position, 1);
	assert_eq!(s, expected_s);
	assert_eq!(t, expected_t);
	assert_eq!(q, expected_q);
	assert_eq!(z, expected_z);
	// All optional-vector output modes have the same pencil factors.
	for want_q in [false, true] {
		for want_z in [false, true] {
			// Transposed backing arrays make the factors row-major strided views.
			let mut ss = source_s.transpose().to_owned();
			let mut tt = source_t.transpose().to_owned();
			let mut qq = Mat::<T>::identity(3, 3);
			let mut zz = qq.clone();
			let got = reorder(
				ss.transpose_mut(),
				tt.transpose_mut(),
				if want_q { Some(qq.as_mut()) } else { None },
				if want_z { Some(zz.as_mut()) } else { None },
				2,
				1,
				MemStack::new(&mut mem),
			)
			.unwrap();
			assert_eq!(got, 1);
			assert_eq!(ss.transpose(), expected_s);
			assert_eq!(tt.transpose(), expected_t);
			if want_q {
				assert_eq!(qq, expected_q);
			}
			if want_z {
				assert_eq!(zz, expected_z);
			}
		}
	}
}
#[test]
fn f32_failure_modes() {
	failure_and_modes::<f32>(|s, t, q, z, i, j, stack| {
		gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
	});
}
#[test]
fn f64_failure_modes() {
	failure_and_modes::<f64>(|s, t, q, z, i, j, stack| {
		gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
	});
}
#[test]
fn c32_failure_modes() {
	failure_and_modes::<c32>(|s, t, q, z, i, j, stack| {
		gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
	});
}
#[test]
fn c64_failure_modes() {
	failure_and_modes::<c64>(|s, t, q, z, i, j, stack| {
		gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
	});
}

#[test]
fn overflowing_stripe_restores_all_factors() {
	let mut s = faer::mat![
		[1.0f64, 0.5, 1.7e308],
		[0.0, 2.0, 1.7e308],
		[0.0, 0.0, 3.0]
	];
	let mut t = Mat::<f64>::identity(3, 3);
	let mut q = t.clone();
	let mut z = t.clone();
	let originals = (s.clone(), t.clone(), q.clone(), z.clone());
	let mut mem = MemBuffer::new(gevd::reorder_schur_scratch::<f64>(3));
	let error = gevd::reorder_schur_real(
		s.as_mut(),
		t.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		0,
		1,
		Par::Seq,
		MemStack::new(&mut mem),
	)
	.unwrap_err();
	assert_eq!(error.position, 0);
	assert_eq!(s, originals.0);
	assert_eq!(t, originals.1);
	assert_eq!(q, originals.2);
	assert_eq!(z, originals.3);
}

#[cfg(feature = "threadteam")]
fn team_case<T: ComplexField>(
	mut reorder: impl FnMut(
		MatMut<'_, T>,
		MatMut<'_, T>,
		Option<MatMut<'_, T>>,
		Option<MatMut<'_, T>>,
		usize,
		usize,
		Par<'_>,
		&mut MemStack,
	) -> Result<usize, gevd::SchurReorderError>,
) {
	let n = 8;
	let imaginary = if T::IS_REAL {
		zero::<T>()
	} else {
		(-one::<T>()).sqrt()
	};
	let original = Mat::<T>::from_fn(n, n, |i, j| {
		if i == j {
			from_f64::<T>((i + 1) as f64)
				+ imaginary.mul_real(from_f64::<T::Real>(0.2))
		} else if i < j {
			from_f64(0.03 * sample(i, j))
		} else {
			zero()
		}
	});
	let identity = Mat::<T>::identity(n, n);
	let mut s = original.clone();
	let mut t = identity.clone();
	let mut q = t.clone();
	let mut z = t.clone();
	let mut mem = MemBuffer::new(gevd::reorder_schur_scratch::<T>(n));
	reorder(
		s.as_mut(),
		t.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		7,
		0,
		Par::Seq,
		MemStack::new(&mut mem),
	)
	.unwrap();
	for threads in [2, 4] {
		faer::team::with_team(threads, |par| {
			let mut a = original.clone();
			let mut b = identity.clone();
			let mut u = identity.clone();
			let mut v = identity.clone();
			let mut mem = MemBuffer::new(gevd::reorder_schur_scratch::<T>(n));
			reorder(
				a.as_mut(),
				b.as_mut(),
				Some(u.as_mut()),
				Some(v.as_mut()),
				7,
				0,
				par,
				MemStack::new(&mut mem),
			)
			.unwrap();
			// Within the family's tolerance, not bit for bit: bitwise agreement is
			// promised at a fixed thread count only (docs/reproducibility.md,
			// tensor4all/faer#327).
			close(a.as_ref(), s.as_ref());
			close(b.as_ref(), t.as_ref());
			close(u.as_ref(), q.as_ref());
			close(v.as_ref(), z.as_ref());
		});
	}
}
/// `close` is what `team_case` compares with, so an infinity on one side must fail it: with
/// `error = inf` and `bound = inf`, `error <= bound` alone would accept the pair.
#[test]
#[should_panic(expected = "non-finite")]
fn close_rejects_an_infinite_reference() {
	let finite = Mat::<f64>::identity(8, 8);
	let mut infinite = finite.clone();
	infinite[(0, 0)] = f64::INFINITY;
	close(finite.as_ref(), infinite.as_ref());
}

#[cfg(feature = "threadteam")]
#[test]
fn sequential_and_team_agree() {
	team_case::<f32>(gevd::reorder_schur_real);
	team_case::<f64>(gevd::reorder_schur_real);
	team_case::<c32>(gevd::reorder_schur_cplx);
	team_case::<c64>(gevd::reorder_schur_cplx);
}

fn exact_range<T: ComplexField>(
	values: &[f64],
	mut reorder: impl FnMut(
		MatMut<'_, T>,
		MatMut<'_, T>,
		Option<MatMut<'_, T>>,
		Option<MatMut<'_, T>>,
		usize,
		usize,
		&mut MemStack,
	) -> Result<usize, gevd::SchurReorderError>,
) {
	for &scale in values {
		// Diagonal pencils have an exact permutation solution even at the
		// smallest subnormal. Distinct alpha/beta pairs pin their order.
		let a = from_f64::<T>(scale);
		let mut s = Mat::<T>::zeros(2, 2);
		s[(0, 0)] = a.copy();
		s[(1, 1)] = a.copy();
		let mut t = Mat::<T>::zeros(2, 2);
		t[(0, 0)] = a.copy();
		let old_s = s.clone();
		let old_t = t.clone();
		let mut q = Mat::identity(2, 2);
		let mut z = q.clone();
		let mut mem = MemBuffer::new(gevd::reorder_schur_scratch::<T>(2));
		assert_eq!(
			reorder(
				s.as_mut(),
				t.as_mut(),
				Some(q.as_mut()),
				Some(z.as_mut()),
				1,
				0,
				MemStack::new(&mut mem)
			)
			.unwrap(),
			0
		);
		assert_eq!(t[(0, 0)], zero());
		assert_ne!(t[(1, 1)], zero());
		assert_eq!(s[(0, 0)].abs(), a.abs());
		assert_eq!(s[(1, 1)].abs(), a.abs());
		assert_eq!(&q * &s * z.adjoint(), old_s);
		assert_eq!(&q * &t * z.adjoint(), old_t);
		unitary(q.as_ref());
		unitary(z.as_ref());
	}
}
#[test]
fn f32_exact_range() {
	exact_range::<f32>(
		&[
			f32::from_bits(1) as f64,
			f32::MIN_POSITIVE as f64,
			2f64.powi(-125),
			2f64.powi(126),
			f32::MAX as f64,
		],
		|s, t, q, z, i, j, stack| {
			gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
		},
	);
}
#[test]
fn f64_exact_range() {
	exact_range::<f64>(
		&[
			f64::from_bits(1),
			f64::MIN_POSITIVE,
			2f64.powi(-1021),
			2f64.powi(1022),
			f64::MAX,
		],
		|s, t, q, z, i, j, stack| {
			gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
		},
	);
}
#[test]
fn c32_exact_range() {
	exact_range::<c32>(
		&[
			f32::from_bits(1) as f64,
			f32::MIN_POSITIVE as f64,
			2f64.powi(-125),
			2f64.powi(126),
			f32::MAX as f64,
		],
		|s, t, q, z, i, j, stack| {
			gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
		},
	);
}
#[test]
fn c64_exact_range() {
	exact_range::<c64>(
		&[
			f64::from_bits(1),
			f64::MIN_POSITIVE,
			2f64.powi(-1021),
			2f64.powi(1022),
			f64::MAX,
		],
		|s, t, q, z, i, j, stack| {
			gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
		},
	);
}

fn clustered<T: ComplexField>(
	mut reorder: impl FnMut(
		MatMut<'_, T>,
		MatMut<'_, T>,
		Option<MatMut<'_, T>>,
		Option<MatMut<'_, T>>,
		usize,
		usize,
		&mut MemStack,
	) -> Result<usize, gevd::SchurReorderError>,
) {
	for separation in [zero::<T::Real>(), eps::<T::Real>().sqrt()] {
		let n = 4;
		let mut s = Mat::<T>::from_fn(n, n, |i, j| {
			if i == j {
				one::<T>()
					+ from_real::<T>(
						&(from_f64::<T::Real>(i as f64) * &separation),
					)
			} else if j == i + 1 {
				one()
			} else {
				zero()
			}
		});
		let old = s.clone();
		let mut t = Mat::<T>::identity(n, n);
		let mut q = t.clone();
		let mut z = t.clone();
		let mut mem = MemBuffer::new(gevd::reorder_schur_scratch::<T>(n));
		let result = reorder(
			s.as_mut(),
			t.as_mut(),
			Some(q.as_mut()),
			Some(z.as_mut()),
			3,
			0,
			MemStack::new(&mut mem),
		);
		// A rejected ill-separated move must still preserve the pencil and
		// a Schur form at its reported last valid position.
		let position = match result {
			Ok(position) => {
				assert_eq!(position, 0);
				position
			},
			Err(error) => error.position,
		};
		close((&q * &s * z.adjoint()).as_ref(), old.as_ref());
		close(
			(&q * &t * z.adjoint()).as_ref(),
			Mat::<T>::identity(n, n).as_ref(),
		);
		unitary(q.as_ref());
		unitary(z.as_ref());
		for j in 0..n {
			for i in j + 1..n {
				assert_eq!(s[(i, j)], zero());
				assert_eq!(t[(i, j)], zero());
			}
		}
		let alpha = from_real::<T>(
			&(one::<T::Real>() + from_f64::<T::Real>(3.0) * &separation),
		);
		assert!(
			(&s[(position, position)] - &alpha * &t[(position, position)])
				.abs() <= tolerance::<T>(n)
		);
	}
}
#[test]
fn f32_clustered() {
	clustered::<f32>(|s, t, q, z, i, j, stack| {
		gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
	});
}
#[test]
fn f64_clustered() {
	clustered::<f64>(|s, t, q, z, i, j, stack| {
		gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
	});
}
#[test]
fn c32_clustered() {
	clustered::<c32>(|s, t, q, z, i, j, stack| {
		gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
	});
}
#[test]
fn c64_clustered() {
	clustered::<c64>(|s, t, q, z, i, j, stack| {
		gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
	});
}

fn saved_real_eigenvalues<T: RealField>(scales: &[f64]) {
	for &sa in scales {
		for &sb in scales {
			let sa = from_f64::<T>(sa);
			let sb = from_f64::<T>(sb);
			// A nontrivial generalized pair, then infinity, 0/0 and a finite
			// singleton. The polynomial oracle does not use the 2x2 helper.
			let mut s = Mat::<T>::zeros(5, 5);
			s[(0, 0)] = from_f64::<T>(0.1) * &sa;
			s[(1, 1)] = from_f64::<T>(0.1) * &sa;
			s[(0, 1)] = sa.copy();
			s[(1, 0)] = -&sa;
			s[(2, 2)] = sa.copy();
			s[(4, 4)] = sa.copy();
			let mut t = Mat::<T>::zeros(5, 5);
			t[(0, 0)] = from_f64::<T>(1.2) * &sb;
			t[(1, 1)] = from_f64::<T>(0.7) * &sb;
			t[(4, 4)] = sb.copy();
			let original = (s.clone(), t.clone());
			// Reversed output columns exercise nonunit/negative strides and
			// previously filled output buffers.
			let mut ar = Col::<T>::from_fn(5, |_| from_f64(42.0));
			let mut ai = ar.clone();
			let mut be = ar.clone();
			gevd::schur_eigenvalues_real(
				s.as_ref(),
				t.as_ref(),
				ar.reverse_rows_mut(),
				ai.reverse_rows_mut(),
				be.reverse_rows_mut(),
			)
			.unwrap();
			let ar = ar.reverse_rows();
			let ai = ai.reverse_rows();
			let be = be.reverse_rows();
			let normalized = |i: usize| {
				let re = &ar[i] / &sa;
				let im = &ai[i] / &sa;
				let beta = &be[i] / &sb;
				let scale = re.abs().fmax(im.abs()).fmax(beta.abs());
				assert!(scale.is_finite() && scale != zero());
				(&re / &scale, &im / &scale, &beta / &scale)
			};
			for i in 0..2 {
				let (re, im, beta) = normalized(i);
				// det(S-lambda*T) = .84 lambda^2 - .19 lambda + 1.01.
				// Homogeneous pairs may share any nonzero scale: normalize
				// before evaluating this polynomial, without loosening its bound.
				let er = from_f64::<T>(0.84) * (&re * &re - &im * &im)
					- from_f64::<T>(0.19) * &re * &beta
					+ from_f64::<T>(1.01) * &beta * &beta;
				let ei = from_f64::<T>(1.68) * &re * &im
					- from_f64::<T>(0.19) * &im * &beta;
				assert!(er.abs().fmax(ei.abs()) <= tolerance::<T>(2));
				assert_ne!(im, zero());
				assert_ne!(beta, zero());
			}
			let (_, im0, b0) = normalized(0);
			let (_, im1, b1) = normalized(1);
			assert!((im0 * b1 + im1 * b0).abs() <= tolerance::<T>(2));
			assert_eq!(
				(ar[2].copy(), ai[2].copy(), be[2].copy()),
				(sa.copy(), zero(), zero())
			);
			assert_eq!(
				(ar[3].copy(), ai[3].copy(), be[3].copy()),
				(zero(), zero(), zero())
			);
			assert_eq!(
				(ar[4].copy(), ai[4].copy(), be[4].copy()),
				(sa.copy(), zero(), sb.copy())
			);
			assert_eq!((s, t), original);
		}
	}
	for n in [0, 1] {
		let s = Mat::<T>::identity(n, n);
		let t = s.clone();
		let mut re = Col::<T>::zeros(n);
		let mut im = re.clone();
		let mut beta = re.clone();
		gevd::schur_eigenvalues_real(
			s.as_ref(),
			t.as_ref(),
			re.as_mut(),
			im.as_mut(),
			beta.as_mut(),
		)
		.unwrap();
		if n == 1 {
			assert_eq!(
				(re[0].copy(), im[0].copy(), beta[0].copy()),
				(one(), zero(), one())
			);
		}
	}
}
#[test]
fn f32_saved_eigenvalues() {
	saved_real_eigenvalues::<f32>(&[1.0, 1e-30, 1e30]);
}
#[test]
fn f64_saved_eigenvalues() {
	saved_real_eigenvalues::<f64>(&[1.0, 1e-300, 1e300]);
}

fn accumulated_vectors<T: ComplexField>(
	phase: T,
	mut reorder: impl FnMut(
		MatMut<'_, T>,
		MatMut<'_, T>,
		Option<MatMut<'_, T>>,
		Option<MatMut<'_, T>>,
		usize,
		usize,
		&mut MemStack,
	) -> Result<usize, gevd::SchurReorderError>,
) {
	let n = 4;
	let mut s = Mat::<T>::from_fn(n, n, |i, j| {
		if i == j {
			from_f64((i + 1) as f64)
		} else if i < j {
			from_f64(0.03 * sample(i, j))
		} else {
			zero()
		}
	});
	let mut t = Mat::<T>::identity(n, n);
	let mut q = Mat::<T>::from_fn(n, n, |i, j| {
		if i == j {
			if i % 2 == 0 {
				phase.copy()
			} else {
				-one::<T>()
			}
		} else {
			zero()
		}
	});
	let mut z = Mat::<T>::from_fn(n, n, |i, j| {
		if i == (j + 1) % n {
			phase.copy()
		} else {
			zero()
		}
	});
	let a = &q * &s * z.adjoint();
	let b = &q * &t * z.adjoint();
	let mut mem = MemBuffer::new(gevd::reorder_schur_scratch::<T>(n));
	reorder(
		s.as_mut(),
		t.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		3,
		0,
		MemStack::new(&mut mem),
	)
	.unwrap();
	close((&q * &s * z.adjoint()).as_ref(), a.as_ref());
	close((&q * &t * z.adjoint()).as_ref(), b.as_ref());
	unitary(q.as_ref());
	unitary(z.as_ref());
	assert!(
		(&s[(0, 0)] - from_f64::<T>(4.0) * &t[(0, 0)]).abs()
			<= tolerance::<T>(n)
	);
}
#[test]
fn f32_accumulated_vectors() {
	accumulated_vectors::<f32>(1.0, |s, t, q, z, i, j, stack| {
		gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
	});
}
#[test]
fn f64_accumulated_vectors() {
	accumulated_vectors::<f64>(1.0, |s, t, q, z, i, j, stack| {
		gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
	});
}
#[test]
fn c32_accumulated_vectors() {
	accumulated_vectors::<c32>(
		c32::new(0.6, 0.8),
		|s, t, q, z, i, j, stack| {
			gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
		},
	);
}
#[test]
fn c64_accumulated_vectors() {
	accumulated_vectors::<c64>(
		c64::new(0.6, 0.8),
		|s, t, q, z, i, j, stack| {
			gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
		},
	);
}

fn repeated_real_pairs<T: RealField>() {
	let mut s = Mat::<T>::zeros(4, 4);
	for start in [0, 2] {
		s[(start, start + 1)] = -one::<T>();
		s[(start + 1, start)] = one();
	}
	s[(0, 2)] = one();
	s[(1, 3)] = one();
	let mut t = Mat::<T>::identity(4, 4);
	let mut q = t.clone();
	let mut z = t.clone();
	let original = (s.clone(), t.clone());
	let mut mem = MemBuffer::new(gevd::reorder_schur_scratch::<T>(4));
	let result = gevd::reorder_schur_real(
		s.as_mut(),
		t.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		2,
		0,
		Par::Seq,
		MemStack::new(&mut mem),
	);
	match result {
		Ok(position) => assert_eq!(position, 0),
		Err(error) => assert_eq!(error.position, 2),
	}
	close((&q * &s * z.adjoint()).as_ref(), original.0.as_ref());
	close((&q * &t * z.adjoint()).as_ref(), original.1.as_ref());
	unitary(q.as_ref());
	unitary(z.as_ref());
	assert_ne!(s[(1, 0)], zero());
	assert_ne!(s[(3, 2)], zero());
	assert_eq!(s[(2, 1)], zero());
}
#[test]
fn f32_repeated_real_pairs() {
	repeated_real_pairs::<f32>();
}
#[test]
fn f64_repeated_real_pairs() {
	repeated_real_pairs::<f64>();
}

fn graded_real_pairs<T: RealField>(
	tiny: f64,
	huge: f64,
	root_tiny: f64,
	root_huge: f64,
) {
	for center in [0.0, 0.1] {
		for (left, right) in [
			([tiny, huge], [1.0, 1.0]),
			([1.0, 1.0], [tiny, huge]),
			([root_tiny, root_huge], [root_huge, root_tiny]),
		] {
			let l = left.map(from_f64::<T>);
			let r = right.map(from_f64::<T>);
			let s = Mat::<T>::from_fn(2, 2, |i, j| {
				from_f64::<T>(if i == j {
					center
				} else if i < j {
					-1.0
				} else {
					1.0
				}) * &l[i] * &r[j]
			});
			let t = Mat::<T>::from_fn(2, 2, |i, j| {
				if i == j { &l[i] * &r[j] } else { zero() }
			});
			let mut re = Col::<T>::zeros(2);
			let mut im = re.clone();
			let mut beta = re.clone();
			gevd::schur_eigenvalues_real(
				s.as_ref(),
				t.as_ref(),
				re.as_mut(),
				im.as_mut(),
				beta.as_mut(),
			)
			.unwrap();
			for i in 0..2 {
				assert!(
					((&re[i] / &beta[i]) - from_f64::<T>(center)).abs()
						< tolerance::<T>(2)
				);
				assert!(
					((&im[i] / &beta[i]).abs() - one::<T>()).abs()
						< tolerance::<T>(2)
				);
			}
		}
	}
}
#[test]
fn f32_graded_real_pairs() {
	graded_real_pairs::<f32>(1e-30, 1e30, 1e-15, 1e15);
}
#[test]
fn f64_graded_real_pairs() {
	graded_real_pairs::<f64>(1e-300, 1e300, 1e-150, 1e150);
}

#[test]
fn unsafe_eigenvalue_scaling_keeps_the_unwritten_suffix() {
	let mut s = Mat::<f64>::zeros(3, 3);
	s[(0, 0)] = 2.0;
	s[(1, 1)] = 1e300;
	s[(2, 2)] = 1e300;
	s[(1, 2)] = -1e300;
	s[(2, 1)] = 1e-300;
	let mut t = Mat::<f64>::identity(3, 3);
	t[(1, 1)] = 1e-300;
	t[(2, 2)] = 1e-300;
	let original = (s.clone(), t.clone());
	let mut re = Col::<f64>::full(3, 42.0);
	let mut im = re.clone();
	let mut beta = re.clone();
	let error = gevd::schur_eigenvalues_real(
		s.as_ref(),
		t.as_ref(),
		re.as_mut(),
		im.as_mut(),
		beta.as_mut(),
	)
	.unwrap_err();
	assert_eq!(error.position, 1);
	assert_eq!((re[0], im[0], beta[0]), (2.0, 0.0, 1.0));
	for i in 1..3 {
		assert_eq!((re[i], im[i], beta[i]), (42.0, 42.0, 42.0));
	}
	assert_eq!((s, t), original);
}
