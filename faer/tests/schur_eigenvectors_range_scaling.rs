//! tensor4all/faer#270: `eigenvectors_from_schur_real` /
//! `eigenvectors_from_schur_cplx` on a Schur form whose entries are far from
//! one, where the back substitution runs on a copy scaled into range that now
//! comes from the caller's scratch. The eigenvectors of `scale * t` are those
//! of `t`, so they are checked against `t` (and `q * t * q^H`): residuals, unit
//! norms, real conjugate pairs and complex phases. A scratch that covers the
//! eigenvector matrix but not the scaled copy is insufficient, and the call
//! panics as on any insufficient scratch.
mod common;
use common::*;
use faer::linalg::evd::{
	eigenvectors_from_schur_cplx, eigenvectors_from_schur_real,
	eigenvectors_from_schur_scratch,
};
use faer::linalg::temp_mat_scratch;
use std::panic::{AssertUnwindSafe, catch_unwind};

const SIZES: [usize; 3] = [2, 8, 33];

fn orthogonal<T: ComplexField>(n: usize) -> Mat<T> {
	Mat::<T>::from_fn(n, n, |i, j| from_f64(sample(i, j)))
		.qr(Par::Seq)
		.compute_Q(Par::Seq)
}

/// A standardized real Schur form with 2 by 2 blocks at rows `1..3`, `4..6`,
/// ...
fn real_schur<T: RealField>(n: usize) -> Mat<T> {
	let pair = |k: usize| k % 3 == 1 && k + 1 < n;
	Mat::from_fn(n, n, |i, j| {
		if i == j {
			let k = if i > 0 && pair(i - 1) { i - 1 } else { i };
			from_f64((k + 1) as f64)
		} else if i + 1 == j && pair(i) {
			from_f64(0.7)
		} else if i == j + 1 && pair(j) {
			from_f64(-0.3)
		} else if i < j {
			from_f64(0.5 * sample(i, j))
		} else {
			zero()
		}
	})
}

/// An upper triangular form whose eigenvalues `(k + 1) e^{0.37 i k}` go round
/// the unit circle.
fn complex_schur<T: ComplexField>(n: usize, make: fn(f64, f64) -> T) -> Mat<T> {
	Mat::from_fn(n, n, |i, j| {
		if i == j {
			let phase = 0.37 * i as f64;
			make((i + 1) as f64 * phase.cos(), (i + 1) as f64 * phase.sin())
		} else if i < j {
			make(0.5 * sample(i, j), 0.5 * sample(j, i))
		} else {
			make(0.0, 0.0)
		}
	})
}

fn scaled<T: ComplexField>(t: &Mat<T>, scale: f64) -> Mat<T> {
	let scale = from_f64::<T::Real>(scale);
	Mat::from_fn(t.nrows(), t.ncols(), |i, j| t[(i, j)].mul_real(&scale))
}

fn real<T: RealField>(scales: [f64; 3]) {
	for n in SIZES {
		let t0 = real_schur::<T>(n);
		let q = orthogonal::<T>(n);
		for scale in scales {
			let t = scaled(&t0, scale);
			for back in [false, true] {
				let a = if back {
					&q * &t0 * q.transpose()
				} else {
					t0.clone()
				};
				let mut x = Mat::zeros(n, n);
				let mut mem =
					MemBuffer::new(eigenvectors_from_schur_scratch::<T>(n));
				eigenvectors_from_schur_real(
					t.as_ref(),
					if back { Some(q.as_ref()) } else { None },
					x.as_mut(),
					Par::Seq,
					MemStack::new(&mut mem),
					Default::default(),
				)
				.unwrap();
				let ax = &a * &x;
				let bound = tolerance::<T>(n) * a.norm_l2();
				let mut j = 0;
				let mut pairs = 0;
				while j < n {
					let paired = j + 1 < n && t0[(j + 1, j)] != zero();
					let wr = t0[(j, j)].copy();
					let wi = if paired {
						(t0[(j, j + 1)].abs() * t0[(j + 1, j)].abs()).sqrt()
					} else {
						zero()
					};
					let norm = if paired {
						x.col(j).norm_l2().hypot(&x.col(j + 1).norm_l2())
					} else {
						x.col(j).norm_l2()
					};
					assert!(
						(&norm - one::<T>()).abs() <= tolerance::<T>(n),
						"n={n} scale={scale:e} column {j}: norm {norm:?}"
					);
					for i in 0..n {
						// `x[:, j] + i x[:, j + 1]` belongs to `wr + i wi`,
						// with `wi > 0`
						let re = &wr * &x[(i, j)]
							- if paired {
								&wi * &x[(i, j + 1)]
							} else {
								zero()
							};
						assert!(
							(&ax[(i, j)] - re).abs() <= bound,
							"n={n} scale={scale:e} back={back} ({i}, {j})"
						);
						if paired {
							let im = &wr * &x[(i, j + 1)] + &wi * &x[(i, j)];
							assert!(
								(&ax[(i, j + 1)] - im).abs() <= bound,
								"n={n} scale={scale:e} back={back} ({i}, {})",
								j + 1
							);
						}
					}
					pairs += paired as usize;
					j += if paired { 2 } else { 1 };
				}
				assert_eq!(pairs, n / 3);
			}
		}
	}
}

fn complex<T: ComplexField>(scales: [f64; 3], make: fn(f64, f64) -> T) {
	for n in SIZES {
		let t0 = complex_schur::<T>(n, make);
		let q = orthogonal::<T>(n);
		for scale in scales {
			let t = scaled(&t0, scale);
			for back in [false, true] {
				let a = if back {
					&q * &t0 * q.adjoint()
				} else {
					t0.clone()
				};
				let mut x = Mat::zeros(n, n);
				let mut mem =
					MemBuffer::new(eigenvectors_from_schur_scratch::<T>(n));
				eigenvectors_from_schur_cplx(
					t.as_ref(),
					if back { Some(q.as_ref()) } else { None },
					x.as_mut(),
					Par::Seq,
					MemStack::new(&mut mem),
					Default::default(),
				)
				.unwrap();
				let mut xw = x.clone();
				for j in 0..n {
					assert!(
						(x.col(j).norm_l2() - one::<T::Real>()).abs()
							<= tolerance::<T>(n),
						"n={n} scale={scale:e} column {j}"
					);
					for i in 0..n {
						xw[(i, j)] *= &t0[(j, j)];
					}
				}
				close((&a * &x).as_ref(), xw.as_ref());
			}
		}
	}
}

#[test]
fn range_scaled_real_schur_pairs() {
	real::<f32>([1.0, 1e30, 1e-30]);
	real::<f64>([1.0, 1e300, 1e-300]);
}

#[test]
fn range_scaled_complex_schur_phases() {
	complex::<c32>([1.0, 1e30, 1e-30], |r, i| c32::new(r as f32, i as f32));
	complex::<c64>([1.0, 1e300, 1e-300], c64::new);
}

type Call<T> = fn(MatRef<'_, T>, faer::MatMut<'_, T>, &mut MemStack);
fn real_call<T: RealField>(
	t: MatRef<'_, T>,
	x: faer::MatMut<'_, T>,
	stack: &mut MemStack,
) {
	eigenvectors_from_schur_real(
		t,
		None,
		x,
		Par::Seq,
		stack,
		Default::default(),
	)
	.unwrap();
}
fn complex_call<T: ComplexField>(
	t: MatRef<'_, T>,
	x: faer::MatMut<'_, T>,
	stack: &mut MemStack,
) {
	eigenvectors_from_schur_cplx(
		t,
		None,
		x,
		Par::Seq,
		stack,
		Default::default(),
	)
	.unwrap();
}

/// Whether `call` panics with a scratch of `bytes` bytes.
fn panics<T: ComplexField>(call: Call<T>, t: &Mat<T>, bytes: usize) -> bool {
	// keep the expected scratch panics out of the output; any other panic is
	// still printed
	static QUIET: std::sync::Once = std::sync::Once::new();
	QUIET.call_once(|| {
		let default = std::panic::take_hook();
		std::panic::set_hook(Box::new(move |info| {
			if !info.to_string().contains("buffer is not large enough") {
				default(info);
			}
		}));
	});
	let n = t.nrows();
	let mut x = Mat::<T>::zeros(n, n);
	let mut mem =
		MemBuffer::new(dyn_stack::StackReq::new_aligned::<u8>(bytes, 128));
	match catch_unwind(AssertUnwindSafe(|| {
		call(t.as_ref(), x.as_mut(), MemStack::new(&mut mem))
	})) {
		Ok(()) => false,
		Err(payload) => {
			// the scratch ran out, and nothing else went wrong
			let message = payload
				.downcast_ref::<String>()
				.map(String::as_str)
				.or(payload.downcast_ref::<&str>().copied())
				.unwrap_or("");
			assert!(
				message.contains("buffer is not large enough"),
				"{message}"
			);
			true
		},
	}
}

fn insufficient<T: ComplexField>(call: Call<T>, t: &Mat<T>, extreme: [f64; 2]) {
	let n = t.nrows();
	let v = temp_mat_scratch::<T>(n, n).size_bytes();
	// less than the eigenvector matrix: insufficient at any scale
	for scale in [1.0, extreme[0], extreme[1]] {
		assert!(panics(call, &scaled(t, scale), v / 2), "scale={scale:e}");
	}
	// the eigenvector matrix but not the scaled copy: insufficient once the
	// copy is needed
	for scale in extreme {
		assert!(panics(call, &scaled(t, scale), v), "scale={scale:e}");
	}
	// the copy is taken only when the entries are out of range
	assert!(!panics(call, &scaled(t, 1.0), v));
	// and the query covers both
	assert!(eigenvectors_from_schur_scratch::<T>(n).size_bytes() >= 2 * v);
}

#[test]
fn insufficient_scratch_panics() {
	insufficient(real_call::<f32>, &real_schur(8), [1e30, 1e-30]);
	insufficient(real_call::<f64>, &real_schur(8), [1e300, 1e-300]);
	insufficient(
		complex_call::<c32>,
		&complex_schur(8, |r, i| c32::new(r as f32, i as f32)),
		[1e30, 1e-30],
	);
	insufficient(
		complex_call::<c64>,
		&complex_schur(8, c64::new),
		[1e300, 1e-300],
	);
}
