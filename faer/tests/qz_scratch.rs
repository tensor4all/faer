//! Exact public scratch for real QZ, including rounded and recursive AED windows.
mod common;
use common::*;
use faer::linalg::gevd::{self, ComputeEigenvectors};
extern "C" fn maximum_window(n: usize, _active: usize) -> usize {
	let w = n.saturating_sub(3) / 3;
	w + w % 2
}
fn check_values<T: RealField>(
	s: MatRef<'_, T>,
	t: MatRef<'_, T>,
	ar: &Col<T>,
	ai: &Col<T>,
	beta: &Col<T>,
) {
	let mut row = 0;
	while row < s.nrows() {
		let size = if row + 1 < s.nrows() && s[(row + 1, row)] != zero() {
			2
		} else {
			1
		};
		let mut scale = zero::<T>();
		for matrix in [s, t] {
			for j in row..row + size {
				for i in row..row + size {
					scale = scale.fmax(matrix[(i, j)].abs());
				}
			}
		}
		let a = Mat::<T>::from_fn(size, size, |i, j| {
			&s[(row + i, row + j)] / &scale
		});
		let b = Mat::<T>::from_fn(size, size, |i, j| {
			&t[(row + i, row + j)] / &scale
		});
		let coefficients = if size == 1 {
			[zero(), -b[(0, 0)].copy(), a[(0, 0)].copy()]
		} else {
			[
				&b[(0, 0)] * &b[(1, 1)] - &b[(0, 1)] * &b[(1, 0)],
				-(&a[(0, 0)] * &b[(1, 1)] + &a[(1, 1)] * &b[(0, 0)]
					- &a[(0, 1)] * &b[(1, 0)]
					- &a[(1, 0)] * &b[(0, 1)]),
				&a[(0, 0)] * &a[(1, 1)] - &a[(0, 1)] * &a[(1, 0)],
			]
		};
		for i in row..row + size {
			assert_ne!(
				beta[i],
				zero(),
				"nonsingular B must not produce an infinite eigenvalue"
			);
			let norm = ar[i].abs().fmax(ai[i].abs()).fmax(beta[i].abs());
			let re = &ar[i] / &norm;
			let im = &ai[i] / &norm;
			let be = &beta[i] / &norm;
			let [c2, c1, c0] = &coefficients;
			let er =
				c2 * (&re * &re - &im * &im) + c1 * &re * &be + c0 * &be * &be;
			let ei = (from_f64::<T>(2.0) * c2 * &re + c1 * &be) * &im;
			let bound = tolerance::<T>(size)
				* (c2.abs() * (&re * &re + &im * &im)
					+ c1.abs() * (re.abs() + im.abs()) * be.abs()
					+ c0.abs() * &be * &be);
			assert!(
				er.abs().fmax(ei.abs()) <= bound,
				"homogeneous block residual at {i}"
			);
			if size == 1 {
				assert_eq!(ai[i], zero());
			}
		}
		if size == 2 {
			assert!(
				(ai[row] > zero() && ai[row + 1] < zero())
					|| (ai[row] < zero() && ai[row + 1] > zero())
			);
		}
		row += size;
	}
}
fn check<T: RealField>(sizes: &[usize], mut par: Par) {
	for &n in sizes {
		for force_maximum in [false, true] {
			let a = Mat::<T>::from_fn(n, n, |i, j| {
				if i == j {
					from_f64(1.0 + i as f64 / n as f64)
				} else if i == j + 1 {
					from_f64(0.2)
				} else if i < j {
					from_f64(0.02 * sample(i, j))
				} else {
					zero()
				}
			});
			let b = Mat::<T>::from_fn(n, n, |i, j| {
				if i == j {
					from_f64(2.0)
				} else if i < j {
					from_f64(0.01 * sample(j, i))
				} else {
					zero()
				}
			});
			let mut params: gevd::GeneralizedSchurParams = faer::auto!(T);
			// Force a small blocked problem to exercise Sylvester scratch as well
			// as the ordinary and recursive maximum-window paths.
			if n == 18 {
				params.blocking_threshold = 15;
			}
			if force_maximum {
				params.recommended_deflation_window = maximum_window;
			}
			let mut reference = None;
			for vectors in [true, false] {
				eprintln!(
					"CASE dtype={} n={n} maximum={force_maximum} vectors={vectors}",
					core::any::type_name::<T>()
				);
				let mut s = a.clone();
				let mut t = b.clone();
				let mut q = Mat::<T>::identity(n, n);
				let mut z = q.clone();
				let mut ar = Col::<T>::from_fn(n, |_| nan::<T>());
				let mut ai = ar.clone();
				let mut beta = ar.clone();
				let mut memory = MemBuffer::new(
					gevd::qz_real::hessenberg_to_qz_scratch::<T>(
						n,
						par.shape(),
						params,
					),
				);
				gevd::qz_real::hessenberg_to_qz(
					s.as_mut(),
					t.as_mut(),
					if vectors { Some(q.as_mut()) } else { None },
					if vectors { Some(z.as_mut()) } else { None },
					ar.as_mut(),
					ai.as_mut(),
					beta.as_mut(),
					ComputeEigenvectors::Yes,
					par.reborrow(),
					params,
					MemStack::new(&mut memory),
				);
				assert!(s.is_all_finite() && t.is_all_finite());
				assert!(
					ar.is_all_finite()
						&& ai.is_all_finite()
						&& beta.is_all_finite()
				);
				for j in 0..n {
					for i in j + 1..n {
						assert_eq!(t[(i, j)], zero());
						if i > j + 1 {
							assert_eq!(s[(i, j)], zero());
						}
					}
				}

				for i in 0..n.saturating_sub(2) {
					assert!(
						s[(i + 1, i)] == zero() || s[(i + 2, i + 1)] == zero(),
						"overlapping real Schur blocks at {i}"
					);
				}
				check_values(s.as_ref(), t.as_ref(), &ar, &ai, &beta);
				if vectors {
					close((&q * &s * z.adjoint()).as_ref(), a.as_ref());
					close((&q * &t * z.adjoint()).as_ref(), b.as_ref());
					unitary(q.as_ref());
					unitary(z.as_ref());
					reference = Some((s, t, ar, ai, beta));
				} else {
					let (ss, tt, re, im, be) = reference.take().unwrap();
					assert_eq!(s, ss);
					assert_eq!(t, tt);
					assert_eq!(ar, re);
					assert_eq!(ai, im);
					assert_eq!(beta, be);
				}
			}
		}
	}
}
#[test]
fn f32_exact_qz_scratch() {
	check::<f32>(&[18, 78, 96], Par::Seq);
}
#[test]
fn f64_exact_qz_scratch() {
	check::<f64>(&[18, 78, 96], Par::Seq);
}

#[test]
#[ignore = "opt-in recursive AED workspace validation; runs in scheduled CI"]
fn f32_recursive_qz_scratch() {
	check::<f32>(&[258], Par::Seq);
}
#[test]
#[ignore = "opt-in recursive AED workspace validation; runs in scheduled CI"]
fn f64_recursive_qz_scratch() {
	check::<f64>(&[258], Par::Seq);
}

#[cfg(feature = "threadteam")]
#[test]
fn dedicated_team_qz_scratch() {
	faer::team::with_team(2, |par| check::<f64>(&[78], par));
}
