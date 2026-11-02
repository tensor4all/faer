mod common;
use common::*;
use faer::linalg::solvers::{
	SelfAdjointGeneralizedEigen, SelfAdjointGevdError,
};

macro_rules! suite {
	($module:ident, $scalar:ty, $make:expr) => {
		mod $module {
			use super::*;

			fn make(r: f64, i: f64) -> $scalar {
				let make: fn(f64, f64) -> $scalar = $make;
				make(r, i)
			}
			fn real(x: <$scalar as ComplexField>::Real) -> f64 {
				x
			}

			/// `||A X - B X Lambda|| / (||A|| + max|lambda| ||B||)`, the only
			/// eigenvector check that is invariant under the phase and the
			/// order of the columns
			pub(super) fn residual(
				a: MatRef<'_, $scalar>,
				b: MatRef<'_, $scalar>,
				eig: &SelfAdjointGeneralizedEigen<$scalar>,
			) -> f64 {
				let (n, u) = (a.nrows(), eig.U());
				let au = a * u;
				let bu = b * u;
				let bul = Mat::from_fn(n, n, |i, j| {
					bu[(i, j)] * make(real(eig.values()[j]), 0.0)
				});
				let lambda = eig
					.values()
					.iter()
					.fold(0.0_f64, |m, x| m.max(real(x.clone()).abs()));
				(&au - &bul).norm_l2()
					/ (a.norm_l2() + from_f64::<f64>(lambda) * b.norm_l2())
			}

			/// `a` is Hermitian, `b` is Hermitian positive definite
			fn pair(n: usize) -> (Mat<$scalar>, Mat<$scalar>) {
				let x = Mat::from_fn(n, n, |i, j| {
					make(sample(i, j), sample(j + 31, i + 17))
				});
				let y = Mat::from_fn(n, n, |i, j| {
					make(sample(i + 71, j + 13), sample(j + 53, i + 97))
				});
				let a = Mat::from_fn(n, n, |i, j| {
					(x[(i, j)] + x[(j, i)].conj()) * make(0.5, 0.0)
				});
				let gram = &y * y.adjoint();
				let b = Mat::from_fn(n, n, |i, j| {
					gram[(i, j)] * make(1.0 / n as f64, 0.0)
						+ make(if i == j { 1.0 } else { 0.0 }, 0.0)
				});
				(a, b)
			}

			#[test]
			fn matches_qz_and_is_b_orthonormal() {
				for n in [0, 1, 2, 4, 8, 32, 128] {
					let (a, b) = pair(n);
					// only the lower triangles are read
					let mut lower_a = a.clone();
					let mut lower_b = b.clone();
					for j in 0..n {
						for i in 0..j {
							lower_a[(i, j)] = make(7.0, 3.0);
							lower_b[(i, j)] = make(-4.0, 2.0);
						}
					}
					let eig = lower_a
						.self_adjoint_generalized_eigen(&lower_b, Par::Seq)
						.unwrap();
					let values = lower_a
						.self_adjoint_generalized_eigenvalues(
							&lower_b,
							Par::Seq,
						)
						.unwrap();
					assert_eq!(eig.values().len(), n);
					assert_eq!(eig.values(), &values[..]);
					for j in 0..n {
						assert!(real(eig.values()[j]).is_finite());
						if j > 0 {
							assert!(eig.values()[j - 1] <= eig.values()[j]);
						}
					}

					if n == 0 {
						assert_eq!(eig.U().nrows(), 0);
						continue;
					}

					let residual = residual(a.as_ref(), b.as_ref(), &eig);
					// x^H B x = 1 for every column of x
					let u = eig.U();
					let gram = &u.adjoint() * (&b * u);
					let orthogonality =
						(&gram - Mat::<$scalar>::identity(n, n)).norm_l2()
							/ (n as f64).sqrt();
					eprintln!(
						"{} n={n}: residual={residual:e}, \
						 B-orthogonality={orthogonality:e}",
						stringify!($scalar)
					);
					assert!(residual < 1e-11, "residual {residual:e}");
					assert!(
						orthogonality < 1e-11,
						"B-orthogonality {orthogonality:e}"
					);

					// the general QZ path is the independent reference
					if (1..=32).contains(&n) {
						let qz = a.generalized_eigen(&b, Par::Seq).unwrap();
						let mut reference: Vec<f64> = (0..n)
							.map(|j| {
								let z = qz.S_a().column_vector()[j]
									/ qz.S_b().column_vector()[j];
								assert!(z.im.abs() < 1e-9 * (1.0 + z.re.abs()));
								z.re
							})
							.collect();
						reference.sort_by(|x, y| x.partial_cmp(y).unwrap());
						for (actual, expected) in
							eig.values().iter().zip(reference)
						{
							let actual = real(actual.clone());
							assert!(
								(actual - expected).abs()
									< 1e-9 * (1.0 + expected.abs()),
								"{actual:?} vs {expected:?}"
							);
						}
					}
				}
			}

			#[test]
			fn non_positive_definite_is_an_error() {
				let a = Mat::<$scalar>::identity(4, 4);
				let mut b = Mat::<$scalar>::identity(4, 4);
				b[(2, 2)] = make(-1.0, 0.0);
				assert_eq!(
					a.self_adjoint_generalized_eigen(&b, Par::Seq).unwrap_err(),
					SelfAdjointGevdError::NonPositiveDefinite
				);
				assert_eq!(
					a.self_adjoint_generalized_eigenvalues(&b, Par::Seq)
						.unwrap_err(),
					SelfAdjointGevdError::NonPositiveDefinite
				);
			}

			/// a conjugated view of a self-adjoint matrix is still self-adjoint
			#[test]
			fn conjugated_view() {
				let (a, b) = pair(4);
				let eig = a
					.conjugate()
					.self_adjoint_generalized_eigen(&b, Par::Seq)
					.unwrap();
				let ax = &a.conjugate() * eig.U();
				let bx = &b * eig.U();
				for j in 0..4 {
					for i in 0..4 {
						let residual = ax[(i, j)]
							- bx[(i, j)] * make(real(eig.values()[j]), 0.0);
						assert!(real(abs(&residual)) < 1e-12);
					}
				}
			}

			#[cfg(feature = "threadteam")]
			#[test]
			fn team_matches_seq() {
				for n in [1, 2, 4, 64] {
					let (a, b) = pair(n);
					let seq =
						a.self_adjoint_generalized_eigen(&b, Par::Seq).unwrap();
					let seq_values = a
						.self_adjoint_generalized_eigenvalues(&b, Par::Seq)
						.unwrap();
					for width in [1, 2, 4, 8] {
						faer::team::with_team(width, |mut par| {
							let team = a
								.self_adjoint_generalized_eigen(
									&b,
									par.reborrow(),
								)
								.unwrap();
							let team_values = a
								.self_adjoint_generalized_eigenvalues(&b, par)
								.unwrap();
							assert_eq!(team.values().len(), n);
							for j in 0..n {
								let want = real(seq.values()[j]);
								let got = real(team.values()[j]);
								assert!(
									(got - want).abs()
										< 1e-12 * (1.0 + want.abs()),
									"width={width} n={n}: {got:?} vs {want:?}"
								);
								let got = real(team_values[j]);
								let want = real(seq_values[j]);
								assert!(
									(got - want).abs()
										< 1e-12 * (1.0 + want.abs()),
									"width={width} n={n}: {got:?} vs {want:?}"
								);
							}
							// the eigenvectors are only defined up to a
							// phase, so compare them through the residual and
							// B-orthonormality rather than their entries
							let error = residual(a.as_ref(), b.as_ref(), &team);
							assert!(
								error < 1e-11,
								"width={width} n={n}: residual {error:e}"
							);
							let u = team.U();
							let gram = &u.adjoint() * (&b * u);
							let error = (&gram
								- Mat::<$scalar>::identity(n, n))
							.norm_l2();
							assert!(
								error < 1e-11,
								"width={width} n={n}: {error:e}"
							);
						});
					}
				}
			}
		}
	};
}

suite!(real, f64, |r, _| r);
suite!(complex, c64, c64::new);

/// the parallel request must reach the factorization, not fall back to a
/// sequential path. One test, because `stats` is a process-wide counter.
#[cfg(feature = "threadteam")]
#[test]
fn team_is_actually_used() {
	use faer::utils::team::stats;
	let n = 256;
	let x = Mat::from_fn(n, n, |i, j| sample(i, j));
	let y = Mat::from_fn(n, n, |i, j| sample(i + 71, j + 13));
	let a = Mat::from_fn(n, n, |i, j| (x[(i, j)] + x[(j, i)]) * 0.5);
	let gram = &y * y.adjoint();
	let b = Mat::from_fn(n, n, |i, j| {
		gram[(i, j)] / n as f64 + if i == j { 1.0 } else { 0.0 }
	});
	faer::team::with_team(8, |par| {
		stats::enable(true);
		let _ = stats::take();
		let eig = a.self_adjoint_generalized_eigen(&b, par).unwrap();
		let evidence = stats::take();
		stats::enable(false);
		let sections: usize = evidence.run.iter().skip(2).sum();
		eprintln!("n=256 width=8 multi-participant sections={sections}");
		assert!(sections > 0, "the team was never dispatched: {evidence:?}");
		let error = real::residual(a.as_ref(), b.as_ref(), &eig);
		assert!(error < 1e-11, "n=256 width=8: residual {error:e}");
	});
}
