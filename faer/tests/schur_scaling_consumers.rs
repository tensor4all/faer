//! #270: the public Arnoldi workspace also covers projected Schur scaling.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::eigen::{
	PartialEigenParams, partial_eigen, partial_eigen_scratch,
};
use faer::{Col, Mat, Par, c32, c64};

macro_rules! partial_case {
	($name:ident, $t:ty, $r:ty, $z:ty, $make:expr, $to_z:expr) => {
		#[test]
		fn $name() {
			let n = 96;
			let k = 3;
			let params = PartialEigenParams {
				min_dim: 32,
				max_dim: 64,
				max_restarts: 100,
				..Default::default()
			};
			// Distinct diagonal spectrum with a deterministic upper coupling;
			// the oracle below uses the dimensionless original operator.
			let normal = Mat::<$t>::from_fn(n, n, |i, j| {
				($make)(if i == j {
					1.0 + i as $r / n as $r
				} else if j == i + 1 {
					0.01
				} else {
					0.0
				})
			});
			for scale in
				[1.0 as $r, (2.0 as $r).powi(30), (2.0 as $r).powi(-30)]
			{
				let a = Mat::<$t>::from_fn(n, n, |i, j| normal[(i, j)] * scale);
				let v0 =
					Col::<$t>::from_fn(n, |_| ($make)(1.0 / (n as $r).sqrt()));
				let mut vectors = Mat::<$z>::zeros(n, k);
				let mut values = vec![<$z>::new(0.0, 0.0); k];
				let mut mem = MemBuffer::new(partial_eigen_scratch(
					&a.as_ref(),
					k,
					Par::Seq.shape(),
					params,
				));
				let info = partial_eigen(
					vectors.as_mut(),
					&mut values,
					&a.as_ref(),
					v0.as_ref(),
					128.0 * <$r>::EPSILON * scale,
					Par::Seq,
					MemStack::new(&mut mem),
					params,
				);
				assert_eq!(info.n_converged_eigen, k, "scale={scale}");
				for j in 0..k {
					let lambda = values[j] / scale;
					let mut err = 0.0 as $r;
					for i in 0..n {
						let mut r = -lambda * vectors[(i, j)];
						for l in 0..n {
							r += ($to_z)(normal[(i, l)]) * vectors[(l, j)];
						}
						err += r.norm_sqr();
					}
					let norm = vectors.col(j).norm_l2();
					// Unlike saved-Schur extraction, partial_eigen does not
					// promise normalization. Test a relative residual.
					assert!(norm.is_finite() && norm > 0.0);
					let expected = 1.0 + (n - 1 - j) as $r / n as $r;
					assert!(
						(lambda - ($to_z)(($make)(expected))).norm()
							<= 256.0 * n as $r * <$r>::EPSILON
					);
					assert!(
						err.sqrt() / norm <= 256.0 * n as $r * <$r>::EPSILON,
						"scale={scale} j={j} residual={}",
						err.sqrt()
					);
				}
			}
		}
	};
}
partial_case!(projected_real_f32, f32, f32, c32, |x| x, |x| c32::new(
	x, 0.0
));
partial_case!(projected_real_f64, f64, f64, c64, |x| x, |x| c64::new(
	x, 0.0
));
partial_case!(
	projected_complex_f32,
	c32,
	f32,
	c32,
	|x| c32::new(x, 0.25 * x),
	|x| x
);
partial_case!(
	projected_complex_f64,
	c64,
	f64,
	c64,
	|x| c64::new(x, 0.25 * x),
	|x| x
);
