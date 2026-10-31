use faer::linalg::matmul::dot::inner_prod;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{DenseSolveCore, Solve, SvdError};
use faer::{Accum, Conj, Mat, Side, c64, col, mat};

fn main() -> Result<(), SvdError> {
	let a = mat![[4.0_f64, 1.0], [1.0, 3.0]];
	let b = mat![[6.0_f64], [7.0]];
	faer::team::with_team(2, |mut par| -> Result<(), SvdError> {
		let lu = a.partial_piv_lu(par.reborrow());
		let chol = a
			.llt(Side::Lower, par.reborrow())
			.expect("leading principal minors 4 and 11 are positive");
		for x in [lu.solve(&b, par.reborrow()), chol.solve(&b, par.reborrow())]
		{
			assert!((x[(0, 0)] - 1.0).abs() < 1.0e-12);
			assert!((x[(1, 0)] - 2.0).abs() < 1.0e-12);
		}
		let rectangular = mat![[1.0_f64, 2.0], [0.0, 1.0], [1.0, 0.0]];
		let qr = rectangular.qr(par.reborrow());
		let q = qr.compute_thin_Q(par.reborrow());
		let mut qr_product = Mat::<f64>::zeros(3, 2);
		matmul(
			&mut qr_product,
			Accum::Replace,
			&q,
			qr.thin_R(),
			1.0,
			par.reborrow(),
		);
		let svd = rectangular.thin_svd(par.reborrow())?;
		assert_eq!((svd.U().nrows(), svd.U().ncols()), (3, 2));
		assert_eq!((svd.V().nrows(), svd.V().ncols()), (2, 2));
		let svd_product = svd.reconstruct(par.reborrow());
		for i in 0..3 {
			for j in 0..2 {
				assert!(
					(qr_product[(i, j)] - rectangular[(i, j)]).abs() < 1.0e-12
				);
				assert!(
					(svd_product[(i, j)] - rectangular[(i, j)]).abs() < 1.0e-12
				);
			}
		}
		Ok(())
	})?;
	let x = col![c64::new(1.0, 1.0), c64::new(2.0, 0.0)];
	let y = col![c64::new(1.0, 0.0), c64::new(0.0, 1.0)];
	let dot =
		inner_prod(x.as_ref().transpose(), Conj::Yes, y.as_ref(), Conj::No);
	assert!((dot - c64::new(1.0, 1.0)).norm() < 1.0e-12);
	Ok(())
}
