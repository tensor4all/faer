use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::EvdError;
use faer::linalg::lu::partial_pivoting::{factor, solve};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par, Side, mat};

fn main() -> Result<(), EvdError> {
	// Two logical rows, two columns, lda = 3. Padding is not matrix data.
	let data = [0.0_f64, 3.0, -999.0, 2.0, 4.0, -999.0];
	let a = MatRef::from_column_major_slice_with_stride(&data, 2, 2, 3);
	let b = mat![[4.0], [11.0]];

	// The pool and the team are built once; each operation lends the team and returns it.
	let pool = threadteam::Pool::builder()
		.threads(2)
		.acquire_timeout(std::time::Duration::from_secs(30))
		.build()
		.expect("a thread pool for the team");
	let team = pool.into_team(2).ok().expect("a newly built pool is idle");
	let mut exec = team.exec(2);
	exec.team(|token| {
		let mut par = Par::Team(token);
		let mut lu = a.to_owned();
		let mut x = b.clone();
		let mut perm = [0_usize; 2];
		let mut perm_inv = [0_usize; 2];
		let factor_req = factor::lu_in_place_scratch::<usize, f64>(
			2,
			2,
			par.shape(),
			Default::default(),
		);
		let solve_req =
			solve::solve_in_place_scratch::<usize, f64>(2, 1, par.shape());
		let mut scratch = MemBuffer::new(factor_req.or(solve_req));
		let (_, row_perm) = factor::lu_in_place(
			lu.as_mut(),
			&mut perm,
			&mut perm_inv,
			par.reborrow(),
			MemStack::new(&mut scratch),
			Default::default(),
		);
		// L and U share packed storage; the routine selects the two triangles.
		solve::solve_in_place(
			lu.as_ref(),
			lu.as_ref(),
			row_perm,
			x.as_mut(),
			par.reborrow(),
			MemStack::new(&mut scratch),
		);
		assert!((x[(0, 0)] - 1.0).abs() < 1e-12);
		assert!((x[(1, 0)] - 2.0).abs() < 1e-12);
		let mut ax = Mat::<f64>::zeros(2, 1);
		matmul(&mut ax, Accum::Replace, a, &x, 1.0, par.reborrow());
		assert!((ax[(0, 0)] - b[(0, 0)]).abs() < 1e-12);
		assert!((ax[(1, 0)] - b[(1, 0)]).abs() < 1e-12);
		assert_eq!(data[2], -999.0);
		assert_eq!(data[5], -999.0);

		let symmetric = mat![[2.0_f64, 1.0], [1.0, 2.0]];
		let eigen = symmetric.self_adjoint_eigen(Side::Lower, par)?;
		let values = eigen.S().column_vector();
		assert!((values[0] - 1.0).abs() < 1e-12);
		assert!((values[1] - 3.0).abs() < 1e-12);
		for j in 0..2 {
			for i in 0..2 {
				let av: f64 =
					(0..2).map(|k| symmetric[(i, k)] * eigen.U()[(k, j)]).sum();
				assert!((av - values[j] * eigen.U()[(i, j)]).abs() < 1e-12);
			}
		}
		Ok(())
	})
	.ok()
	.expect("the team is idle between calls")
}
