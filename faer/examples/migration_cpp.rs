use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{LdltError, Solve};
use faer::{Accum, Mat, Par, Side, mat};

fn main() -> Result<(), LdltError> {
	let a = mat![[4.0_f64, 1.0], [1.0, 3.0]];
	let b = mat![[6.0_f64], [7.0]];
	let expected = mat![[1.0_f64], [2.0]];

	// Sequential work does not acquire parallel authority.
	let lu = a.partial_piv_lu(Par::Seq);
	let x = lu.solve(&b, Par::Seq);
	for i in 0..2 {
		assert!((x[(i, 0)] - expected[(i, 0)]).abs() < 1.0e-12);
	}

	// The pool and the team are built once; each operation lends the team and returns it.
	let pool = threadteam::Pool::builder()
		.threads(2)
		.acquire_timeout(std::time::Duration::from_secs(30))
		.build()
		.expect("a thread pool for the team");
	let team = pool.into_team(2).ok().expect("a newly built pool is idle");
	let mut exec = team.exec(2);
	exec.team(|token| -> Result<(), LdltError> {
		let mut par = Par::Team(token);
		let ldlt = a.ldlt(Side::Lower, par.reborrow())?;
		let x = ldlt.solve(&b, par.reborrow());
		let mut storage = Mat::<f64>::zeros(3, 2);
		{
			let block = storage.as_mut().submatrix_mut(1, 1, 2, 1);
			matmul(block, Accum::Replace, &a, &x, 1.0, par.reborrow());
		}
		let block = storage.as_ref().submatrix(1, 1, 2, 1);
		for i in 0..2 {
			assert!((x[(i, 0)] - expected[(i, 0)]).abs() < 1.0e-12);
			assert!((block[(i, 0)] - b[(i, 0)]).abs() < 1.0e-12);
			assert_eq!(storage[(i, 0)], 0.0);
		}
		assert_eq!(storage[(0, 1)], 0.0);
		Ok(())
	})
	.ok()
	.expect("the team is idle between calls")?;
	Ok(())
}
