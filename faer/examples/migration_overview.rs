use faer::linalg::solvers::Solve;
use faer::{Mat, Par, mat};

fn solve_checked(a: &Mat<f64>, b: &Mat<f64>, mut par: Par<'_>) {
	let lu = a.partial_piv_lu(par.reborrow());
	let x = lu.solve(b, par);
	assert!((x[(0, 0)] - 1.0).abs() < 1e-12);
	assert!((x[(1, 0)] - 2.0).abs() < 1e-12);
}

fn main() {
	let a = mat![[4.0, 1.0], [1.0, 3.0]];
	let b = mat![[6.0], [7.0]];
	solve_checked(&a, &b, Par::Seq);

	// The pool and the team are built once. Each call takes an execution token from the
	// team, lends it, and gets it back when the operation returns.
	let pool = threadteam::Pool::builder()
		.threads(4)
		.acquire_timeout(std::time::Duration::from_secs(30))
		.build()
		.expect("a thread pool for the team");
	let team = pool.into_team(4).ok().expect("a newly built pool is idle");

	for _ in 0..2 {
		let mut exec = team.exec(4);
		exec.team(|token| solve_checked(&a, &b, Par::Team(token)))
			.ok()
			.expect("the team is idle between calls");
	}

	// One call split into two children of two participants each.
	let mut exec = team.exec(4);
	exec.team(|token| {
		token
			.split(&[2, 2], |_, child| solve_checked(&a, &b, Par::Team(child)))
			.expect("two children of size two fit the parent of size four");
	})
	.ok()
	.expect("the team is idle between calls");
}
