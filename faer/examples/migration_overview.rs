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

	faer::team::with_team(4, |mut par| {
		solve_checked(&a, &b, par.reborrow());
		solve_checked(&a, &b, par.reborrow());

		// The two children each own two participants from the same four.
		let Par::Team(team) = par else {
			unreachable!("with_team(4) supplies a team");
		};
		team.split(&[2, 2], |_, child| {
			solve_checked(&a, &b, Par::Team(child));
		})
		.expect("two children of size two fit the parent of size four");
	});
}
