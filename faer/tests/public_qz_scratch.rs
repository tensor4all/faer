//! The public real-QZ workspace query has to cover the deflation window the
//! kernel is allowed to use, not only the unrounded `(n - 3) / 3`.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::gevd::{self, ComputeEigenvectors};
use faer::{Col, Mat, Par, ParShape};

extern "C" fn window(_: usize, _: usize) -> usize {
	32
}

fn pencil(n: usize) -> (Mat<f64>, Mat<f64>) {
	let a = Mat::<f64>::from_fn(n, n, |i, j| {
		if i <= j + 1 {
			((17 * i + 13 * j + 7) % 101) as f64 / 101.0 - 0.5
		} else {
			0.0
		}
	});
	let b = Mat::<f64>::from_fn(n, n, |i, j| {
		if i == j {
			2.0
		} else if i < j {
			0.01 * ((7 * i + 11 * j + 3) % 97) as f64 / 97.0
		} else {
			0.0
		}
	});
	(a, b)
}

#[test]
fn scratch_query_covers_the_requested_deflation_window() {
	// n = 96 is the size from the report: the query took `(96 - 3) / 3 = 31` elements per
	// window while the kernel clamped to the even-rounded 32. n = 95 is the same defect and
	// passes, but it costs about as much as n = 96 on its own (3.2 s here, several minutes
	// under llvm-cov), so the sweep keeps 96 and the cheap neighbours and leaves 95 out.
	for n in [31, 32, 96, 97] {
		let (mut a, mut b) = pencil(n);
		let mut q = Mat::<f64>::identity(n, n);
		let mut z = Mat::<f64>::identity(n, n);
		let mut ar = Col::<f64>::zeros(n);
		let mut ai = Col::<f64>::zeros(n);
		let mut beta = Col::<f64>::zeros(n);
		let mut params: gevd::GeneralizedSchurParams = faer::auto!(f64);
		params.recommended_deflation_window = window;

		let req = gevd::qz_real::hessenberg_to_qz_scratch::<f64>(
			n,
			ParShape::Seq,
			params,
		);
		let mut mem = MemBuffer::new(req);
		gevd::qz_real::hessenberg_to_qz(
			a.as_mut(),
			b.as_mut(),
			Some(q.as_mut()),
			Some(z.as_mut()),
			ar.as_mut(),
			ai.as_mut(),
			beta.as_mut(),
			ComputeEigenvectors::No,
			Par::Seq,
			params,
			MemStack::new(&mut mem),
		);

		assert!(
			(0..n).all(|i| beta[i].is_finite()),
			"beta is not finite at n={n}"
		);
		assert!(
			(0..n).all(|i| ar[i].is_finite() && ai[i].is_finite()),
			"eigenvalues are not finite at n={n}",
		);
	}
}
