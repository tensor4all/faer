use crate::internal_prelude::*;
use crate::matrix_free::LinOp;

/// computes the `n_eigval` eigenvalues of largest magnitude of the self-adjoint
/// operator `A`, when the projection space is the whole space (`max_dim >= n`).
///
/// the operator is materialised one column at a time into the workspace, and the
/// resulting dense matrix is passed to the self-adjoint eigendecomposition. this
/// branch costs O(n^2) memory and O(n^3) time, and it is meant for the small
/// problems whose projection space covers the whole space. the iterative branch
/// keeps the matrix-free path for larger problems.
///
/// the outputs are written only after the decomposition has succeeded, so a
/// failure leaves the caller's buffers untouched and is reported as a zero
/// count. a non-finite operator or `n_eigval == 0` also produce a zero count.
///
/// the pairs are ordered by descending eigenvalue magnitude, ties broken by
/// ascending index, so the first `k` pairs do not depend on `k`.
pub fn partial_full_spectrum<T: ComplexField>(
	eigvecs: MatMut<'_, T>,
	eigvals: &mut [T],
	A: &dyn LinOp<T>,
	n_eigval: usize,
	mut par: Par,
	stack: &mut MemStack,
) -> usize {
	let n = A.nrows();
	let k = Ord::min(n_eigval, n);
	if k == 0 {
		return 0;
	}
	let (mut dense, stack) = temp_mat_zeroed::<T, _, _>(n, n, stack);
	let mut dense = dense.as_mat_mut();
	let (mut e, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
	let mut e = e.as_mat_mut().col_mut(0);
	for j in 0..n {
		e.fill(zero());
		e[j] = one();
		A.apply(
			dense.rb_mut().col_mut(j).as_mat_mut(),
			e.rb().as_mat(),
			par.reborrow(),
			stack,
		);
	}
	let mut finite = true;
	for j in 0..n {
		for i in 0..n {
			let x = &dense[(i, j)];
			finite = finite && is_finite(x);
		}
	}
	if !finite {
		return 0;
	}
	let (mut u, stack) = temp_mat_zeroed::<T, _, _>(n, n, stack);
	let mut u = u.as_mat_mut();
	let (mut s, stack) = stack.make_with(n, |_| zero::<T>());
	let ok = linalg::evd::self_adjoint_evd(
		dense.rb(),
		DiagMut::from_slice_mut(&mut *s),
		Some(u.rb_mut()),
		par.reborrow(),
		stack,
		default(),
	)
	.is_ok();
	if !ok {
		return 0;
	}
	let (mut perm, stack) = stack.make_with(n, |j| j);
	let perm = &mut *perm;
	perm.sort_unstable_by(|&i, &j| {
		let (norm_i, norm_j) = (s[i].abs(), s[j].abs());
		if norm_i > norm_j {
			core::cmp::Ordering::Less
		} else if norm_i < norm_j {
			core::cmp::Ordering::Greater
		} else {
			i.cmp(&j)
		}
	});
	let mut eigvecs = eigvecs;
	for idx in 0..k {
		let j = perm[idx];
		eigvecs.rb_mut().col_mut(idx).copy_from(u.rb().col(j));
		eigvals[idx] = copy(&s[j]);
	}
	let _ = stack;
	k
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::matrix_free::eigen::{
		PartialEigenParams, partial_self_adjoint_eigen, partial_self_adjoint_eigen_scratch,
	};
	use crate::prelude::*;
	use dyn_stack::MemBuffer;

	/// runs the public driver and checks the residual and the orthonormality of
	/// the returned pairs. the eigenvectors of a degenerate spectrum are not
	/// unique, so only the residual and the orthogonality are asserted.
	fn check<T: ComplexField>(A: MatRef<'_, T>, k: usize, params: PartialEigenParams) -> Vec<T> {
		let n = A.nrows();
		let mut V = Mat::<T>::zeros(n, k);
		let mut w = vec![zero::<T>(); k];
		let v0 = Col::<T>::from_fn(n, |i| one::<T>() + from_f64::<T>(i as f64));
		let mut mem = MemBuffer::new(partial_self_adjoint_eigen_scratch(
			&A,
			k,
			Par::Seq.shape(),
			params,
		));
		let info = partial_self_adjoint_eigen(
			V.as_mut(),
			&mut w,
			&A,
			v0.as_ref(),
			from_f64::<T::Real>(1e-13),
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		);
		core::assert!(info.n_converged_eigen == k);
		let tol = from_f64::<T::Real>(1e-12) * A.norm_l2();
		for j in 0..k {
			let residual = (A * V.col(j) - Scale(copy(&w[j])) * V.col(j)).norm_l2();
			core::assert!(residual < tol);
		}
		let one_tol = from_f64::<T::Real>(1e-12);
		for i in 0..k {
			for j in 0..k {
				let g: T = V.col(i).adjoint() * V.col(j);
				let expected: T = if i == j { one() } else { zero() };
				let d = g - expected;
				core::assert!(real(&d).abs() < one_tol && imag(&d).abs() < one_tol);
			}
		}
		w
	}

	// the MWE of #204: 4x4 identity, default parameters, one eigenpair
	#[test]
	fn mwe_identity_default_params() {
		let n = 4;
		let A = Mat::<f64>::identity(n, n);
		let w = check(A.as_ref(), 1, default());
		core::assert!((w[0] - 1.0).abs() < 1e-12);
	}

	// a degenerate spectrum: only the residual and the orthogonality are checked
	#[test]
	fn degenerate_spectrum() {
		let n = 6;
		let A = Mat::<f64>::identity(n, n);
		let _ = check(A.as_ref(), 1, default());
		let _ = check(A.as_ref(), n - 1, default());
		let _ = check(A.as_ref(), n, default());
	}

	#[test]
	fn diagonal_small_n() {
		let diag = [4.0f64, -1.0, 3.0, 2.0];
		let n = diag.len();
		let A = Mat::<f64>::from_fn(n, n, |i, j| if i == j { diag[i] } else { 0.0 });
		for k in 1..=n {
			let w = check(A.as_ref(), k, default());
			// descending magnitude
			core::assert!((w[0] - 4.0).abs() < 1e-12);
			if k >= 2 {
				core::assert!((w[1] - 3.0).abs() < 1e-12);
			}
			if k >= 3 {
				core::assert!((w[2] - 2.0).abs() < 1e-12);
			}
			if k >= 4 {
				core::assert!((w[3] + 1.0).abs() < 1e-12);
			}
		}
	}

	// the leading pairs must not depend on k
	#[test]
	fn prefix_stable_in_k() {
		let diag = [4.0f64, -1.0, 3.0, 2.0];
		let n = diag.len();
		let A = Mat::<f64>::from_fn(n, n, |i, j| if i == j { diag[i] } else { 0.0 });
		let full = check(A.as_ref(), n, default());
		for k in 1..n {
			let partial = check(A.as_ref(), k, default());
			for j in 0..k {
				core::assert!((partial[j] - full[j]).abs() < 1e-12);
			}
		}
	}

	#[test]
	fn n_equals_one() {
		let A = Mat::<f64>::from_fn(1, 1, |_, _| 2.5);
		let w = check(A.as_ref(), 1, default());
		core::assert!((w[0] - 2.5).abs() < 1e-12);
	}

	// an explicit max_dim >= n must take the same branch without panicking
	#[test]
	fn explicit_full_projection() {
		let n = 3;
		let A = Mat::<f64>::from_fn(n, n, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
		let params = PartialEigenParams {
			max_dim: 8,
			..default()
		};
		let w = check(A.as_ref(), n, params);
		core::assert!((w[0] - 3.0).abs() < 1e-12);
	}

	#[test]
	fn complex_hermitian() {
		let n = 4;
		let A = Mat::<c64>::from_fn(n, n, |i, j| {
			if i == j {
				c64::new((i + 1) as f64, 0.0)
			} else if i == j + 1 {
				c64::new(0.5, 0.25)
			} else if j == i + 1 {
				c64::new(0.5, -0.25)
			} else {
				c64::ZERO
			}
		});
		let _ = check(A.as_ref(), 1, default());
		let _ = check(A.as_ref(), n - 1, default());
		let _ = check(A.as_ref(), n, default());
	}

	// a non-finite operator must be reported as a zero count, not as a panic, and
	// must leave the outputs untouched
	#[test]
	fn non_finite_operator_is_reported() {
		let n = 2;
		let A = Mat::<f64>::from_fn(n, n, |_, _| f64::NAN);
		let mut V = Mat::<f64>::from_fn(n, 1, |_, _| 7.0);
		let mut w = [7.0f64];
		let v0 = Col::<f64>::from_fn(n, |i| (i + 1) as f64);
		let params: PartialEigenParams = default();
		let mut mem = MemBuffer::new(partial_self_adjoint_eigen_scratch(
			&A,
			1,
			Par::Seq.shape(),
			params,
		));
		let info = partial_self_adjoint_eigen(
			V.as_mut(),
			&mut w,
			&A,
			v0.as_ref(),
			1e-12,
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		);
		core::assert!(info.n_converged_eigen == 0);
		core::assert!(V[(0, 0)] == 7.0);
		core::assert!(w[0] == 7.0);
	}
}
