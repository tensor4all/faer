use crate::internal_prelude::*;
use crate::matrix_free::BiLinOp;

/// computes the `n_sval` singular triplets of largest singular value of `A`,
/// when the projection space is the whole space (`max_dim >= min(m, n)`).
///
/// the operator is materialised one column at a time into the workspace, and the
/// resulting dense matrix is passed to the dense SVD. this branch costs O(m n)
/// memory and O(m n min(m, n)) time, and it is meant for the small problems whose
/// projection space covers the whole space. the iterative branch keeps the
/// matrix-free path for larger problems.
///
/// the outputs are written only after the decomposition has succeeded, so a
/// failure leaves the caller's buffers untouched and is reported as a zero
/// count. a non-finite operator or `n_sval == 0` also produce a zero count.
///
/// the triplets are ordered by descending singular value, ties broken by
/// ascending index, so the first `k` triplets do not depend on `k`.
pub fn partial_full_svd<T: ComplexField>(
	left_singular_vecs: MatMut<'_, T>,
	right_singular_vecs: MatMut<'_, T>,
	singular_vals: &mut [T],
	A: &dyn BiLinOp<T>,
	n_sval: usize,
	mut par: Par,
	stack: &mut MemStack,
) -> usize {
	let m = A.nrows();
	let n = A.ncols();
	let size = Ord::min(m, n);
	let k = Ord::min(n_sval, size);
	let k = Ord::min(k, singular_vals.len());
	let k = Ord::min(
		k,
		Ord::min(left_singular_vecs.ncols(), right_singular_vecs.ncols()),
	);
	if k == 0 {
		return 0;
	}
	let (mut dense, stack) = temp_mat_zeroed::<T, _, _>(m, n, stack);
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
		for i in 0..m {
			let x = &dense[(i, j)];
			finite = finite && is_finite(x);
		}
	}
	if !finite {
		return 0;
	}
	let (mut u, stack) = temp_mat_zeroed::<T, _, _>(m, size, stack);
	let mut u = u.as_mat_mut();
	let (mut v, stack) = temp_mat_zeroed::<T, _, _>(n, size, stack);
	let mut v = v.as_mat_mut();
	let (mut s, stack) = stack.make_with(size, |_| zero::<T>());
	let ok = linalg::svd::svd(
		dense.rb(),
		DiagMut::from_slice_mut(&mut *s),
		Some(u.rb_mut()),
		Some(v.rb_mut()),
		par.reborrow(),
		stack,
		default(),
	)
	.is_ok();
	if !ok {
		return 0;
	}
	let (mut perm, stack) = stack.make_with(size, |j| j);
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
	let mut left_singular_vecs = left_singular_vecs;
	let mut right_singular_vecs = right_singular_vecs;
	for idx in 0..k {
		let j = perm[idx];
		left_singular_vecs
			.rb_mut()
			.col_mut(idx)
			.copy_from(u.rb().col(j));
		right_singular_vecs
			.rb_mut()
			.col_mut(idx)
			.copy_from(v.rb().col(j));
		singular_vals[idx] = copy(&s[j]);
	}
	let _ = stack;
	k
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::matrix_free::eigen::{
		PartialEigenParams, partial_svd, partial_svd_scratch,
	};
	use crate::prelude::*;
	use dyn_stack::MemBuffer;

	/// runs the public driver and checks both residuals and the orthonormality
	/// of the returned singular vectors. the singular vectors of a degenerate
	/// spectrum are not unique, so only the residuals and the orthogonality are
	/// asserted.
	fn check<T: ComplexField>(
		A: MatRef<'_, T>,
		k: usize,
		params: PartialEigenParams,
	) -> Vec<T> {
		let (m, n) = A.shape();
		let mut U = Mat::<T>::zeros(m, k);
		let mut V = Mat::<T>::zeros(n, k);
		let mut s = vec![zero::<T>(); k];
		let v0 = Col::<T>::from_fn(n, |i| one::<T>() + from_f64::<T>(i as f64));
		let mut mem = MemBuffer::new(partial_svd_scratch(
			&A,
			k,
			Par::Seq.shape(),
			params,
		));
		let info = partial_svd(
			U.as_mut(),
			V.as_mut(),
			&mut s,
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
			let left_residual =
				(A * V.col(j) - Scale(copy(&s[j])) * U.col(j)).norm_l2();
			core::assert!(left_residual < tol);
			let right_residual = (A.adjoint() * U.col(j)
				- Scale(copy(&s[j])) * V.col(j))
			.norm_l2();
			core::assert!(right_residual < tol);
		}
		let one_tol = from_f64::<T::Real>(1e-12);
		for x in [U.rb(), V.rb()] {
			for i in 0..k {
				for j in 0..k {
					let g: T = x.col(i).adjoint() * x.col(j);
					let expected: T = if i == j { one() } else { zero() };
					let d = g - expected;
					core::assert!(
						real(&d).abs() < one_tol && imag(&d).abs() < one_tol
					);
				}
			}
		}
		s
	}

	// the MWE of #213: 4x4 identity, default parameters, one singular triplet
	#[test]
	fn mwe_identity_default_params() {
		let n = 4;
		let A = Mat::<f64>::identity(n, n);
		let s = check(A.as_ref(), 1, default());
		core::assert!((s[0] - 1.0).abs() < 1e-12);
	}

	// a degenerate spectrum: only the residuals and the orthogonality are checked
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
		let A = Mat::<f64>::from_fn(
			n,
			n,
			|i, j| if i == j { diag[i] } else { 0.0 },
		);
		for k in 1..=n {
			let s = check(A.as_ref(), k, default());
			// descending magnitude
			core::assert!((s[0] - 4.0).abs() < 1e-12);
			if k >= 2 {
				core::assert!((s[1] - 3.0).abs() < 1e-12);
			}
			if k >= 3 {
				core::assert!((s[2] - 2.0).abs() < 1e-12);
			}
			if k >= 4 {
				core::assert!((s[3] - 1.0).abs() < 1e-12);
			}
		}
	}

	// the leading triplets must not depend on k
	#[test]
	fn prefix_stable_in_k() {
		let diag = [4.0f64, -1.0, 3.0, 2.0];
		let n = diag.len();
		let A = Mat::<f64>::from_fn(
			n,
			n,
			|i, j| if i == j { diag[i] } else { 0.0 },
		);
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
		let s = check(A.as_ref(), 1, default());
		core::assert!((s[0] - 2.5).abs() < 1e-12);
	}

	#[test]
	fn rectangular() {
		// tall and wide, both with min(m, n) <= 2 * MIN_DIM
		let tall = Mat::<f64>::from_fn(6, 3, |i, j| ((i + 1) * (j + 2)) as f64);
		let _ = check(tall.as_ref(), 1, default());
		let _ = check(tall.as_ref(), 3, default());
		let wide = tall.adjoint();
		let _ = check(wide, 1, default());
		let _ = check(wide, 3, default());
	}

	// an explicit max_dim >= min(m, n) must take the same branch without panicking
	#[test]
	fn explicit_full_projection() {
		let A = Mat::<f64>::from_fn(3, 3, |i, j| {
			if i == j { (i + 1) as f64 } else { 0.0 }
		});
		let params = PartialEigenParams {
			max_dim: 8,
			..default()
		};
		let s = check(A.as_ref(), 3, params);
		core::assert!((s[0] - 3.0).abs() < 1e-12);
	}

	#[test]
	fn complex_input() {
		let A = Mat::<c64>::from_fn(4, 3, |i, j| {
			c64::new((i + j) as f64, (i as f64) - 0.5 * (j as f64))
		});
		let _ = check(A.as_ref(), 1, default());
		let _ = check(A.as_ref(), 3, default());
	}

	// a non-finite operator must be reported as a zero count, not as a panic, and
	// must leave the outputs untouched
	#[test]
	fn non_finite_operator_is_reported() {
		let (m, n) = (2, 2);
		let A = Mat::<f64>::from_fn(m, n, |_, _| f64::NAN);
		let mut U = Mat::<f64>::from_fn(m, 1, |_, _| 7.0);
		let mut V = Mat::<f64>::from_fn(n, 1, |_, _| 7.0);
		let mut s = [7.0f64];
		let v0 = Col::<f64>::from_fn(n, |i| (i + 1) as f64);
		let params: PartialEigenParams = default();
		let mut mem = MemBuffer::new(partial_svd_scratch(
			&A,
			1,
			Par::Seq.shape(),
			params,
		));
		let info = partial_svd(
			U.as_mut(),
			V.as_mut(),
			&mut s,
			&A,
			v0.as_ref(),
			1e-12,
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		);
		core::assert!(info.n_converged_eigen == 0);
		core::assert!(U[(0, 0)] == 7.0);
		core::assert!(V[(0, 0)] == 7.0);
		core::assert!(s[0] == 7.0);
	}

	// the boundary of the full-projection clip: at `min(m, n) == 2 * MIN_DIM`
	// the driver clips `max_dim` to `min(m, n)` and takes the dense branch; one
	// larger keeps `max_dim` below it and takes the iterative branch (#229)
	#[test]
	fn projection_clip_boundary() {
		// `MIN_DIM` is private to `operator::eigen` and is 32, so the boundary
		// is 64 here
		let n = 64;
		let A = Mat::<f64>::identity(n, n);
		let s = check(A.as_ref(), 1, default());
		core::assert!((s[0] - 1.0).abs() < 1e-12);

		// one larger: `max_dim` stays at 64 and the iterative branch runs
		let n = 65;
		let A = Mat::<f64>::from_fn(n, n, |i, j| {
			if i == j {
				(n - i) as f64
			} else {
				0.0
			}
		});
		let s = check(A.as_ref(), 1, default());
		core::assert!((s[0] - n as f64).abs() < 1e-9);
	}

	// an empty operator has `min(m, n) == 0`: the driver must return a zero
	// count before the clipped `max_dim` can violate the projection assert, and
	// must leave the outputs untouched (#229)
	#[test]
	fn empty_shape_is_reported() {
		let A = Mat::<f64>::zeros(0, 0);
		let mut U = Mat::<f64>::zeros(0, 1);
		let mut V = Mat::<f64>::zeros(0, 1);
		let mut s = [7.0f64];
		let v0 = Col::<f64>::zeros(0);
		let params: PartialEigenParams = default();
		let mut mem = MemBuffer::new(partial_svd_scratch(&A, 1, Par::Seq.shape(), params));
		let info = partial_svd(
			U.as_mut(),
			V.as_mut(),
			&mut s,
			&A,
			v0.as_ref(),
			1e-12,
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		);
		core::assert!(info.n_converged_eigen == 0);
		core::assert!(s[0] == 7.0);
	}
}
