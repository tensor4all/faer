# From Fortran LAPACK and BLAS to t4a-faer

The Rust package is `t4a-faer`, imported as `faer`. These are mathematical mappings, not LAPACK ABI replacements.

| LAPACK or BLAS construct | t4a-faer equivalent |
| --- | --- |
| `DGETRF(M, N, A, LDA, IPIV, INFO)` | `a.partial_piv_lu(par)`, or `linalg::lu::partial_pivoting::factor::lu_in_place` for packed factors and caller scratch |
| `DGETRS('N', N, NRHS, A, LDA, IPIV, B, LDB, INFO)` | Import `linalg::solvers::Solve`; use `lu.solve(&b, par)` or the low level `solve::solve_in_place` with factors and a `PermRef` |
| `DGETRS('T', ...)` | `lu.solve_transpose(&b, par)`; the low level module also provides `solve_transpose_in_place` |
| `DSYEVD('V', 'L', ...)` | `a.self_adjoint_eigen(Side::Lower, par)?`; `U()` holds eigenvectors and `S().column_vector()` holds ascending eigenvalues |
| `DSYEVD('N', 'U', ...)` | `a.self_adjoint_eigenvalues(Side::Upper, par)?` |
| `DGEMM('N', 'N', ..., ALPHA, A, ..., B, ..., BETA, C, ...)` | `linalg::matmul::matmul(&mut c, Accum::Replace, &a, &b, alpha, par)` for `beta = 0`; `Accum::Add` for `beta = 1` |
| `DGEMM` transpose flags | Pass `a.transpose()` or `b.transpose()`; for complex conjugate transpose use `adjoint()` |
| `A(LDA, N)` with padding between columns | `MatRef::from_column_major_slice_with_stride(&data, m, n, lda)`; use the matching `MatMut` constructor to write |
| `WORK`, `LWORK`, and a workspace query with `LWORK = -1` | Call the matching `*_scratch` function, allocate `dyn_stack::MemBuffer`, and lend it through `MemStack` |
| `IPIV` with indices starting at one | A faer permutation and inverse with indices starting at zero, not LAPACK's pivot swap history |

## Parallel regions belong to the caller

A BLAS thread count setting does not select faer's execution resources.
Create a region with `faer::team::with_team(width, |mut par| ...)`, then lend its token
with `par.reborrow()` to factorization, solve, and product calls in succession.
The width includes the calling thread, and dropping the scoped team returns its reservation.
For a repeated workload, hold the pool and team outside the loop instead of acquiring them per call.

`Par::Team` is exclusive authority over that reservation, not an integer thread count.
It cannot be copied or cloned. Concurrent nested work must split the reservation and use the
child token it receives. It must not capture and reuse the parent token.
The [parallel regions guide](README.md#parallel-regions) has a working split and a type checked
example of the forbidden parent reuse. A second `with_team` call would acquire an independent
team rather than borrowing the existing budget.

Use `Par::Seq` when the call must stay on the calling thread. It bypasses team acquisition
and worker dispatch; it does not enter a global pool at width one.
Upstream faer exposed global parallelism settings and `Par::Rayon`; this fork removes them
and requires trailing `Par` arguments on factorization and solver methods.
Use this fork's [API reference](https://docs.rs/t4a-faer), not upstream signatures, and the
[API migration history](../par-team-migration.md).
Published timing results and conditions live on the [benchmark site](https://tensor4all.org/faer-benchmark/).

## Storage, scratch, and error conventions

Fortran arrays use indices starting at one; faer uses indices starting at zero.
The element at Fortran `A(i,j)` has Rust offset `(i - 1) + (j - 1) * lda` in a column storage buffer.
The leading dimension is measured in elements, not bytes. A padded slice constructor borrows that
storage without repacking it. For nonempty LAPACK style matrices, use `lda >= m` and provide
storage through the last logical entry; mutable columns must not overlap.
A `Mat` owns its storage and may have padding too; query its
`col_stride()` rather than assuming it equals `nrows()`.

The owning factorization methods allocate their factors and working memory.
The low level routines below accept `MatMut` and scratch explicitly and overwrite the factors or right hand side.
Scratch queries return a `StackReq` describing size and alignment, not an array of `f64` values or an `lwork` integer.
Use the same dimensions, scalar type, parameters, and parallel shape for the query and call.
Sequential phases can share one buffer sized with `factor_req.or(solve_req)`; concurrent children need separate scratch.
Workspace queries take `par.shape()`, which describes resources but does not grant execution authority.

There is no universal `INFO` output. Eigenvalue computation returns a `Result` that must be handled.
The low level LU routine returns factorization information and a permutation, not a LAPACK singular pivot status.
Dimension checks can panic, and an LU solve requires a nonsingular factorization.
Do not treat successful LU factor construction as a singularity test, or reinterpret `IPIV` as a `PermRef`.

## Complete example: padded input, LU scratch, a product, and eigenvectors

The source is [`faer/examples/migration_fortran.rs`](../../faer/examples/migration_fortran.rs).
Run it with `cargo run -p t4a-faer --example migration_fortran -j8`.
It factors a matrix that requires a row interchange, solves a known system, and checks `A * x = b`.
The last calculation checks the eigenvalue equations for a symmetric matrix.

```rust
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::EvdError;
use faer::linalg::lu::partial_pivoting::{factor, solve};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Side, mat};

fn main() -> Result<(), EvdError> {
	// Two logical rows, two columns, lda = 3. Padding is not matrix data.
	let data = [0.0_f64, 3.0, -999.0, 2.0, 4.0, -999.0];
	let a = MatRef::from_column_major_slice_with_stride(&data, 2, 2, 3);
	let b = mat![[4.0], [11.0]];

	faer::team::with_team(2, |mut par| {
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
}
```

## Open gaps

- `t4a-faer` does not export the LAPACK or BLAS Fortran ABI. Translate calls and data ownership in Rust.
- There is no `LWORK = -1` protocol or shared `INFO` convention. Use scratch requests and each routine's return type.
- There is no direct `IPIV` interchange API in the factorization calls above. Retain faer's permutation pair when using its low level factor and solve routines.
- `Accum` accepts replace or add, not an arbitrary BLAS `beta`. For other values, scale `c` explicitly before adding the product.
- `self_adjoint_eigen` computes the mathematical result of the mapped `DSYEVD` call, not a promise to use LAPACK's driver or workspace layout.
