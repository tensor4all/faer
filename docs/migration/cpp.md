# From Eigen and BLAS

This page translates common C++ dense matrix operations to this fork's Rust API.

| Eigen or BLAS operation | faer counterpart |
| --- | --- |
| `Eigen::MatrixXd` | `Mat<f64>` owns a dense matrix; `MatRef` and `MatMut` borrow it. |
| `a.ldlt()` | `a.ldlt(Side::Lower, par)` computes an **unpivoted** LDL factorization; see the pivoting distinction below. |
| `a.partialPivLu()` | `a.partial_piv_lu(par)` returns row pivoted LU; `Solve::solve(&b, par)` solves a system. |
| `a.selfadjointView<Eigen::Lower>()` | Pass `Side::Lower` to a self adjoint factorization, or `Side::Upper` for the upper triangle. There is no dedicated view wrapper. |
| `a.block(i, j, rows, cols)` | `a.as_ref().submatrix(i, j, rows, cols)`; use `as_mut().submatrix_mut(...)` for mutation. |
| BLAS GEMM | `linalg::matmul::matmul(dst, accum, lhs, rhs, alpha, par)` computes `dst = alpha * lhs * rhs` or adds that product. |

## Storage and factors

`Mat<f64>` is a growable owned column major matrix. Index with `a[(i, j)]`,
starting at zero. `MatRef<'_, f64>` is a shared view and `MatMut<'_, f64>` is an
exclusive mutable view. A block borrows storage rather than copying it;
`to_owned()` makes an independent matrix. Views can have strides, so a block is
not necessarily contiguous. Rust prevents resizing or conflicting mutation of
the owner while a borrowed view is in use.

Unlike an Eigen expression template, a view is not a deferred expression tree.
Use `transpose()` or `adjoint()` views for GEMM operands. In particular,
`adjoint()` conjugates complex entries, while `transpose()` does not.
`Accum::Replace` replaces the destination and `Accum::Add` adds to it.
For general BLAS `beta`, scale the destination yourself first and then add;
there is no arbitrary `beta` scalar argument to this `matmul` API.

The owning factorization methods copy their input; they do not overwrite `a`.
Partial pivoted LU uses `P A = L U`, with factors available through `P()`, `L()`
and `U()`. Solve with the factor object instead of manually applying pivots.

For real symmetric matrices, `ldlt` gives `A = L D L.transpose()` with unit
lower triangular `L` and diagonal `D`. For complex Hermitian matrices the
transpose in that identity is an adjoint. This is mathematically the same
factorization family as Eigen's LDLT, not the same pivoting algorithm:
Eigen's LDLT uses pivoting, while this fork's `ldlt` has no permutation.
A matrix that needs a pivot can therefore fail here even if Eigen succeeds.
For an indefinite problem that needs symmetric pivoting, inspect `lblt`,
which uses Bunch Kaufman pivoting and block diagonal factors. It is not a
promise of the same factors or decisions as Eigen's LDLT.

`Side` tells the factorization which triangle contains the input. The other
triangle is ignored, not checked for symmetry. It does not create a general
Eigen style self adjoint expression for subsequent operations.

## The caller owns parallel execution

The important change is not the spelling of the factorization. Factorization
and solve calls in this fork take a trailing `Par`. `Par::Seq` executes on the
calling thread, with no hidden dispatch to a worker pool. This is an execution
contract, not a claim of performance parity with Eigen or a BLAS library.
Measurements are published at <https://tensor4all.org/faer-benchmark/>.

`Par::Team` is non-Copy and non-Clone exclusive authority to use a worker team.
`with_team(2, |mut par| ...)` scopes that ownership to the closure. Repeated
operations lend the authority with `par.reborrow()` instead of copying it or
acquiring a separate team for each call. The borrow ends when each call returns,
so the next operation can borrow the same authority. The matrix and factor
owners remain ordinary Rust values; holding a matrix does not grant worker
access.

If work is split into nested children, each child must use its child token,
not a captured parent token. This keeps the parent authority unavailable while
its children own the subdivisions. See the verified compile-fail example in
[parallel regions](README.md#parallel-regions).

These signatures belong to the tensor4all fork, not upstream faer. There is no
`get_global_parallelism`, `set_global_parallelism`, or `Par::Rayon` here.
Upstream examples are not a source for these call signatures. The detailed
history is in [the Par migration guide](../par-team-migration.md).

## A solve, a borrowed block, and GEMM

The symmetric matrix below is positive definite: its leading principal minors
are 4 and 11. Both LU and unpivoted LDLT should recover the known solution.
The last product checks a mutable block without changing the surrounding cells.
The source is [`faer/examples/migration_cpp.rs`](../../faer/examples/migration_cpp.rs).
Run it with `cargo run -p t4a-faer --example migration_cpp -j8`.

```rust
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

	faer::team::with_team(2, |mut par| -> Result<(), LdltError> {
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
	})?;
	Ok(())
}
```

## Open gaps

There is no dedicated `selfadjointView` wrapper, and `ldlt` is not a direct
replacement for Eigen's pivoted LDLT algorithm. General BLAS GEMM `beta`
requires an explicit destination scaling step. The mapped block and partial
pivoted LU operations have direct counterparts, subject to Rust borrowing and
the explicit `Par` argument.
