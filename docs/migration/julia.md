# From Julia's LinearAlgebra

This page maps dense `LinearAlgebra.jl` workflows to this fork's Rust API.

| Julia operation | faer counterpart |
| --- | --- |
| `lu(A)` | `a.partial_piv_lu(par)`, with `P A = L U`; access `P()`, `L()`, `U()`. |
| `cholesky(Hermitian(A, :L))` | `a.llt(Side::Lower, par)`, returning `Result`; `L()` gives the lower factor. |
| `qr(A)` | `a.qr(par)` is unpivoted; `compute_Q(par)` materializes Q and `R()` borrows R. |
| `svd(A)` | `a.thin_svd(par)` gives thin factors; `svd(par)` gives full factors. Both return `Result`. |
| `dot(x, y)` | `linalg::matmul::dot::inner_prod(x.transpose(), Conj::Yes, y, Conj::No)` for column views. The first operand is conjugated. |
| `size(A)`, `A[i, j]` | `a.nrows()`, `a.ncols()`, `a[(i, j)]`; indices start at zero. |
| `view(A, rows, cols)` | Borrow with `as_ref().submatrix(...)` or `as_mut().submatrix_mut(...)`. |

## Shapes, storage, and factor conventions

An owned `Mat<f64>` is column major, like a Julia `Matrix{Float64}`. Use `c64`
for complex double precision. Julia indexing starts at one; faer starts at zero.
A column is a `Col` or `ColRef`, not a matrix with an interchangeable vector
shape. `mat![[a, b], [c, d]]` writes rows even though storage is column major.

Julia slicing normally copies, while `view` aliases the parent. In faer,
`MatRef` and `MatMut` borrow the parent and `to_owned()` explicitly copies.
A mutable view is exclusive: Rust rejects another overlapping borrow while it
is in use. Keep the owned matrix alive while its views are used. Strided blocks
do not imply contiguous storage. The high level factorization methods create
owned factors without modifying their input, unlike Julia's `lu!` and similar
mutating routines.

For LU, Julia's `A[F.p, :] = F.L * F.U` corresponds to faer's `P A = L U`.
Do not copy permutation indices without accounting for indexing and direction;
use `Solve::solve` on the factor object to solve systems. Cholesky returns the
lower factor `L`, so `A = L L.adjoint()` for Hermitian input. `Side::Lower` or
`Side::Upper` selects the triangle to read, rather than creating a `Hermitian`
wrapper. The unused triangle is ignored, not validated.

Unpivoted QR uses `A = Q R`. Q is stored as Householder data, not as a dense
matrix: call `compute_Q(par)` for a full Q, or `compute_thin_Q(par)` together
with `thin_R()` for thin factors. For column pivoting, use `col_piv_qr(par)`
rather than assuming `qr(par)` chooses pivots.

SVD uses `A = U S V.adjoint()`. `S()` is a diagonal view, not Julia's vector
of singular values, and `V()` is V, not Julia's `Vt`. Singular values are
nonnegative and sorted in descending order. For an input with `m` rows and `n` columns,
write `k = min(m, n)`. Full `svd` returns U with shape `m` by `m` and V with shape `n` by `n`.
`thin_svd` returns U with shape `m` by `k`, V with shape `n` by `k`, and `k` singular values,
matching Julia's usual thin factor shapes. Signs or complex phases of singular vectors
need not match between implementations.

For complex `dot`, conjugating the first input is essential. A plain transpose
alone does not implement Julia's `dot`. The scalar `inner_prod` primitive has
no `Par` argument and runs locally; its explicit `Conj` arguments determine
conjugation. `adjoint()` is the conjugating matrix view elsewhere in the API.

## The caller owns parallel execution

This fork makes execution authority explicit, rather than consulting a global
BLAS thread setting. High level factorizations and solves take a trailing
`Par`. `Par::Seq` runs on the calling thread without hidden worker dispatch.
It is an execution contract, not a performance parity claim against Julia or
its BLAS. See <https://tensor4all.org/faer-benchmark/> for measurements.

`Par::Team` is non-Copy and non-Clone exclusive authority to use a worker team.
`with_team(2, |mut par| ...)` scopes that authority to its closure. Keep the
region around a sequence of operations and lend its token using
`par.reborrow()`. Each completed call releases the borrow, allowing the next
factorization or solve to use the same authority. A factor object owns its
numerical data, but it does not own a team or silently choose parallelism.
The separate choice of `Par::Seq` needs no team region at all.

A nested child must receive and use the child token, not capture the parent
token. While child authority is active the parent is unavailable. This prevents
two concurrent operations from spending the same authority. See the verified
compile-fail example under [parallel regions](README.md#parallel-regions).

This is the tensor4all fork, not upstream faer: these trailing `Par` signatures
are fork APIs. There is no `get_global_parallelism`, `set_global_parallelism`,
or `Par::Rayon`. Do not transplant upstream documentation's signatures.
See [the Par migration guide](../par-team-migration.md) for detailed API history.

## Reuse one region for several factorizations

The source is [`faer/examples/migration_julia.rs`](../../faer/examples/migration_julia.rs).
Run it with `cargo run -p t4a-faer --example migration_julia -j8`.
It checks a known solution, rectangular QR and SVD reconstruction, and complex dot conjugation.
The Cholesky input has positive leading principal minors 4 and 11; the example asserts that
known success condition and propagates SVD nonconvergence as its original typed error.

```rust
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
```

## Open gaps

The mapped numerical operations have counterparts, but there is no Julia style
`Hermitian` wrapper or automatic one based indexing translation. Julia's
mutating factorization functions do not map to the owning methods shown here;
use the separate low level in place APIs if preserving that storage behavior
is necessary. Julia factor field names and shapes are not interchangeable
with faer's factor accessors, especially SVD's diagonal `S()` and `V()`.
