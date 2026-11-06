# Migrating to t4a-faer for HPC dense linear algebra

`t4a-faer` provides dense matrix types, factorizations, solves, and matrix products in Rust.
Choose a guide by the library you already use. The tables map mathematical operations, not binary interfaces.

| Source ecosystem or construct | t4a-faer equivalent |
| --- | --- |
| [C++: Eigen and BLAS](cpp.md) | `Mat`, `MatRef`, `MatMut`, factorization objects, and `matmul` |
| [Fortran: LAPACK and BLAS](fortran.md) | Factor and solve routines with explicit scratch, or owning factorization objects |
| [Julia: LinearAlgebra.jl](julia.md) | `partial_piv_lu`, `llt`, `qr`, `svd`, and vector products |
| An owning matrix | `Mat<T>`; borrowed read and write views are `MatRef` and `MatMut` |
| A library or vendor thread count setting | A caller owned `threadteam` team, passed to calls as `Par::Team` |
| A call that must stay on the calling thread | `Par::Seq` |

## Dependency and upstream API differences

The package is named `t4a-faer`; its Rust library name is `faer`.
In a consuming project's `Cargo.toml`, use:

```toml
[dependencies]
faer = { package = "t4a-faer", version = "0.28" }
```

These examples use the default features, which include `std` and `threadteam`.
Without `threadteam`, there is no `Par::Team` or `faer::team` module.
The examples build a `threadteam::Pool` directly, so add `threadteam` alongside the faer
dependency, at the version faer itself resolves: `Par::Team` carries a `threadteam::TeamExec`, and
two threadteam minors are two distinct types.

Upstream faer exposed process wide parallelism controls backed by a global Rayon pool.
This fork removes `set_global_parallelism`, `get_global_parallelism`, and `Par::Rayon`.
Its factorization and solve methods take a trailing `Par`, including calls such as
`a.partial_piv_lu(par)` and `lu.solve(&b, par)`. Do not copy those signatures from upstream documentation.
The [API migration history](../par-team-migration.md) records these changes.
Import `faer::linalg::solvers::Solve` to call the solver trait methods.

## Parallel regions

The application owns the thread budget. Owning a team is the primary route: build a
`threadteam::Pool` once, reserve its members once with `pool.into_team(width)`, and take a fresh
execution token for each call with `team.exec(width)`. `Exec::team` runs the closure on that token,
and the operation it is lent to consumes it and gives the team back. That is the route for a sequence
of calls, and the one the [Rust migration guide](../migration-guide.md#explicit-parallelism)
measures. The width includes the calling thread. A width of one executes sequentially.
There is no implicit global pool selection inside a faer operation.
Threadteam may reuse cached workers, which is different from choosing a global execution team.
The [parallelism section](../../README.md#parallelism) shows the same construction.

`faer::team::with_team(width, |par| ...)` is the one-off route: it builds a pool, reserves its
members, runs one closure, and releases them. Use it for a single call, or for a script that runs a
single operation. The Rust guide gives its cost per call beside a team the caller holds.

`Par` is neither `Copy` nor `Clone`. A call consumes its token.
For successive operations in one region, lend it with `par.reborrow()`.
That borrow ends when the operation returns, so the next phase can reuse the same team.

Two pools do not share their members. A second `with_team` call, or a second pool, acquires its own
members instead of borrowing the budget already held.

Nested work must use the permission it receives. Split a team into disjoint child teams
before running concurrent children; each child passes its own `Par::Team` to faer.
Do not capture the parent permission inside a child while that parent is borrowed or consumed.
Nesting does not add a team: a call made from inside an operation that holds a team gets the token
passed down to it, and a token that carries no members, or a team narrowed to one participant, is a
serial token that runs inline on the calling thread. Do not acquire the same team recursively: `threadteam::Team::exec` panics when the calling
thread already holds a token of that team.
The type system protects a token's authority, not the total number of teams an application creates.

## Complete example: sequential, an owned team, and nested solves

The complete source is [`faer/examples/migration_overview.rs`](../../faer/examples/migration_overview.rs).
Run it with `cargo run -p t4a-faer --example migration_overview -j8` from the repository root.
The documentation check requires this code block to match that source.

```rust
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
```

The following is intentionally rejected and checked as a `compile_fail` doctest.
The child already borrows the parent, so moving the parent into the same nested call would
hand out overlapping authority. Pass only the child token, as in the example above.
This ownership rule applies to `Par::Seq` as well as `Par::Team`.

```compile_fail,E0505
use faer::Par;

fn nested_work(_child: Par<'_>, _parent: Par<'_>) {}

fn main() {
    let mut parent = Par::Seq;
    let child = parent.reborrow();
    nested_work(child, parent);
}
```

## Open gaps

These guides describe Rust APIs, not a source or binary compatibility layer for Eigen,
LAPACK, BLAS, or Julia. The per ecosystem pages list the differences that need explicit translation.
They do not establish performance parity with another library; consult the published benchmark results.
