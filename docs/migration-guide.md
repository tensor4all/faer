# Migrating Rust linear algebra to t4a faer

This guide targets users of upstream faer, nalgebra, and ndarray with ndarray-linalg. It describes the API in the repository at version 0.28.0. This fork can differ from upstream and incompatible releases can change its API.

## Why consider the fork?

The library implements dense and sparse linear algebra in Rust without requiring a BLAS installation. Parallel execution is explicit: the caller supplies `Par::Seq` or lends a `threadteam` team through `Par::Team`. This makes the thread budget visible at the call site. The [published measurements](#published-measurements) show both wins and cases where another implementation is faster; they are not a promise for every workload.

## Add the dependency

In `Cargo.toml`, use the package name `t4a-faer` and the Rust crate name `faer`:

```toml
[dependencies]
faer = { package = "t4a-faer", version = "0.28" }
```

The default features include dense and sparse linear algebra and `threadteam`. Rust 1.89 or newer is required. If you disable default features, enable the features needed by your application. Without `threadteam` only sequential execution is available. To manage a pool directly, also add a compatible `threadteam` 0.3 dependency. The fork and upstream are distinct Cargo packages, so their matrix and parallelism types are not interchangeable.

## Common operations

The table gives comparable operations, not a guarantee that their return types, error handling, layout, or decomposition conventions are identical. The t4a examples use `Par::Seq`; pass a borrowed team for parallel execution. Matrix multiplication operators are sequential.

| Task | nalgebra | ndarray and ndarray-linalg | t4a faer |
| :--- | :--- | :--- | :--- |
| Construct a dense matrix | `DMatrix::from_fn(rows, cols, f)` | `Array2::from_shape_fn((rows, cols), f)` | `Mat::from_fn(rows, cols, f)` |
| Multiply matrices | `&a * &b` | `a.dot(&b)` | `&a * &b` |
| Factor for a square solve | `a.lu()` | `a.solve_into(b)` | `a.partial_piv_lu(par)` then `lu.solve(&b, par)` |
| QR factorization | `a.qr()` | `a.qr()` via `QR` | `a.qr(par)` |
| Singular values and vectors | `a.svd(true, true)` | `a.svd(true, true)` via `SVD` | `a.svd(par)` |

Nalgebra constructors use a closure with two indices; ndarray uses a single tuple. The `Solve` trait must be in scope for the faer solve method. SVD returns a `Result` in t4a faer. Check the dimensions and vector conventions before replacing decomposition consumers: faer represents the SVD as `U S V^H`, while other APIs may return `V^H` directly. Prefer solving a system to explicitly computing its inverse.

This complete example constructs and solves a small system. The assertions check the numerical result:

```rust
use faer::{mat, Mat, Par};
use faer::linalg::solvers::Solve;

fn main() {
    let a = mat![[3.0_f64, 1.0], [1.0, 2.0]];
    let b = Mat::from_fn(2, 1, |i, _| [9.0, 8.0][i]);
    let lu = a.partial_piv_lu(Par::Seq);
    let x = lu.solve(&b, Par::Seq);
    assert!((x[(0, 0)] - 2.0).abs() < 1e-12);
    assert!((x[(1, 0)] - 3.0).abs() < 1e-12);
}
```

## From upstream faer

Keep imports such as `use faer::Mat`, but change the Cargo dependency. Replace global parallelism settings and implicit parallel calls with an explicit `Par` at each operation that takes one.

| Upstream pattern | t4a faer pattern |
| :--- | :--- |
| `a.svd()` | `a.svd(Par::Seq)` |
| `a.partial_piv_lu()` | `a.partial_piv_lu(Par::Seq)` |
| `lu.solve(&b)` | `lu.solve(&b, Par::Seq)` |
| Global parallelism setting | Pass `Par::Seq` or a team to each operation |

This fork has no `Par::rayon`, `Par::Rayon`, or global faer parallelism setter. Replace old Rayon width tokens with `faer::team::with_team` or a caller owned team. For scratch queries use `par.shape()`, not the execution token itself. See the current API in `faer/src/lib.rs` and the [changelog](../CHANGELOG.md) for the release history.

## Explicit parallelism

A scoped team is convenient for a sequence of calls. The requested width includes the calling thread. A width of one, or zero, selects sequential execution. The default `threadteam` feature is required.

```rust
use faer::{mat, Accum, Mat};
use faer::linalg::matmul::matmul;

fn main() {
    let a = mat![[1.0_f64, 2.0], [3.0, 4.0]];
    let mut product = Mat::<f64>::zeros(2, 2);
    let mut again = Mat::<f64>::zeros(2, 2);
    faer::team::with_team(4, |mut par| {
        matmul(product.as_mut(), Accum::Replace, &a, &a, 1.0, par.reborrow());
        matmul(again.as_mut(), Accum::Replace, &a, &a, 1.0, par.reborrow());
    });
    assert!((product[(0, 0)] - 7.0).abs() < 1e-12);
    assert!((again[(1, 1)] - 22.0).abs() < 1e-12);
}
```

`Par` is consumed by an operation and is neither `Copy` nor `Clone`. `reborrow()` lends its exclusive authority for one call; the parent cannot be used until that borrow ends. A team token is valid only in its scope and cannot escape it. Do not acquire or reuse the same team recursively inside an active team operation; pass the existing authority through the call chain instead. Independent nested pools can oversubscribe the machine. Internal algorithms may split a team into disjoint children; that is not permission to duplicate a parent token.

For repeated parallel calls, construct and retain a `threadteam::Pool` and `Team`, then obtain a fresh execution token for each scope as shown in the [README](../README.md#parallelism). `with_team` acquires a pool and team for each scope; threadteam may reuse cached workers. There is no implicit global faer execution pool or global faer thread setting, but this does not mean threadteam has no process level worker cache. Short or small operations can execute sequentially even with a team when parallel dispatch does not pay.

## Migration pitfalls

* **Storage and views:** `Mat` is owned, while matrix views borrow their storage. Do not assume an ndarray layout or a contiguous slice matches faer's owned layout. Copy by index or use an appropriate strided view and verify dimensions and strides.
* **Parallelism:** `&a * &b` is sequential. Use `linalg::matmul::matmul` and a team when explicit parallel execution is needed. `Accum::Replace` overwrites the destination; `Accum::Add` accumulates.
* **Ownership:** nalgebra operations may consume their inputs, while faer uses owned matrices and borrowed views. Check the method signature and preserve data needed later.
* **Numerical results:** factorization signs, eigenvector phases, eigenvalue ordering, and error types can differ. Validate reconstruction and residuals rather than comparing factors element by element. The general eigenvalue ordering is unspecified.
* **Thread budgets:** do not nest separate pools around a team call without a coordinated budget. Retain a pool for repeated work rather than acquiring one for each small operation.
* **Feature and version matching:** a `threadteam` token must come from the compatible dependency version used by faer. The `rayon` feature alone does not provide `Par::Team`. See `faer/Cargo.toml` for the current feature definitions.

## Published measurements

The [benchmark index](https://github.com/tensor4all/faer-benchmark/blob/main/result/INDEX.md) links published results and conditions. As one concrete example, the [Zen 5 large kernel report](https://github.com/tensor4all/faer-benchmark/blob/main/result/zen5-cpu/faer-kernels-large/0e15729ad011-0539947ffb75-mkl-timing-v2.md) measured `f64` row major 1024 by 1024 QR factorization on four participants on an AMD Ryzen AI 9 HX 470. The fork at commit `0e15729ad011` took 25.1275 ms, upstream at `0539947ffb75` took 74.1278 ms, and MKL 2026.1 took 20.9194 ms. In the same report, singular values took 189.915 ms for the fork, 189.863 ms upstream, and 121.591 ms with MKL.

These are historical measurements of t4a faer 0.25.0, not the current release. The report labels the development host results provisional. It reports medians of process medians, with five calls per process and three process sets, after a 250 ms priming period; the vendor row uses a separate build and row major conversions inside the vendor API remain timed. Inputs were verified before timing. Hardware, shape, layout, thread count, versions, and measurement boundaries matter, so consult the full report and measure your own workload before drawing conclusions.
