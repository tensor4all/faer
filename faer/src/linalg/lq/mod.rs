//! the $LQ$ decomposition decomposes a matrix $A$ into the product
//! $$A = LQ$$
//! where $L$ is a lower trapezoidal matrix and $Q$ is a matrix with orthonormal
//! rows (represented using block householder sequences).
//!
//! the factorization is the $QR$ decomposition of the transposed view of $A$,
//! read transposed: `factor::lq_in_place` runs the blocked $QR$ kernel on
//! `A.transpose_mut()`, so the triangular factor $L$ lands in the lower triangle
//! of $A$ and the householder vectors of $Q$ land along its rows. Each panel, a
//! block of rows of $A$, is factored in a buffer that holds those rows
//! contiguous, and the trailing rows are updated in place by matrix products. `unitary`
//! builds $Q$ (thin or full) and its orthonormal complement, `reconstruct`
//! rebuilds $A = LQ$, and `factor::copy_l` copies $L$ out.
pub mod no_pivoting;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod lapack;
