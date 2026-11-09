// Driver for the OpenBLAS BLAS-level matmul ports (TRMM, rank-k update, rank-1
// update). Provenance and the source revision are recorded in each case file
// and in docs/blas-test-inventory.md.
use crate::internal_prelude::*;
use crate::linalg::matmul::matmul;
use crate::linalg::matmul::triangular::{BlockStructure, matmul_with_conj};
use crate::{Accum, Conj, Mat, MatRef, Par};
macro_rules! suite {
	($name:ident,$ty:ty,$real:ty,$make:expr) => {
		#[allow(dead_code, unused_imports)]
		mod $name {
			use super::*;
			type Scalar = $ty;
			type Real = $real;
			fn scalar(re: Real, im: Real) -> Scalar {
				($make)(re, im)
			}
			fn z(re: Real) -> Scalar {
				scalar(re, 0.0)
			}
			include!("../../tests/blas_support.rs");
			include!("blas_trmm.rs");
			include!("blas_rank_update.rs");
		}
	};
}
suite!(f64, f64, f64, |re, _im| re);
suite!(f32, f32, f32, |re, _im| re);
suite!(c64, crate::c64, f64, crate::c64::new);
suite!(c32, crate::c32, f32, crate::c32::new);
