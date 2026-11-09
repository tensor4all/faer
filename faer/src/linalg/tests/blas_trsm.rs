// Driver for the OpenBLAS TRSM/TRSV port. Provenance and the source revision
// are recorded in blas_trsm_cases.rs and docs/blas-test-inventory.md.
use crate::internal_prelude::*;
use crate::linalg::matmul::triangular::BlockStructure;
use crate::linalg::triangular_solve as ts;
use crate::{Conj, Mat, MatMut, Par};
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
			include!("blas_support.rs");
			include!("blas_trsm_cases.rs");
		}
	};
}
suite!(f64, f64, f64, |re, _im| re);
suite!(f32, f32, f32, |re, _im| re);
suite!(c64, crate::c64, f64, crate::c64::new);
suite!(c32, crate::c32, f32, crate::c32::new);
