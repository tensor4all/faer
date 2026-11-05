//! Public finite-pencil regression for safe generalized-Hessenberg rotations.
mod common;
use common::*;
use faer::linalg::gevd::gen_hessenberg::{self, GeneralizedHessenbergParams};
fn normalized<T: ComplexField>(x: &T, scale: &T::Real, imaginary: &T) -> T {
	from_real::<T>(&(&x.real() / scale))
		+ imaginary.mul_real(&(&x.imag() / scale))
}
fn check<T: ComplexField>(make: fn(f64, f64) -> T, tiny: f64, large: f64) {
	let imaginary = if T::IS_REAL {
		zero::<T>()
	} else {
		(-one::<T>()).sqrt()
	};
	let pairs = [
		(make(tiny, 0.0), make(1.0, 0.0)),
		(make(1.0, 0.0), make(tiny, 0.0)),
		(make(tiny, 0.0), make(tiny, tiny)),
		(zero(), make(tiny, tiny)),
		(make(-3.0, 0.2), make(4.0, -0.5)),
		(make(large, 0.0), make(large, large * 0.2)),
	];
	for (f, g) in pairs {
		let mut a = Mat::<T>::identity(3, 3);
		a[(1, 0)] = f;
		a[(2, 0)] = g;
		let original = a.clone();
		let mut b = Mat::<T>::identity(3, 3);
		let mut q = b.clone();
		let mut z = b.clone();
		let params: GeneralizedHessenbergParams = faer::auto!(T);
		let mut memory = MemBuffer::new(
			gen_hessenberg::generalized_hessenberg_scratch::<T>(3, params),
		);
		gen_hessenberg::generalized_hessenberg(
			a.as_mut(),
			b.as_mut(),
			Some(q.as_mut()),
			Some(z.as_mut()),
			Par::Seq,
			MemStack::new(&mut memory),
			params,
		);
		assert!(
			a.is_all_finite()
				&& b.is_all_finite()
				&& q.is_all_finite()
				&& z.is_all_finite()
		);
		assert_eq!(a[(2, 0)], zero());
		for j in 0..3 {
			for i in j + 1..3 {
				assert_eq!(b[(i, j)], zero());
			}
		}
		unitary(q.as_ref());
		unitary(z.as_ref());
		let mut scale = one::<T::Real>();
		for j in 0..3 {
			for i in 0..3 {
				scale = scale
					.fmax(original[(i, j)].real().abs())
					.fmax(original[(i, j)].imag().abs());
			}
		}
		let an = Mat::<T>::from_fn(3, 3, |i, j| {
			normalized(&a[(i, j)], &scale, &imaginary)
		});
		let source = Mat::<T>::from_fn(3, 3, |i, j| {
			normalized(&original[(i, j)], &scale, &imaginary)
		});
		close((&q * &an * z.adjoint()).as_ref(), source.as_ref());
		close(
			(&q * &b * z.adjoint()).as_ref(),
			Mat::<T>::identity(3, 3).as_ref(),
		);
	}
}
#[test]
fn f32_safe_rotations() {
	check::<f32>(|r, _| r as f32, f32::from_bits(1) as f64, 1e30);
}
#[test]
fn f64_safe_rotations() {
	check::<f64>(|r, _| r, 1e-320, 1e300);
}
#[test]
fn c32_safe_rotations() {
	check::<c32>(
		|r, i| c32::new(r as f32, i as f32),
		f32::from_bits(1) as f64,
		1e30,
	);
}
#[test]
fn c64_safe_rotations() {
	check::<c64>(c64::new, 1e-320, 1e300);
}
