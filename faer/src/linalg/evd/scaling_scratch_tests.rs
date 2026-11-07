//! #270: the optional range copy uses stack storage, fully initialized before use.
use super::*;
use crate::assert;
use dyn_stack::MemBuffer;

fn check_copy<T: ComplexField>(value: fn(f64, f64) -> T, scales: &[f64]) {
	let n = 4;
	for subdiag in [0, 1] {
		for &s in scales {
			let scale = from_f64::<T::Real>(s);
			let input = Mat::<T>::from_fn(n, n, |i, j| {
				let re = if i == j { (i + 1) as f64 } else { 2.0 };
				value(re * s, s)
			});
			let original = input.clone();
			let mut memory = MemBuffer::new(evd_from_schur_scratch::<T>(n));
			// Poison reusable storage so an unwritten band/tail cannot appear
			// valid.
			for byte in memory.iter_mut() {
				byte.write(0xFF);
			}
			let (scaled, _) = scaled_into_range(
				input.as_ref(),
				subdiag,
				MemStack::new(&mut memory),
			);
			if s == 1.0 || s == 2.0f64.powi(-22) || s == 2.0f64.powi(18) {
				assert!(scaled.is_none());
			} else {
				let scaled = scaled.as_ref().unwrap().as_mat_ref();
				for j in 0..n {
					for i in 0..n {
						let expected = if i < Ord::min(j + 1 + subdiag, n) {
							value(
								(if i == j { (i + 1) as f64 } else { 2.0 })
									/ 4.0,
								0.25,
							)
						} else {
							zero::<T>()
						};
						assert!(
							scaled[(i, j)] == expected,
							"scale {scale:?}, ({i},{j}): {:?} != {expected:?}",
							scaled[(i, j)]
						);
					}
				}
			}
			assert!(input == original);
		}
	}
}
#[test]
fn range_copy_initializes_all_entries_and_preserves_thresholds() {
	let single = [
		1.0,
		2.0f64.powi(-22),
		2.0f64.powi(18),
		2.0f64.powi(-23),
		2.0f64.powi(19),
		2.0f64.powi(125),
		2.0f64.powi(-120),
		f32::from_bits(1) as f64,
	];
	let double = [
		1.0,
		2.0f64.powi(-22),
		2.0f64.powi(18),
		2.0f64.powi(-23),
		2.0f64.powi(19),
		2.0f64.powi(1021),
		2.0f64.powi(-1020),
		f64::from_bits(1),
	];
	check_copy::<f32>(|r, _| r as f32, &single);
	check_copy::<f64>(|r, _| r, &double);
	check_copy::<crate::c32>(
		|r, i| crate::c32::new(r as f32, i as f32),
		&single,
	);
	check_copy::<crate::c64>(crate::c64::new, &double);
}
#[test]
fn empty_and_zero_schur_inputs_need_no_range_copy() {
	for n in [0, 1, 3] {
		let a = Mat::<f64>::zeros(n, n);
		let mut memory = MemBuffer::new(StackReq::EMPTY);
		let (scaled, _) =
			scaled_into_range(a.as_ref(), 1, MemStack::new(&mut memory));
		assert!(scaled.is_none());
	}
}
