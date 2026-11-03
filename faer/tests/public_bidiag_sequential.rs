//! `bidiag_in_place` on the sequential-only build.
//!
//! The blocked path used to lose the `Y` update outside a `rayon` build (#203), so
//! `A = Hl B Hr^H` only held when `rayon` was compiled in. The library does build
//! without it (`--no-default-features --features std,linalg`), so this file pins the
//! contract for that configuration. It runs under any feature set; run it as
//! `cargo test -p t4a-faer --no-default-features --features std,linalg --test public_bidiag_sequential`
//! to exercise the path the bug lived in.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder as hh;
use faer::linalg::svd::bidiag;
use faer::{Conj, Mat, Par};
use reborrow::*;

/// `A - U B V^H`, with `B` the bidiagonal part of the factorized `UV`.
fn reconstruction_error(m: usize, n: usize, bl: usize) -> f64 {
	let a = Mat::<f64>::from_fn(m, n, |i, j| {
		(((i * 73 + j * 31 + i * j * 7) % 1009) as f64) / 504.0 - 1.0
	});
	let size = Ord::min(m, n);
	let mut uv = a.clone();
	let mut hl = Mat::<f64>::zeros(bl, size);
	let mut hr = Mat::<f64>::zeros(bl, size - 1);
	let mut mem = MemBuffer::new(bidiag::bidiag_in_place_scratch::<f64>(
		m,
		n,
		Par::Seq.shape(),
		Default::default(),
	));
	bidiag::bidiag_in_place(
		uv.as_mut(),
		hl.as_mut(),
		hr.as_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	);
	let mut check = a.clone();
	let mut mem = MemBuffer::new(
		hh::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(m, bl, n),
	);
	hh::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
		uv.as_ref().get(.., ..size),
		hl.as_ref(),
		Conj::Yes,
		check.as_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
	);
	let v = uv.as_ref().get(..size - 1, 1..size);
	let mut b = check.as_mut().get_mut(.., 1..size);
	let mut mem = MemBuffer::new(
		hh::apply_block_householder_sequence_on_the_right_in_place_scratch::<f64>(n - 1, bl, m),
	);
	hh::apply_block_householder_sequence_on_the_right_in_place_with_conj(
		v.transpose(),
		hr.as_ref(),
		Conj::Yes,
		b.rb_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
	);
	let b = b.to_owned();
	let mut err = 0.0f64;
	for j in 0..n {
		for i in 0..m {
			let expected = if j == 0 {
				if i == 0 {
					check[(i, 0)]
				} else {
					0.0
				}
			} else if i + 1 == j || i == j {
				b[(i, j - 1)]
			} else {
				0.0
			};
			err = err.max((check[(i, j)] - expected).abs());
		}
	}
	// every retained entry of `B` must be finite as well
	for j in 0..n.saturating_sub(1) {
		for i in 0..m {
			if (i + 1 == j || i == j) && !b[(i, j)].is_finite() {
				return f64::INFINITY;
			}
		}
	}
	err
}

/// Blocked, multi-panel shapes: the blocked path is what needs the `Y` update.
#[test]
fn blocked_bidiagonalization_reconstructs() {
	for (m, n, bl) in [(131, 129, 1), (300, 300, 4), (400, 300, 3)] {
		let err = reconstruction_error(m, n, bl);
		assert!(err <= 1e-12, "({m}, {n}): |A - U B V^H| = {err:e}");
	}
}

/// The unblocked path must keep working too.
#[test]
fn small_bidiagonalization_reconstructs() {
	for (m, n, bl) in [(8, 4, 1), (8, 8, 1)] {
		let err = reconstruction_error(m, n, bl);
		assert!(err <= 1e-12, "({m}, {n}): |A - U B V^H| = {err:e}");
	}
}
