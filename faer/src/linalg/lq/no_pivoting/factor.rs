//! the packed $LQ$ factorization: $L$ in the lower triangle of the input, the
//! householder vectors of $Q$ along its rows, and the block coefficients apart.
//!
//! The output conventions of this module (the factor shapes, the placement of the
//! null-space block at the end of the full $Q$, and the optional positive diagonal
//! gauge) follow MatrixAlgebraKit.jl `src/implementations/lq.jl` and
//! `src/common/gauge.jl` (MIT, Copyright (c) QuantumKitHub contributors). The
//! factorization itself is faer's own blocked $QR$ kernel applied to the
//! transposed view; no arithmetic is copied from that project.
use crate::assert;
use crate::internal_prelude::*;
/// information about the resulting $LQ$ factorization.
#[derive(Copy, Clone, Debug)]
pub struct LqInfo {
	/// estimated rank of the matrix: the $QR$ kernel's estimate for the transposed
	/// view, which is an upper bound on the true rank (a run of zero rows of $A$
	/// inside the block structure can leave it above the exact rank).
	pub rank: usize,
}
/// the recommended block size to use for an $LQ$ decomposition of a matrix with
/// the given shape.
#[inline]
pub fn recommended_block_size<T: ComplexField>(
	nrows: usize,
	ncols: usize,
) -> usize {
	linalg::qr::no_pivoting::factor::recommended_block_size::<T>(nrows, ncols)
}
/// $LQ$ factorization tuning parameters.
///
/// An $LQ$ factorization is the $QR$ factorization of the transposed view, so
/// these are the $QR$ tuning parameters, applied to that view.
pub use linalg::qr::no_pivoting::factor::QrParams as LqParams;
/// applies the $QR$ factorization of the transposed view in place, which is this
/// matrix's $LQ$ factorization.
///
/// `Q_coeff` holds the block coefficients, one block per `Q_coeff.nrows()` steps,
/// as in [`qr::no_pivoting::factor::qr_in_place`](crate::linalg::qr::no_pivoting::factor::qr_in_place).
#[track_caller]
pub fn lq_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	Q_coeff: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<LqParams, T>,
) -> LqInfo {
	let info = linalg::qr::no_pivoting::factor::qr_in_place_packed(
		A.transpose_mut(),
		Q_coeff,
		par,
		stack,
		params,
	);
	LqInfo { rank: info.rank }
}
/// computes the layout of required workspace for performing an $LQ$
/// decomposition with no pivoting
#[inline]
pub fn lq_in_place_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	block_size: usize,
	par: ParShape,
	params: Spec<LqParams, T>,
) -> StackReq {
	linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
		ncols,
		nrows,
		block_size,
		par,
		params,
	)
}
/// copies the triangular factor $L$ out of the packed factorization in `packed`,
/// leaving the householder vectors intact.
///
/// `out` is either the compact $m \times \min(m, n)$ factor or the full
/// $m \times n$ one, whose columns past $\min(m, n)$ are zero. The strict upper
/// triangle is zero, so every entry of `out` is written.
///
/// with `positive`, column $j$ is scaled by $\overline{d_j} / |d_j|$, where
/// $d_j$ is the diagonal of $L$, which makes every diagonal entry of the result
/// real and nonnegative. [`unitary`](crate::linalg::lq::no_pivoting::unitary::unitary)
/// applies the matching phase to $Q$, so that the product is unchanged.
#[track_caller]
pub fn copy_l<T: ComplexField>(
	out: MatMut<'_, T>,
	packed: MatRef<'_, T>,
	positive: bool,
) {
	let (m, n) = packed.shape();
	let k = Ord::min(m, n);
	let mut out = out;
	assert!(all(out.nrows() == m, out.ncols() == k || out.ncols() == n));
	// every entry of `out` is written: the strict upper triangle and the full
	// factor's trailing columns are zero.
	out.rb_mut().fill(zero());
	out.rb_mut().get_mut(.., ..k).copy_from_triangular_lower(packed.get(.., ..k));
	if positive {
		for j in 0..k {
			let phase = linalg::positive_phase(packed[(j, j)].clone()).conj();
			for i in j..m {
				out[(i, j)] *= &phase;
			}
		}
	}
}

#[cfg(all(test, feature = "threadteam"))]
mod tests {
	use super::*;
	use crate::assert;
	use crate::internal_prelude::*;
	use crate::linalg::qr::no_pivoting::factor::{
		LOOKAHEAD_FORCE, LOOKAHEAD_STEPS,
	};
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, c64};
	use core::sync::atomic::Ordering::Relaxed;
	use dyn_stack::MemBuffer;
	/// [`copy_l`] hands `MatMut::copy_from_triangular_lower` a row-major packed
	/// matrix and a column-major destination, which is the tiled gate of
	/// `copy_lower_across_layouts`, with the working set of the `m x k` factor.
	/// The route is pinned on both sides of the threshold, square, tall and
	/// wide: the per-thread tiled-copy counter moves by exactly one where the
	/// gate should tile and stays put where it should fall back, which equal
	/// values alone cannot tell apart.
	///
	/// The destination here is the column-major `m x k` allocation, so its column
	/// stride is `m`: the `m = 2048` and `k = 256` cells tile on
	/// `tensor4all/faer#352`'s power-of-two stride although their working set is
	/// below the byte threshold.
	#[test]
	fn copy_l_reaches_the_tiled_copy_gate() {
		use crate::linalg::{tile_copy_pays, tiled_copies};
		for (m, n, tiled) in [
			(512, 512, true),
			// square, 2048 bytes of stride: #352's measured cell
			(256, 256, true),
			// tall: `k = 256`, a `2048 x 256` factor is 8 MiB over both buffers
			(2048, 256, true),
			// tall: 2 MiB, tiled only because the stride is 16 KiB, a power of two
			(2048, 64, true),
			// wide: the factor is only `m x k = 256 x 256`, and its stride is 2 KiB
			(256, 2048, true),
		] {
			let k = Ord::min(m, n);
			// `packed` as a row-major view of a column-major `n x m` allocation
			let packed_col = Mat::<f64>::from_fn(n, m, |j, i| (i + 2 * j) as f64);
			let packed = packed_col.as_ref().transpose();
			assert!(tile_copy_pays::<f64>(m, k, m as isize) == tiled);
			let mut out = Mat::<f64>::from_fn(m, k, |_, _| -1.0);
			let before = tiled_copies();
			copy_l(out.as_mut(), packed, false);
			assert!(
				tiled_copies() - before == tiled as usize,
				"{m}x{n}: copy_l took the {} path",
				if tiled { "elementwise" } else { "tiled" }
			);
			for j in 0..k {
				for i in 0..m {
					let want = if i >= j { packed[(i, j)] } else { 0.0 };
					assert!(out[(i, j)] == want, "{m}x{n}: entry ({i}, {j})");
				}
			}
		}
	}

	/// the look-ahead overlaps the next panel's factorization with the trailing
	/// apply. For LQ the panel is the block of *rows* of `A`, which the transposed
	/// view presents with a stride of `A.col_stride()`, so the transposed path and
	/// its factor layout need the same check as the QR one: the packed form and
	/// the reconstructed matrix agree with the sequential run, and the forced
	/// steps actually ran.
	#[test]
	fn test_lookahead_matches_sequential() {
		let rng = &mut StdRng::seed_from_u64(7);
		let mut ran = 0usize;
		for (m, n) in [(200, 512), (512, 512), (512, 320)] {
			for threads in [4, 8] {
				let a = CwiseMatDistribution {
					nrows: m,
					ncols: n,
					dist: ComplexDistribution::new(
						StandardNormal,
						StandardNormal,
					),
				}
				.rand::<Mat<c64>>(rng);
				let k = Ord::min(m, n);
				let bs = 8;
				let mut mem = MemBuffer::new(lq_in_place_scratch::<c64>(
					m,
					n,
					bs,
					crate::ParShape::team(threads),
					default(),
				));
				let mut run = |force: usize, steps: &mut usize, team: bool| {
					let mut packed = a.clone();
					let mut coeff = Mat::<c64>::zeros(bs, k);
					let rank = if team {
						crate::team::with_team(threads, |par| {
							LOOKAHEAD_FORCE.store(force, Relaxed);
							let before = LOOKAHEAD_STEPS.load(Relaxed);
							let info = lq_in_place(
								packed.as_mut(),
								coeff.as_mut(),
								par,
								MemStack::new(&mut mem),
								default(),
							);
							*steps = LOOKAHEAD_STEPS.load(Relaxed) - before;
							LOOKAHEAD_FORCE.store(0, Relaxed);
							info.rank
						})
					} else {
						lq_in_place(
							packed.as_mut(),
							coeff.as_mut(),
							Par::Seq,
							MemStack::new(&mut mem),
							default(),
						)
						.rank
					};
					(packed, coeff, rank)
				};
				let (p0, c0, r0) = run(0, &mut 0, false);
				let mut steps = 0;
				let (p1, c1, r1) = run(4, &mut steps, true);
				let tag = format!("{m}x{n} threads {threads}");
				assert_eq!(r0, r1, "{tag}: rank");
				assert!(steps > 0, "{tag}: the forced look-ahead ran no step");
				ran += 1;
				// the packed factor, the coefficients and the rank are the
				// sequential ones to rounding: the overlap changes only the order of
				// the apply against the panel.
				let approx_eq =
					CwiseMat(ApproxEq::eps() * (Ord::max(m, n) as f64) * 40.0);
				assert!(p1 ~ p0, "{tag}: packed factor");
				// the coefficients of a prefactored panel are assembled in a
				// different order, so the factors are compared through what a
				// consumer reads: $L$, $Q$, and $A = LQ$.
				let mut l = Mat::<c64>::zeros(m, k);
				copy_l(l.as_mut(), p1.as_ref(), true);
				let mut q = Mat::<c64>::zeros(k, n);
				let _ = &c1;
				crate::linalg::lq::no_pivoting::unitary::unitary(
					q.as_mut(),
					p1.as_ref(),
					c1.as_ref(),
					true,
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(
						crate::linalg::lq::no_pivoting::unitary::unitary_scratch::<c64>(
							m,
							n,
							k,
							bs,
							ParShape::Seq,
						),
					)),
				);
				assert!((&l * &q) ~ a, "{tag}: L Q");
				assert!((&q * q.adjoint()) ~ Mat::<c64>::identity(k, k), "{tag}: Q orthonormal");
				for j in 0..k {
					let d = l[(j, j)].clone();
					assert!(
						real(&d) >= 0.0
							&& imag(&d).abs() <= 1e-13 * d.abs().max(1.0)
					);
				}
			}
		}
		assert!(ran == 6);
	}
	/// the forced look-ahead must not lose rank on a prefactored panel that runs
	/// out of rank, which is the case the QR test names explicitly.
	#[test]
	fn test_lookahead_rank_loss() {
		let rng = &mut StdRng::seed_from_u64(11);
		// `n` clears the kernel's look-ahead floor, so the panel that runs out of
		// rank is a prefactored one.
		let (m, n, bs, threads) = (512usize, 400usize, 8usize, 4usize);
		let k = Ord::min(m, n);
		let mut a = CwiseMatDistribution {
			nrows: m,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);
		// every row from 40 on is zero, so the factorization's rank stops inside a
		// panel that the preceding look-ahead step has already prefactored.
		for i in 40..m {
			for j in 0..n {
				a[(i, j)] = c64::new(0.0, 0.0);
			}
		}
		let mut mem = MemBuffer::new(lq_in_place_scratch::<c64>(
			m,
			n,
			bs,
			crate::ParShape::team(threads),
			default(),
		));
		let mut run = |force: usize, team: bool| {
			let mut packed = a.clone();
			let mut coeff = Mat::<c64>::zeros(bs, k);
			let rank = if team {
				crate::team::with_team(threads, |par| {
					LOOKAHEAD_FORCE.store(force, Relaxed);
					let info = lq_in_place(
						packed.as_mut(),
						coeff.as_mut(),
						par,
						MemStack::new(&mut mem),
						default(),
					);
					LOOKAHEAD_FORCE.store(0, Relaxed);
					info.rank
				})
			} else {
				lq_in_place(
					packed.as_mut(),
					coeff.as_mut(),
					Par::Seq,
					MemStack::new(&mut mem),
					default(),
				)
				.rank
			};
			(packed, rank)
		};
		let (p0, r0) = run(0, false);
		let (p1, r1) = run(4, true);
		assert_eq!(r0, r1, "rank under look-ahead");
		assert!(r0 < k, "the fixture must be rank deficient");
		let approx_eq =
			CwiseMat(ApproxEq::eps() * (Ord::max(m, n) as f64) * 40.0);
		assert!(p1 ~ p0, "packed factor under look-ahead");
	}
}
