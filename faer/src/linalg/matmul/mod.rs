//! matrix multiplication
use super::temp_mat_scratch;
use crate::col::ColRef;
use crate::internal_prelude::*;
use crate::mat::{MatMut, MatRef, MatUninitMut};
use crate::row::RowRef;
use crate::utils::bound::Dim;
use crate::utils::simd::SimdCtx;
use crate::{Conj, ContiguousFwd, Par, Shape};
use core::mem::MaybeUninit;
use dyn_stack::{MemBuffer, MemStack};
use equator::assert;
use faer_traits::{ByRef, ComplexField, Conjugate};
use pulp::Simd;
use reborrow::*;
const NANO_GEMM_THRESHOLD: usize = 16 * 16 * 16;
pub(crate) mod internal;
/// triangular matrix multiplication module, where some of the operands are
/// treated as triangular matrices
pub mod triangular;
mod matmul_shared {
	use super::*;
	pub const NC: usize = 2048;
	pub const KC: usize = 128;
	pub struct SimdLaneCount<T: ComplexField> {
		pub __marker: core::marker::PhantomData<fn() -> T>,
	}
	impl<T: ComplexField> pulp::WithSimd for SimdLaneCount<T> {
		type Output = usize;

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let _ = simd;
			core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>()
		}
	}
	pub struct MicroKernelShape<T: ComplexField> {
		pub __marker: core::marker::PhantomData<fn() -> T>,
	}
	impl<T: ComplexField> MicroKernelShape<T> {
		pub const IS_1X1: bool = Self::MAX_MR_DIV_N == 2 && Self::MAX_NR == 1;
		pub const IS_2X1: bool = Self::MAX_MR_DIV_N == 2 && Self::MAX_NR == 1;
		pub const IS_2X2: bool = Self::MAX_MR_DIV_N == 2 && Self::MAX_NR == 2;
		pub const MAX_MR_DIV_N: usize = Self::SHAPE.0;
		pub const MAX_NR: usize = Self::SHAPE.1;
		pub const SHAPE: (usize, usize) = {
			if const { size_of::<T>() / size_of::<T::Unit>() <= 2 } {
				(2, 2)
			} else if const { size_of::<T>() / size_of::<T::Unit>() == 4 } {
				(2, 1)
			} else {
				(1, 1)
			}
		};
	}
}
mod matmul_vertical {
	use super::*;
	use matmul_shared::*;
	struct Ukr<'a, const MR_DIV_N: usize, const NR: usize, T: ComplexField> {
		dst: MatMut<'a, T, usize, usize, ContiguousFwd>,
		a: MatRef<'a, T, usize, usize, ContiguousFwd>,
		b: MatRef<'a, T, usize, usize>,
		conj_lhs: Conj,
		conj_rhs: Conj,
		alpha: &'a T,
		beta: Accum,
	}
	impl<const MR_DIV_N: usize, const NR: usize, T: ComplexField> pulp::WithSimd
		for Ukr<'_, MR_DIV_N, NR, T>
	{
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let Self {
				dst,
				a,
				b,
				conj_lhs,
				conj_rhs,
				alpha,
				beta,
			} = self;
			if const {
				!(T::IS_NATIVE_F64
					|| T::IS_NATIVE_F32
					|| T::IS_NATIVE_C64
					|| T::IS_NATIVE_C32)
					&& !cfg!(test)
			} {
				panic!();
			} else {
				with_dim!(M, a.nrows());
				with_dim!(N, b.ncols());
				with_dim!(K, a.ncols());
				let a = a.as_shape(M, K);
				let b = b.as_shape(K, N);
				let mut dst = dst.as_shape_mut(M, N);
				let simd =
					SimdCtx::<T, S>::new_force_mask(T::simd_ctx(simd), M);
				let (_, body, tail) = simd.indices();
				let tail = tail.unwrap();
				let mut local_acc = [[simd.zero(); MR_DIV_N]; NR];
				if conj_lhs == conj_rhs {
					for depth in K.indices() {
						let mut a_uninit =
							[MaybeUninit::<T::SimdVec<S>>::uninit(); MR_DIV_N];
						for (dst, src) in
							core::iter::zip(&mut a_uninit, body.clone())
						{
							*dst =
								MaybeUninit::new(simd.read(a.col(depth), src));
						}
						a_uninit[MR_DIV_N - 1] =
							MaybeUninit::new(simd.read(a.col(depth), tail));
						let a: [T::SimdVec<S>; MR_DIV_N] = unsafe {
							crate::hacks::transmute::<
								[MaybeUninit<T::SimdVec<S>>; MR_DIV_N],
								[T::SimdVec<S>; MR_DIV_N],
							>(a_uninit)
						};
						for j in N.indices() {
							let b = simd.splat(&b[(depth, j)]);
							for i in 0..MR_DIV_N {
								let local_acc = &mut local_acc[*j][i];
								*local_acc = simd.mul_add(b, a[i], *local_acc);
							}
						}
					}
				} else {
					for depth in K.indices() {
						let mut a_uninit: [_; MR_DIV_N] =
							[MaybeUninit::<T::SimdVec<S>>::uninit(); MR_DIV_N];
						for (dst, src) in
							core::iter::zip(&mut a_uninit, body.clone())
						{
							*dst =
								MaybeUninit::new(simd.read(a.col(depth), src));
						}
						a_uninit[MR_DIV_N - 1] =
							MaybeUninit::new(simd.read(a.col(depth), tail));
						let a: [T::SimdVec<S>; MR_DIV_N] = unsafe {
							crate::hacks::transmute::<
								[MaybeUninit<T::SimdVec<S>>; MR_DIV_N],
								[T::SimdVec<S>; MR_DIV_N],
							>(a_uninit)
						};
						for j in N.indices() {
							let b = simd.splat(&b[(depth, j)]);
							for i in 0..MR_DIV_N {
								let local_acc = &mut local_acc[*j][i];
								*local_acc =
									simd.conj_mul_add(b, a[i], *local_acc);
							}
						}
					}
				}
				if conj_lhs.is_conj() {
					for x in &mut local_acc {
						for x in x {
							*x = simd.conj(*x);
						}
					}
				}
				let alpha = simd.splat(alpha);
				match beta {
					Accum::Add => {
						for (result, j) in
							core::iter::zip(&local_acc, N.indices())
						{
							for (result, i) in
								core::iter::zip(result, body.clone())
							{
								let mut val = simd.read(dst.rb().col(j), i);
								val = simd.mul_add(alpha, *result, val);
								simd.write(dst.rb_mut().col_mut(j), i, val);
							}
							let i = tail;
							let result = &result[MR_DIV_N - 1];
							let mut val = simd.read(dst.rb().col(j), i);
							val = simd.mul_add(alpha, *result, val);
							simd.write(dst.rb_mut().col_mut(j), i, val);
						}
					},
					Accum::Replace => {
						for (result, j) in
							core::iter::zip(&local_acc, N.indices())
						{
							for (result, i) in
								core::iter::zip(result, body.clone())
							{
								let val = simd.mul(alpha, *result);
								simd.write(dst.rb_mut().col_mut(j), i, val);
							}
							let i = tail;
							let result = &result[MR_DIV_N - 1];
							let val = simd.mul(alpha, *result);
							simd.write(dst.rb_mut().col_mut(j), i, val);
						}
					},
				}
			}
		}
	}
	pub fn matmul_simd<'M, 'N, 'K, T: ComplexField>(
		dst: MatMut<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		beta: Accum,
		lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: MatRef<'_, T, Dim<'K>, Dim<'N>>,
		conj_rhs: Conj,
		alpha: &T,
		mut par: Par,
	) {
		let dst = dst.as_dyn_mut();
		let lhs = lhs.as_dyn();
		let rhs = rhs.as_dyn();
		let (m, n) = dst.shape();
		let k = lhs.ncols();
		let arch = T::Arch::default();
		let lane_count = arch.dispatch(SimdLaneCount::<T> {
			__marker: core::marker::PhantomData,
		});
		let nr = MicroKernelShape::<T>::MAX_NR;
		let mr_div_n = MicroKernelShape::<T>::MAX_MR_DIV_N;
		let mr = mr_div_n * lane_count;
		let mut col_outer = 0;
		while col_outer < n {
			let n_chunk = Ord::min(n - col_outer, NC);
			let mut beta = beta;
			let mut depth = 0;
			while depth < k {
				let k_chunk = Ord::min(k - depth, KC);
				let job = |row: usize, col_inner: usize| {
					let nrows = Ord::min(m - row, mr);
					let ukr_i = nrows.div_ceil(lane_count);
					let ncols = Ord::min(n_chunk - col_inner, nr);
					let ukr_j = ncols;
					let dst = unsafe { dst.rb().const_cast() }.submatrix_mut(
						row,
						col_outer + col_inner,
						nrows,
						ncols,
					);
					let a = lhs.submatrix(row, depth, nrows, k_chunk);
					let b = rhs.submatrix(
						depth,
						col_outer + col_inner,
						k_chunk,
						ncols,
					);
					macro_rules! call {
						($M:expr, $N:expr) => {
							arch.dispatch(Ukr::<'_, $M, $N, T> {
								dst,
								a,
								b,
								conj_lhs,
								conj_rhs,
								alpha,
								beta,
							})
						};
					}
					if const { MicroKernelShape::<T>::IS_2X2 } {
						match (ukr_i, ukr_j) {
							(2, 2) => call!(2, 2),
							(1, 2) => call!(1, 2),
							(2, 1) => call!(2, 1),
							(1, 1) => call!(1, 1),
							_ => unreachable!(),
						}
					} else if const { MicroKernelShape::<T>::IS_2X1 } {
						match (ukr_i, ukr_j) {
							(2, 1) => call!(2, 1),
							(1, 1) => call!(1, 1),
							_ => unreachable!(),
						}
					} else if const { MicroKernelShape::<T>::IS_1X1 } {
						call!(1, 1)
					} else {
						unreachable!()
					}
				};
				let job_count = m.div_ceil(mr) * n_chunk.div_ceil(nr);
				let d = n_chunk.div_ceil(nr);
				match par {
					Par::Seq => {
						for job_idx in 0..job_count {
							let col_inner = nr * (job_idx % d);
							let row = mr * (job_idx / d);
							job(row, col_inner);
						}
					},
	#[cfg(feature = "rayon")]
					_ => {
					let nthreads = core::num::NonZeroUsize::new(par.degree()).unwrap();
						let nthreads = nthreads.get();
						use rayon::prelude::*;
						let job_idx = core::sync::atomic::AtomicUsize::new(0);
						crate::utils::thread::for_each(par.reborrow(), 
							nthreads,
							(0..nthreads).into_par_iter(),
							|_| {
								loop {
									let job_idx = job_idx.fetch_add(
										1,
										core::sync::atomic::Ordering::Relaxed,
									);
									if job_idx < job_count {
										let col_inner = nr * (job_idx % d);
										let row = mr * (job_idx / d);
										job(row, col_inner);
									} else {
										return;
									}
								}
							},
						);
					},
				}
				beta = Accum::Add;
				depth += k_chunk;
			}
			col_outer += n_chunk;
		}
	}
}
mod matmul_horizontal {
	use super::*;
	use matmul_shared::*;
	struct Ukr<'a, const MR: usize, const NR: usize, T: ComplexField> {
		dst: MatMut<'a, T, usize, usize>,
		a: MatRef<'a, T, usize, usize, isize, ContiguousFwd>,
		b: MatRef<'a, T, usize, usize, ContiguousFwd, isize>,
		conj_lhs: Conj,
		conj_rhs: Conj,
		alpha: &'a T,
		beta: Accum,
	}
	impl<const MR: usize, const NR: usize, T: ComplexField> pulp::WithSimd
		for Ukr<'_, MR, NR, T>
	{
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			if const {
				!(T::IS_NATIVE_F64
					|| T::IS_NATIVE_F32
					|| T::IS_NATIVE_C64
					|| T::IS_NATIVE_C32)
					&& !cfg!(test)
			} {
				panic!();
			} else {
				let Self {
					dst,
					a,
					b,
					conj_lhs,
					conj_rhs,
					alpha,
					beta,
				} = self;
				with_dim!(M, a.nrows());
				with_dim!(N, b.ncols());
				with_dim!(K, a.ncols());
				let a = a.as_shape(M, K);
				let b = b.as_shape(K, N);
				let mut dst = dst.as_shape_mut(M, N);
				let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), K);
				let (_, body, tail) = simd.indices();
				let mut local_acc = [[simd.zero(); MR]; NR];
				let mut is: [_; MR] = [M.idx(0usize); MR];
				let mut js: [_; NR] = [N.idx(0usize); NR];
				for (idx, i) in is.iter_mut().enumerate() {
					*i = M.idx(idx);
				}
				for (idx, j) in js.iter_mut().enumerate() {
					*j = N.idx(idx);
				}
				if conj_lhs == conj_rhs {
					macro_rules! do_it {
						($depth:expr) => {{
							let depth = $depth;
							let a = is.map(
								#[inline(always)]
								|i| simd.read(a.row(i).transpose(), depth),
							);
							let b = js.map(
								#[inline(always)]
								|j| simd.read(b.col(j), depth),
							);
							for i in 0..MR {
								for j in 0..NR {
									local_acc[j][i] = simd.mul_add(
										b[j],
										a[i],
										local_acc[j][i],
									);
								}
							}
						}};
					}
					for depth in body {
						do_it!(depth);
					}
					if let Some(depth) = tail {
						do_it!(depth);
					}
				} else {
					macro_rules! do_it {
						($depth:expr) => {{
							let depth = $depth;
							let a = is.map(
								#[inline(always)]
								|i| simd.read(a.row(i).transpose(), depth),
							);
							let b = js.map(
								#[inline(always)]
								|j| simd.read(b.col(j), depth),
							);
							for i in 0..MR {
								for j in 0..NR {
									local_acc[j][i] = simd.conj_mul_add(
										b[j],
										a[i],
										local_acc[j][i],
									);
								}
							}
						}};
					}
					for depth in body {
						do_it!(depth);
					}
					if let Some(depth) = tail {
						do_it!(depth);
					}
				}
				if conj_lhs.is_conj() {
					for x in &mut local_acc {
						for x in x {
							*x = simd.conj(*x);
						}
					}
				}
				let result = local_acc;
				let result = result.map(
					#[inline(always)]
					|result| {
						result.map(
							#[inline(always)]
							|result| simd.reduce_sum(result),
						)
					},
				);
				match beta {
					Accum::Add => {
						for (result, j) in core::iter::zip(&result, js) {
							for (result, i) in core::iter::zip(result, is) {
								dst[(i, j)] = alpha * result + &dst[(i, j)];
							}
						}
					},
					Accum::Replace => {
						for (result, j) in core::iter::zip(&result, js) {
							for (result, i) in core::iter::zip(result, is) {
								dst[(i, j)] = alpha * result;
							}
						}
					},
				}
			}
		}
	}
	pub fn matmul_simd<'M, 'N, 'K, T: ComplexField>(
		dst: MatMut<'_, T, Dim<'M>, Dim<'N>>,
		beta: Accum,
		lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: MatRef<'_, T, Dim<'K>, Dim<'N>, ContiguousFwd, isize>,
		conj_rhs: Conj,
		alpha: &T,
		mut par: Par,
	) {
		let dst = dst.as_dyn_mut();
		let lhs = lhs.as_dyn();
		let rhs = rhs.as_dyn();
		let (m, n) = dst.shape();
		let k = lhs.ncols();
		let nr = MicroKernelShape::<T>::MAX_NR;
		let mr = MicroKernelShape::<T>::MAX_MR_DIV_N;
		let arch = T::Arch::default();
		let lane_count = arch.dispatch(SimdLaneCount::<T> {
			__marker: core::marker::PhantomData,
		});
		let kc = KC * lane_count;
		let mut col_outer = 0;
		while col_outer < n {
			let n_chunk = Ord::min(n - col_outer, NC);
			let mut beta = beta;
			let mut depth = 0;
			while depth < k {
				let k_chunk = Ord::min(k - depth, kc);
				let job = |row: usize, col_inner: usize| {
					let nrows = Ord::min(m - row, mr);
					let ukr_i = nrows;
					let ncols = Ord::min(n_chunk - col_inner, nr);
					let ukr_j = ncols;
					let dst = unsafe { dst.rb().const_cast() }.submatrix_mut(
						row,
						col_outer + col_inner,
						nrows,
						ncols,
					);
					let a = lhs.submatrix(row, depth, nrows, k_chunk);
					let b = rhs.submatrix(
						depth,
						col_outer + col_inner,
						k_chunk,
						ncols,
					);
					macro_rules! call {
						($M:expr, $N:expr) => {
							arch.dispatch(Ukr::<'_, $M, $N, T> {
								dst,
								a,
								b,
								conj_lhs,
								conj_rhs,
								alpha,
								beta,
							})
						};
					}
					if const { MicroKernelShape::<T>::IS_2X2 } {
						match (ukr_i, ukr_j) {
							(2, 2) => call!(2, 2),
							(1, 2) => call!(1, 2),
							(2, 1) => call!(2, 1),
							(1, 1) => call!(1, 1),
							_ => unreachable!(),
						}
					} else if const { MicroKernelShape::<T>::IS_2X1 } {
						match (ukr_i, ukr_j) {
							(2, 1) => call!(2, 1),
							(1, 1) => call!(1, 1),
							_ => unreachable!(),
						}
					} else if const { MicroKernelShape::<T>::IS_1X1 } {
						call!(1, 1)
					} else {
						unreachable!()
					}
				};
				let job_count = m.div_ceil(mr) * n.div_ceil(nr);
				let d = n.div_ceil(nr);
				match par {
					Par::Seq => {
						for job_idx in 0..job_count {
							let col_inner = nr * (job_idx % d);
							let row = mr * (job_idx / d);
							job(row, col_inner);
						}
					},
	#[cfg(feature = "rayon")]
					_ => {
					let nthreads = core::num::NonZeroUsize::new(par.degree()).unwrap();
						let nthreads = nthreads.get();
						use rayon::prelude::*;
						let job_idx = core::sync::atomic::AtomicUsize::new(0);
						crate::utils::thread::for_each(par.reborrow(), 
							nthreads,
							(0..nthreads).into_par_iter(),
							|_| {
								loop {
									let job_idx = job_idx.fetch_add(
										1,
										core::sync::atomic::Ordering::Relaxed,
									);
									if job_idx < job_count {
										let col_inner = nr * (job_idx % d);
										let row = mr * (job_idx / d);
										job(row, col_inner);
									} else {
										return;
									}
								}
							},
						);
					},
				}
				beta = Accum::Add;
				depth += k_chunk;
			}
			col_outer += n_chunk;
		}
	}
}
/// dot product
pub mod dot {
	use super::*;
	use faer_traits::SimdArch;
	/// returns `lhs * rhs`, implicitly conjugating the operands if needed
	pub fn inner_prod<K: Shape, T: ComplexField>(
		lhs: RowRef<T, K>,
		conj_lhs: Conj,
		rhs: ColRef<T, K>,
		conj_rhs: Conj,
	) -> T {
		pub fn imp<'K, T: ComplexField>(
			lhs: RowRef<T, Dim<'K>>,
			conj_lhs: Conj,
			rhs: ColRef<T, Dim<'K>>,
			conj_rhs: Conj,
		) -> T {
			if const { T::SIMD_CAPABILITIES.is_simd() } {
				if let (Some(lhs), Some(rhs)) =
					(lhs.try_as_row_major(), rhs.try_as_col_major())
				{
					inner_prod_slice::<T>(
						lhs.ncols(),
						lhs.transpose(),
						conj_lhs,
						rhs,
						conj_rhs,
					)
				} else {
					inner_prod_schoolbook(lhs, conj_lhs, rhs, conj_rhs)
				}
			} else {
				inner_prod_schoolbook(lhs, conj_lhs, rhs, conj_rhs)
			}
		}
		with_dim!(K, lhs.ncols().unbound());
		imp(lhs.as_col_shape(K), conj_lhs, rhs.as_row_shape(K), conj_rhs)
	}
	#[inline(always)]
	fn inner_prod_slice<'K, T: ComplexField>(
		len: Dim<'K>,
		lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
		conj_rhs: Conj,
	) -> T {
		struct Impl<'a, 'K, T: ComplexField> {
			len: Dim<'K>,
			lhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
			conj_lhs: Conj,
			rhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
			conj_rhs: Conj,
		}
		impl<'a, 'K, T: ComplexField> pulp::WithSimd for Impl<'_, '_, T> {
			type Output = T;

			#[inline(always)]
			fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
				let Self {
					len,
					lhs,
					conj_lhs,
					rhs,
					conj_rhs,
				} = self;
				let simd = SimdCtx::new(T::simd_ctx(simd), len);
				let mut tmp = if conj_lhs == conj_rhs {
					inner_prod_no_conj_simd::<T, S>(simd, lhs, rhs)
				} else {
					inner_prod_conj_lhs_simd::<T, S>(simd, lhs, rhs)
				};
				if conj_rhs == Conj::Yes {
					tmp = tmp.conj();
				}
				tmp
			}
		}
		dispatch!(
			Impl {
				len,
				lhs,
				rhs,
				conj_lhs,
				conj_rhs
			},
			Impl,
			T
		)
	}
	#[inline(always)]
	pub(crate) fn inner_prod_no_conj_simd<'K, T: ComplexField, S: Simd>(
		simd: SimdCtx<'K, T, S>,
		lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
		rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
	) -> T {
		let mut acc = [simd.zero(); 4];
		simd_iter!(for (IDX, i) in [simd.batch_indices(); 4] {
			let l = simd.read(lhs, i);
			let r = simd.read(rhs, i);
			acc[IDX] = simd.mul_add(l, r, acc[IDX]);
		});

		let acc0 = simd.add(acc[0], acc[1]);
		let acc2 = simd.add(acc[2], acc[3]);
		let acc0 = simd.add(acc0, acc2);
		simd.reduce_sum(acc0)
	}
	#[inline(always)]
	pub(crate) fn inner_prod_conj_lhs_simd<'K, T: ComplexField, S: Simd>(
		simd: SimdCtx<'K, T, S>,
		lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
		rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
	) -> T {
		let mut acc = [simd.zero(); 4];
		simd_iter!(for (IDX, i) in [simd.batch_indices(); 4] {
			let l = simd.read(lhs, i);
			let r = simd.read(rhs, i);
			acc[IDX] = simd.conj_mul_add(l, r, acc[IDX]);
		});

		let acc0 = simd.add(acc[0], acc[1]);
		let acc2 = simd.add(acc[2], acc[3]);
		let acc0 = simd.add(acc0, acc2);
		simd.reduce_sum(acc0)
	}
	pub(crate) fn inner_prod_schoolbook<'K, T: ComplexField>(
		lhs: RowRef<'_, T, Dim<'K>>,
		conj_lhs: Conj,
		rhs: ColRef<'_, T, Dim<'K>>,
		conj_rhs: Conj,
	) -> T {
		let mut acc = zero();
		for k in lhs.ncols().indices() {
			if const { T::IS_REAL } {
				acc = &lhs[k] * &rhs[k] + acc;
			} else {
				match (conj_lhs, conj_rhs) {
					(Conj::No, Conj::No) => {
						acc = &lhs[k] * &rhs[k] + acc;
					},
					(Conj::No, Conj::Yes) => {
						acc = &lhs[k] * rhs[k].conj() + acc;
					},
					(Conj::Yes, Conj::No) => {
						acc = lhs[k].conj() * &rhs[k] + acc;
					},
					(Conj::Yes, Conj::Yes) => {
						acc = (&lhs[k] * &rhs[k]).conj() + acc;
					},
				}
			}
		}
		acc
	}
}
mod matvec_rowmajor {
	use super::*;
	use crate::col::ColMut;
	use faer_traits::SimdArch;
	pub fn matvec<'M, 'K, T: ComplexField>(
		dst: ColMut<'_, T, Dim<'M>>,
		beta: Accum,
		lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
		conj_rhs: Conj,
		alpha: &T,
		par: Par,
	) {
		core::assert!(const { T::SIMD_CAPABILITIES.is_simd() });
		// One section (a split of the rows or columns); the unit is a matrix element. The
		// product reads each element once, so the rate is the streaming one.
		let size = *lhs.nrows() * *lhs.ncols();
		let width = crate::utils::cost::par_width::<T>(
			&par,
			size,
			crate::utils::cost::rate::matvec(),
			1,
		);
		crate::utils::thread::with_width(par, width, |par| {
			match par {
				Par::Seq => {
					pub struct Impl<'a, 'M, 'K, T: ComplexField> {
						dst: ColMut<'a, T, Dim<'M>>,
						beta: Accum,
						lhs: MatRef<'a, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
						conj_lhs: Conj,
						rhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
						conj_rhs: Conj,
						alpha: &'a T,
					}
					impl<'a, 'M, 'K, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'K, T> {
						type Output = ();

						#[inline(always)]
						fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
							let Self {
								dst,
								beta,
								lhs,
								conj_lhs,
								rhs,
								conj_rhs,
								alpha,
							} = self;
							let simd = T::simd_ctx(simd);
							let mut dst = dst;
							let K = lhs.ncols();
							let simd = SimdCtx::new(simd, K);
							for i in lhs.nrows().indices() {
								let dst = &mut dst[i];
								let lhs = lhs.row(i);
								let rhs = rhs;
								let mut tmp = if conj_lhs == conj_rhs {
									dot::inner_prod_no_conj_simd::<T, S>(
										simd,
										lhs.transpose(),
										rhs,
									)
								} else {
									dot::inner_prod_conj_lhs_simd::<T, S>(
										simd,
										lhs.transpose(),
										rhs,
									)
								};
								if conj_rhs == Conj::Yes {
									tmp = tmp.conj();
								}
								tmp = alpha * tmp;
								if let Accum::Add = beta {
									tmp = &*dst + tmp;
								}
								*dst = tmp;
							}
						}
					}
					dispatch!(
						Impl {
							dst,
							beta,
							lhs,
							conj_lhs,
							rhs,
							conj_rhs,
							alpha,
						},
						Impl,
						T
					);
				},
		#[cfg(feature = "rayon")]
				_ => {
				let nthreads = core::num::NonZeroUsize::new(par.degree()).unwrap();
					let nthreads = nthreads.get();
					use rayon::prelude::*;
					crate::utils::thread::for_each(par, 
						nthreads,
						dst.par_partition_mut(nthreads)
							.zip_eq(lhs.par_row_partition(nthreads)),
						|(dst, lhs)| {
							make_guard!(M);
							let nrows = dst.nrows().bind(M);
							let dst = dst.as_row_shape_mut(nrows);
							let lhs = lhs.as_row_shape(nrows);
							matvec(
								dst,
								beta,
								lhs,
								conj_lhs,
								rhs,
								conj_rhs,
								alpha,
								Par::Seq,
							);
						},
					)
				},
			}
		});
	}
}
mod matvec_colmajor {
	use super::*;
	use crate::col::ColMut;
	use crate::utils::bound::IdxInc;

	use faer_traits::SimdArch;
	pub fn matvec<'M, 'K, T: ComplexField>(
		dst: ColMut<'_, T, Dim<'M>, ContiguousFwd>,
		beta: Accum,
		lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, ContiguousFwd, isize>,
		conj_lhs: Conj,
		rhs: ColRef<'_, T, Dim<'K>>,
		conj_rhs: Conj,
		alpha: &T,
		par: Par,
	) {
		core::assert!(const { T::SIMD_CAPABILITIES.is_simd() });
		// One section (a split of the rows or columns); the unit is a matrix element. The
		// product reads each element once, so the rate is the streaming one.
		let size = *lhs.nrows() * *lhs.ncols();
		let width = crate::utils::cost::par_width::<T>(
			&par,
			size,
			crate::utils::cost::rate::matvec(),
			1,
		);
		crate::utils::thread::with_width(par, width, |par| {
			match par {
				Par::Seq => {
					pub struct Impl<'a, 'M, 'K, T: ComplexField> {
						dst: ColMut<'a, T, Dim<'M>, ContiguousFwd>,
						beta: Accum,
						lhs: MatRef<'a, T, Dim<'M>, Dim<'K>, ContiguousFwd, isize>,
						conj_lhs: Conj,
						rhs: ColRef<'a, T, Dim<'K>>,
						conj_rhs: Conj,
						alpha: &'a T,
					}
					impl<'a, 'M, 'K, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'K, T> {
						type Output = ();

						#[inline(always)]
						fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
							let Self {
								dst,
								beta,
								lhs,
								conj_lhs,
								rhs,
								conj_rhs,
								alpha,
							} = self;
							let simd = T::simd_ctx(simd);
							let M = lhs.nrows();
							let simd = SimdCtx::<T, S>::new(simd, M);
							let indices = simd.indices();
							let mut dst = dst;
							match beta {
								Accum::Add => {},
								Accum::Replace => {
									simd_iter!(for i in [indices] {
										simd.write(dst.rb_mut(), i, simd.zero());
									});
								},
							}
							for j in lhs.ncols().indices() {
								let mut dst = dst.rb_mut();
								let lhs = lhs.col(j);
								let rhs = &rhs[j];
								let rhs = conj_rhs.apply_rt(rhs);
								let rhs = rhs * alpha;
								let vrhs = simd.splat(&rhs);
								if conj_lhs == Conj::Yes {
									simd_iter!(for i in [indices] {
										let y = simd.read(dst.rb(), i);
										let x = simd.read(lhs, i);
										simd.write(
											dst.rb_mut(),
											i,
											simd.conj_mul_add(x, vrhs, y),
										);
									});
								} else {
									simd_iter!(for i in [indices] {
										let y = simd.read(dst.rb(), i);
										let x = simd.read(lhs, i);
										simd.write(
											dst.rb_mut(),
											i,
											simd.mul_add(x, vrhs, y),
										);
									});
								}
							}
						}
					}
					dispatch!(
						Impl {
							dst,
							lhs,
							conj_lhs,
							rhs,
							conj_rhs,
							beta,
							alpha,
						},
						Impl,
						T
					)
				},
		#[cfg(feature = "rayon")]
				_ => {
				let nthreads = core::num::NonZeroUsize::new(par.degree()).unwrap();
					use rayon::prelude::*;
					let nthreads = nthreads.get();
					let mut mem = MemBuffer::new(temp_mat_scratch::<T>(
						dst.nrows().unbound(),
						nthreads,
					));
					let stack = MemStack::new(&mut mem);
					alloca!('stack: {
						let tmp =
							unsafe { mat![uninit::<T>, dst.nrows(), nthreads] };
					});
					let mut tmp = tmp.try_as_col_major_mut().unwrap();
					let mut dst = dst;
					make_guard!(Z);
					let Z = 0usize.bind(Z);
					let z = IdxInc::new_checked(0, lhs.ncols());
					crate::utils::thread::for_each(par, 
						nthreads,
						tmp.rb_mut()
							.par_col_iter_mut()
							.zip_eq(lhs.par_col_partition(nthreads))
							.zip_eq(rhs.par_partition(nthreads)),
						|((dst, lhs), rhs)| {
							make_guard!(K);
							let K = lhs.ncols().bind(K);
							let lhs = lhs.as_col_shape(K);
							let rhs = rhs.as_row_shape(K);
							matvec(
								dst,
								Accum::Replace,
								lhs,
								conj_lhs,
								rhs,
								conj_rhs,
								alpha,
								Par::Seq,
							);
						},
					);
					matvec(
						dst.rb_mut(),
						beta,
						lhs.subcols(z, Z),
						conj_lhs,
						rhs.subrows(z, Z),
						conj_rhs,
						&zero(),
						Par::Seq,
					);
					for j in 0..nthreads {
						zip!(dst.rb_mut(), tmp.rb().col(j))
							.for_each(|unzip!(dst, src)| *dst = &*dst + &*src)
					}
				},
			}
		});
	}
}
mod rank_update {
	use super::*;
	use crate::assert;
	fn rank_update_imp<'M, 'N, T: ComplexField>(
		dst: MatMut<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		beta: Accum,
		lhs: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: RowRef<'_, T, Dim<'N>>,
		conj_rhs: Conj,
		alpha: &T,
	) {
		assert!(T::SIMD_CAPABILITIES.is_simd());
		struct Impl<'a, 'M, 'N, T: ComplexField> {
			dst: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
			beta: Accum,
			lhs: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
			conj_lhs: Conj,
			rhs: RowRef<'a, T, Dim<'N>>,
			conj_rhs: Conj,
			alpha: &'a T,
		}
		impl<T: ComplexField> pulp::WithSimd for Impl<'_, '_, '_, T> {
			type Output = ();

			#[inline(always)]
			fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
				let Self {
					mut dst,
					beta,
					lhs,
					conj_lhs,
					rhs,
					conj_rhs,
					alpha,
				} = self;
				let (m, n) = dst.shape();
				let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), m);
				let indices = simd.indices();
				for j in n.indices() {
					let mut dst = dst.rb_mut().col_mut(j);
					let rhs = alpha * conj_rhs.apply_rt(&rhs[j]);
					let rhs = simd.splat(&rhs);
					if conj_lhs.is_conj() {
						match beta {
							Accum::Add => {
								simd_iter!(for i in [indices] {
									let mut acc = simd.read(dst.rb(), i);
									acc = simd.conj_mul_add(
										simd.read(lhs, i),
										rhs,
										acc,
									);
									simd.write(dst.rb_mut(), i, acc);
								});
							},
							Accum::Replace => {
								simd_iter!(for i in [indices] {
									let acc =
										simd.conj_mul(simd.read(lhs, i), rhs);
									simd.write(dst.rb_mut(), i, acc);
								});
							},
						}
					} else {
						match beta {
							Accum::Add => {
								simd_iter!(for i in [indices] {
									let mut acc = simd.read(dst.rb(), i);
									acc = simd.mul_add(
										simd.read(lhs, i),
										rhs,
										acc,
									);
									simd.write(dst.rb_mut(), i, acc);
								});
							},
							Accum::Replace => {
								simd_iter!(for i in [indices] {
									let acc = simd.mul(simd.read(lhs, i), rhs);
									simd.write(dst.rb_mut(), i, acc);
								});
							},
						}
					}
				}
			}
		}
		dispatch!(
			Impl {
				dst,
				lhs,
				conj_lhs,
				rhs,
				conj_rhs,
				beta,
				alpha,
			},
			Impl,
			T
		)
	}
	pub fn rank_update<'M, 'N, T: ComplexField>(
		dst: MatMut<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		beta: Accum,
		lhs: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: RowRef<'_, T, Dim<'N>>,
		conj_rhs: Conj,
		alpha: &T,
		par: Par,
	) {
		match par {
			Par::Seq => {
				rank_update_imp(dst, beta, lhs, conj_lhs, rhs, conj_rhs, alpha);
			},
	#[cfg(feature = "rayon")]
			_ => {
			let nthreads = core::num::NonZeroUsize::new(par.degree()).unwrap();
				let nthreads = nthreads.get();
				use rayon::prelude::*;
				crate::utils::thread::for_each(par, 
					nthreads,
					dst.par_col_partition_mut(nthreads)
						.zip(rhs.par_partition(nthreads)),
					|(dst, rhs)| {
						with_dim!(N, dst.ncols());
						rank_update_imp(
							dst.as_col_shape_mut(N),
							beta,
							lhs,
							conj_lhs,
							rhs.as_col_shape(N),
							conj_rhs,
							alpha,
						);
					},
				);
			},
		}
	}
}
fn matmul_imp<'M, 'N, 'K, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'M>, Dim<'N>>,
	beta: Accum,
	lhs: MatRef<'_, T, Dim<'M>, Dim<'K>>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, Dim<'K>, Dim<'N>>,
	conj_rhs: Conj,
	alpha: &T,
	par: Par,
) {
	let mut dst = dst;
	let M = dst.nrows();
	let N = dst.ncols();
	let K = lhs.ncols();
	if *M == 0 || *N == 0 {
		return;
	}
	if *K == 0 {
		if beta == Accum::Replace {
			dst.fill(zero());
		}
		return;
	}
	let mut lhs = lhs;
	let mut rhs = rhs;
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if dst.row_stride() < 0 {
			dst = dst.reverse_rows_mut();
			lhs = lhs.reverse_rows();
		}
		if dst.col_stride() < 0 {
			dst = dst.reverse_cols_mut();
			rhs = rhs.reverse_cols();
		}
		if lhs.col_stride() < 0 {
			lhs = lhs.reverse_cols();
			rhs = rhs.reverse_rows();
		}
		if dst.ncols().unbound() == 1 {
			let first = dst.ncols().check(0);
			if let (Some(dst), Some(lhs)) =
				(dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major())
			{
				matvec_colmajor::matvec(
					dst.col_mut(first),
					beta,
					lhs,
					conj_lhs,
					rhs.col(first),
					conj_rhs,
					alpha,
					par,
				);
				return;
			}
			if let (Some(rhs), Some(lhs)) =
				(rhs.try_as_col_major(), lhs.try_as_row_major())
			{
				matvec_rowmajor::matvec(
					dst.col_mut(first),
					beta,
					lhs,
					conj_lhs,
					rhs.col(first),
					conj_rhs,
					alpha,
					par,
				);
				return;
			}
		}
		if dst.nrows().unbound() == 1 {
			let mut dst = dst.rb_mut().transpose_mut();
			let (rhs, lhs) = (lhs.transpose(), rhs.transpose());
			let (conj_rhs, conj_lhs) = (conj_lhs, conj_rhs);
			let first = dst.ncols().check(0);
			if let (Some(dst), Some(lhs)) =
				(dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major())
			{
				matvec_colmajor::matvec(
					dst.col_mut(first),
					beta,
					lhs,
					conj_lhs,
					rhs.col(first),
					conj_rhs,
					alpha,
					par,
				);
				return;
			}
			if let (Some(rhs), Some(lhs)) =
				(rhs.try_as_col_major(), lhs.try_as_row_major())
			{
				matvec_rowmajor::matvec(
					dst.col_mut(first),
					beta,
					lhs,
					conj_lhs,
					rhs.col(first),
					conj_rhs,
					alpha,
					par,
				);
				return;
			}
		}
		if *K == 1 {
			let z = K.idx(0);
			if let (Some(dst), Some(lhs)) =
				(dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major())
			{
				rank_update::rank_update(
					dst,
					beta,
					lhs.col(z),
					conj_lhs,
					rhs.row(z),
					conj_rhs,
					alpha,
					par,
				);
				return;
			}
			if let (Some(dst), Some(rhs)) =
				(dst.rb_mut().try_as_row_major_mut(), rhs.try_as_row_major())
			{
				let dst = dst.transpose_mut();
				let rhs = rhs.row(z).transpose();
				let lhs = lhs.col(z).transpose();
				rank_update::rank_update(
					dst, beta, rhs, conj_rhs, lhs, conj_lhs, alpha, par,
				);
				return;
			}
		}
		macro_rules! gemm_call {
			($kind:ident, $ty:ty, $nanogemm:ident) => {
				unsafe {
					let dst = core::mem::transmute_copy::<
						MatMut<'_, T, Dim<'M>, Dim<'N>>,
						MatMut<'_, $ty, Dim<'M>, Dim<'N>>,
					>(&dst);
					let lhs = core::mem::transmute_copy::<
						MatRef<'_, T, Dim<'M>, Dim<'K>>,
						MatRef<'_, $ty, Dim<'M>, Dim<'K>>,
					>(&lhs);
					let rhs = core::mem::transmute_copy::<
						MatRef<'_, T, Dim<'K>, Dim<'N>>,
						MatRef<'_, $ty, Dim<'K>, Dim<'N>>,
					>(&rhs);
					let alpha = *core::mem::transmute_copy::<&T, &$ty>(&alpha);
					if (*M).saturating_mul(*N).saturating_mul(*K)
						<= NANO_GEMM_THRESHOLD
					{
						nano_gemm::planless::$nanogemm(
							*M,
							*N,
							*K,
							dst.as_ptr_mut(),
							dst.row_stride(),
							dst.col_stride(),
							lhs.as_ptr(),
							lhs.row_stride(),
							lhs.col_stride(),
							rhs.as_ptr(),
							rhs.row_stride(),
							rhs.col_stride(),
							match beta {
								Accum::Replace => core::mem::zeroed(),
								Accum::Add => {
									<$ty as ComplexFieldExt>::from_f64(1.0)
								},
							},
							alpha,
							conj_lhs == Conj::Yes,
							conj_rhs == Conj::Yes,
						);
						return;
					} else {
						let _ = match par {
							#[cfg(feature = "threadteam")]
							Par::Team(team) => {
							// the team path runs both kernels below on the team. On
							// x86_64 the `private_gemm_x86` kernel runs on the team's own members
							// (its packing and job queue, gated by `pays(.., 2 sections)`); elsewhere
							// one sequential gemm per member on its own tile of the output
							crate::utils::team::gemm_team::<$ty>(
								team,
								M.unbound(),
								N.unbound(),
								K.unbound(),
								$crate::utils::cost::gemm_ns_per_mac::<T>(),
								dst.as_ptr_mut(),
								dst.col_stride(),
								dst.row_stride(),
								beta != Accum::Replace,
								lhs.as_ptr(),
								lhs.col_stride(),
								lhs.row_stride(),
								rhs.as_ptr(),
								rhs.col_stride(),
								rhs.row_stride(),
								match beta {
									Accum::Replace => core::mem::zeroed(),
									Accum::Add => {
										<$ty as ComplexFieldExt>::from_f64(1.0)
									},
								},
								alpha,
								conj_lhs == Conj::Yes,
								conj_rhs == Conj::Yes,
							);
							return;
							},
							other => other,
						};
						#[cfg(all(target_arch = "x86_64", feature = "std"))]
						{
							use private_gemm_x86::*;
							let feat = if std::arch::is_x86_feature_detected!(
								"avx512f"
							) {
								Some(InstrSet::Avx512)
							} else if std::arch::is_x86_feature_detected!(
								"avx2"
							) && std::arch::is_x86_feature_detected!(
								"fma"
							) {
								Some(InstrSet::Avx256)
							} else {
								None
							};
							if let Some(feat) = feat {
								gemm(
									DType::$kind,
									IType::U64,
									feat,
									*M,
									*N,
									*K,
									dst.as_ptr_mut() as *mut (),
									dst.row_stride(),
									dst.col_stride(),
									core::ptr::null(),
									core::ptr::null(),
									DstKind::Full,
									match beta {
										$crate::Accum::Replace => {
											Accum::Replace
										},
										$crate::Accum::Add => Accum::Add,
									},
									lhs.as_ptr() as *const (),
									lhs.row_stride(),
									lhs.col_stride(),
									conj_lhs == Conj::Yes,
									core::ptr::null(),
									0,
									rhs.as_ptr() as *const (),
									rhs.row_stride(),
									rhs.col_stride(),
									conj_rhs == Conj::Yes,
									&raw const alpha as *const (),
									// one thread: a team returned above (`gemm_team`), so `par`
									// is sequential here
									1,
									// a team is the only parallel route: the `gemm` crate's own
									// parallelism is never handed a thread (the team runs the
									// kernel on its members instead, or the kernel stays serial)
								);
								return;
							}
						}
						{
							gemm::gemm(
								M.unbound(),
								N.unbound(),
								K.unbound(),
								dst.as_ptr_mut(),
								dst.col_stride(),
								dst.row_stride(),
								beta != Accum::Replace,
								lhs.as_ptr(),
								lhs.col_stride(),
								lhs.row_stride(),
								rhs.as_ptr(),
								rhs.col_stride(),
								rhs.row_stride(),
								match beta {
									Accum::Replace => core::mem::zeroed(),
									Accum::Add => {
										<$ty as ComplexFieldExt>::from_f64(1.0)
									},
								},
								alpha,
								false,
								conj_lhs == Conj::Yes,
								conj_rhs == Conj::Yes,
								gemm::Parallelism::None,
							);
							return;
						}
					}
				};
			};
		}
		if const { T::IS_NATIVE_F64 } {
			gemm_call!(F64, f64, execute_f64);
		}
		if const { T::IS_NATIVE_C64 } {
			gemm_call!(C64, num_complex::Complex<f64>, execute_c64);
		}
		if const { T::IS_NATIVE_F32 } {
			gemm_call!(F32, f32, execute_f32);
		}
		if const { T::IS_NATIVE_C32 } {
			gemm_call!(C32, num_complex::Complex<f32>, execute_c32);
		}
		if const {
			!(T::IS_NATIVE_F64
				|| T::IS_NATIVE_F32
				|| T::IS_NATIVE_C64
				|| T::IS_NATIVE_C32)
		} {
			if let (Some(dst), Some(lhs)) =
				(dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major())
			{
				matmul_vertical::matmul_simd(
					dst, beta, lhs, conj_lhs, rhs, conj_rhs, alpha, par,
				);
				return;
			}
			if let (Some(dst), Some(rhs)) =
				(dst.rb_mut().try_as_row_major_mut(), rhs.try_as_row_major())
			{
				matmul_vertical::matmul_simd(
					dst.transpose_mut(),
					beta,
					rhs.transpose(),
					conj_rhs,
					lhs.transpose(),
					conj_lhs,
					alpha,
					par,
				);
				return;
			}
			if let (Some(lhs), Some(rhs)) =
				(lhs.try_as_row_major(), rhs.try_as_col_major())
			{
				matmul_horizontal::matmul_simd(
					dst, beta, lhs, conj_lhs, rhs, conj_rhs, alpha, par,
				);
				return;
			}
		}
	}
	match par {
		Par::Seq => {
			for j in dst.ncols().indices() {
				for i in dst.nrows().indices() {
					let dst = &mut dst[(i, j)];
					let mut acc = dot::inner_prod_schoolbook(
						lhs.row(i),
						conj_lhs,
						rhs.col(j),
						conj_rhs,
					);
					acc = alpha * acc;
					if let Accum::Add = beta {
						acc = &*dst + acc;
					}
					*dst = acc;
				}
			}
		},
	#[cfg(feature = "rayon")]
		_ => {
		let nthreads = core::num::NonZeroUsize::new(par.degree()).unwrap();
			use rayon::prelude::*;
			let nthreads = nthreads.get();
			let m = *dst.nrows();
			let n = *dst.ncols();
			let task_count = m * n;
			let task_per_thread = task_count.msrv_div_ceil(nthreads);
			let dst = dst.rb();
			crate::utils::thread::for_each(par, nthreads, (0..nthreads).into_par_iter(), |tid| {
				let task_idx = tid * task_per_thread;
				if task_idx >= task_count {
					return;
				}
				let ntasks = Ord::min(task_per_thread, task_count - task_idx);
				for ij in 0..ntasks {
					let ij = task_idx + ij;
					let i = dst.nrows().check(ij % m);
					let j = dst.ncols().check(ij / m);
					let mut dst = unsafe { dst.const_cast() };
					let dst = &mut dst[(i, j)];
					let mut acc = dot::inner_prod_schoolbook(
						lhs.row(i),
						conj_lhs,
						rhs.col(j),
						conj_rhs,
					);
					acc = alpha * acc;
					if let Accum::Add = beta {
						acc = &*dst + acc;
					}
					*dst = acc;
				}
			});
		},
	}
}
/// `alpha * sum_p op(lhs[i, p]) * op(rhs[p, j])` written into the uninitialized slot `(i, j)` of
/// `dst`: read through raw pointers, written with `ptr::write`, no reference to the slot.
///
/// # Safety
/// The pointers and strides describe valid `m x k`, `k x n` and `m x n` matrices with `i < m`,
/// `j < n`; `dst` does not overlap the operands.
#[inline(always)]
unsafe fn matmul_uninit_one<T: ComplexField>(
	dst: *mut MaybeUninit<T>,
	(dst_rs, dst_cs): (isize, isize),
	lhs: *const T,
	(lhs_rs, lhs_cs): (isize, isize),
	rhs: *const T,
	(rhs_rs, rhs_cs): (isize, isize),
	k: usize,
	(i, j): (usize, usize),
	(conj_lhs, conj_rhs): (Conj, Conj),
	alpha: &T,
) {
	let mut acc: T = zero();
	for p in 0..k {
		let l = unsafe { &*lhs.offset(i as isize * lhs_rs + p as isize * lhs_cs) };
		let r = unsafe { &*rhs.offset(p as isize * rhs_rs + j as isize * rhs_cs) };
		acc = &conj_lhs.apply_rt(l) * &conj_rhs.apply_rt(r) + acc;
	}
	unsafe {
		dst.offset(i as isize * dst_rs + j as isize * dst_cs)
			.write(MaybeUninit::new(alpha * &acc));
	}
}

/// The body of [`matmul_with_conj_uninit`], on dynamic shapes: every logical element of `dst` is
/// written through raw pointers before it returns, and its prior contents are never read.
fn matmul_uninit_imp<T: ComplexField>(
	dst: MatUninitMut<'_, T>,
	lhs: MatRef<'_, T>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T>,
	conj_rhs: Conj,
	alpha: T,
	par: Par,
) {
	let m = dst.nrows();
	let n = dst.ncols();
	let k = lhs.ncols();
	precondition(m, n, lhs.nrows(), k, rhs.nrows(), rhs.ncols());
	if m == 0 || n == 0 {
		return;
	}
	let ptr = dst.as_ptr_mut();
	let (dst_rs, dst_cs) = (dst.row_stride(), dst.col_stride());
	// zeros through raw writes, after which the destination is initialized
	let fill_zero = || {
		for j in 0..n {
			for i in 0..m {
				unsafe {
					ptr.offset(i as isize * dst_rs + j as isize * dst_cs)
						.write(MaybeUninit::new(zero()));
				}
			}
		}
	};
	if k == 0 {
		// an empty product is zero; the kernels below write nothing when the depth is empty
		fill_zero();
		return;
	}
	// The native types on x86_64 with AVX2 or AVX-512: `t4a-private-gemm-x86` with
	// `Accum::Replace`, which writes every element in its first depth block and never loads the
	// destination before that (on a `Par::Team`, `gemm_team` takes the same kernel). The `gemm`
	// and `nano-gemm` crates are not given uninitialized storage: `gemm`'s GEMV forms a
	// `&mut [T]` of the destination and its Apple AMX kernel copies partial tiles of it before
	// checking `alpha`, and `nano-gemm` scales it by `beta`.
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	{
		macro_rules! gemm_call {
			($kind:ident, $ty:ty) => {
				unsafe {
					use private_gemm_x86::*;
					let feat = if std::arch::is_x86_feature_detected!("avx512f") {
						Some(InstrSet::Avx512)
					} else if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
						Some(InstrSet::Avx256)
					} else {
						None
					};
					if let Some(feat) = feat {
						let alpha: $ty = core::mem::transmute_copy(&alpha);
						let dst = ptr as *mut $ty;
						let (lhs_p, rhs_p) = (lhs.as_ptr() as *const $ty, rhs.as_ptr() as *const $ty);
						#[cfg(feature = "threadteam")]
						if let Par::Team(team) = par {
							// the same instruction set test as `gemm_team`'s, so it takes the kernel
							crate::utils::team::gemm_team::<$ty>(
								team,
								m,
								n,
								k,
								crate::utils::cost::gemm_ns_per_mac::<T>(),
								dst,
								dst_cs,
								dst_rs,
								false,
								lhs_p,
								lhs.col_stride(),
								lhs.row_stride(),
								rhs_p,
								rhs.col_stride(),
								rhs.row_stride(),
								core::mem::zeroed(),
								alpha,
								conj_lhs == Conj::Yes,
								conj_rhs == Conj::Yes,
							);
							return;
						}
						gemm(
							DType::$kind,
							IType::U64,
							feat,
							m,
							n,
							k,
							dst as *mut (),
							dst_rs,
							dst_cs,
							core::ptr::null(),
							core::ptr::null(),
							DstKind::Full,
							Accum::Replace,
							lhs_p as *const (),
							lhs.row_stride(),
							lhs.col_stride(),
							conj_lhs == Conj::Yes,
							core::ptr::null(),
							0,
							rhs_p as *const (),
							rhs.row_stride(),
							rhs.col_stride(),
							conj_rhs == Conj::Yes,
							&raw const alpha as *const (),
							1,
						);
						return;
					}
				}
			};
		}
		if const { T::IS_NATIVE_F64 } {
			gemm_call!(F64, f64);
		}
		if const { T::IS_NATIVE_C64 } {
			gemm_call!(C64, num_complex::Complex<f64>);
		}
		if const { T::IS_NATIVE_F32 } {
			gemm_call!(F32, f32);
		}
		if const { T::IS_NATIVE_C32 } {
			gemm_call!(C32, num_complex::Complex<f32>);
		}
	}
	if const { T::IS_NATIVE_F64 || T::IS_NATIVE_C64 || T::IS_NATIVE_F32 || T::IS_NATIVE_C32 } {
		// a native type without that kernel: initialize, then the ordinary product, which
		// overwrites the zeros without reading them (the zero pass is the cost the kernel route
		// avoids)
		fill_zero();
		let dst = unsafe { MatMut::from_raw_parts_mut(ptr as *mut T, m, n, dst_rs, dst_cs) };
		matmul_with_conj(dst, Accum::Replace, lhs, conj_lhs, rhs, conj_rhs, alpha, par);
		return;
	}
	// any other scalar: one dot product per element, by columns (a column per task on a team)
	let (dst_w, lhs_w, rhs_w) = (
		crate::utils::thread::Ptr(ptr),
		crate::utils::thread::Ptr(lhs.as_ptr() as *mut T),
		crate::utils::thread::Ptr(rhs.as_ptr() as *mut T),
	);
	let lhs_s = (lhs.row_stride(), lhs.col_stride());
	let rhs_s = (rhs.row_stride(), rhs.col_stride());
	let alpha = &alpha;
	let column = move |j: usize| {
		let (dst_w, lhs_w, rhs_w) = (dst_w, lhs_w, rhs_w);
		for i in 0..m {
			// SAFETY: the shapes were checked by `precondition`; `dst` is exclusive and does not
			// overlap the operands (the caller's contract); each column is written by one task
			unsafe {
				matmul_uninit_one(dst_w.0, (dst_rs, dst_cs), lhs_w.0, lhs_s, rhs_w.0, rhs_s, k, (i, j), (conj_lhs, conj_rhs), alpha)
			}
		}
	};
	match par {
		Par::Seq => (0..n).for_each(column),
		#[cfg(feature = "threadteam")]
		par => {
			use rayon::prelude::*;
			let degree = par.degree();
			crate::utils::thread::for_each(par, degree, (0..n).into_par_iter(), column);
		},
		#[cfg(not(feature = "threadteam"))]
		_ => (0..n).for_each(column),
	}
}

/// computes `alpha * op(lhs) * op(rhs)` into a destination whose elements may be uninitialized,
/// and returns it as an initialized [`MatMut`].
///
/// This is the overwrite-only entry for callers that hold `MaybeUninit<T>` storage and do not
/// want an output-sized initialization pass before the product (upstream request
/// sarah-quinones/faer-rs#269). The destination's prior contents are never read, no `&T` or
/// `&mut T` to one of its elements is formed before the element is written, and every logical
/// element is written before the function returns. Only then is the returned [`MatMut`] formed.
/// If the call panics, the storage stays `MaybeUninit<T>` (some elements may have been written)
/// and nothing is exposed as initialized.
///
/// The output-sized initialization pass is skipped for the native types on x86_64 with AVX2 or
/// AVX-512, and for other scalar types. On other targets the native types are zero-filled first
/// and then multiplied as by [`matmul_with_conj`].
///
/// The parallelism is the explicit `par`. The operands must not overlap the destination, and the
/// destination's elements must not overlap each other: the safe slice constructors of
/// [`MatUninitMut`] check this, and the raw constructor requires it.
///
/// # example
///
/// ```
/// use core::mem::MaybeUninit;
/// use faer::linalg::matmul::matmul_with_conj_uninit;
/// use faer::{mat, Conj, MatUninitMut, Par};
///
/// let lhs = mat![[1.0_f64, 2.0], [3.0, 4.0]];
/// let rhs = mat![[5.0_f64, 6.0], [7.0, 8.0]];
/// let mut storage = [MaybeUninit::<f64>::uninit(); 4];
/// let dst = MatUninitMut::from_column_major_slice_mut(&mut storage, 2, 2);
/// let result = matmul_with_conj_uninit(dst, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, 1.0, Par::Seq);
/// assert_eq!(result[(0, 0)], 19.0);
/// ```
#[track_caller]
#[inline]
pub fn matmul_with_conj_uninit<'a, T: ComplexField, M: Shape, N: Shape, K: Shape, RStride: Stride, CStride: Stride>(
	dst: MatUninitMut<'a, T, M, N, RStride, CStride>,
	lhs: impl AsMatRef<T = T, Rows = M, Cols = K>,
	conj_lhs: Conj,
	rhs: impl AsMatRef<T = T, Rows = K, Cols = N>,
	conj_rhs: Conj,
	alpha: T,
	par: Par,
) -> MatMut<'a, T, M, N, RStride, CStride> {
	let ptr = dst.as_ptr_mut();
	let (nrows, ncols) = (dst.nrows(), dst.ncols());
	let (row_stride, col_stride) = (dst.row_stride(), dst.col_stride());
	matmul_uninit_imp(
		dst.as_dyn(),
		lhs.as_mat_ref().as_dyn(),
		conj_lhs,
		rhs.as_mat_ref().as_dyn(),
		conj_rhs,
		alpha,
		par,
	);
	// SAFETY: `matmul_uninit_imp` returned, so every logical element has been written; the view
	// keeps the destination's exclusive borrow `'a`
	unsafe { MatMut::from_raw_parts_mut(ptr.cast(), nrows, ncols, row_stride, col_stride) }
}

#[track_caller]
fn precondition<M: Shape, N: Shape, K: Shape>(
	dst_nrows: M,
	dst_ncols: N,
	lhs_nrows: M,
	lhs_ncols: K,
	rhs_nrows: K,
	rhs_ncols: N,
) {
	assert!(all(
		dst_nrows == lhs_nrows,
		dst_ncols == rhs_ncols,
		lhs_ncols == rhs_nrows,
	));
}
/// computes the matrix product `[beta * acc] + alpha * lhs * rhs` and stores
/// the result in `acc`
///
/// performs the operation:
/// - `acc = alpha * lhs * rhs` if `beta` is `Accum::Replace` (in this case, the
///   preexisting
/// values in `acc` are not read)
/// - `acc = acc + alpha * lhs * rhs` if `beta` is `Accum::Add`
///
/// # panics
///
/// panics if the matrix dimensions are not compatible for matrix
/// multiplication. i.e.
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
/// # Example
///
/// ```
/// use faer::linalg::matmul::matmul;
/// use faer::{Accum, Conj, Mat, Par, mat, unzip, zip};
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
/// 	[
/// 		2.5 * (lhs[(0, 0)] * rhs[(0, 0)] + lhs[(0, 1)] * rhs[(1, 0)]),
/// 		2.5 * (lhs[(0, 0)] * rhs[(0, 1)] + lhs[(0, 1)] * rhs[(1, 1)]),
/// 	],
/// 	[
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 0)] + lhs[(1, 1)] * rhs[(1, 0)]),
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 1)] + lhs[(1, 1)] * rhs[(1, 1)]),
/// 	],
/// ];
/// matmul(&mut acc, Accum::Replace, &lhs, &rhs, 2.5, Par::Seq);
/// zip!(&acc, &target)
/// 	.for_each(|unzip!(acc, target)| assert!((acc - target).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn matmul<
	T: ComplexField,
	LhsT: Conjugate<Canonical = T>,
	RhsT: Conjugate<Canonical = T>,
	M: Shape,
	N: Shape,
	K: Shape,
>(
	dst: impl AsMatMut<T = T, Rows = M, Cols = N>,
	beta: Accum,
	lhs: impl AsMatRef<T = LhsT, Rows = M, Cols = K>,
	rhs: impl AsMatRef<T = RhsT, Rows = K, Cols = N>,
	alpha: T,
	par: Par,
) {
	let mut dst = dst;
	let dst = dst.as_mat_mut();
	let lhs = lhs.as_mat_ref();
	let rhs = rhs.as_mat_ref();
	precondition(
		dst.nrows(),
		dst.ncols(),
		lhs.nrows(),
		lhs.ncols(),
		rhs.nrows(),
		rhs.ncols(),
	);
	make_guard!(M);
	make_guard!(N);
	make_guard!(K);
	let M = dst.nrows().bind(M);
	let N = dst.ncols().bind(N);
	let K = lhs.ncols().bind(K);
	matmul_imp(
		dst.as_dyn_stride_mut().as_shape_mut(M, N),
		beta,
		lhs.as_dyn_stride().canonical().as_shape(M, K),
		const { Conj::get::<LhsT>() },
		rhs.as_dyn_stride().canonical().as_shape(K, N),
		const { Conj::get::<RhsT>() },
		&alpha,
		par,
	);
}
/// computes the matrix product `[beta * acc] + alpha * lhs * rhs` (implicitly
/// conjugating the operands if needed) and stores the result in `acc`
///
/// performs the operation:
/// - `acc = alpha * lhs * rhs` if `beta` is `Accum::Replace` (in this case, the
///   preexisting
/// values in `acc` are not read)
/// - `acc = acc + alpha * lhs * rhs` if `beta` is `Accum::Add`
///
/// # panics
///
/// panics if the matrix dimensions are not compatible for matrix
/// multiplication. i.e.
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
/// # example
///
/// ```
/// use faer::linalg::matmul::matmul_with_conj;
/// use faer::{Accum, Conj, Mat, Par, mat, unzip, zip};
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
/// 	[
/// 		2.5 * (lhs[(0, 0)] * rhs[(0, 0)] + lhs[(0, 1)] * rhs[(1, 0)]),
/// 		2.5 * (lhs[(0, 0)] * rhs[(0, 1)] + lhs[(0, 1)] * rhs[(1, 1)]),
/// 	],
/// 	[
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 0)] + lhs[(1, 1)] * rhs[(1, 0)]),
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 1)] + lhs[(1, 1)] * rhs[(1, 1)]),
/// 	],
/// ];
/// matmul_with_conj(
/// 	&mut acc,
/// 	Accum::Replace,
/// 	&lhs,
/// 	Conj::No,
/// 	&rhs,
/// 	Conj::No,
/// 	2.5,
/// 	Par::Seq,
/// );
/// zip!(&acc, &target)
/// 	.for_each(|unzip!(acc, target)| assert!((acc - target).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn matmul_with_conj<T: ComplexField, M: Shape, N: Shape, K: Shape>(
	dst: impl AsMatMut<T = T, Rows = M, Cols = N>,
	beta: Accum,
	lhs: impl AsMatRef<T = T, Rows = M, Cols = K>,
	conj_lhs: Conj,
	rhs: impl AsMatRef<T = T, Rows = K, Cols = N>,
	conj_rhs: Conj,
	alpha: T,
	par: Par,
) {
	let mut dst = dst;
	let dst = dst.as_mat_mut();
	let lhs = lhs.as_mat_ref();
	let rhs = rhs.as_mat_ref();
	precondition(
		dst.nrows(),
		dst.ncols(),
		lhs.nrows(),
		lhs.ncols(),
		rhs.nrows(),
		rhs.ncols(),
	);
	make_guard!(M);
	make_guard!(N);
	make_guard!(K);
	let M = dst.nrows().bind(M);
	let N = dst.ncols().bind(N);
	let K = lhs.ncols().bind(K);
	matmul_imp(
		dst.as_dyn_stride_mut().as_shape_mut(M, N),
		beta,
		lhs.as_dyn_stride().canonical().as_shape(M, K),
		conj_lhs,
		rhs.as_dyn_stride().canonical().as_shape(K, N),
		conj_rhs,
		&alpha,
		par,
	);
}
#[cfg(test)]
mod tests {
	use super::triangular::{BlockStructure, DiagonalKind};
	use super::*;
	use crate::mat::{Mat, MatMut, MatRef};
	use crate::stats::prelude::*;
	use crate::{assert, c32, c64, fx128};
	use core::mem::MaybeUninit;
	use std::num::NonZeroUsize;
	/// Below the flop floor `matmul` runs the sequential kernel whatever budget
	/// the caller passes, and above it the kernel threads: both must compute the
	/// same product, at the sizes on each side of the floor.
	#[test]
	fn thread_budget_does_not_change_the_product() {
		for (m, k, n) in [(32, 32, 32), (64, 64, 64), (17, 23, 31), (128, 128, 128)]
		{
			let lhs = Mat::<f64>::from_fn(m, k, |i, j| {
				((i * 7 + j * 13) % 11) as f64 * 0.5 - 2.5
			});
			let rhs = Mat::<f64>::from_fn(k, n, |i, j| {
				((i * 5 + j * 3) % 7) as f64 * 0.25 - 1.0
			});
			let mut sequential = Mat::<f64>::zeros(m, n);
			matmul(
				sequential.as_mut(),
				Accum::Replace,
				lhs.as_ref(),
				rhs.as_ref(),
				1.0,
				Par::Seq,
			);
			let mut threaded = Mat::<f64>::zeros(m, n);
			crate::team::with_team(8, |par| {
				matmul(
					threaded.as_mut(),
					Accum::Replace,
					lhs.as_ref(),
					rhs.as_ref(),
					1.0,
					par,
				)
			});
			for i in 0..m {
				for j in 0..n {
					let d = (sequential[(i, j)] - threaded[(i, j)]).abs();
					assert!(d < 1e-12, "({i}, {j}): {d}");
				}
			}
		}
	}
	#[test]
	#[ignore = "takes too long"]
	fn test_matmul() {
		let rng = &mut StdRng::seed_from_u64(0);
		if option_env!("CI") == Some("true") {
			return;
		}
		/// the sequential arm, then the parallel one on a team of four
		fn for_each_par(mut f: impl FnMut(Par<'_>)) {
			f(Par::Seq);
			crate::team::with_team(4, |par| f(par));
		}
		let betas = [Accum::Replace, Accum::Add];
		#[cfg(not(miri))]
		let bools = [false, true];
		#[cfg(not(miri))]
		let alphas = [c32::ONE, c32::ZERO, c32::new(21.04, -12.13)];
		#[cfg(not(miri))]
		let conjs = [Conj::Yes, Conj::No];
		#[cfg(miri)]
		let bools = [true];
		#[cfg(miri)]
		let alphas = [c32::new(0.3218, -1.217489)];
		#[cfg(miri)]
		let par = [Par::Seq];
		#[cfg(miri)]
		let conjs = [Conj::Yes];
		let big0 = 127;
		let big1 = 128;
		let big2 = 129;
		let mid0 = 15;
		let mid1 = 16;
		let mid2 = 17;
		for (m, n, k) in [
			(big0, big1, 5),
			(big1, big0, 5),
			(big0, big2, 5),
			(big2, big0, 5),
			(mid0, mid0, 5),
			(mid1, mid1, 5),
			(mid2, mid2, 5),
			(mid0, mid1, 5),
			(mid1, mid0, 5),
			(mid0, mid2, 5),
			(mid2, mid0, 5),
			(mid0, 1, 1),
			(1, mid0, 1),
			(1, 1, mid0),
			(1, mid0, mid0),
			(mid0, 1, mid0),
			(mid0, mid0, 1),
			(1, 1, 1),
		] {
			let distribution =
				ComplexDistribution::new(StandardNormal, StandardNormal);
			let a = CwiseMatDistribution {
				nrows: m,
				ncols: k,
				dist: distribution,
			}
			.rand::<Mat<c32>>(rng);
			let b = CwiseMatDistribution {
				nrows: k,
				ncols: n,
				dist: distribution,
			}
			.rand::<Mat<c32>>(rng);
			let mut acc_init = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: distribution,
			}
			.rand::<Mat<c32>>(rng);
			let a = a.as_ref();
			let b = b.as_ref();
			for reverse_acc_cols in bools {
				for reverse_acc_rows in bools {
					for reverse_b_cols in bools {
						for reverse_b_rows in bools {
							for reverse_a_cols in bools {
								for reverse_a_rows in bools {
									for a_colmajor in bools {
										for b_colmajor in bools {
											for acc_colmajor in bools {
												let a = if a_colmajor {
													a
												} else {
													a.transpose()
												};
												let mut a = if a_colmajor {
													a
												} else {
													a.transpose()
												};
												let b = if b_colmajor {
													b
												} else {
													b.transpose()
												};
												let mut b = if b_colmajor {
													b
												} else {
													b.transpose()
												};
												if reverse_a_rows {
													a = a.reverse_rows();
												}
												if reverse_a_cols {
													a = a.reverse_cols();
												}
												if reverse_b_rows {
													b = b.reverse_rows();
												}
												if reverse_b_cols {
													b = b.reverse_cols();
												}
												for conj_a in conjs {
													for conj_b in conjs {
														for_each_par(|mut par| {
															for beta in betas {
																for alpha in
																	alphas
																{
																	test_matmul_impl(
																		reverse_acc_cols,
																		reverse_acc_rows,
																		acc_colmajor,
																		m,
																		n,
																		conj_a,
																		conj_b,
																		par.reborrow(),
																		beta,
																		alpha,
																		acc_init.as_mut(),
																		a,
																		b,
																	);
																}
															}
																										});
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	#[test]
	fn test_matmul_with_conj_uninit() {
		let lhs = mat![[1.0_f64, 2.0], [3.0, 4.0]];
		let rhs = mat![[5.0_f64, 6.0], [7.0, 8.0]];
		let mut data = [MaybeUninit::<f64>::uninit(); 4];
		let dst = MatUninitMut::from_column_major_slice_mut(&mut data, 2, 2);
		let result = matmul_with_conj_uninit(dst, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, 1.0, Par::Seq);
		assert!(result[(0, 0)] == 19.0);
		assert!(result[(1, 0)] == 43.0);
		assert!(result[(0, 1)] == 22.0);
		assert!(result[(1, 1)] == 50.0);
	}

	/// A matrix-vector product into uninitialized storage: the shape for which the `gemm` crate
	/// would take its GEMV path, which forms a slice of the destination (run it under Miri).
	#[test]
	fn test_matmul_with_conj_uninit_matrix_vector() {
		let lhs = mat![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
		let rhs = mat![[1.0_f64], [-1.0], [2.0]];
		let mut data = [MaybeUninit::<f64>::uninit(); 2];
		let dst = MatUninitMut::from_column_major_slice_mut(&mut data, 2, 1);
		let result = matmul_with_conj_uninit(dst, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, 1.0, Par::Seq);
		assert!(result[(0, 0)] == 5.0);
		assert!(result[(1, 0)] == 11.0);
	}

	#[test]
	fn test_matmul_with_conj_uninit_strided_and_conjugated() {
		let lhs = mat![[c64::new(1.0, 2.0), c64::new(2.0, -1.0)], [c64::new(3.0, 1.0), c64::new(-2.0, 4.0)],];
		let rhs = mat![[c64::new(5.0, -1.0), c64::new(1.0, 3.0)], [c64::new(2.0, 2.0), c64::new(-4.0, 1.0)],];
		let mut expected = Mat::<c64>::zeros(2, 2);
		matmul_with_conj(expected.as_mut(), Accum::Replace, lhs.as_ref(), Conj::Yes, rhs.as_ref(), Conj::No, c64::new(2.0, 0.0), Par::Seq);
		let mut data = [MaybeUninit::<c64>::uninit(); 7];
		let dst = unsafe { MatUninitMut::from_raw_parts_mut(data.as_mut_ptr(), 2, 2, 3, 1) };
		let result = matmul_with_conj_uninit(dst, lhs.as_ref(), Conj::Yes, rhs.as_ref(), Conj::No, c64::new(2.0, 0.0), Par::Seq);
		for j in 0..2 {
			for i in 0..2 {
				assert!((result[(i, j)] - expected[(i, j)]).norm() < 1e-12);
			}
		}
	}

	#[test]
	fn test_matmul_with_conj_uninit_negative_destination_stride() {
		let lhs = mat![[1.0_f64, 2.0], [3.0, 4.0]];
		let rhs = mat![[5.0_f64, 6.0], [7.0, 8.0]];
		let mut data = [MaybeUninit::<f64>::uninit(); 4];
		let dst = unsafe { MatUninitMut::from_raw_parts_mut(data.as_mut_ptr().add(1), 2, 2, -1, 2) };
		let result = matmul_with_conj_uninit(dst, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, 1.0, Par::Seq);
		assert!(result[(0, 0)] == 19.0);
		assert!(result[(1, 0)] == 43.0);
		assert!(result[(0, 1)] == 22.0);
		assert!(result[(1, 1)] == 50.0);
	}

	#[test]
	fn test_matmul_with_conj_uninit_empty_inner_dimension() {
		let lhs = Mat::<f64>::zeros(2, 0);
		let rhs = Mat::<f64>::zeros(0, 3);
		let mut data = [MaybeUninit::<f64>::uninit(); 6];
		let dst = MatUninitMut::from_row_major_slice_mut(&mut data, 2, 3);
		let result = matmul_with_conj_uninit(dst, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, 2.0, Par::Seq);
		for i in 0..2 {
			for j in 0..3 {
				assert!(result[(i, j)] == 0.0);
			}
		}
	}

	#[test]
	fn test_matmul_with_conj_uninit_generic_scalar() {
		let from = |x: f64| <fx128 as ComplexFieldExt>::from_f64(x);
		let lhs = Mat::<fx128>::from_fn(2, 2, |i, j| from((i + 2 * j + 1) as f64));
		let rhs = Mat::<fx128>::from_fn(2, 2, |i, j| from((3 * i + j + 1) as f64));
		let mut data = [MaybeUninit::<fx128>::uninit(); 4];
		let dst = MatUninitMut::from_column_major_slice_mut(&mut data, 2, 2);
		let result = matmul_with_conj_uninit(dst, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, from(2.0), Par::Seq);
		assert!(result[(0, 0)] == from(26.0));
		assert!(result[(1, 0)] == from(36.0));
		assert!(result[(0, 1)] == from(34.0));
		assert!(result[(1, 1)] == from(48.0));
	}

	/// The native kernels at sizes below and above the small-matrix and tiling thresholds: the destination starts as NaN, which a kernel that
	/// read it would carry into the product, and its padding rows must be left untouched.
	fn check_uninit_native<T: ComplexField + Copy + core::fmt::Debug>(
		from: impl Fn(f64, f64) -> T,
		dist: impl Fn(T, T) -> f64,
		tol: f64,
	) {
		let nan = <T as ComplexFieldExt>::from_f64(f64::NAN);
		for (m, k, n) in [(1, 1, 1), (3, 4, 5), (17, 23, 31), (64, 64, 64), (130, 70, 90), (5, 300, 7), (9, 600, 7)] {
			for (conj_lhs, conj_rhs) in [(Conj::No, Conj::No), (Conj::Yes, Conj::No), (Conj::No, Conj::Yes)] {
				for row_major in [false, true] {
					let lhs = Mat::<T>::from_fn(m, k, |i, j| from(((i * 7 + j * 3) % 11) as f64 - 5.0, ((i + 2 * j) % 5) as f64 - 2.0));
					let rhs = Mat::<T>::from_fn(k, n, |i, j| from(((i * 5 + j) % 13) as f64 - 6.0, ((3 * i + j) % 7) as f64 - 3.0));
					let alpha = from(0.5, -1.5);
					let mut expected = Mat::<T>::zeros(m, n);
					matmul_with_conj(expected.as_mut(), Accum::Replace, lhs.as_ref(), conj_lhs, rhs.as_ref(), conj_rhs, alpha, Par::Seq);
					// a padded leading dimension: three NaN elements between columns (or rows)
					let (rs, cs, len) = if row_major { (n as isize + 3, 1, m * (n + 3)) } else { (1, m as isize + 3, n * (m + 3)) };
					let mut data = alloc::vec![MaybeUninit::new(nan); len];
					let dst = unsafe { MatUninitMut::<T>::from_raw_parts_mut(data.as_mut_ptr(), m, n, rs, cs) };
					let result = matmul_with_conj_uninit(dst, lhs.as_ref(), conj_lhs, rhs.as_ref(), conj_rhs, alpha, Par::Seq);
					for j in 0..n {
						for i in 0..m {
							let err = dist(result[(i, j)], expected[(i, j)]);
							::core::assert!(err <= tol * k as f64 * 200.0, "{m}x{k}x{n} ({i}, {j}): {:?} vs {:?}", result[(i, j)], expected[(i, j)]);
						}
					}
					let inner = if row_major { n } else { m };
					for (p, x) in data.iter().enumerate() {
						if p % (inner + 3) >= inner {
							let x = unsafe { x.assume_init() };
							::core::assert!(x != x, "{m}x{k}x{n}: padding element {p} was written");
						}
					}
				}
			}
		}
	}

	#[test]
	fn test_matmul_with_conj_uninit_native_never_reads_destination() {
		check_uninit_native(|x, _| x as f32, |a: f32, b| (a - b).abs() as f64, 1e-6);
		check_uninit_native(|x, _| x, |a: f64, b| (a - b).abs(), 1e-14);
		check_uninit_native(|x, y| c32::new(x as f32, y as f32), |a: c32, b| (a - b).norm() as f64, 1e-6);
		check_uninit_native(|x, y| c64::new(x, y), |a: c64, b| (a - b).norm(), 1e-14);
	}

	#[cfg(feature = "threadteam")]
	#[test]
	fn test_matmul_with_conj_uninit_team() {
		crate::utils::team::with_team(3, |mut par| {
			{
				let lhs = Mat::<f64>::from_fn(130, 70, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
				let rhs = Mat::<f64>::from_fn(70, 90, |i, j| ((i * 5 + j) % 13) as f64 - 6.0);
				let mut expected = Mat::<f64>::zeros(130, 90);
				matmul_with_conj(expected.as_mut(), Accum::Replace, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, 1.0, Par::Seq);
				let mut data = alloc::vec![MaybeUninit::new(f64::NAN); 130 * 90];
				let dst = MatUninitMut::from_column_major_slice_mut(&mut data, 130, 90);
				let result = matmul_with_conj_uninit(dst, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, 1.0, par.reborrow());
				assert!(result == expected);
			}
			{
				let from = |x: f64| <fx128 as ComplexFieldExt>::from_f64(x);
				let lhs = Mat::<fx128>::from_fn(9, 6, |i, j| from((i + 2 * j) as f64));
				let rhs = Mat::<fx128>::from_fn(6, 11, |i, j| from((3 * i + j) as f64));
				let mut expected = Mat::<fx128>::zeros(9, 11);
				matmul_with_conj(expected.as_mut(), Accum::Replace, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, from(1.0), Par::Seq);
				let mut data = alloc::vec![MaybeUninit::<fx128>::uninit(); 9 * 11];
				let dst = MatUninitMut::from_column_major_slice_mut(&mut data, 9, 11);
				let result = matmul_with_conj_uninit(dst, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, from(1.0), par.reborrow());
				assert!(result == expected);
			}
		});
	}

	/// A batched destination: one buffer holding `batch` matrices at a batch stride, each filled
	/// through its own strided view.
	#[test]
	fn test_matmul_with_conj_uninit_batched() {
		let (batch, m, k, n) = (3, 5, 4, 6);
		let batch_stride = m * n + 2;
		let mut data = alloc::vec![MaybeUninit::new(f64::NAN); batch * batch_stride];
		for b in 0..batch {
			let lhs = Mat::<f64>::from_fn(m, k, |i, j| (i + j + b) as f64);
			let rhs = Mat::<f64>::from_fn(k, n, |i, j| (i * j + b) as f64 - 2.0);
			let mut expected = Mat::<f64>::zeros(m, n);
			matmul_with_conj(expected.as_mut(), Accum::Replace, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, 1.0, Par::Seq);
			// row major inside each batch slot
			let dst = unsafe { MatUninitMut::<f64>::from_raw_parts_mut(data.as_mut_ptr().add(b * batch_stride), m, n, n as isize, 1) };
			let result = matmul_with_conj_uninit(dst, lhs.as_ref(), Conj::No, rhs.as_ref(), Conj::No, 1.0, Par::Seq);
			assert!(result == expected);
		}
		for b in 0..batch {
			for p in m * n..batch_stride {
				assert!(unsafe { data[b * batch_stride + p].assume_init() }.is_nan());
			}
		}
	}

	fn matmul_with_conj_fallback<T: Copy + ComplexField>(
		acc: MatMut<'_, T>,
		a: MatRef<'_, T>,
		conj_a: Conj,
		b: MatRef<'_, T>,
		conj_b: Conj,
		beta: Accum,
		alpha: T,
	) {
		let m = acc.nrows();
		let n = acc.ncols();
		let k = a.ncols();
		let job = |idx: usize| {
			let i = idx % m;
			let j = idx / m;
			let acc = acc.rb().submatrix(i, j, 1, 1);
			let mut acc = unsafe { acc.const_cast() };
			let mut local_acc = zero::<T>();
			for depth in 0..k {
				let a = &a[(i, depth)];
				let b = &b[(depth, j)];
				local_acc = local_acc
					+ match conj_a {
						Conj::Yes => a.conj(),
						Conj::No => a.copy(),
					} * match conj_b {
						Conj::Yes => b.conj(),
						Conj::No => b.copy(),
					};
			}
			match beta {
				Accum::Add => acc[(0, 0)] = acc[(0, 0)] + local_acc * alpha,
				Accum::Replace => acc[(0, 0)] = local_acc * alpha,
			}
		};
		for i in 0..m * n {
			job(i);
		}
	}
	fn test_matmul_impl(
		reverse_acc_cols: bool,
		reverse_acc_rows: bool,
		acc_colmajor: bool,
		m: usize,
		n: usize,
		conj_a: Conj,
		conj_b: Conj,
		mut par: Par,
		beta: Accum,
		alpha: c32,
		acc_init: MatMut<c32>,
		a: MatRef<c32>,
		b: MatRef<c32>,
	) {
		let acc = if acc_colmajor {
			acc_init
		} else {
			acc_init.transpose_mut()
		};
		let mut acc = if acc_colmajor {
			acc
		} else {
			acc.transpose_mut()
		};
		if reverse_acc_rows {
			acc = acc.reverse_rows_mut();
		}
		if reverse_acc_cols {
			acc = acc.reverse_cols_mut();
		}
		let mut target = acc.rb().to_owned();
		matmul_with_conj_fallback(
			target.as_mut(),
			a,
			conj_a,
			b,
			conj_b,
			beta,
			alpha,
		);
		let target = target.rb();
		{
			let mut acc = acc.cloned();
			let a = a.cloned();
			{
				with_dim!(M, a.nrows());
				with_dim!(N, b.ncols());
				with_dim!(K, a.ncols());
				let mut acc = acc.rb_mut().as_shape_mut(M, N);
				let a = a.as_shape(M, K);
				let b = b.as_shape(K, N);
				matmul_vertical::matmul_simd(
					acc.rb_mut().try_as_col_major_mut().unwrap(),
					beta,
					a.try_as_col_major().unwrap(),
					conj_a,
					b,
					conj_b,
					&alpha,
					par.reborrow(),
				);
			}
			for j in 0..n {
				for i in 0..m {
					let acc = acc[(i, j)];
					let target = target[(i, j)];
					assert!((acc.re - target.re).abs() < 1e-3);
					assert!((acc.im - target.im).abs() < 1e-3);
				}
			}
		}
		{
			let mut acc = acc.cloned();
			let a = a.transpose().cloned();
			let a = a.transpose();
			let b = b.cloned();
			{
				with_dim!(M, a.nrows());
				with_dim!(N, b.ncols());
				with_dim!(K, a.ncols());
				let mut acc = acc.rb_mut().as_shape_mut(M, N);
				let a = a.as_shape(M, K);
				let b = b.as_shape(K, N);
				matmul_horizontal::matmul_simd(
					acc.rb_mut(),
					beta,
					a.try_as_row_major().unwrap(),
					conj_a,
					b.try_as_col_major().unwrap(),
					conj_b,
					&alpha,
					par.reborrow(),
				);
			}
			for j in 0..n {
				for i in 0..m {
					let acc = acc[(i, j)];
					let target = target[(i, j)];
					assert!((acc.re - target.re).abs() < 1e-3);
					assert!((acc.im - target.im).abs() < 1e-3);
				}
			}
		}
		matmul_with_conj(acc.rb_mut(), beta, a, conj_a, b, conj_b, alpha, par);
		for j in 0..n {
			for i in 0..m {
				let acc = acc[(i, j)];
				let target = target[(i, j)];
				assert!((acc.re - target.re).abs() < 1e-3);
				assert!((acc.im - target.im).abs() < 1e-3);
			}
		}
	}
	fn generate_structured_matrix(
		is_dst: bool,
		nrows: usize,
		ncols: usize,
		structure: BlockStructure,
	) -> Mat<f64> {
		let rng = &mut StdRng::seed_from_u64(0);
		let mut mat = CwiseMatDistribution {
			nrows,
			ncols,
			dist: StandardNormal,
		}
		.rand::<Mat<f64>>(rng);
		if !is_dst {
			let kind = structure.diag_kind();
			if structure.is_lower() {
				for j in 0..ncols {
					for i in 0..j {
						mat[(i, j)] = 0.0;
					}
				}
			} else if structure.is_upper() {
				for j in 0..ncols {
					for i in j + 1..nrows {
						mat[(i, j)] = 0.0;
					}
				}
			}
			match kind {
				triangular::DiagonalKind::Zero => {
					for i in 0..nrows {
						mat[(i, i)] = 0.0;
					}
				},
				triangular::DiagonalKind::Unit => {
					for i in 0..nrows {
						mat[(i, i)] = 1.0;
					}
				},
				triangular::DiagonalKind::Generic => {},
			}
		}
		mat
	}
	fn run_test_problem(
		m: usize,
		n: usize,
		k: usize,
		dst_structure: BlockStructure,
		lhs_structure: BlockStructure,
		rhs_structure: BlockStructure,
	) {
		let mut dst = generate_structured_matrix(true, m, n, dst_structure);
		let mut dst_target = dst.as_ref().to_owned();
		let dst_orig = dst.as_ref().to_owned();
		let lhs = generate_structured_matrix(false, m, k, lhs_structure);
		let rhs = generate_structured_matrix(false, k, n, rhs_structure);
		let mut run = |mut par: Par| {
			triangular::matmul_with_conj(
				dst.as_mut(),
				dst_structure,
				Accum::Replace,
				lhs.as_ref(),
				lhs_structure,
				Conj::No,
				rhs.as_ref(),
				rhs_structure,
				Conj::No,
				2.5,
				par.reborrow(),
			);
			matmul_with_conj(
				dst_target.as_mut(),
				Accum::Replace,
				lhs.as_ref(),
				Conj::No,
				rhs.as_ref(),
				Conj::No,
				2.5,
				par,
			);
			if dst_structure.is_dense() {
				for j in 0..n {
					for i in 0..m {
						assert!(
							(dst[(i, j)] - dst_target[(i, j)]).abs() < 1e-10
						);
					}
				}
			} else if dst_structure.is_lower() {
				for j in 0..n {
					if matches!(
						dst_structure.diag_kind(),
						DiagonalKind::Generic
					) {
						for i in 0..j {
							assert!(
								(dst[(i, j)] - dst_orig[(i, j)]).abs() < 1e-10
							);
						}
						for i in j..n {
							assert!(
								(dst[(i, j)] - dst_target[(i, j)]).abs()
									< 1e-10
							);
						}
					} else {
						for i in 0..=j {
							assert!(
								(dst[(i, j)] - dst_orig[(i, j)]).abs() < 1e-10
							);
						}
						for i in j + 1..n {
							assert!(
								(dst[(i, j)] - dst_target[(i, j)]).abs()
									< 1e-10
							);
						}
					}
				}
			} else {
				for j in 0..n {
					if matches!(
						dst_structure.diag_kind(),
						DiagonalKind::Generic
					) {
						for i in 0..=j {
							assert!(
								(dst[(i, j)] - dst_target[(i, j)]).abs()
									< 1e-10
							);
						}
						for i in j + 1..n {
							assert!(
								(dst[(i, j)] - dst_orig[(i, j)]).abs() < 1e-10
							);
						}
					} else {
						for i in 0..j {
							assert!(
								(dst[(i, j)] - dst_target[(i, j)]).abs()
									< 1e-10
							);
						}
						for i in j..n {
							assert!(
								(dst[(i, j)] - dst_orig[(i, j)]).abs() < 1e-10
							);
						}
					}
				}
			}
		};
		run(Par::Seq);
		crate::team::with_team(8, run);
	}
	#[test]
	fn test_triangular() {
		use BlockStructure::*;
		let structures = [
			Rectangular,
			TriangularLower,
			TriangularUpper,
			StrictTriangularLower,
			StrictTriangularUpper,
			UnitTriangularLower,
			UnitTriangularUpper,
		];
		for dst in structures {
			for lhs in structures {
				for rhs in structures {
					#[cfg(not(miri))]
					let big = 100;
					#[cfg(miri)]
					let big = 31;
					for _ in 0..3 {
						let m = rand::random::<u32>() as usize % big;
						let mut n = rand::random::<u32>() as usize % big;
						let mut k = rand::random::<u32>() as usize % big;
						match (
							!dst.is_dense(),
							!lhs.is_dense(),
							!rhs.is_dense(),
						) {
							(true, true, _)
							| (true, _, true)
							| (_, true, true) => {
								n = m;
								k = m;
							},
							_ => {},
						}
						if !dst.is_dense() {
							n = m;
						}
						if !lhs.is_dense() {
							k = m;
						}
						if !rhs.is_dense() {
							k = n;
						}
						run_test_problem(m, n, k, dst, lhs, rhs);
					}
				}
			}
		}
	}
}
