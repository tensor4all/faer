use crate::internal_prelude::*;
use crate::linalg::matmul::triangular::DiagonalKind;
use crate::{MatRef, assert};
use linalg::matmul::triangular::BlockStructure;
pub fn has_spicy_matmul<T: ComplexField>() -> bool {
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	if const {
		T::IS_NATIVE_F64
			|| T::IS_NATIVE_F32
			|| T::IS_NATIVE_C64
			|| T::IS_NATIVE_C32
	} {
		if std::is_x86_feature_detected!("avx2")
			&& std::is_x86_feature_detected!("fma")
		{
			return true;
		}
	}
	false
}
pub fn spicy_matmul_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	depth: usize,
	gather: bool,
	diag: bool,
) -> StackReq {
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	if const {
		T::IS_NATIVE_F64
			|| T::IS_NATIVE_F32
			|| T::IS_NATIVE_C64
			|| T::IS_NATIVE_C32
	} {
		if std::is_x86_feature_detected!("avx2")
			&& std::is_x86_feature_detected!("fma")
		{
			return StackReq::EMPTY;
		}
	}
	spicy_matmul_generic_scratch::<T>(nrows, ncols, depth, gather, diag)
}
/// The scratch of [`spicy_matmul_generic`].
fn spicy_matmul_generic_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	depth: usize,
	gather: bool,
	diag: bool,
) -> StackReq {
	let diag = temp_mat_scratch::<T>(nrows, if diag { depth } else { 0 });
	let gather = temp_mat_scratch::<T>(nrows, if gather { ncols } else { 0 });
	diag.and(gather)
}
pub fn spicy_matmul<I: Index, T: ComplexField>(
	C: MatMut<'_, T>,
	C_block: BlockStructure,
	row_idx: Option<&[I]>,
	col_idx: Option<&[I]>,
	beta: Accum,
	A: MatRef<'_, T>,
	conj_A: Conj,
	B: MatRef<'_, T>,
	conj_B: Conj,
	D: Option<DiagRef<'_, T>>,
	alpha: T,
	par: Par,
	stack: &mut MemStack,
) {
	let mut C = C;
	assert!(all(
		A.ncols() == B.nrows(),
		A.nrows() == row_idx.map(|idx| idx.len()).unwrap_or(C.nrows()),
		B.ncols() == col_idx.map(|idx| idx.len()).unwrap_or(C.ncols()),
	));
	let nrows = A.nrows();
	let ncols = B.ncols();
	let depth = A.ncols();
	if nrows == 0 || ncols == 0 {
		return;
	}
	// One section; the unit is a multiply-add at the GEMM rate of the dtype. A team token
	// that can run the x86 kernel decides below, with the kernel's own two sections and the
	// width `width_for` chooses; the one-section gate at the full degree is applied to it only
	// where there is no kernel (`team_gate_pending`).
	let units = (nrows * ncols).saturating_mul(depth);
	#[cfg(feature = "threadteam")]
	let team_gate_pending = matches!(&par, Par::Team(_)) && par.degree() >= 2;
	#[cfg(not(feature = "threadteam"))]
	let team_gate_pending = false;
	// a token that is not a team is sequential: nothing to decide before the kernel
	let mut par = par;
	if let Some(row_idx) = row_idx {
		for &i in row_idx {
			assert!(i.zx() < C.nrows());
		}
	}
	if let Some(col_idx) = col_idx {
		for &j in col_idx {
			assert!(j.zx() < C.ncols());
		}
	}
	if depth == 0 {
		if beta == Accum::Replace {
			let diagonal = matches!(C_block.diag_kind(), DiagonalKind::Generic);
			for j in 0..ncols {
				for i in 0..nrows {
					if C_block.is_dense()
						|| (C_block.is_lower() && i > j)
						|| (C_block.is_upper() && i < j)
						|| (diagonal && i == j)
					{
						C[(
							row_idx.map_or(i, |rows| rows[i].zx()),
							col_idx.map_or(j, |cols| cols[j].zx()),
						)] = T::zero();
					}
				}
			}
		}
		return;
	}
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	if const {
		T::IS_NATIVE_F64
			|| T::IS_NATIVE_F32
			|| T::IS_NATIVE_C64
			|| T::IS_NATIVE_C32
	} {
		use private_gemm_x86::*;
		let feat = if std::arch::is_x86_feature_detected!("avx512f") {
			Some(InstrSet::Avx512)
		} else if std::arch::is_x86_feature_detected!("avx2")
			&& std::arch::is_x86_feature_detected!("fma")
		{
			Some(InstrSet::Avx256)
		} else {
			None
		};
		if let Some(feat) = feat {
			let mut C = C;
			let mut A = A;
			let mut B = B;
			let mut row_idx = row_idx;
			let mut col_idx = col_idx;
			if matches!(
				C_block,
				BlockStructure::StrictTriangularLower
					| BlockStructure::UnitTriangularLower
			) {
				if nrows == 0 {
					return;
				}
				A = A.get(1.., ..);
				if let Some(row_idx) = &mut row_idx {
					*row_idx = &row_idx[1..];
				} else {
					C = C.get_mut(1.., ..);
				}
			}
			if matches!(
				C_block,
				BlockStructure::StrictTriangularUpper
					| BlockStructure::UnitTriangularUpper
			) {
				if ncols == 0 {
					return;
				}
				B = B.get(.., 1..);
				if let Some(col_idx) = &mut col_idx {
					*col_idx = &col_idx[1..];
				} else {
					C = C.get_mut(.., 1..);
				}
			}
			let nrows = A.nrows();
			let ncols = B.ncols();
			if nrows == 0 || ncols == 0 {
				return;
			}
			macro_rules! go {
				($call:path, $last:expr) => {
					$call(
						const {
							if T::IS_NATIVE_F64 {
								DType::F64
							} else if T::IS_NATIVE_F32 {
								DType::F32
							} else if T::IS_NATIVE_C64 {
								DType::C64
							} else {
								DType::C32
							}
						},
						const {
							if size_of::<I>() == 4 {
								IType::U32
							} else if size_of::<I>() == 8 {
								IType::U64
							} else {
								unreachable!()
							}
						},
						feat,
						nrows,
						ncols,
						depth,
						C.as_ptr_mut() as _,
						C.row_stride(),
						C.col_stride(),
						row_idx
							.map(|idx| idx.as_ptr())
							.unwrap_or(core::ptr::null()) as _,
						col_idx
							.map(|idx| idx.as_ptr())
							.unwrap_or(core::ptr::null()) as _,
						match C_block {
							BlockStructure::Rectangular => DstKind::Full,
							BlockStructure::TriangularLower => DstKind::Lower,
							BlockStructure::StrictTriangularLower => {
								DstKind::Lower
							},
							BlockStructure::UnitTriangularLower => {
								DstKind::Lower
							},
							BlockStructure::TriangularUpper => DstKind::Upper,
							BlockStructure::StrictTriangularUpper => {
								DstKind::Upper
							},
							BlockStructure::UnitTriangularUpper => {
								DstKind::Upper
							},
						},
						match beta {
							crate::Accum::Replace => Accum::Replace,
							crate::Accum::Add => Accum::Add,
						},
						A.as_ptr() as _,
						A.row_stride(),
						A.col_stride(),
						conj_A == Conj::Yes,
						D.map(|D| D.column_vector().as_ptr() as _)
							.unwrap_or(core::ptr::null()),
						D.map(|D| D.column_vector().row_stride()).unwrap_or(0),
						B.as_ptr() as _,
						B.row_stride(),
						B.col_stride(),
						conj_B == Conj::Yes,
						&raw const alpha as _,
						$last,
					)
				};
			}
			// a `Team` token runs the kernel on its own members (`Fork` of the team), gated by the
			// kernel's two sections at the width `width_for` chooses; below the gate the kernel
			// runs on the calling thread. It never hands its size to a team.
			#[cfg(feature = "threadteam")]
			if let Par::Team(team) = par {
				let macs = nrows.saturating_mul(ncols).saturating_mul(depth);
				let macs = if matches!(C_block, BlockStructure::Rectangular) {
					macs
				} else {
					macs / 2
				};
				let estimate = crate::utils::cost::serial_time(
					macs,
					crate::utils::cost::rate::gemm_mac().of::<T>(),
				);
				match crate::utils::team::kernel::kernel_width(&team, estimate)
				{
					Some(width) => crate::utils::team::kernel::with_fork(
						team,
						width,
						|fork| unsafe {
							go!(private_gemm_x86::gemm_with, fork)
						},
					),
					None => unsafe { go!(private_gemm_x86::gemm, 1) },
				}
				return;
			}
			unsafe {
				go!(private_gemm_x86::gemm, par.degree());
			}
			return;
		}
	}
	// no kernel for a team here: the one-section gate of the generic path, at the widest width
	// that pays; a narrower width reruns the call on a sub-team of that width, where the same
	// decision then takes the whole sub-team
	if team_gate_pending {
		let width = crate::utils::cost::par_width::<T>(
			&par,
			units,
			crate::utils::cost::rate::gemm_mac(),
			1,
		);
		if width < par.degree() {
			return crate::utils::thread::with_width(par, width, |par| {
				spicy_matmul(
					C, C_block, row_idx, col_idx, beta, A, conj_A, B, conj_B, D, alpha, par, stack,
				)
			});
		}
	}
	spicy_matmul_generic(
		C, C_block, row_idx, col_idx, beta, A, conj_A, B, conj_B, D, alpha, par, stack,
	);
}

/// The product without a native kernel: every target other than x86_64 with AVX2 and FMA, and
/// every non-native scalar type. It needs the scratch of [`spicy_matmul_scratch`] for the gather
/// output and the scaled `A`. Separate from [`spicy_matmul`] so that its tests run on every host.
fn spicy_matmul_generic<I: Index, T: ComplexField>(
	mut C: MatMut<'_, T>,
	C_block: BlockStructure,
	row_idx: Option<&[I]>,
	col_idx: Option<&[I]>,
	beta: Accum,
	A: MatRef<'_, T>,
	conj_A: Conj,
	B: MatRef<'_, T>,
	conj_B: Conj,
	D: Option<DiagRef<'_, T>>,
	alpha: T,
	mut par: Par,
	stack: &mut MemStack,
) {
	let nrows = A.nrows();
	let ncols = B.ncols();
	let depth = A.ncols();
	let (mut out, stack) = unsafe {
		temp_mat_uninit::<T, _, _>(
			nrows,
			if row_idx.is_some() || col_idx.is_some() {
				ncols
			} else {
				0
			},
			stack,
		)
	};
	let mut out = out.as_mat_mut();
	let (mut scaled, stack) = unsafe {
		temp_mat_uninit::<T, _, _>(
			nrows,
			if D.is_some() { depth } else { 0 },
			stack,
		)
	};
	let mut scaled = scaled.as_mat_mut();
	let _ = stack;
	let A = if let Some(D) = D {
		for k in 0..depth {
			let ref d = D[k].real();
			zip!(scaled.rb_mut().col_mut(k), A.col(k))
				.for_each(|unzip!(x, y)| *x = y.mul_real(d));
		}
		scaled.rb()
	} else {
		A
	};
	{
		let mut C = if row_idx.is_some() || col_idx.is_some() {
			out.rb_mut()
		} else {
			C.rb_mut()
		};
		let size = Ord::min(nrows, ncols);
		if C_block.is_dense() {
			linalg::matmul::triangular::matmul_with_conj(
				C,
				C_block,
				if row_idx.is_some() || col_idx.is_some() {
					Accum::Replace
				} else {
					beta
				},
				A,
				BlockStructure::Rectangular,
				conj_A,
				B,
				BlockStructure::Rectangular,
				conj_B,
				alpha,
				par,
			);
		} else {
			linalg::matmul::triangular::matmul_with_conj(
				C.rb_mut().get_mut(..size, ..size),
				C_block,
				if row_idx.is_some() || col_idx.is_some() {
					Accum::Replace
				} else {
					beta
				},
				A.get(..size, ..),
				BlockStructure::Rectangular,
				conj_A,
				B.get(.., ..size),
				BlockStructure::Rectangular,
				conj_B,
				alpha.copy(),
				par.reborrow(),
			);
			if C_block.is_lower() && nrows > ncols {
				linalg::matmul::matmul_with_conj(
					C.rb_mut().get_mut(size.., ..size),
					if row_idx.is_some() || col_idx.is_some() {
						Accum::Replace
					} else {
						beta
					},
					A.get(size.., ..),
					conj_A,
					B.get(.., ..size),
					conj_B,
					alpha,
					par,
				);
			} else if C_block.is_upper() && ncols > nrows {
				// the columns right of the square part are strictly upper: a lower block
				// must not write them
				linalg::matmul::matmul_with_conj(
					C.rb_mut().get_mut(..size, size..),
					if row_idx.is_some() || col_idx.is_some() {
						Accum::Replace
					} else {
						beta
					},
					A.get(..size, ..),
					conj_A,
					B.get(.., size..),
					conj_B,
					alpha,
					par,
				);
			}
		}
	}
	let lower = C_block.is_lower();
	let upper = C_block.is_upper();
	let diag = matches!(C_block.diag_kind(), DiagonalKind::Generic) as usize;
	// a rectangular block is neither lower nor upper: every gathered entry is stored
	let dense = C_block.is_dense();
	match (row_idx, col_idx) {
		(Some(row_idx), Some(col_idx)) => match beta {
			Accum::Replace => {
				for (j, &jj) in col_idx.iter().enumerate() {
					for (i, &ii) in row_idx.iter().enumerate() {
						if dense || (lower && j < i + diag) || (upper && i < j + diag) {
							C[(ii.zx(), jj.zx())] = out[(i, j)].copy();
						}
					}
				}
			},
			Accum::Add => {
				for (j, &jj) in col_idx.iter().enumerate() {
					for (i, &ii) in row_idx.iter().enumerate() {
						if dense || (lower && j < i + diag) || (upper && i < j + diag) {
							C[(ii.zx(), jj.zx())] += &out[(i, j)];
						}
					}
				}
			},
		},
		(Some(row_idx), None) => match beta {
			Accum::Replace => {
				for j in 0..ncols {
					for (i, &ii) in row_idx.iter().enumerate() {
						if dense || (lower && j < i + diag) || (upper && i < j + diag) {
							C[(ii.zx(), j)] = out[(i, j)].copy();
						}
					}
				}
			},
			Accum::Add => {
				for j in 0..ncols {
					for (i, &ii) in row_idx.iter().enumerate() {
						if dense || (lower && j < i + diag) || (upper && i < j + diag) {
							C[(ii.zx(), j)] += &out[(i, j)];
						}
					}
				}
			},
		},
		(None, Some(col_idx)) => match beta {
			Accum::Replace => {
				for (j, &jj) in col_idx.iter().enumerate() {
					for i in 0..nrows {
						if dense || (lower && j < i + diag) || (upper && i < j + diag) {
							C[(i, jj.zx())] = out[(i, j)].copy();
						}
					}
				}
			},
			Accum::Add => {
				for (j, &jj) in col_idx.iter().enumerate() {
					for i in 0..nrows {
						if dense || (lower && j < i + diag) || (upper && i < j + diag) {
							C[(i, jj.zx())] += &out[(i, j)];
						}
					}
				}
			},
		},
		(None, None) => {},
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The product through [`spicy_matmul`], or directly through the generic path, which
	/// x86_64 with AVX2 and FMA otherwise never reaches for a native type. The scratch is the
	/// route's own request; the generic path takes its temporaries from it.
	fn product<T: ComplexField>(
		generic: bool,
		C: MatMut<'_, T>,
		C_block: BlockStructure,
		row_idx: Option<&[usize]>,
		col_idx: Option<&[usize]>,
		beta: Accum,
		A: MatRef<'_, T>,
		conj_A: Conj,
		B: MatRef<'_, T>,
		conj_B: Conj,
		D: Option<DiagRef<'_, T>>,
		alpha: T,
	) {
		let (m, n, depth) = (A.nrows(), B.ncols(), A.ncols());
		let gather = row_idx.is_some() || col_idx.is_some();
		let req = if generic {
			spicy_matmul_generic_scratch::<T>(m, n, depth, gather, D.is_some())
		} else {
			spicy_matmul_scratch::<T>(m, n, depth, gather, D.is_some())
		};
		let mut mem = dyn_stack::MemBuffer::new(req);
		let stack = MemStack::new(&mut mem);
		if generic {
			spicy_matmul_generic(
				C, C_block, row_idx, col_idx, beta, A, conj_A, B, conj_B, D, alpha, Par::Seq,
				stack,
			);
		} else {
			spicy_matmul(
				C, C_block, row_idx, col_idx, beta, A, conj_A, B, conj_B, D, alpha, Par::Seq,
				stack,
			);
		}
	}

	#[test]
	fn empty_strict_triangle_preserves_parent_guards() {
		for generic in [false, true] {
			empty_strict_triangle_preserves_parent_guards_on(generic);
		}
	}

	fn empty_strict_triangle_preserves_parent_guards_on(generic: bool) {
		let a = Mat::from_fn(2, 1, |i, _| 2.0_f64 + i as f64);
		let b = Mat::from_fn(1, 2, |_, j| 4.0_f64 + j as f64);
		for block in [
			BlockStructure::StrictTriangularLower,
			BlockStructure::UnitTriangularLower,
			BlockStructure::StrictTriangularUpper,
			BlockStructure::UnitTriangularUpper,
		] {
			let mut c = Mat::from_fn(2, 2, |_, _| 7.0_f64);
			let expected = c.clone();
			product::<f64>(
				generic,
				c.as_mut().submatrix_mut(0, 0, 1, 1),
				block,
				None,
				None,
				Accum::Replace,
				a.as_ref().submatrix(0, 0, 1, 1),
				Conj::No,
				b.as_ref().submatrix(0, 0, 1, 1),
				Conj::No,
				None,
				1.0,
			);
			std::assert_eq!(c, expected);
		}
	}

	fn check_strict_triangles<T: ComplexField<Real = f64>>(
		scalar: impl Fn(f64, f64) -> T,
		depths: &[usize],
	) {
		for generic in [false, true] {
			check_strict_triangles_on(generic, &scalar, depths);
		}
	}

	fn check_strict_triangles_on<T: ComplexField<Real = f64>>(
		generic: bool,
		scalar: &impl Fn(f64, f64) -> T,
		depths: &[usize],
	) {
		for (m, n) in [
			(0, 3),
			(3, 0),
			(1, 1),
			(3, 3),
			(1, 3),
			(3, 1),
			(2, 4),
			(4, 2),
		] {
			for &depth in depths {
				let a = Mat::from_fn(m + 1, depth.max(1), |i, k| {
					scalar(0.5 + i as f64, 0.25 + k as f64)
				});
				let b = Mat::from_fn(depth.max(1), n + 1, |k, j| {
					scalar(1.0 + j as f64, -0.5 - k as f64)
				});
				let d =
					Mat::from_fn(depth, 1, |k, _| scalar(1.25 + k as f64, 0.0));
				let rows: Vec<usize> = (0..m).map(|i| m - i).collect();
				let cols: Vec<usize> = (0..n).map(|j| n - j).collect();
				for block in [
					BlockStructure::Rectangular,
					BlockStructure::TriangularLower,
					BlockStructure::TriangularUpper,
					BlockStructure::StrictTriangularLower,
					BlockStructure::UnitTriangularLower,
					BlockStructure::StrictTriangularUpper,
					BlockStructure::UnitTriangularUpper,
				] {
					for (ri, ci) in [
						(false, false),
						(true, false),
						(false, true),
						(true, true),
					] {
						for beta in [Accum::Replace, Accum::Add] {
							let mut c = Mat::from_fn(m + 2, n + 2, |_, _| {
								scalar(7.0, 2.0)
							});
							let mut expected = c.clone();
							let alpha = scalar(0.75, -0.25);
							for j in 0..n {
								for i in 0..m {
									let stored = match block {
										BlockStructure::Rectangular => true,
										BlockStructure::TriangularLower => i >= j,
										BlockStructure::TriangularUpper => i <= j,
										BlockStructure::StrictTriangularLower | BlockStructure::UnitTriangularLower => i > j,
										BlockStructure::StrictTriangularUpper | BlockStructure::UnitTriangularUpper => i < j,
									};
									if stored {
										let mut sum = T::zero();
										for k in 0..depth {
											sum += a[(i, k)]
												.conj()
												.mul_real(&d[(k, 0)].real())
												* b[(k, j)].clone();
										}
										let dst = &mut expected[(
											if ri { rows[i] } else { i },
											if ci { cols[j] } else { j },
										)];
										let value = alpha.clone() * sum;
										if beta == Accum::Replace {
											*dst = value;
										} else {
											*dst += value;
										}
									}
								}
							}
							product::<T>(
								generic,
								c.as_mut().submatrix_mut(
									0,
									0,
									if ri { m + 1 } else { m },
									if ci { n + 1 } else { n },
								),
								block,
								ri.then_some(rows.as_slice()),
								ci.then_some(cols.as_slice()),
								beta,
								a.as_ref().submatrix(0, 0, m, depth),
								Conj::Yes,
								b.as_ref().submatrix(0, 0, depth, n),
								Conj::No,
								Some(d.as_ref().col(0).as_diagonal()),
								alpha,
							);
							std::assert!(c.as_ref().is_all_finite());
							let error =
								(c.as_ref() - expected.as_ref()).norm_max();
							std::assert!(
								error < 1e-10,
								"generic={generic} block={block:?} m={m} n={n} depth={depth} lower={} ri={ri} ci={ci} replace={} real={} error={error}; c={c:?}; expected={expected:?}",
								block.is_lower(),
								beta == Accum::Replace,
								T::IS_REAL
							);
						}
					}
				}
			}
		}
	}

	#[test]
	fn strict_triangle_maps_preserve_stored_entries_and_parent_guards() {
		check_strict_triangles::<f64>(|re, _| re, &[2]);
		check_strict_triangles::<c64>(c64::new, &[2]);
	}

	#[test]
	fn strict_triangle_zero_depth_preserves_replace_semantics() {
		check_strict_triangles::<f64>(|re, _| re, &[0]);
		check_strict_triangles::<c64>(c64::new, &[0]);
	}

	#[test]
	fn strict_triangle_preserves_nonempty_parent_guards() {
		let a = Mat::from_fn(4, 1, |i, _| 2.0_f64 + i as f64);
		let b = Mat::from_fn(1, 4, |_, j| 4.0_f64 + j as f64);
		for generic in [false, true] {
			let mut c = Mat::from_fn(4, 4, |_, _| 7.0_f64);
			product::<f64>(
				generic,
				c.as_mut().submatrix_mut(0, 0, 3, 3),
				BlockStructure::StrictTriangularLower,
				None,
				None,
				Accum::Replace,
				a.as_ref().submatrix(0, 0, 3, 1),
				Conj::No,
				b.as_ref().submatrix(0, 0, 1, 3),
				Conj::No,
				None,
				1.0,
			);
			for j in 0..4 {
				std::assert_eq!(c[(3, j)], 7.0);
			}
			// the strict upper part of the 3 x 3 block stays as it was
			for j in 0..3 {
				for i in 0..j + 1 {
					std::assert_eq!(c[(i, j)], 7.0, "generic={generic} ({i}, {j})");
				}
			}
		}
	}
}
#[cfg(test)]
mod reversed_depth_tests {
	use super::*;
	use crate::{Col, Mat};

	/// `C += A D B` with `A`'s columns and `B`'s rows stored in either direction, into a
	/// column-major and a row-major `C`, against the plain sum: the x86 kernel reversed a negative
	/// depth stride without reversing `D` (#113).
	#[test]
	fn a_diagonal_product_reads_the_depth_in_any_direction() {
		let (m, k) = (61usize, 5usize);
		let a = Mat::<f64>::from_fn(m, k, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
		let b = Mat::<f64>::from_fn(k, m, |i, j| ((i * 5 + j * 2) % 13) as f64 - 6.0);
		let d = Col::<f64>::from_fn(k, |i| 1.0 + i as f64);
		let mut want = Mat::<f64>::zeros(m, m);
		for i in 0..m {
			for j in 0..m {
				want[(i, j)] = (0..k).map(|t| a[(i, t)] * d[t] * b[(t, j)]).sum::<f64>();
			}
		}
		// the operands stored with their depth reversed: `view` turns them back, so the logical
		// matrices are `a` and `b`, read through a negative depth stride
		let a_rev = Mat::<f64>::from_fn(m, k, |i, j| a[(i, k - 1 - j)]);
		let b_rev = Mat::<f64>::from_fn(k, m, |i, j| b[(k - 1 - i, j)]);
		for a_reversed in [false, true] {
			for b_reversed in [false, true] {
				for row_major in [false, true] {
					let av = if a_reversed { a_rev.as_ref().reverse_cols() } else { a.as_ref() };
					let bv = if b_reversed { b_rev.as_ref().reverse_rows() } else { b.as_ref() };
					let mut buf = Mat::<f64>::zeros(m, m);
					let c = if row_major { buf.as_mut().transpose_mut() } else { buf.as_mut() };
					spicy_matmul::<usize, f64>(
						c,
						BlockStructure::Rectangular,
						None,
						None,
						crate::Accum::Add,
						av,
						Conj::No,
						bv,
						Conj::No,
						Some(d.as_ref().as_diagonal()),
						1.0,
						Par::Seq,
						MemStack::new(&mut dyn_stack::MemBuffer::new(spicy_matmul_scratch::<f64>(m, m, k, false, true))),
					);
					let got = if row_major { buf.transpose().to_owned() } else { buf };
					// integer-valued operands: exact, and a NaN fails the comparison
					for i in 0..m {
						for j in 0..m {
							core::assert!(
								got[(i, j)] == want[(i, j)],
								"a reversed {a_reversed}, b reversed {b_reversed}, row major {row_major}: ({i}, {j}) {} against {}",
								got[(i, j)],
								want[(i, j)],
							);
						}
					}
				}
			}
		}
	}
}
