use crate::internal_prelude::*;
/// jacobi rotation matrix
///
/// $$ \begin{bmatrix} c & -\bar s \\\\ s & c \end{bmatrix} $$
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct JacobiRotation<T> {
	/// cosine
	pub c: T,
	/// sine
	pub s: T,
}
#[allow(dead_code)]
impl<T: ComplexField> JacobiRotation<T> {
	#[doc(hidden)]
	pub fn make_givens(p: T, q: T) -> Self
	where
		T: RealField,
	{
		if q == zero::<T>() {
			let c = if p < zero::<T>() {
				-one::<T>()
			} else {
				one::<T>()
			};
			let s = zero::<T>();
			Self { c, s }
		} else if p == zero::<T>() {
			let c = zero::<T>();
			let s = if q < zero::<T>() {
				one::<T>()
			} else {
				-one::<T>()
			};
			Self { c, s }
		} else if p.abs() > q.abs() {
			let t = q / &p;
			let mut u = t.hypot(one::<T>());
			if p < zero::<T>() {
				u = -u;
			}
			let c = u.recip();
			let s = -t * &c;
			Self { c, s }
		} else {
			let t = p / &q;
			let mut u = t.hypot(one::<T>());
			if q < zero::<T>() {
				u = -u;
			}
			let s = -u.recip();
			let c = -t * &s;
			Self { c, s }
		}
	}

	#[doc(hidden)]
	pub fn rotg(p: T, q: T) -> (Self, T) {
		if const { T::IS_REAL } {
			let p = p.real();
			let q = q.real();
			if q == zero::<T::Real>() {
				let c = one::<T::Real>();
				let s = zero::<T::Real>();
				let c = c.to_cplx();
				let s = s.to_cplx();
				(Self { c, s }, p.to_cplx())
			} else if p == zero::<T::Real>() {
				let c = zero::<T::Real>();
				let s = -one::<T::Real>();
				let c = c.to_cplx();
				let s = s.to_cplx();
				(Self { c, s }, q.to_cplx())
			} else {
				let ref safmin = min_positive::<T::Real>();
				let ref safmax = safmin.recip();
				let ref a = p;
				let ref b = q;
				let ref scl = a.abs().fmax(b.abs()).fmax(safmin).fmin(safmax);
				let r = scl * ((a / scl).abs2() + (b / scl).abs2()).sqrt();
				let ref r = if a.abs() > b.abs() {
					if *a > zero::<T::Real>() { r } else { -r }
				} else {
					if *b > zero::<T::Real>() { r } else { -r }
				};
				let c = (a / r).to_cplx();
				let s = (-b / r).to_cplx();
				let r = r.to_cplx();
				(Self { c, s }, r)
			}
		} else {
			let a = p;
			let b = q;
			if b == zero::<T>() {
				// `J = I` maps `[a, 0]` to `[a, 0]`, so the returned `r` is `a`,
				// not the constant one.
				return (
					Self {
						c: one::<T>(),
						s: zero::<T>(),
					},
					a,
				);
			}
			let ref eps = eps::<T::Real>();
			let ref sml = min_positive::<T::Real>();
			let ref big = max_positive::<T::Real>();
			let ref rtmin = (sml / eps).sqrt();
			let ref rtmax = rtmin.recip();
			let (c, s, r);
			if a == zero::<T>() {
				c = zero::<T>();
				let ref g1 = b.real().abs().fmax(b.imag().abs());
				if g1 > rtmin && g1 < rtmax {
					let g2 = b.abs2();
					let d = g2.sqrt();
					s = b.conj().mul_real(d.recip());
					r = d.to_cplx();
				} else {
					let ref u = g1.fmax(sml).fmin(big);
					let ref uu = u.recip();
					let gs = b.mul_real(uu);
					let g2 = gs.abs2();
					let d = g2.sqrt();
					s = gs.conj().mul_real(d.recip());
					r = (d * u).to_cplx();
				}
			} else {
				let ref f1 = a.real().abs().fmax(a.imag().abs());
				let ref g1 = b.real().abs().fmax(b.imag().abs());
				if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
					let ref f2 = a.abs2();
					let ref g2 = b.abs2();
					let ref h2 = f2 + g2;
					let d = if f2 > rtmin && h2 < rtmax {
						(f2 * h2).sqrt()
					} else {
						f2.sqrt() * h2.sqrt()
					};
					let ref p = d.recip();
					c = (f2 * p).to_cplx();
					s = b.conj() * a.mul_real(p);
					r = a.mul_real(h2 * p);
				} else {
					let u = f1.fmax(g1).fmax(sml).fmin(big);
					let ref uu = u.recip();
					let ref gs = b.mul_real(uu);
					let ref g2 = gs.abs2();
					let (f2, h2, w);
					let fs;
					if f1 * uu < *rtmin {
						let v = f1.fmax(sml).fmin(big);
						let vv = v.recip();
						w = v * uu;
						fs = a.mul_real(vv);
						f2 = fs.abs2();
						h2 = (&f2 * &w) * &w + g2;
					} else {
						w = one::<T::Real>();
						fs = a.mul_real(uu);
						f2 = fs.abs2();
						h2 = &f2 + g2;
					}
					let d = if f2 > *rtmin && h2 < *rtmax {
						(&f2 * &h2).sqrt()
					} else {
						f2.sqrt() * h2.sqrt()
					};
					let ref p = d.recip();
					c = ((&f2 * p) * &w).to_cplx();
					s = gs.conj() * fs.mul_real(p);
					r = fs.mul_real(h2 * p).mul_real(u);
				}
			}
			(Self { c, s: -s.conj() }, r)
			// (Self { c, s }.transpose(), r)
			// (Self { c, s }, r)
		}
	}

	/// apply to the given matrix from the left
	///
	/// $$ J \begin{bmatrix} m_{00} & m_{01} \\\\ m_{10} & m_{11} \end{bmatrix}
	/// $$
	#[inline]
	pub fn apply_on_the_left_2x2(
		&self,
		m00: T,
		m01: T,
		m10: T,
		m11: T,
	) -> (T, T, T, T) {
		let Self { c, s } = self;
		(
			&m00 * c - &m10 * s.conj(),
			&m01 * c - &m11 * s.conj(),
			c * m10 + s * m00,
			c * m11 + s * m01,
		)
	}

	/// apply to the given matrix from the right
	///
	/// $$ \begin{bmatrix} m_{00} & m_{01} \\\\ m_{10} & m_{11} \end{bmatrix} J
	/// $$
	#[inline]
	pub fn apply_on_the_right_2x2(
		&self,
		m00: T,
		m01: T,
		m10: T,
		m11: T,
	) -> (T, T, T, T) {
		let (r00, r01, r10, r11) =
			self.transpose().apply_on_the_left_2x2(m00, m10, m01, m11);
		(r00, r10, r01, r11)
	}

	#[inline]
	fn apply_on_the_left_in_place_impl<'N>(
		&self,
		(x, y): (RowMut<'_, T, Dim<'N>>, RowMut<'_, T, Dim<'N>>),
	) {
		let mut x = x;
		let mut y = y;
		if const { T::SIMD_CAPABILITIES.is_simd() } {
			struct Impl<'a, 'N, T: ComplexField> {
				this: &'a JacobiRotation<T>,
				x: RowMut<'a, T, Dim<'N>, ContiguousFwd>,
				y: RowMut<'a, T, Dim<'N>, ContiguousFwd>,
			}
			impl<T: ComplexField> pulp::WithSimd for Impl<'_, '_, T> {
				type Output = ();

				#[inline(always)]
				fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
					let Self { this, x, y } = self;
					let simd = SimdCtx::new(T::simd_ctx(simd), x.ncols());
					this.apply_on_the_left_in_place_with_simd(simd, x, y);
				}
			}
			if x.col_stride() < 0 && y.col_stride() < 0 {
				x = x.reverse_cols_mut();
				y = y.reverse_cols_mut();
			}
			if let (Some(x), Some(y)) = (
				x.rb_mut().try_as_row_major_mut(),
				y.rb_mut().try_as_row_major_mut(),
			) {
				dispatch!(Impl { this: self, x, y }, Impl, T);
				return;
			}
		}
		self.apply_on_the_left_in_place_fallback(x, y);
	}

	/// apply from the left to $x$ and $y$
	#[inline]
	pub fn apply_on_the_left_in_place<N: Shape>(
		&self,
		(x, y): (RowMut<'_, T, N>, RowMut<'_, T, N>),
	) {
		with_dim!(N, x.ncols().unbound());
		self.apply_on_the_left_in_place_impl((
			x.as_col_shape_mut(N),
			y.as_col_shape_mut(N),
		));
	}

	/// apply from the right to $x$ and $y$
	#[inline]
	pub fn apply_on_the_right_in_place<N: Shape>(
		&self,
		(x, y): (ColMut<'_, T, N>, ColMut<'_, T, N>),
	) {
		with_dim!(N, x.nrows().unbound());
		let x = x.as_row_shape_mut(N);
		let y = y.as_row_shape_mut(N);
		self.transpose()
			.apply_on_the_left_in_place((x.transpose_mut(), y.transpose_mut()));
	}

	#[inline(never)]
	fn apply_on_the_left_in_place_fallback<'N>(
		&self,
		x: RowMut<'_, T, Dim<'N>>,
		y: RowMut<'_, T, Dim<'N>>,
	) {
		let Self { c, s } = self;
		zip!(x, y).for_each(move |unzip!(x, y)| {
			let x_ = c * &*x - conj(s) * &*y;
			let y_ = c * &*y + s * &*x;
			*x = x_;
			*y = y_;
		});
	}

	#[inline(always)]
	pub(crate) fn apply_on_the_right_in_place_with_simd<'N, S: pulp::Simd>(
		&self,
		simd: SimdCtx<'N, T, S>,
		x: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
		y: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
	) {
		self.adjoint().apply_on_the_left_in_place_with_simd(
			simd,
			x.transpose_mut(),
			y.transpose_mut(),
		);
	}

	#[inline(always)]
	pub(crate) fn apply_on_the_left_in_place_with_simd<'N, S: pulp::Simd>(
		&self,
		simd: SimdCtx<'N, T, S>,
		x: RowMut<'_, T, Dim<'N>, ContiguousFwd>,
		y: RowMut<'_, T, Dim<'N>, ContiguousFwd>,
	) {
		let Self { c, s } = self;
		if *c == one::<T>() && *s == zero::<T>() {
			return;
		}
		let mut x = x.transpose_mut();
		let mut y = y.transpose_mut();
		let c = simd.splat_real(c.real());
		let s = simd.splat(s);
		let indices = simd.indices();
		simd_iter!(for i in [indices] {
			let mut xx = simd.read(x.rb(), i);
			let mut yy = simd.read(y.rb(), i);
			(xx, yy) = (
				simd.conj_mul_add(simd.neg(s), yy, simd.mul_real(xx, c)),
				simd.mul_add(s, xx, simd.mul_real(yy, c)),
			);
			simd.write(x.rb_mut(), i, xx);
			simd.write(y.rb_mut(), i, yy);
		});
	}

	/// returns the transpose of `self`
	#[inline]
	pub fn transpose(&self) -> Self {
		Self {
			c: self.c.copy(),
			s: -self.s.conj(),
		}
	}

	/// returns the conjugate of `self`
	#[inline]
	pub fn conjugate(&self) -> Self {
		Self {
			c: self.c.copy(),
			s: self.s.conj(),
		}
	}

	/// returns the adjoint of `self`
	#[inline]
	pub fn adjoint(&self) -> Self {
		Self {
			c: self.c.copy(),
			s: -&self.s,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{assert, c64};

	fn assert_close(actual: c64, expected: c64) {
		let tol = 1e-14 * actual.abs().max(expected.abs()).max(1.0);
		if (actual - expected).abs() > tol {
			panic!("actual={actual:?} expected={expected:?}");
		}
	}

	/// `rotg(p, 0)` must return the rotation that leaves `p` alone and report
	/// `r == p`; for complex coefficients the constant one is wrong.
	#[test]
	fn complex_rotg_with_zero_q_returns_p() {
		for (re, im) in [(3.0, 2.0), (-1.5, 0.75), (0.0, 2.0), (0.0, 0.0)] {
			let p = c64::new(re, im);
			let (j, r) = JacobiRotation::rotg(p, c64::new(0.0, 0.0));
			assert_close(r, p);
			// `J * [p, 0] = [r, 0]`
			let (x, y, _, _) = j.apply_on_the_left_2x2(
				p,
				c64::new(0.0, 0.0),
				c64::new(0.0, 0.0),
				c64::new(0.0, 0.0),
			);
			assert_close(x, r);
			assert_close(y, c64::new(0.0, 0.0));
		}
	}

	/// `A * J` has to apply `J` itself, not its conjugate.
	#[test]
	fn complex_apply_on_the_right_2x2_matches_explicit_product() {
		let rotations = [
			JacobiRotation {
				c: c64::new(0.6, 0.0),
				s: c64::new(0.0, 0.8),
			},
			JacobiRotation {
				c: c64::new(0.0, 0.0),
				s: c64::new(1.0, 0.0),
			},
			JacobiRotation {
				c: c64::new(0.6, 0.0),
				s: c64::new(0.3, -0.7),
			},
		];
		let blocks = [
			[
				c64::new(1.0, 0.0),
				c64::new(0.0, 0.0),
				c64::new(0.0, 0.0),
				c64::new(1.0, 0.0),
			],
			[
				c64::new(1.0, 2.0),
				c64::new(-0.5, 0.25),
				c64::new(3.0, -1.0),
				c64::new(0.75, 0.0),
			],
		];
		for j in rotations {
			// `J = [[c, -conj(s)], [s, c]]` by the convention of
			// `apply_on_the_left_2x2`.
			let jm = [[j.c, -j.s.conj()], [j.s, j.c]];
			for [m00, m01, m10, m11] in blocks {
				let (r00, r01, r10, r11) =
					j.apply_on_the_right_2x2(m00, m01, m10, m11);
				let expected = [
					[
						m00 * jm[0][0] + m01 * jm[1][0],
						m00 * jm[0][1] + m01 * jm[1][1],
					],
					[
						m10 * jm[0][0] + m11 * jm[1][0],
						m10 * jm[0][1] + m11 * jm[1][1],
					],
				];
				assert_close(r00, expected[0][0]);
				assert_close(r01, expected[0][1]);
				assert_close(r10, expected[1][0]);
				assert_close(r11, expected[1][1]);
			}
		}
	}

	/// The real path is unaffected by the complex-only fix.
	#[test]
	fn real_apply_on_the_right_2x2_matches_explicit_product() {
		let j = JacobiRotation {
			c: 0.6 as f64,
			s: 0.8 as f64,
		};
		let (m00, m01, m10, m11) = (1.0, 2.0, 3.0, 4.0);
		let (r00, r01, r10, r11) = j.apply_on_the_right_2x2(m00, m01, m10, m11);
		assert!((r00 - (m00 * j.c + m01 * j.s)).abs() <= 1e-14);
		assert!((r01 - (-m00 * j.s + m01 * j.c)).abs() <= 1e-14);
		assert!((r10 - (m10 * j.c + m11 * j.s)).abs() <= 1e-14);
		assert!((r11 - (-m10 * j.s + m11 * j.c)).abs() <= 1e-14);
	}
}
