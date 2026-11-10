// Reference BLAS Level-1 CHECK2 (DDOT/SDOT, CDOTC/ZDOTC and CDOTU/ZDOTU) in
// test/{s,d,c,z}blat1.f, distributed in OpenMathLib/OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Source URLs:
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/dblat1.f
// https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/test/zblat1.f
// Original test program: Copyright (c) 1989 Jack Dongarra, Iain Duff, Jeremy
// Du Croz, Sven Hammarling. Netlib BLAS test-program terms:
// faer/COPYING.BLAS.NETLIB. This Rust adaptation is new code under the crate's
// MIT licence.
// Material changes: the source's DX1/DY1 fixture and its DT7/CT6/CT7 expected
// values are replaced by an exact-integer fixture and an explicit sum. The
// length and increment lists are the source's own: `NS = 0, 1, 2, 4` and
// `INCXS = 1, 2, -2, -1` / `INCYS = 1, -2, 1, -2` (`dblat1.f:347-350`,
// `zblat1.f:347-350`). faer has no dot entry point that takes an increment: the
// increment is the view's, so a strided `RowRef`/`ColRef` carries the source's
// `INCX`/`INCY` and a negative increment is the same view built from its last
// element. `?dotc` and `?dotu` are the `Conj::Yes`/`Conj::No` arm of the same
// call.

/// the offset of the `n`-entry body of a view with signed `stride` in its
/// buffer. The buffer carries `n` entries of padding on each side (the source
/// allocates `NMAX*INCMAX` for its strided vectors), so a kernel that ignores
/// the increment and walks the buffer forward reads a sentinel inside the
/// allocation instead of running off it
fn body_offset(n: usize, stride: isize) -> usize {
	n + if n == 0 || stride > 0 {
		0
	} else {
		(n - 1) * stride.unsigned_abs()
	}
}

/// the buffer for `n` entries of the fixture at signed `stride`: the referred
/// entries hold `integer(k, j)`, and every other entry holds the source's rogue
/// value (here NaN), so an increment the kernel ignores reads a sentinel and
/// the result stops being exact
fn strided_buffer(n: usize, stride: isize, j: usize) -> Col<Scalar> {
	let start = body_offset(n, stride) as isize;
	let body = if n == 0 { 1 } else { (n - 1) * stride.unsigned_abs() + 1 };
	Col::<Scalar>::from_fn(2 * n + body, |i| {
		let off = i as isize - start;
		let k = off / stride;
		if n > 0 && off % stride == 0 && k >= 0 && k < n as isize {
			integer(k as usize, j)
		} else {
			nan()
		}
	})
}

fn strided_row(x: &Col<Scalar>, n: usize, stride: isize) -> RowRef<'_, Scalar> {
	// SAFETY: the body ends `n * |stride|` entries into a buffer that carries
	// `n` more entries, so the last referred entry is inside it, and the buffer
	// outlives the view
	unsafe {
		RowRef::from_raw_parts(x.as_ptr().add(body_offset(n, stride)), n, stride)
	}
}

fn strided_col(x: &Col<Scalar>, n: usize, stride: isize) -> ColRef<'_, Scalar> {
	// SAFETY: as in `strided_row`
	unsafe {
		ColRef::from_raw_parts(x.as_ptr().add(body_offset(n, stride)), n, stride)
	}
}

#[test]
fn dot_matches_the_source_stride_sweep() {
	// CHECK2: `?dot` over the driver's length list and its increment pairs.
	// The increments are read as `INCX = INCXS(KI)`, `INCY = INCYS(KI)` in the
	// same loop (`dblat1.f:598-600`, `zblat1.f:517-519`), so the pairs are the
	// four below. Length 0 is the "reads nothing" case, and the two negative
	// increments are the ones a kernel that walks the buffer forward gets
	// wrong. `?dotc` conjugates the left operand in reference BLAS, and faer's
	// two `Conj` flags also cover the mixed arms the source cannot express; for
	// the real types conjugation is the identity, so there only the one arm is
	// the source's check.
	for n in [0, 1, 2, 4, 7] {
		for (incx, incy) in [(1, 1), (2, -2), (-2, 1), (-1, -2)] {
			for (name, conj_lhs, conj_rhs) in [
				("dot", Conj::No, Conj::No),
				("dotc", Conj::Yes, Conj::No),
				("dot conj rhs", Conj::No, Conj::Yes),
				("dot conj both", Conj::Yes, Conj::Yes),
			] {
				if Scalar::IS_REAL
					&& (conj_lhs == Conj::Yes || conj_rhs == Conj::Yes)
				{
					continue;
				}
				let x = strided_buffer(n, incx, 1);
				let y = strided_buffer(n, incy, 2);
				let got = inner_prod(
					strided_row(&x, n, incx),
					conj_lhs,
					strided_col(&y, n, incy),
					conj_rhs,
				);
				let want = (0..n).fold(zero(), |acc, k| {
					acc + conjugated(integer(k, 1), conj_lhs)
						* conjugated(integer(k, 2), conj_rhs)
				});
				crate::assert!(
					same_number(got, want),
					"{name} n={n} incx={incx} incy={incy} \
					 got={:?} want={:?}",
					real(&got),
					real(&want)
				);
			}
		}
	}
}

#[test]
fn dot_reads_every_referenced_entry() {
	// The source's `LDE` compares an operand against its pre-call copy; for a
	// dot product the analogous property is that the result moves when any
	// entry the view refers to moves. One NaN written into a referred entry of
	// either operand must reach the sum as a NaN, and one infinity must leave
	// the sum non-finite. Each runs on stride 1, 2 and -2 views of length 4 and
	// on every conjugation arm, so a kernel that skips a referred entry, or the
	// conjugation of one operand, cannot pass.
	let n = 4;
	for stride in [1isize, 2, -2] {
		for at in 0..n {
			for (conj_lhs, conj_rhs) in [
				(Conj::No, Conj::No),
				(Conj::Yes, Conj::No),
				(Conj::No, Conj::Yes),
				(Conj::Yes, Conj::Yes),
			] {
				if Scalar::IS_REAL
					&& (conj_lhs == Conj::Yes || conj_rhs == Conj::Yes)
				{
					continue;
				}
				for poison_lhs in [true, false] {
					let mut x = strided_buffer(n, stride, 1);
					let mut y = strided_buffer(n, 1, 2);
					let dot = |x: &Col<Scalar>, y: &Col<Scalar>| {
						inner_prod(
							strided_row(x, n, stride),
							conj_lhs,
							strided_col(y, n, 1),
							conj_rhs,
						)
					};
					let finite = |v: &Scalar| {
						real(v).is_finite() && imag(v).is_finite()
					};
					let label = format!(
						"stride={stride} at={at} poison_lhs={poison_lhs} \
						 conj_lhs={conj_lhs:?} conj_rhs={conj_rhs:?}"
					);
					crate::assert!(
						finite(&dot(&x, &y)),
						"{label}: the sentinel is not referred to"
					);
					let buf_stride = if poison_lhs { stride } else { 1 };
					let referred = (body_offset(n, buf_stride) as isize
						+ at as isize * buf_stride)
						as usize;
					let poison = |x: &mut Col<Scalar>,
					               y: &mut Col<Scalar>,
					               v: Scalar| {
						if poison_lhs {
							*x.as_mut().get_mut(referred) = v;
						} else {
							*y.as_mut().get_mut(referred) = v;
						}
					};
					poison(&mut x, &mut y, nan());
					crate::assert!(
						real(&dot(&x, &y)).is_nan(),
						"{label}: a NaN in a referred entry did not reach the \
						 sum"
					);
					poison(&mut x, &mut y, from_f64(f64::INFINITY));
					crate::assert!(
						!finite(&dot(&x, &y)),
						"{label}: an infinity in a referred entry did not \
						 reach the sum"
					);
				}
			}
		}
	}
}
