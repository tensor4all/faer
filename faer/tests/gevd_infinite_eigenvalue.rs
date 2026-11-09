//! faer#324: `gevd_cplx` returned wrong left eigenvectors for a pencil with an
//! infinite eigenvalue, while its eigenvalues and right eigenvectors were right.
//!
//! When the complex QZ iteration's unblocked path (`hessenberg_to_qz_unblocked`,
//! which the default parameters run for `n < max(15, blocking_threshold)`, i.e.
//! `n < 75`, and which the blocked path also calls at the end) finds a zero on
//! the diagonal of `B` at `j == ilo` or below a negligible subdiagonal of `A`,
//! it chases that zero to the bottom with rotations applied to rows
//! `jch, jch + 1` of `A` and `B`. `Q` must then take the adjoint rotation on
//! columns `jch, jch + 1`; it rotated rows instead. `Q` stayed unitary but
//! `A = Q S Z^H` no longer held, so every left eigenvector `Q * y` was wrong.
//! `Z` was untouched, so the right eigenvectors were fine.
//!
//! Whether the iteration takes that branch depends on rounding: the issue's
//! `c32` pencil takes it on x86-64 with AVX2/FMA and not on aarch64 or on
//! x86-64 without AVX, so this test guards the x86-64 CI path. The branch
//! itself is covered on every target by
//! `linalg::gevd::qz_cplx::tests::unblocked_zero_chase_updates_q_columns`,
//! which hands the unblocked iteration an exact zero on the diagonal of `B`.
//!
//! The test checks, for every column, the relative residuals
//! `||beta u^H A - alpha u^H B|| / (||u|| (|beta| ||A|| + |alpha| ||B||))` and
//! `||beta A v - alpha B v|| / (||v|| (|beta| ||A|| + |alpha| ||B||))`,
//! computed in `c64`, against `10 * n * eps` of the scalar type. A `NaN`
//! residual or `beta` fails the test instead of being dropped by `f64::max`.
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{gevd_cplx, gevd_scratch};
use faer::traits::ComplexField;
use faer::{Mat, Par, Scale, c32, c64};

/// `A` and `B` of faer#324, 12 x 12 column-major `c32` as `(re, im)` f32 bit
/// patterns. Made in Julia by `rng = Xoshiro(23)`, `U` and `V` the Q factors of
/// `randn(rng, ComplexF32, 12, 12)`, `A = randn(rng, ComplexF32, 12, 12)` and
/// `B = U * Diagonal([ones(Float32, 11); 0]) * V'`.
const A_BITS: [u32; 288] = [
	0xbf4f6163, 0xbfaf8d0f, 0xbf284eb8, 0x3e5450f6, 0x3f792599, 0x3f468b81,
	0xbfd6e280, 0x3e5ac4dd, 0x3e801a4f, 0x3ed3389f, 0x3f3b7e67, 0x3ee2cce8,
	0xbf18d76b, 0xbfec8d86, 0x3dfade19, 0x3ea50f5a, 0xbe51783e, 0x3ebe98a9,
	0xbec60d56, 0xbfcd7c07, 0xbf391480, 0xbcdd4260, 0x3f2eb1f3, 0x3edba827,
	0x3eb9a75f, 0x3eabe882, 0x3ef40f6c, 0x3f170c99, 0xbf469536, 0xbd56f76f,
	0x3ea18fa1, 0xbf3f9766, 0x3f39edcf, 0x3f2b1e16, 0xbf8cc624, 0xbf00995e,
	0xbf848bb3, 0x3e452b55, 0x3f07d692, 0xbf15e47f, 0xbf78c96b, 0x3dac89b0,
	0x3f7b5738, 0xbdbf1b5d, 0xbee59f63, 0x3f126a87, 0xbf10c015, 0x40010505,
	0xbf3eb471, 0xbf0e8290, 0xbedcd6f8, 0x3f8fd58d, 0xbf9e60eb, 0xbec11184,
	0xbf84f3ec, 0xbf86c7a2, 0x3f32ee92, 0xbe982796, 0x3ebdfbfc, 0xbdd556d5,
	0x3f813d6e, 0xbe680d0d, 0xbf8b6f2f, 0x3ed8a27d, 0xbef1f445, 0xbf228e89,
	0xbe0dbcda, 0xbe5c7b1e, 0x3dcd9628, 0x3ed9b4eb, 0xbd8ba4cf, 0xbe5272d0,
	0xbe6f596a, 0xbe0c7a03, 0x3eede3c1, 0x3e97e81b, 0xbef31572, 0xbf089f76,
	0xbed23891, 0xbeb51c5a, 0xbeb03be6, 0x3f98a2cf, 0xbea89963, 0xbed6e037,
	0xbed59ec8, 0x3ea35097, 0xbf20ced1, 0xbfb19680, 0x3d6d6559, 0xbed0f1e4,
	0xbeade97f, 0x3ddf8a4e, 0xbfe078b6, 0x3d60fda8, 0xbf8e0d51, 0x3f363eb8,
	0xbe77a522, 0xbf6625ee, 0xbf4d0900, 0xbdde8b22, 0xbe6923d1, 0x3ddbb595,
	0x3c9e5539, 0xbecf00a1, 0xbc918357, 0xbf9d3f59, 0xbf5e30c5, 0x3f9699f5,
	0x3d27c04e, 0xbee2252a, 0x3f85e4e5, 0xbfc15574, 0x3e4ea842, 0x3f2f37c4,
	0xbf8bce39, 0xbf9317e6, 0xbf3531f5, 0x3f7276ad, 0xbe91378b, 0x3f32558b,
	0x3f2246fb, 0x3e2b5f34, 0xbee5ef32, 0x3c876061, 0xbedd50c2, 0x3f16c641,
	0xbf8ee74f, 0x3df84626, 0x3f4285c0, 0x3f8c59c3, 0xbe96eada, 0x3e116f7f,
	0x3f558f6c, 0xbbc2eb96, 0x3e9dbe69, 0xbf847a15, 0x3ee472ae, 0x3fad3734,
	0x3f51c643, 0x3e886971, 0xbedb1ddd, 0xbefa6b43, 0x3f091bbd, 0xbf73555a,
	0x3f406c28, 0xbe96084c, 0x3e841a5b, 0x4002c851, 0xbf60d371, 0xbf879afb,
	0x3f0316a2, 0x3d39df0b, 0x3eb84e69, 0x3e8d8780, 0x3e1fc026, 0xbe67b5f9,
	0xbee8ca57, 0xbf523235, 0xbe983fad, 0x3fa56928, 0xbfef51dd, 0x3ef021bc,
	0x3bf816ac, 0xbf083b99, 0x3eadcc26, 0xbdd7c5d8, 0xbe9fbcc4, 0x3f1576cd,
	0x3e867594, 0x3f43aea7, 0xbf6e2796, 0x3e1d360e, 0xbf6e38c8, 0x3f2046ab,
	0x3fbf11c5, 0xbfbfc438, 0x3f99b1ee, 0xbc612ab1, 0x3ec37e6f, 0x3e7754d3,
	0xbb6cb962, 0xbf73023d, 0x3d91c1f4, 0x3e817cf4, 0x3ce5ebcb, 0xbf2105dc,
	0x3db5639c, 0xbf8052b5, 0xbf440e73, 0xbec72e16, 0xbf3e555d, 0xbf677c26,
	0x3edf571d, 0x3f157aad, 0xbfa69ded, 0xbf30c8b2, 0xbf359983, 0xbdedffdb,
	0x3f5b4fed, 0xbe725d63, 0x3e4a4fd9, 0xbf3a1e8e, 0x3eced515, 0x3d2fe6e3,
	0x3dae1562, 0x3eccd06e, 0xbf4f71bc, 0x3dee43b8, 0x3eaec0da, 0xbe238bad,
	0x3e1921ae, 0x3f8d09a0, 0xbf3a55d5, 0xbf047854, 0x3f8e4e32, 0x3fae52b6,
	0x3f28457c, 0xbc905c85, 0x3f139872, 0xbf5e587a, 0xbf2fecf0, 0x3e07d7a8,
	0x3e197c0e, 0x3f12897a, 0x3fa47f64, 0x3ddf12b2, 0x3e9abf88, 0xbe4869d7,
	0x3f89204a, 0x3e9de574, 0x3da829bd, 0xba92e9c3, 0x3eb443ae, 0x3f893eef,
	0x3ea84a00, 0xbdd90d75, 0x3f3de106, 0xbe28e6ae, 0x3e74f4b7, 0x3f361e14,
	0xbf6f0f53, 0xbf77ef24, 0xbf85828d, 0xbf40f1a9, 0x3ea16a7e, 0xbfa51ce5,
	0x3beb5de9, 0x3f45e6dc, 0xbf3ba6a4, 0x3f5134b5, 0xbc3c375b, 0xbf3ef52a,
	0x3c8b5053, 0xbeb80981, 0x3f40ac6a, 0x3f1b1eb7, 0x3f54b29d, 0x3f23070c,
	0xc02b5709, 0xbe80d831, 0xbf830608, 0x3d6bbfbf, 0x3e877dd7, 0xbe190d19,
	0x3f54c252, 0x3f3c7e74, 0xbeb1582b, 0xbf0cda46, 0x3f262f3b, 0x3ea8b9a9,
	0x3eacc3db, 0xbeb5fb1b, 0x3e1f302a, 0xbde174ef, 0x3eeea84d, 0xbf6eec14,
	0xbf3444f5, 0xbf1df5eb, 0x3f22c2c6, 0x3f8c34c9, 0x3efdf863, 0x3d1f5bcd,
	0x3f2a0e32, 0xbf2f3100, 0x3ea0e062, 0x3d961833, 0xbf885b14, 0xbe826741,
];
const B_BITS: [u32; 288] = [
	0xbe214c0a, 0x3c180ab2, 0x3e53942f, 0xbe2d70a4, 0x3d7eace2, 0x3ee59f96,
	0xbdaa50ce, 0xbe2bd95e, 0xbe3f58ca, 0xbe05a0c8, 0x3ef358df, 0xbe83a756,
	0xbe8fdf06, 0x3d48267f, 0x3e149e40, 0x3e096784, 0xbdaadec9, 0xbdfa972b,
	0x3e984722, 0xbb3e4340, 0x3def0a2c, 0xbc41c210, 0x3e276ff4, 0x3be8f226,
	0x3de57fbc, 0x3ebbd2e2, 0x3e4dbeb1, 0x3da2dab4, 0xbdf5d5b0, 0x3e05ec2d,
	0x3e049296, 0x3e104455, 0xbe8b4ac4, 0xbe4a33f6, 0xbd837fcc, 0xbdecdcc3,
	0x3efaf2b9, 0xbda655c6, 0x3e6281e7, 0xbc4e524c, 0x3e766e16, 0xbe4565eb,
	0xbdd643d4, 0xbe014954, 0xbe126970, 0xbdc8f2b7, 0x3eaf43a6, 0x3e3f3c1d,
	0x3de8c144, 0xbc3926b6, 0xbeafd6ac, 0x3d744101, 0xbdd2b237, 0x3e2221fe,
	0xbd8ff132, 0xbe86f6c0, 0x3ea7e3b2, 0x3ed46ff9, 0xbd2550bc, 0xbea942b3,
	0x3e33eca9, 0xbe269600, 0xbe036129, 0x3e070a86, 0x3e87b637, 0xbd8cc3a4,
	0x3e34b6e9, 0xbcabfb6c, 0xbd2d2511, 0xbdbd2efd, 0x3cc7281c, 0x3e4b2214,
	0x3df2d99f, 0x3e40d4df, 0x3ec5fdac, 0x3e93ac9f, 0x3ec75b99, 0xbe1eaa3e,
	0x3da2549b, 0xbe490acb, 0x3d66abd0, 0x3c3f094b, 0x3de20da6, 0xbd9efde0,
	0x3c12aed3, 0xbe52fbb5, 0xbd9b8bbb, 0xbde4e7ae, 0x3eaa4ae8, 0x3ebd3359,
	0x3e67831e, 0x3e4f015c, 0xbd98e34b, 0xba928700, 0xbdbe9ac8, 0xbd1166ed,
	0xbdc5900c, 0x3e5f9b7e, 0xbe362eee, 0xbc9b602e, 0xbdf7c275, 0xbe28665d,
	0x3ea67435, 0x3e304614, 0x3e90fa5e, 0xbd103f64, 0x3eb1bed0, 0xbe474954,
	0xbe437fa1, 0xbe22a1e1, 0xbd953260, 0xbe05e26c, 0x3d4d570f, 0xbe8d674e,
	0xbe358232, 0x3d27dc00, 0xbcc7c07f, 0xbed99233, 0x3c4f2690, 0xbe44443d,
	0xbd665cb8, 0x3ec32396, 0x3c08c567, 0xbe142bc2, 0xbeb5bfe9, 0x3d687635,
	0x3d87f331, 0xbdfe645e, 0xbdb0537c, 0x3e3a031d, 0xbe80bde0, 0x3d7e0321,
	0xbd60dde7, 0xbdcf81ae, 0x3e6fc8d6, 0x3e4e0e0e, 0xbd119fd0, 0x3b566150,
	0xbcd944ca, 0x3db25f92, 0x3e14174c, 0x3e446615, 0xbe9c0a74, 0x3db603cd,
	0x3d960d0c, 0x3e952cdf, 0xbea1bbd4, 0xbe36be2f, 0x3e46a5f1, 0xbd3fc041,
	0x3e8a9a2c, 0x3e8750e5, 0x3dc74849, 0xbe6cba24, 0xbd049520, 0x3e1df621,
	0xbe239c0c, 0xbd44ce9b, 0x3e16b6d3, 0xbe39eb22, 0x3d7421f0, 0xbd81a260,
	0x3ee11557, 0xbd2ba59d, 0xbe06ec40, 0x3e95ccae, 0xbe1455c7, 0x3dad765a,
	0x3dfc3786, 0xbe0a1c67, 0x3e91699f, 0xbe7f44ab, 0x3e35997a, 0x3e56074a,
	0x3e48ba1f, 0xbda86bb0, 0x3e619686, 0x3d6b51ee, 0x3de74721, 0x3e70e215,
	0x3e1e871e, 0x3e8480af, 0x3c71663b, 0xbcfdecf8, 0x3c790499, 0xbdda71ec,
	0xbe69086f, 0xbe666ca7, 0xbeae5070, 0xbe32facd, 0xbebf909a, 0x3e0acc7a,
	0xbe09fa07, 0xbd188d12, 0xbe586622, 0x3e07db2b, 0xbe0d881f, 0x3d8510b4,
	0xbe12139c, 0xbe607175, 0xbd4383ca, 0x3c6848f5, 0x3e7fedf8, 0x3d73fd22,
	0xbd4cd5a5, 0xbe0075a8, 0x3ebe613e, 0xbd4b5a62, 0x3e928ef4, 0x3e13ef02,
	0xbe0fbbdb, 0xbec3be67, 0xbea68554, 0x3e69adc4, 0xbe0308c2, 0xbeaf1abc,
	0x3e4ff81a, 0xbdf6996b, 0xbcf3e916, 0x3da93af8, 0x3dd91840, 0x3dffefc5,
	0x3d8f6344, 0x3e506fe1, 0x3db32152, 0xbd7a5993, 0x3e443a4b, 0xbdbcd106,
	0x3ef0897e, 0xbd28b80c, 0xbd8e2c30, 0x3eab0c19, 0xbda095fc, 0xbe54b551,
	0x3e59276c, 0xbd096141, 0x3e7a8623, 0x3e5fab98, 0xbe9d6296, 0xbea3a79e,
	0x3ec1b7cb, 0xbeaf0bd4, 0x3d6ba9d5, 0x3daa1207, 0xbe3755bc, 0xbe18176b,
	0x3de716ca, 0x3ecb1289, 0x3d346a84, 0x3e170f90, 0x3e1075aa, 0xbe8bcacc,
	0xbe0f039f, 0x3e19172a, 0x3ed7aead, 0x3db7bf46, 0x3cba2610, 0x3e448594,
	0xbd80054b, 0x3e3de570, 0xbe09613f, 0x3e0b7ea0, 0x3dc99b48, 0x3dd54d7d,
	0xbe85bba1, 0xbdb9fe3e, 0x3c123b6f, 0x3e4e185d, 0x3df3054a, 0x3e39ebc0,
	0x3e42fed0, 0x3ea190a8, 0xbe6dd6ee, 0x3ebb508b, 0xbc42d4bd, 0x3caa4f60,
	0xbe40490b, 0xbdc9682a, 0xbe9a10be, 0x3e13860e, 0x3ccfbf18, 0xbdce51b2,
	0x3dbc018d, 0xbe434aac, 0xbe9bc951, 0x3e1ff480, 0x3d1472b4, 0x3e9a2e82,
];

fn from_bits(bits: &[u32; 288]) -> Mat<c32> {
	Mat::from_fn(12, 12, |i, j| {
		let k = 2 * (i + 12 * j);
		c32::new(f32::from_bits(bits[k]), f32::from_bits(bits[k + 1]))
	})
}

fn to_c32(m: &Mat<c64>) -> Mat<c32> {
	Mat::from_fn(m.nrows(), m.ncols(), |i, j| {
		c32::new(m[(i, j)].re as f32, m[(i, j)].im as f32)
	})
}

fn to_c64(m: &Mat<c32>) -> Mat<c64> {
	Mat::from_fn(m.nrows(), m.ncols(), |i, j| {
		c64::new(m[(i, j)].re as f64, m[(i, j)].im as f64)
	})
}

/// returns the largest left and right relative residuals over all columns, and
/// the smallest `|beta| / (|alpha| + |beta|)`
fn residuals<T: ComplexField>(
	a: &Mat<T>,
	b: &Mat<T>,
	up: impl Fn(&T) -> c64,
) -> (f64, f64, f64) {
	let n = a.nrows();
	let (mut a1, mut b1) = (a.clone(), b.clone());
	let mut s = Diag::<T>::zeros(n);
	let mut beta = Diag::<T>::zeros(n);
	let mut ul = Mat::<T>::zeros(n, n);
	let mut ur = Mat::<T>::zeros(n, n);
	let params = Default::default();
	let mut mem = MemBuffer::new(gevd_scratch::<T>(
		n,
		ComputeEigenvectors::Yes,
		ComputeEigenvectors::Yes,
		Par::Seq.shape(),
		params,
	));
	gevd_cplx(
		a1.as_mut(),
		b1.as_mut(),
		s.as_mut(),
		beta.as_mut(),
		Some(ul.as_mut()),
		Some(ur.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		params,
	)
	.unwrap();
	let a = Mat::from_fn(n, n, |i, j| up(&a[(i, j)]));
	let b = Mat::from_fn(n, n, |i, j| up(&b[(i, j)]));
	let (na, nb) = (a.norm_l2(), b.norm_l2());
	let (mut left, mut right, mut beta_min) = (0.0f64, 0.0f64, f64::INFINITY);
	for j in 0..n {
		let (al, be) = (up(&s[j]), up(&beta[j]));
		// `f64::max` and `f64::min` drop a NaN operand, so a `0/0` residual
		// would leave the maximum untouched and read as a perfect one.
		let ratio = be.norm() / (al.norm() + be.norm());
		assert!(
			ratio.is_finite(),
			"column {j}: |beta|/(|alpha|+|beta|) = {ratio:e}, alpha = {al:?}, beta = {be:?}"
		);
		beta_min = beta_min.min(ratio);
		let u = Mat::from_fn(n, 1, |i, _| up(&ul[(i, j)]));
		let v = Mat::from_fn(n, 1, |i, _| up(&ur[(i, j)]));
		let lu = u.adjoint() * &a * Scale(be) - u.adjoint() * &b * Scale(al);
		let rv = &a * &v * Scale(be) - &b * &v * Scale(al);
		let den = be.norm() * na + al.norm() * nb;
		let (nlu, nrv) = (lu.norm_l2(), rv.norm_l2());
		let (nu, nv) = (u.norm_l2(), v.norm_l2());
		assert!(
			nlu.is_finite(),
			"column {j}: left residual numerator {nlu:e}, alpha = {al:?}, beta = {be:?}"
		);
		assert!(
			nrv.is_finite(),
			"column {j}: right residual numerator {nrv:e}, alpha = {al:?}, beta = {be:?}"
		);
		assert!(
			nu > 0.0,
			"column {j}: left eigenvector has norm {nu:e}, alpha = {al:?}, beta = {be:?}"
		);
		assert!(
			nv > 0.0,
			"column {j}: right eigenvector has norm {nv:e}, alpha = {al:?}, beta = {be:?}"
		);
		assert!(
			den > 0.0,
			"column {j}: residual denominator {den:e}, ||A|| = {na:e}, ||B|| = {nb:e}, alpha = {al:?}, beta = {be:?}"
		);
		let (lj, rj) = (nlu / (nu * den), nrv / (nv * den));
		// `f64::max` returns the other operand when one is `NaN`, so a `NaN`
		// folded into `left`/`right` would leave it unchanged and read as a
		// perfect residual instead of failing.
		assert!(
			lj.is_finite(),
			"column {j}: left residual {lj:e} (numerator {nlu:e}, ||u|| {nu:e}, denominator {den:e})"
		);
		assert!(
			rj.is_finite(),
			"column {j}: right residual {rj:e} (numerator {nrv:e}, ||v|| {nv:e}, denominator {den:e})"
		);
		left = left.max(lj);
		right = right.max(rj);
	}
	(left, right, beta_min)
}

fn check(a: Mat<c64>, b: Mat<c64>) {
	let n = a.nrows() as f64;
	let up32 = |x: &c32| c64::new(x.re as f64, x.im as f64);
	let (l, r, beta_min) = residuals(&to_c32(&a), &to_c32(&b), up32);
	let tol = 10.0 * n * f32::EPSILON as f64;
	assert!(l < tol, "c32 left residual {l:e} >= {tol:e}");
	assert!(r < tol, "c32 right residual {r:e} >= {tol:e}");
	// `B` is singular up to the rounding of its entries to f32
	assert!(beta_min < tol, "c32 smallest beta {beta_min:e} >= {tol:e}");
	let (l, r, beta_min) = residuals(&a, &b, |x: &c64| *x);
	let tol = 10.0 * n * f64::EPSILON;
	assert!(l < tol, "c64 left residual {l:e} >= {tol:e}");
	assert!(r < tol, "c64 right residual {r:e} >= {tol:e}");
	assert!(beta_min < 10.0 * n * f32::EPSILON as f64);
}

#[test]
fn left_eigenvectors_of_issue_324_pencil() {
	let (a, b) = (from_bits(&A_BITS), from_bits(&B_BITS));
	check(to_c64(&a), to_c64(&b));
}
