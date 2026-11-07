//! A zero `PivLltParams::block_size` used to leave the panel loop's `k` where it was, so
//! `cholesky_in_place` never returned even for `A = [1]` (tensor4all/faer#283). The public
//! contract is a positive block size, rejected before the input is touched.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt_pivoting::factor::{
	PivLltParams, cholesky_in_place, cholesky_in_place_scratch,
};
use faer::{Mat, Par};

fn params(block_size: usize) -> PivLltParams {
	PivLltParams {
		block_size,
		..Default::default()
	}
}

fn factor(n: usize, block_size: usize) -> (usize, Mat<f64>) {
	let a = Mat::<f64>::from_fn(n, n, |i, j| {
		if i == j {
			4.0
		} else if i.abs_diff(j) == 1 {
			1.0
		} else {
			0.0
		}
	});
	let reference = a.clone();
	let mut l = a;
	let mut perm = vec![0usize; n];
	let mut perm_inv = vec![0usize; n];
	let req = cholesky_in_place_scratch::<usize, f64>(
		n,
		Par::Seq.shape(),
		params(block_size).into(),
	);
	let mut mem = MemBuffer::new(req);
	let (info, p) = cholesky_in_place(
		l.as_mut(),
		&mut perm,
		&mut perm_inv,
		Par::Seq,
		MemStack::new(&mut mem),
		params(block_size).into(),
	)
	.unwrap();
	assert_eq!(info.rank, n);
	// `p` borrows `perm`; the forward array says which original index sits at `i`
	let perm = {
		let _ = p;
		perm.to_vec()
	};
	// `A = P L L^H P^T` for a positive definite input, whatever the panel width is
	let mut reconstructed = Mat::<f64>::zeros(n, n);
	for j in 0..n {
		for i in j..n {
			reconstructed[(i, j)] = l[(i, j)];
		}
	}
	// `perm[i]` is the original index that now sits at `i`, so `L L^H` is the input
	// permuted by it
	let mut permuted = Mat::<f64>::zeros(n, n);
	for j in 0..n {
		for i in 0..n {
			permuted[(i, j)] = reference[(perm[i], perm[j])];
		}
	}
	assert!(
		(&reconstructed * reconstructed.transpose() - &permuted).norm_max()
			< 1e-12,
		"block_size={block_size} n={n}"
	);
	(info.rank, reference)
}

#[test]
#[should_panic(expected = "params.config.block_size > 0")]
fn zero_block_size_panics_on_the_issue_input() {
	// the reported shape: A = [1]
	factor(1, 0);
}

#[test]
#[should_panic(expected = "params.config.block_size > 0")]
fn zero_block_size_panics_on_empty_input() {
	factor(0, 0);
}

#[test]
#[should_panic(expected = "params.config.block_size > 0")]
fn zero_block_size_panics_in_the_scratch_query() {
	let _ = cholesky_in_place_scratch::<usize, f64>(
		1,
		Par::Seq.shape(),
		params(0).into(),
	);
}

#[test]
fn positive_block_sizes_factor() {
	// a control for every panel shape the loop can take: below, at and above `n`
	for block_size in [1, 2, 3, 5, 128] {
		factor(1, block_size);
		factor(5, block_size);
		factor(17, block_size);
	}
}

#[test]
#[should_panic(expected = "a.nrows() == a.ncols()")]
fn non_square_input_still_panics_on_its_own_contract() {
	let mut a = Mat::<f64>::zeros(2, 3);
	let mut perm = [0usize; 2];
	let mut perm_inv = [0usize; 2];
	let mut mem = MemBuffer::new(cholesky_in_place_scratch::<usize, f64>(
		2,
		Par::Seq.shape(),
		params(1).into(),
	));
	let _ = cholesky_in_place(
		a.as_mut(),
		&mut perm,
		&mut perm_inv,
		Par::Seq,
		MemStack::new(&mut mem),
		params(1).into(),
	);
}
