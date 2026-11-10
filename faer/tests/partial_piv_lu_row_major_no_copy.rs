//! tensor4all/faer#314 slice 4: the partial-pivoting LU factors a row-major
//! matrix in place, without a column-major copy of the whole matrix, on one
//! thread and on small matrices, and returns the factorization the
//! column-major input gets.
//!
//! - The allocator records the largest single allocation the calling thread
//!   makes while `lu_in_place` runs. The copy is one `m * n` buffer; the
//!   in-place path allocates the leaf panel only, `m * min(n,
//!   recursion_threshold)` elements. On a team the copy stays from `256 * 256`
//!   (it is faster there, see `takes_the_copy`), and a generic-stride view
//!   (neither stride 1) keeps it everywhere: those are the controls, where the
//!   counter has to see it.
//! - Agreement: the same matrix, column-major and row-major, gives the same
//!   permutation, the same transposition count, and factors and solutions that
//!   agree to the tolerances stated at [`check_agreement`].
#![cfg(feature = "std")]
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::{factor as plu, solve as plu_solve};
use faer::perm::PermRef;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::traits::ComplexField;
use faer::traits::math_utils::zero;
use faer::{Mat, MatMut, MatRef, Par, c64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct LargestOnThisThread;
thread_local! {
	static COUNTING: Cell<bool> = const { Cell::new(false) };
	static LARGEST: Cell<usize> = const { Cell::new(0) };
}
fn record(size: usize) {
	// `try_with`: the allocator also runs while a thread's locals are torn down
	let _ = COUNTING.try_with(|on| {
		if on.get() {
			let _ = LARGEST.try_with(|l| l.set(Ord::max(l.get(), size)));
		}
	});
}
unsafe impl GlobalAlloc for LargestOnThisThread {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		record(layout.size());
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		record(layout.size());
		unsafe { System.alloc_zeroed(layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		record(new_size);
		unsafe { System.realloc(ptr, layout, new_size) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: LargestOnThisThread = LargestOnThisThread;

/// A deterministic value in `[-0.5, 0.5)` for the `k`-th element: the
/// splitmix64 finalizer of `k`, so the entries are not a linear sequence (a
/// multiplicative hash alone gives a near low-rank matrix).
fn value(k: usize) -> f64 {
	let mut z = (k as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
	z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
	z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
	z ^= z >> 31;
	(z >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

/// Panics unless the positive strides `(rs, cs)` give an `m x n` view whose
/// entries address distinct elements: the rows are laid out one after another
/// (`rs >= n * cs`) or the columns are (`cs >= m * rs`).
fn assert_non_overlapping(m: usize, n: usize, (rs, cs): (usize, usize)) {
	assert!(
		rs >= n * cs || cs >= m * rs,
		"{m}x{n} strides ({rs}, {cs}) overlap: neither {rs} >= {n} * {cs} nor \
		 {cs} >= {m} * {rs}"
	);
}

/// The largest single allocation, in bytes, the calling thread makes while an
/// `m x n` f64 matrix with element strides `(rs, cs)` (both positive) is
/// factored (permutations and scratch are allocated before).
fn largest_allocation_strided(
	m: usize,
	n: usize,
	(rs, cs): (usize, usize),
	mut par: Par<'_>,
) -> usize {
	let len = (m - 1) * rs + (n - 1) * cs + 1;
	let mut a: Vec<f64> = (0..len).map(value).collect();
	let mut warm = a.clone();
	let (mut p, mut pi) = (vec![0usize; m], vec![0usize; m]);
	let mut mem = MemBuffer::new(plu::lu_in_place_scratch::<usize, f64>(
		m,
		n,
		par.shape(),
		Default::default(),
	));
	assert_non_overlapping(m, n, (rs, cs));
	// SAFETY: `a` holds `(m - 1) * rs + (n - 1) * cs + 1` elements, and the
	// strides address distinct elements (`assert_non_overlapping` above)
	let view = |a: &mut Vec<f64>| unsafe {
		MatMut::from_raw_parts_mut(
			a.as_mut_ptr(),
			m,
			n,
			rs as isize,
			cs as isize,
		)
	};
	// a first call on the same shape: the first factorization a thread runs
	// makes one allocation of 2796202 bytes, column-major input included, and
	// the second none (on aarch64 macOS), so it is the thread's lazily
	// allocated state and not the factorization's
	plu::lu_in_place(
		view(&mut warm),
		&mut p,
		&mut pi,
		par.reborrow(),
		MemStack::new(&mut mem),
		Default::default(),
	);
	LARGEST.with(|l| l.set(0));
	COUNTING.with(|c| c.set(true));
	plu::lu_in_place(
		view(&mut a),
		&mut p,
		&mut pi,
		par.reborrow(),
		MemStack::new(&mut mem),
		Default::default(),
	);
	COUNTING.with(|c| c.set(false));
	LARGEST.with(|l| l.get())
}

#[test]
fn row_major_partial_piv_lu_does_not_copy_the_matrix() {
	// `(m, n, row stride)`: the shapes that took the copy before (`m >= n`,
	// `m * n >= 64 * 64`), a wide one, and contiguous rows with a padded
	// leading dimension, and whether a team keeps the copy for them
	let shapes = [
		(64, 64, 64, false),
		(128, 128, 128, false),
		(256, 32, 32, false),
		(96, 96, 100, false),
		(256, 4096, 4096, false),
		// a team keeps the copy from `256 * 256` elements
		(256, 256, 256, true),
		(1024, 64, 64, true),
		(4096, 256, 256, true),
	];
	let run = |mut par: Par<'_>, team: bool| {
		for (m, n, rs, copied_on_a_team) in shapes {
			let whole = m * n * size_of::<f64>();
			let largest =
				largest_allocation_strided(m, n, (rs, 1), par.reborrow());
			let copied = team && copied_on_a_team;
			println!(
				"{} {m}x{n} rs {rs}: largest allocation {largest} bytes, the \
				 matrix is {whole}",
				if team { "team" } else { "seq" }
			);
			if copied {
				assert!(
					largest >= whole,
					"{m}x{n}, strides ({rs}, 1), team: largest allocation \
					 {largest} bytes, the copy is {whole}"
				);
			} else {
				assert!(
					largest < whole,
					"{m}x{n}, strides ({rs}, 1): an allocation of {largest} \
					 bytes, the whole matrix is {whole}"
				);
			}
		}
		// the control: a generic-stride view keeps the copy, and the counter
		// sees it
		// (neither stride 1, the entries distinct: `cs = 128 * rs` and
		// `rs = 128 * cs`)
		for (rs, cs) in [(2, 128 * 2), (128 * 2, 2)] {
			let whole = 128 * 128 * size_of::<f64>();
			let largest =
				largest_allocation_strided(128, 128, (rs, cs), par.reborrow());
			println!(
				"128x128 strides ({rs}, {cs}): largest allocation {largest} \
				 bytes, the matrix is {whole}"
			);
			assert!(
				largest >= whole,
				"128x128, strides ({rs}, {cs}): largest allocation {largest} \
				 bytes, the copy is {whole}"
			);
		}
	};
	run(Par::Seq, false);
	#[cfg(feature = "threadteam")]
	faer::team::with_team(4, |par| run(par, true));
}

trait Entry: ComplexField + Copy {
	fn from(re: f64, im: f64) -> Self;
	fn modulus(self) -> f64;
}
impl Entry for f64 {
	fn from(re: f64, _: f64) -> Self {
		re
	}

	fn modulus(self) -> f64 {
		self.abs()
	}
}
impl Entry for c64 {
	fn from(re: f64, im: f64) -> Self {
		c64::new(re, im)
	}

	fn modulus(self) -> f64 {
		self.norm()
	}
}

struct Factor<T> {
	lu: Mat<T>,
	perm: Vec<usize>,
	perm_inv: Vec<usize>,
	n_trans: usize,
}

/// Factors `a` laid out column-major or row-major and returns the factor as a
/// column-major `Mat`.
fn factor<T: Entry>(
	a: MatRef<'_, T>,
	row_major: bool,
	mut par: Par<'_>,
) -> Factor<T> {
	let (m, n) = a.shape();
	let mut buf = vec![zero::<T>(); m * n];
	let mut view = if row_major {
		MatMut::from_row_major_slice_mut(&mut buf, m, n)
	} else {
		MatMut::from_column_major_slice_mut(&mut buf, m, n)
	};
	view.copy_from(a);
	let (mut perm, mut perm_inv) = (vec![0usize; m], vec![0usize; m]);
	let mut mem = MemBuffer::new(plu::lu_in_place_scratch::<usize, T>(
		m,
		n,
		par.shape(),
		Default::default(),
	));
	let (info, _) = plu::lu_in_place(
		view.rb_mut(),
		&mut perm,
		&mut perm_inv,
		par.reborrow(),
		MemStack::new(&mut mem),
		Default::default(),
	);
	Factor {
		lu: view.rb().to_owned(),
		perm,
		perm_inv,
		n_trans: info.transposition_count,
	}
}

/// `max |P^-1 L U - A|` over the entries.
fn reconstruction_error<T: Entry>(a: MatRef<'_, T>, f: &Factor<T>) -> f64 {
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let l = Mat::<T>::from_fn(m, k, |i, j| {
		if i > j {
			f.lu[(i, j)]
		} else if i == j {
			T::from(1.0, 0.0)
		} else {
			zero()
		}
	});
	let u = Mat::<T>::from_fn(
		k,
		n,
		|i, j| {
			if i <= j { f.lu[(i, j)] } else { zero() }
		},
	);
	let lu = &l * &u;
	// row `i` of `L U` is row `perm[i]` of `A`
	finite_max("|P^-1 L U - A|", m, n, |i, j| {
		(lu[(i, j)] - a[(f.perm[i], j)]).modulus()
	})
}

/// The solution of `A x = b` from the factor, through the same solve for both
/// layouts (the factor is column-major in `f`).
fn solve<T: Entry>(f: &Factor<T>, b: MatRef<'_, T>) -> Mat<T> {
	let n = f.lu.nrows();
	let mut x = b.to_owned();
	let mut mem =
		MemBuffer::new(plu_solve::solve_in_place_scratch::<usize, T>(
			n,
			b.ncols(),
			Par::Seq.shape(),
		));
	// SAFETY: `perm` and `perm_inv` are the factorization's, inverse to each other
	let p = unsafe { PermRef::new_unchecked(&f.perm, &f.perm_inv, n) };
	plu_solve::solve_in_place(
		f.lu.as_ref(),
		f.lu.as_ref(),
		p,
		x.as_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
	);
	x
}

/// The largest of `entry(i, j)` over an `m x n` index range, panicking on the
/// first value that is not finite.
///
/// `f64::max` returns the other operand when one is NaN, so a fold with it
/// reduces an all-NaN factor or solution to `0` and every `<= tol` check
/// passes; each value is checked before it is compared.
fn finite_max(
	what: &str,
	m: usize,
	n: usize,
	mut entry: impl FnMut(usize, usize) -> f64,
) -> f64 {
	let mut max = 0.0f64;
	for j in 0..n {
		for i in 0..m {
			let v = entry(i, j);
			assert!(
				v.is_finite(),
				"{what}: non-finite value {v} at ({i}, {j})"
			);
			if v > max {
				max = v;
			}
		}
	}
	max
}

fn max_abs<T: Entry>(a: MatRef<'_, T>) -> f64 {
	finite_max("|a|", a.nrows(), a.ncols(), |i, j| a[(i, j)].modulus())
}

fn max_diff<T: Entry>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> f64 {
	assert_eq!(a.shape(), b.shape());
	finite_max("|a - b|", a.nrows(), a.ncols(), |i, j| {
		(a[(i, j)] - b[(i, j)]).modulus()
	})
}

/// The column-major and the row-major factorization of `a` agree.
///
/// - The permutation and the transposition count are equal: the row-major leaf
///   runs the column-major leaf on a copy of its panel, so its pivot search is
///   the same, ties included.
/// With `k = min(m, n)`, `eps` the `f64` machine epsilon and `rho =
/// max(1, |LU|_max / |A|_max)` the growth of the computed factor:
///
/// - Factor, `|LU_row - LU_col|_max <= 16 * k * eps * rho * |A|_max`. The leaf
///   is the column-major leaf on a copy of the panel, but the triangular solves
///   and products of the trailing updates take row-major operands as they are,
///   so a product's sums need not round the same way (observed on aarch64:
///   `f64` bit for bit on every matrix below, `c64` not).
///   The scale is the backward error of the LU (Higham, *Accuracy and
///   Stability*, Thm 9.3: `|P A - L U| <= gamma_k |L| |U|`, `|L| <= 1`), whose
///   max-norm form is `k * gamma_k * |U|_max`; the constant `16 k` is tighter
///   than that worst case, and every observed value is printed.
/// - Reconstruction, `|P^-1 L U - A|_max` under the same bound, for each layout.
/// - Solve (square, nonsingular): `|x_row - x_col|_max <= 16 * k * eps * rho *
///   kappa * |x|_max`, with `kappa = |A|_max |x|_max / |b|_max` the cheap
///   estimate of the condition number the solution itself gives (forward error
///   <= condition times backward error).
///
/// Returns `(factor diff, max reconstruction error, solve diff)`.
fn check_agreement<T: Entry>(
	name: &str,
	a: MatRef<'_, T>,
	solve_too: bool,
	par: Par<'_>,
) -> (f64, f64, f64) {
	let (m, n) = a.shape();
	let k = Ord::min(m, n).max(1) as f64;
	let eps = f64::EPSILON;
	let norm_a = max_abs(a).max(f64::MIN_POSITIVE);
	let (cm, rm) = match par {
		Par::Seq => (factor(a, false, Par::Seq), factor(a, true, Par::Seq)),
		par => {
			let mut par = par;
			(
				factor(a, false, par.reborrow()),
				factor(a, true, par.reborrow()),
			)
		},
	};
	assert_eq!(cm.perm, rm.perm, "{name}: permutation");
	assert_eq!(cm.n_trans, rm.n_trans, "{name}: transposition count");

	let growth = max_abs(cm.lu.as_ref()) / norm_a;
	let tol = 16.0 * k * eps * norm_a * growth.max(1.0);
	let d_factor = max_diff(cm.lu.as_ref(), rm.lu.as_ref());
	let e_cm = reconstruction_error(a, &cm);
	let e_rm = reconstruction_error(a, &rm);
	assert!(
		d_factor <= tol,
		"{name}: |LU_row - LU_col| = {d_factor:e} > {tol:e}"
	);
	assert!(
		e_cm <= tol,
		"{name}: column-major |PA - LU| = {e_cm:e} > {tol:e}"
	);
	assert!(
		e_rm <= tol,
		"{name}: row-major |PA - LU| = {e_rm:e} > {tol:e}"
	);
	let mut d_solve = 0.0;
	let mut d_solve_rel = 0.0;
	if solve_too {
		let b = Mat::<T>::from_fn(m, 2, |i, j| {
			T::from(value(3 * i + j + 7), value(5 * i + j))
		});
		let x_cm = solve(&cm, b.as_ref());
		let x_rm = solve(&rm, b.as_ref());
		let norm_x = max_abs(x_cm.as_ref());
		let cond = norm_a * norm_x / max_abs(b.as_ref());
		d_solve = max_diff(x_cm.as_ref(), x_rm.as_ref());
		d_solve_rel = d_solve / norm_x;
		let tol_x = 16.0 * k * eps * growth.max(1.0) * cond.max(1.0) * norm_x;
		assert!(
			d_solve <= tol_x,
			"{name}: |x_row - x_col| = {d_solve:e} > {tol_x:e}"
		);
	}
	println!(
		"{name}: perm equal, n_trans {}, |dLU| {d_factor:.3e} (rel {:.3e}, tol {tol:.3e}), \
		 |PA-LU| col {e_cm:.3e} row {e_rm:.3e}, growth {growth:.2}, |dx| {d_solve:.3e} \
		 (rel {d_solve_rel:.3e}), bitwise {}",
		cm.n_trans,
		d_factor / norm_a,
		cm.lu == rm.lu,
	);
	// both are finite (`finite_max`), so `f64::max` drops no NaN here
	(d_factor, e_cm.max(e_rm), d_solve)
}

fn random<T: Entry>(m: usize, n: usize, seed: usize) -> Mat<T> {
	Mat::from_fn(m, n, |i, j| {
		let k = seed * 1_000_003 + i * n + j;
		T::from(value(2 * k), value(2 * k + 1))
	})
}

fn matrix_set<T: Entry>() -> Vec<(String, Mat<T>, bool)> {
	let mut set = Vec::new();
	// a permutation needs every row swapped, with a dominant entry off the
	// diagonal of each column
	let p = Mat::<T>::from_fn(5, 5, |i, j| {
		if (i + 2) % 5 == j {
			T::from(4.0, 1.0)
		} else {
			T::from(value(i * 5 + j), 0.0)
		}
	});
	set.push(("row swaps 5x5".to_string(), p, true));
	// singular: the first column is exactly zero (the leaf skips the zero
	// pivot), and a column that repeats another
	let s = Mat::<T>::from_fn(128, 128, |i, j| match j {
		0 => zero(),
		7 => T::from(value(i * 128 + 3), value(i)),
		_ => T::from(value(i * 128 + j), value(i + j)),
	});
	let mut s = s;
	for i in 0..128 {
		s[(i, 9)] = s[(i, 3)];
	}
	set.push(("singular 128x128".to_string(), s, false));
	set.push((
		"1x1".to_string(),
		Mat::from_fn(1, 1, |_, _| T::from(-3.0, 0.5)),
		true,
	));
	set.push(("1x300".to_string(), random(1, 300, 1), false));
	set.push(("300x1".to_string(), random(300, 1, 2), false));
	// exact ties: small integers, so several rows hold the largest entry of a
	// column; the first of them is the pivot in both layouts
	let t = Mat::<T>::from_fn(96, 96, |i, j| {
		T::from(((i * 7 + j * 3) % 5) as f64, 0.0)
	});
	set.push(("ties 96x96 (singular)".to_string(), t, false));
	for n in [64, 128, 512] {
		set.push((format!("{n}x{n}"), random(n, n, n), true));
	}
	set.push(("4096x256".to_string(), random(4096, 256, 11), false));
	set.push(("256x4096".to_string(), random(256, 4096, 12), false));
	set
}

fn agreement<T: Entry>(dtype: &str) {
	for (name, a, solve_too) in matrix_set::<T>() {
		check_agreement(
			&format!("{dtype} {name} seq"),
			a.as_ref(),
			solve_too,
			Par::Seq,
		);
		#[cfg(feature = "threadteam")]
		faer::team::with_team(4, |par| {
			check_agreement(
				&format!("{dtype} {name} team4"),
				a.as_ref(),
				solve_too,
				par,
			)
		});
	}
}

#[test]
fn row_major_partial_piv_lu_agrees_with_column_major_f64() {
	agreement::<f64>("f64");
}

#[test]
fn row_major_partial_piv_lu_agrees_with_column_major_c64() {
	agreement::<c64>("c64");
}

/// The first pivot on a zero matrix with two entries of the same magnitude in
/// column 0 is the smaller row, row-major or column-major: the row-major leaf
/// is the column-major leaf on the copied panel.
#[test]
fn row_major_partial_piv_lu_tie_goes_to_the_first_row() {
	for row_major in [false, true] {
		let a = Mat::<f64>::from_fn(64, 64, |i, j| match (i, j) {
			(5, 0) => -2.0,
			(2, 0) => 2.0,
			(i, j) if j > 0 => value(i * 64 + j),
			_ => 0.0,
		});
		let f = factor(a.as_ref(), row_major, Par::Seq);
		assert_eq!(f.perm[0], 2, "row_major = {row_major}");
	}
}

/// The `128 x 128` control with strides `(2, 192)` overlapped: entries
/// `(96, 0)` and `(0, 1)` both address element `192`. The check rejects it.
#[test]
#[should_panic(expected = "overlap")]
fn overlapping_strides_are_rejected() {
	assert_non_overlapping(128, 128, (2, 192));
}

/// An all-NaN factor fails the agreement checks instead of reducing to a zero
/// difference.
#[test]
#[should_panic(expected = "non-finite")]
fn nan_factor_is_rejected() {
	let a = random::<f64>(4, 4, 3);
	let nan = Mat::<f64>::from_fn(4, 4, |_, _| f64::NAN);
	max_diff(a.as_ref(), nan.as_ref());
}
