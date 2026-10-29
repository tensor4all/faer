//! #111: a parallel region allocates nothing on the heap. Each counted call below runs on a held
//! team with its regions forced parallel (`ParThreshold::Fixed(0)`), after a warm-up, and the
//! allocator counts every allocation of the process (any thread) while the call runs. The team,
//! the matrices and the scratch are built outside the counted interval, and the check that the
//! regions did dispatch reads the team counters after it.
#![cfg(all(feature = "std", feature = "threadteam"))]
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::utils::cost::ParThreshold;
use faer::utils::team::stats;
use faer::{Auto, Mat, Par};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counting;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.alloc(layout) }
	}
	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.alloc_zeroed(layout) }
	}
	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.realloc(ptr, layout, new_size) }
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations during `f`, and the number of parallel sections and splits it dispatched.
fn counted(mut f: impl FnMut()) -> (usize, usize) {
	f(); // warm-up: first-use allocations (thread locals, lazy statics) are not the region's
	stats::enable(true);
	let _ = stats::take();
	ALLOCATIONS.store(0, Ordering::SeqCst);
	COUNTING.store(true, Ordering::SeqCst);
	f();
	COUNTING.store(false, Ordering::SeqCst);
	let n = ALLOCATIONS.load(Ordering::SeqCst);
	stats::enable(false);
	let s = stats::take();
	let parallel = s.run.iter().skip(2).sum::<usize>() + s.split.iter().skip(2).sum::<usize>();
	(n, parallel)
}

fn sample(m: usize, n: usize, seed: u64) -> Mat<f64> {
	let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
	Mat::from_fn(m, n, |_, _| {
		x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
	})
}

// one test function: the counter is process wide, so nothing else may run while it counts
#[test]
fn parallel_regions_allocate_nothing() {
	use faer::linalg::cholesky::lblt::factor as lblt;
	use faer::linalg::lu::full_pivoting::factor as flu;
	for width in [4usize, 8, 70] {
		faer::team::with_team(width, |mut par: Par<'_>| {
			// full pivoting LU: the per-step pivot search (`for_each` over column blocks)
			let n = 96;
			let a0 = sample(n, n, 1);
			let mut a = a0.clone();
			let (mut p, mut pi, mut q, mut qi) = (vec![0usize; n], vec![0usize; n], vec![0usize; n], vec![0usize; n]);
			let params = flu::FullPivLuParams { par_threshold: ParThreshold::Fixed(0), ..<flu::FullPivLuParams as Auto<f64>>::auto() };
			let mut mem = MemBuffer::new(flu::lu_in_place_scratch::<usize, f64>(n, n, par.shape(), params.into()));
			let (allocs, parallel) = counted(|| {
				a.copy_from(&a0);
				flu::lu_in_place(a.as_mut(), &mut p, &mut pi, &mut q, &mut qi, par.reborrow(), MemStack::new(&mut mem), params.into());
			});
			assert!(parallel > 0, "full pivoting LU, width {width}: no parallel region");
			assert_eq!(allocs, 0, "full pivoting LU, width {width}");

			// Bunch-Kaufman with full pivoting: rank-one and rank-two steps
			let x = sample(n, n, 2);
			let s0 = &x + x.transpose();
			let mut s = s0.clone();
			let mut sub = faer::diag::Diag::<f64>::zeros(n);
			let params = lblt::LbltParams {
				pivoting: lblt::PivotingStrategy::Full,
				par_threshold: ParThreshold::Fixed(0),
				..<lblt::LbltParams as Auto<f64>>::auto()
			};
			let mut mem = MemBuffer::new(lblt::cholesky_in_place_scratch::<usize, f64>(n, par.shape(), params.into()));
			let (allocs, parallel) = counted(|| {
				s.copy_from(&s0);
				lblt::cholesky_in_place(s.as_mut(), sub.as_mut(), &mut p, &mut pi, par.reborrow(), MemStack::new(&mut mem), params.into());
			});
			assert!(parallel > 0, "Bunch-Kaufman, width {width}: no parallel region");
			assert_eq!(allocs, 0, "Bunch-Kaufman, width {width}");
			// both update ranks ran: a 2x2 pivot leaves a nonzero subdiagonal entry, a 1x1 a zero
			let subs: Vec<f64> = (0..n).map(|i| sub[i]).collect();
			assert!(subs.iter().any(|&x| x != 0.0) && subs.iter().any(|&x| x == 0.0), "Bunch-Kaufman, width {width}: a pivot size never occurred");

			// `join_sized`, the LU swap stage's split, with closures that capture
			let mut left = vec![0usize; 4];
			let mut right = vec![0usize; 4];
			let (allocs, parallel) = counted(|| {
				faer::utils::thread::join_sized(
					par.reborrow(),
					width / 2,
					|_| left.iter_mut().for_each(|x| *x += 1),
					|_| right.iter_mut().for_each(|x| *x += 2),
				);
			});
			assert!(parallel > 0, "join_sized, width {width}: no split");
			assert_eq!(allocs, 0, "join_sized, width {width}");
		});
	}
}
