#![allow(missing_docs)]
/// compile-time bound-checked indexing types
pub mod bound;
/// the parallel rule: dispatch cost of a shape against the serial time of a region
pub mod cost;
/// the parallel route onto a threadteam team, with [`team::with_team`] for the caller
#[cfg(feature = "threadteam")]
pub mod team;
#[doc(hidden)]
pub mod thread {
	use crate::*;
	/// A closure handed to exactly one concurrent child: each index is taken once, which is what
	/// the split below guarantees. `alloc` and `core` only, so the hand-off does not need `std`.
	struct Slot<T>(core::cell::UnsafeCell<Option<T>>);
	// SAFETY: every index is taken by one child; `Slot::take` is the only accessor.
	unsafe impl<T: Send> Sync for Slot<T> {}
	impl<T> Slot<T> {
		fn new(value: T) -> Self {
			Self(core::cell::UnsafeCell::new(Some(value)))
		}
		/// # Safety
		/// The caller takes each index at most once.
		unsafe fn take(&self) -> T {
			unsafe { (*self.0.get()).take() }.expect("one child per slot")
		}
	}
	/// executes the two operations, possibly in parallel, while splitting the
	/// amount of parallelism between the two
	#[inline]
	pub fn join_raw(
		op_a: impl Send + FnOnce(Par),
		op_b: impl Send + FnOnce(Par),
		parallelism: Par,
	) {
		fn implementation<'a>(
			op_a: &'a mut (dyn Send + FnMut(Par)),
			op_b: &'a mut (dyn Send + FnMut(Par)),
			mut parallelism: Par,
		) {
			match parallelism {
				Par::Seq => {
					(op_a(parallelism.reborrow()), op_b(parallelism));
				},
				#[cfg(feature = "threadteam")]
				Par::Team(team) => {
					let n_threads = team.size().get();
					if n_threads == 1 {
						(op_a(Par::Team(team)), op_b(Par::Seq));
					} else {
						// each half runs on the leader of a sub-team of its width and owns that
						// sub-team
						let half = n_threads / 2;
						let ops = [Slot::new(op_a), Slot::new(op_b)];
						super::team::note_split(2);
						let _ = team.split(&[n_threads - half, half], |c, child| {
							// SAFETY: the split hands index `c` to one child only, and `c < 2`
							unsafe { (ops[c].take())(Par::Team(child)) }
						});
					}
				},
			};
		}
		let mut op_a = Some(op_a);
		let mut op_b = Some(op_b);
		implementation(
			&mut |parallelism| op_a.take().unwrap()(parallelism),
			&mut |parallelism| op_b.take().unwrap()(parallelism),
			parallelism,
		)
	}
	/// Runs `f` on a part of `par` of `width` participants: `par` itself when `width` is its
	/// degree or more, `Par::Seq` below two, otherwise a sub-team of `width` (a split into one
	/// child, which is led by the caller and dispatches nothing; the members it leaves out stay
	/// idle). A region decides its width with [`cost::par_width`](crate::utils::cost) and runs
	/// here, so only that region is narrowed and the caller keeps its whole `par`.
	pub fn with_width<R>(par: Par, width: usize, f: impl FnOnce(Par) -> R) -> R {
		if width >= par.degree() {
			return f(par);
		}
		if width < 2 {
			return f(Par::Seq);
		}
		match par {
			Par::Seq => f(Par::Seq),
			#[cfg(feature = "threadteam")]
			Par::Team(team) => {
				super::team::note_width(width);
				struct Once<F, R>(core::cell::UnsafeCell<(Option<F>, Option<R>)>);
				// SAFETY: a split into the single child `[width]` runs its callback once, on the
				// calling thread (child 0 is led by the caller), so the cell is never shared.
				unsafe impl<F, R> Sync for Once<F, R> {}
				impl<F, R> Once<F, R> {
					fn get(&self) -> *mut (Option<F>, Option<R>) {
						self.0.get()
					}
				}
				let once = Once(core::cell::UnsafeCell::new((Some(f), None)));
				team.split(&[width], |_, child| {
					// SAFETY: see `Once`.
					let (f, out) = unsafe { &mut *once.get() };
					*out = Some((f.take().expect("one child"))(Par::Team(child)));
				})
				.expect("2 <= width < size is a valid split");
				once.0.into_inner().1.expect("the child ran")
			},
			#[cfg(not(feature = "threadteam"))]
			Par::__Lifetime(x, _) => match x {},
		}
	}
	/// unsafe [`Send`] and [`Sync`] pointer type
	pub struct Ptr<T>(pub *mut T);
	unsafe impl<T> Send for Ptr<T> {}
	unsafe impl<T> Sync for Ptr<T> {}
	impl<T> Copy for Ptr<T> {}
	impl<T> Clone for Ptr<T> {
		#[inline]
		fn clone(&self) -> Self {
			*self
		}
	}
	/// the amount of threads that should ideally execute an operation with the
	/// given parallelism
	#[inline]
	pub fn parallelism_degree(parallelism: &Par) -> usize {
		parallelism.degree()
	}

	/// A flat item loop on the team of `par`: the team's members process contiguous blocks,
	/// the caller alone for `Seq`
	#[cfg(feature = "rayon")]
	pub fn for_each<
		T: Send,
		I: rayon::iter::IntoParallelIterator<Item = T>,
	>(
		par: Par,
		n_jobs: usize,
		iter: I,
		f: impl Sync + Fn(T),
	) where
		I::Iter: rayon::iter::IndexedParallelIterator,
	{
		match par {
			Par::Seq => {
				use rayon::iter::plumbing::{Producer, ProducerCallback};
				struct Each<F>(F);
				impl<T, F: Fn(T)> ProducerCallback<T> for Each<F> {
					type Output = ();
					fn callback<P: Producer<Item = T>>(self, p: P) {
						p.into_iter().for_each(self.0);
					}
				}
				rayon::iter::IndexedParallelIterator::with_producer(
					iter.into_par_iter(),
					Each(f),
				);
			},
			#[cfg(feature = "threadteam")]
			Par::Team(team) => super::team::for_each(team, n_jobs, iter, f),
		}
	}
	/// `for_each` whose closure also receives the parallelism of its job: the share of `par`
	/// that one of `n_jobs` concurrent jobs may use (sequential below two threads)
	#[cfg(feature = "rayon")]
	pub fn for_each_with_par<
		T: Send,
		I: rayon::iter::IntoParallelIterator<Item = T>,
	>(
		par: Par,
		n_jobs: usize,
		iter: I,
		f: impl Sync + Fn(T, Par),
	) where
		I::Iter: rayon::iter::IndexedParallelIterator,
	{
		match par {
			Par::Seq => for_each(Par::Seq, n_jobs, iter, |item| f(item, Par::Seq)),
			#[cfg(feature = "threadteam")]
			Par::Team(team) => {
				// the item runs on the leader of its own sub-team and receives it
				super::team::for_each_child(team, n_jobs, iter, |item, child| {
					f(item, Par::Team(child))
				})
			},
		}
	}

	/// runs the two operations concurrently on a split of `par` in which the first receives
	/// `size_a` threads (clamped so that both get at least one) and the second the rest
	#[cfg(feature = "rayon")]
	pub fn join_sized(
		par: Par,
		size_a: usize,
		op_a: impl Send + FnOnce(Par),
		op_b: impl Send + FnOnce(Par),
	) {
		let degree = par.degree();
		if degree < 2 {
			op_a(Par::Seq);
			op_b(Par::Seq);
			return;
		}
		let size_a = size_a.clamp(1, degree - 1);
		// a typed slot per closure, so nothing is boxed (#111: no allocation per region)
		let (a, b) = (Slot::new(op_a), Slot::new(op_b));
		split(par, &[size_a, degree - size_a], &|c, child| {
			// SAFETY: the split hands index `c` to one child only, and `c < 2`
			unsafe {
				if c == 0 {
					(a.take())(child)
				} else {
					(b.take())(child)
				}
			}
		});
	}

	/// runs `f(c, child)` for every `c < sizes.len()` concurrently, where `child` is the part of
	/// `par` of width `sizes[c]` (sequential for a width of one). The widths must be non-zero and
	/// add up to at most the degree of `par`; the children never use more than they are handed.
	#[cfg(feature = "rayon")]
	pub fn split(par: Par, sizes: &[usize], f: &(dyn Sync + Fn(usize, Par))) {
		core::assert!(sizes.iter().all(|&s| s > 0));
		match par {
			Par::Seq => {
				for c in 0..sizes.len() {
					f(c, Par::Seq);
				}
			},
			#[cfg(feature = "threadteam")]
			Par::Team(team) => {
				if sizes.iter().sum::<usize>() <= team.size().get() {
					super::team::note_split(sizes.len());
					let _ = team.split(sizes, |c, child| f(c, Par::Team(child)));
				} else {
					// the team is too small for the requested widths: run the parts in turn
					for c in 0..sizes.len() {
						f(c, Par::Seq);
					}
				}
			},
		}
	}

	/// returns the start and length of a subsegment of `0..n`, split between
	/// `chunk_count` consumers, for the consumer at index `idx`
	///
	/// for the same `n` and `chunk_count`, different values of `idx` between in
	/// `0..chunk_count` will represent distinct subsegments
	#[inline]
	pub fn par_split_indices(
		n: usize,
		idx: usize,
		chunk_count: usize,
	) -> (usize, usize) {
		let chunk_size = n / chunk_count;
		let rem = n % chunk_count;
		let idx_to_col_start = move |idx| {
			if idx < rem {
				idx * (chunk_size + 1)
			} else {
				rem + idx * chunk_size
			}
		};
		let start = idx_to_col_start(idx);
		let end = idx_to_col_start(idx + 1);
		(start, end - start)
	}

	#[cfg(test)]
	mod tests {
		use super::{for_each, for_each_with_par, join_raw, join_sized, split};
		use crate::Par;
		use core::sync::atomic::{AtomicUsize, Ordering};

		struct Leaves {
			count: AtomicUsize,
			active: AtomicUsize,
			peak: AtomicUsize,
		}

		// splits until each branch has a single thread, then holds the
		// leaf long enough for the leaves to overlap
		fn walk(parallelism: Par, leaves: &Leaves) {
			if parallelism.degree() > 1 {
				join_raw(
					|par| walk(par, leaves),
					|par| walk(par, leaves),
					parallelism,
				);
			} else {
				leaves.count.fetch_add(1, Ordering::Relaxed);
				let now = leaves.active.fetch_add(1, Ordering::AcqRel) + 1;
				leaves.peak.fetch_max(now, Ordering::AcqRel);
				std::thread::sleep(std::time::Duration::from_millis(5));
				leaves.active.fetch_sub(1, Ordering::AcqRel);
			}
		}

		#[test]
		fn join_raw_leaves_do_not_exceed_requested_width() {
			// the team is wider than every requested width, so only the split itself can bound the
			// concurrency
			crate::team::with_team(8, |mut par| {
				for n in [2usize, 3, 5, 6, 7, 8] {
					let l = Leaves {
						count: AtomicUsize::new(0),
						active: AtomicUsize::new(0),
						peak: AtomicUsize::new(0),
					};
					split(par.reborrow(), &[n], &|_c, child| walk(child, &l));
					assert_eq!(l.count.into_inner(), n, "leaves for n = {n}");
					assert!(
						l.peak.into_inner() <= n,
						"concurrent leaves exceed n = {n}"
					);
				}
			});
		}

		// a region handed a `Par::Team` budget runs on the members of that team and not on the
		// caller alone: the caller's own item cannot finish until another participant has entered
		// one, so a caller-alone dispatch runs into the deadline instead of passing by luck on a
		// quiet machine (the block Householder apply's row split was dispatched that way: 4096x256
		// f64 at eight threads, 14.6 ms against 5.9)
		#[test]
		fn for_each_region_reaches_the_team() {
			use std::time::{Duration, Instant};
			let active = AtomicUsize::new(0);
			let peak = AtomicUsize::new(0);
			let deadline = Instant::now() + Duration::from_secs(2);
			crate::team::with_team(4, |par| {
				use rayon::prelude::*;
				for_each(par, 4, (0..4u32).into_par_iter(), |_| {
					let now = active.fetch_add(1, Ordering::AcqRel) + 1;
					peak.fetch_max(now, Ordering::AcqRel);
					while peak.load(Ordering::Acquire) < 2 && Instant::now() < deadline {
						std::hint::spin_loop();
					}
					std::thread::sleep(Duration::from_millis(2));
					active.fetch_sub(1, Ordering::AcqRel);
				});
			});
			assert!(
				peak.into_inner() > 1,
				"no member of the team of 4 entered the region: it ran on the caller alone"
			);
		}

		// the two halves of a sized join run on the widths they are handed: their children
		// never overlap beyond the budget, and each half sees the width it asked for
		#[test]
		fn join_sized_children_stay_within_their_share() {
			for n in [2usize, 3, 5, 8] {
				crate::team::with_team(n, |mut par| {
					use rayon::prelude::*;
					for left in 0..=n + 1 {
						let active = AtomicUsize::new(0);
						let peak = AtomicUsize::new(0);
						let widths = [AtomicUsize::new(0), AtomicUsize::new(0)];
						let side = |which: usize, par: Par| {
							let jobs = par.degree();
							widths[which].store(jobs, Ordering::Relaxed);
							for_each_with_par(par, jobs, (0..jobs).into_par_iter(), |_, _| {
								let now = active.fetch_add(1, Ordering::AcqRel) + 1;
								peak.fetch_max(now, Ordering::AcqRel);
								std::thread::sleep(std::time::Duration::from_millis(3));
								active.fetch_sub(1, Ordering::AcqRel);
							});
						};
						join_sized(
							par.reborrow(),
							left,
							|par| side(0, par),
							|par| side(1, par),
						);
						let total = widths[0].load(Ordering::Relaxed)
							+ widths[1].load(Ordering::Relaxed);
						assert_eq!(total, n, "n = {n}, left = {left}");
						assert!(
							peak.load(Ordering::Relaxed) <= n,
							"concurrent jobs exceed n = {n} with left = {left}"
						);
					}
				});
			}
		}
	}
}
/// approximate comparators for testing purposes
pub mod approx;
/// simd helper utilities based on lifetime-bound indices
pub mod simd;
