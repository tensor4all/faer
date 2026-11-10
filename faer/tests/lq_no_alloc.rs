//! tensor4all/faer#130: the packed LQ factorization and the factor extraction
//! (`copy_l`, `unitary`, `null`, `reconstruct`) allocate nothing on the heap once
//! the caller's `MemStack` is sized by the matching `*_scratch` query, on the
//! serial path.
//!
//! The transposed view that `lq_in_place` factors is never column-major, so a
//! per-call copy would be allocated by the QR layout adaptation on every wide
//! call; the LQ entry reaches the packed kernel directly instead. The allocator
//! counts every allocation **of the counting thread** while the call runs; the
//! matrices, the coefficients and the scratch are built outside the counted
//! interval.
//!
//! Not covered here, and inherited from the QR kernel: `upgrade_householder_factor`
//! allocates one `partials` buffer when the `V^H V` reduction is split over more
//! than one thread, and the matmul fallback on hosts without AVX2/FMA allocates
//! (PATCHSET rows #58/#56). Both apply to a `Par::Team` call, not to this one.
//!
//! A third path is host-dependent and does reach this one: on an aarch64 host the
//! `gemm` crate dispatches to its SME kernel, whose pack buffer is per thread but
//! not free. `t4a-gemm-common 0.20.4` (`Cargo.lock:2709-2712`) reuses the buffer in
//! the thread-local slot only when that one can hold the request (`src/gemm.rs:84-97`),
//! and its `Drop` keeps a buffer only up to `CACHE_INFO[1].cache_bytes`
//! (`src/gemm.rs:114-129`), which that build reports as 2796202 B on this host. The
//! `60 x 300` pipeline allocates 3993600 and 4390912 B buffers for f64, above that
//! cap, so each pass leaves the slot empty; the smaller requests that follow
//! (57344, 114688, 65536, 131072, 147456 B) are each larger than the buffer then in
//! the slot. That accounts for the 13 allocations counted for f64, and for the same
//! reason 12 (f32, 7895040 and 8781824 B buffers), 17 (c64) and 17 (c32) — measured
//! on an aarch64 macOS host, identical at `620b96a` and at head. The same test passes
//! on `x86_64`, which does not dispatch to that kernel.
//!
//! tensor4all/faer#365 attributed every one of those allocations by backtrace to
//! `PackBuffer::take` (`src/gemm.rs:92`), none to faer, and `gemm::gemm` takes no
//! caller buffer that `lq_scratch` could size. The harness therefore splits each
//! count into the `gemm` pack buffer's and faer's own, and the `60 x 300` cell
//! asserts only faer's own on an Apple aarch64 host (see the cell); every other
//! cell, and that one on every other target, still asserts the whole count.
#![cfg(feature = "std")]
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lq::no_pivoting::{factor, reconstruct, unitary};
use faer::traits::ComplexField;
use faer::{Auto, Mat, Par, ParShape, c32, c64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
// Per thread, not per process: the call under test is `Par::Seq`, so every
// allocation it makes is made on this thread, and a line the harness prints on
// another thread (its watchdog, a result line) is not the call's.
thread_local! {
	static COUNTING: Cell<bool> = const { Cell::new(false) };
}
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// Every counted allocation also records its size and the backtrace of the
// allocating call site (tensor4all/faer#365), so that an allocation can be
// attributed to the `gemm` pack buffer or to faer after the window closes. The
// capture itself allocates, so counting is switched off on this thread while it
// runs. `LQ_ALLOC_TRACE=1` prints every record.
thread_local! {
	static TRACE: RefCell<Vec<(&'static str, usize, Backtrace)>> =
		const { RefCell::new(Vec::new()) };
}

#[inline]
fn record(kind: &'static str, size: usize) {
	if !counting() {
		return;
	}
	ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
	COUNTING.with(|c| c.set(false));
	let bt = Backtrace::force_capture();
	TRACE.with(|t| t.borrow_mut().push((kind, size, bt)));
	COUNTING.with(|c| c.set(true));
}

/// The frame of `t4a-gemm-common`'s per-thread pack buffer (`PackBuffer::take`,
/// `src/gemm.rs:84-97` in 0.20.4). An allocation whose backtrace does not name
/// it, including one whose backtrace could not be symbolized, is counted as
/// faer's own.
const GEMM_PACK_FRAME: &str = "gemm_common::gemm::PackBuffer::take";

/// Allocations counted in one window, split by owner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Allocations {
	/// Allocations made outside the `gemm` pack buffer: faer's own.
	faer: usize,
	/// Allocations of the `gemm` crate's per-thread pack buffer.
	gemm_pack: usize,
}

#[inline]
fn counting() -> bool {
	COUNTING.with(Cell::get)
}
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		record("alloc", layout.size());
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		record("alloc_zeroed", layout.size());
		unsafe { System.alloc_zeroed(layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		record("realloc", new_size);
		unsafe { System.realloc(ptr, layout, new_size) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations of one complete LQ pipeline on `a`, after a warm-up call.
fn lq_allocations<T: ComplexField>(a: &Mat<T>) -> Allocations {
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let params = <factor::LqParams as Auto<T>>::auto().into();
	let block_size =
		Ord::max(factor::recommended_block_size::<T>(m, n), 1);
	let mut factor_mem = MemBuffer::new(factor::lq_in_place_scratch::<T>(
		m,
		n,
		block_size,
		ParShape::Seq,
		params,
	));
	let mut reconstruct_mem = MemBuffer::new(reconstruct::reconstruct_scratch::<T>(
		m,
		n,
		block_size,
		ParShape::Seq,
	));
	let mut unitary_mem = MemBuffer::new(unitary::unitary_scratch::<T>(
		m,
		n,
		k,
		block_size,
		ParShape::Seq,
	));
	let mut full_mem = MemBuffer::new(unitary::unitary_scratch::<T>(
		m,
		n,
		n,
		block_size,
		ParShape::Seq,
	));
	let mut null_mem = MemBuffer::new(unitary::null_scratch::<T>(
		m,
		n,
		block_size,
		ParShape::Seq,
	));
	let mut packed = a.clone();
	let mut coeff = Mat::<T>::zeros(block_size, k);
	let mut l = Mat::<T>::zeros(m, k);
	let mut l_full = Mat::<T>::zeros(m, n);
	let mut q = Mat::<T>::zeros(k, n);
	let mut q_full = Mat::<T>::zeros(n, n);
	let mut null = Mat::<T>::zeros(n - k, n);
	let mut rebuilt = Mat::<T>::zeros(m, n);
	let mut run = || {
		factor::lq_in_place(
			packed.as_mut(),
			coeff.as_mut(),
			Par::Seq,
			MemStack::new(&mut factor_mem),
			params,
		);
		factor::copy_l(l.as_mut(), packed.as_ref(), true);
		factor::copy_l(l_full.as_mut(), packed.as_ref(), true);
		unitary::unitary(
			q.as_mut(),
			packed.as_ref(),
			coeff.as_ref(),
			true,
			Par::Seq,
			MemStack::new(&mut unitary_mem),
		);
		unitary::unitary(
			q_full.as_mut(),
			packed.as_ref(),
			coeff.as_ref(),
			true,
			Par::Seq,
			MemStack::new(&mut full_mem),
		);
		unitary::null(
			null.as_mut(),
			packed.as_ref(),
			coeff.as_ref(),
			Par::Seq,
			MemStack::new(&mut null_mem),
		);
		reconstruct::reconstruct(
			rebuilt.as_mut(),
			packed.as_ref(),
			coeff.as_ref(),
			Par::Seq,
			MemStack::new(&mut reconstruct_mem),
		);
	};
	run(); // warm-up: first-use allocations (lazy statics, feature detection) are not the call's
	TRACE.with(|t| t.borrow_mut().clear());
	ALLOCATIONS.store(0, Ordering::SeqCst);
	COUNTING.with(|c| c.set(true));
	run();
	COUNTING.with(|c| c.set(false));
	let count = ALLOCATIONS.load(Ordering::SeqCst);
	let records = TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()));
	assert_eq!(
		records.len(),
		count,
		"every counted allocation has a record"
	);
	let print = std::env::var_os("LQ_ALLOC_TRACE").is_some();
	let name = std::any::type_name::<T>();
	let mut out = Allocations::default();
	for (i, (kind, size, bt)) in records.iter().enumerate() {
		let bt = bt.to_string();
		if bt.contains(GEMM_PACK_FRAME) {
			out.gemm_pack += 1;
		} else {
			out.faer += 1;
		}
		if print {
			eprintln!(
				"=== LQ_ALLOC_TRACE {m}x{n} {name} #{i} {kind} {size} B\n{bt}"
			);
		}
	}
	out
}

fn sample(m: usize, n: usize, seed: u64) -> Mat<f64> {
	let mut x = seed
		.wrapping_mul(6364136223846793005)
		.wrapping_add(1442695040888963407);
	Mat::from_fn(m, n, |_, _| {
		x = x
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
	})
}

/// Serializes the counted windows. The counter is process wide, so no other
/// test of this binary may allocate while one counts.
static GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Which allocations a cell asserts to be none.
#[derive(Clone, Copy)]
enum Contract {
	/// Every allocation of the pipeline.
	All,
	/// faer's own allocations everywhere, and the `gemm` pack buffer's as well
	/// except on an Apple aarch64 host, where `gemm` can dispatch to its SME
	/// kernel; see the cell that uses it.
	FaerOnlyOnAppleAarch64,
}

/// Allocations of the four scalar types on an `m x n` matrix, asserted to be
/// none under `contract`.
fn check_cell(m: usize, n: usize, contract: Contract) {
	let _turn = GATE.lock().unwrap_or_else(|e| e.into_inner());
	let re = sample(m, n, 130);
	let im = sample(m, n, 131);
	let f32s = Mat::<f32>::from_fn(m, n, |i, j| re[(i, j)] as f32);
	let c64s =
		Mat::<c64>::from_fn(m, n, |i, j| c64::new(re[(i, j)], im[(i, j)]));
	let c32s = Mat::<c32>::from_fn(m, n, |i, j| {
		c32::new(re[(i, j)] as f32, im[(i, j)] as f32)
	});
	let counts = [
		lq_allocations(&re),
		lq_allocations(&f32s),
		lq_allocations(&c64s),
		lq_allocations(&c32s),
	];
	let faer = counts.map(|c| c.faer);
	let gemm_pack = counts.map(|c| c.gemm_pack);
	let pack_exempt = matches!(contract, Contract::FaerOnlyOnAppleAarch64)
		&& cfg!(all(target_arch = "aarch64", target_vendor = "apple"));
	if pack_exempt {
		// Reported, not asserted: the count is the `gemm` crate's (see the cell).
		eprintln!(
			"{m} x {n}: gemm pack-buffer allocations (f64, f32, c64, c32): {gemm_pack:?}"
		);
		assert_eq!(
			faer, [0; 4],
			"{m} x {n}: allocations outside the gemm pack buffer (f64, f32, c64, c32)"
		);
	} else {
		let all = counts.map(|c| c.faer + c.gemm_pack);
		assert_eq!(all, [0; 4], "{m} x {n}: allocations (f64, f32, c64, c32)");
	}
}

// One test per shape, because the harness reports a test that runs for over 60
// seconds from inside the test process, and its watchdog thread allocates while
// the counting window is open.
macro_rules! cells {
	($($name:ident: ($m:expr, $n:expr) $(, $contract:ident)?;)*) => {
		$(
			#[test]
			fn $name() {
				check_cell($m, $n, cells!(@contract $($contract)?));
			}
		)*
	};
	(@contract) => { Contract::All };
	(@contract $contract:ident) => { Contract::$contract };
}
// 1 x 1, small, square past the recursion threshold, tall and wide enough for
// the blocked panel to have a trailing update, and a shape that is a multiple of
// 1 KiB per row (128 f64), which the transposed panel walks with a 1 KiB stride.
cells! {
	lq_allocates_nothing_1x1: (1, 1);
	lq_allocates_nothing_2x2: (2, 2);
	lq_allocates_nothing_17x17: (17, 17);
	lq_allocates_nothing_40x33: (40, 33);
	lq_allocates_nothing_33x40: (33, 40);
	lq_allocates_nothing_128x128: (128, 128);
	lq_allocates_nothing_300x60: (300, 60);
	// Narrowed on Apple aarch64, not dropped (tensor4all/faer#365). On an SME
	// host every allocation this cell counts there is the `gemm` crate's
	// per-thread pack buffer (`t4a-gemm-common 0.20.4`, `src/gemm.rs:92`): the
	// SME products of the `60 x 300` pipeline (from
	// `apply_block_householder_on_the_left_in_place_generic`, in `unitary` with
	// `k = n` and in `null`) need a 3993600 / 4390912 B buffer (f64), above the
	// size the slot keeps, so each one allocates and leaves the slot empty, and
	// the smaller NEON products after it allocate again. `gemm::gemm` takes no
	// caller buffer, so `lq_scratch` cannot size this; the cell still asserts
	// that faer itself allocates nothing, and the whole count stays asserted on
	// every other target.
	lq_allocates_nothing_60x300: (60, 300), FaerOnlyOnAppleAarch64;
}
