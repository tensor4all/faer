//! The parallel rule: a region runs in parallel when its serial time is at least `PAYS` times
//! the dispatch cost of every section it opens (design record: `threadteam`
//! `docs/design/thresholds.md`).
//!
//! The dispatch cost belongs to the token ([`ParShape::dispatch_cost`]); the serial time of a
//! kernel family is its own throughput constant, below. `PAYS` mirrors `threadteam::PAYS` so that
//! a build without the parallel route still decides the way the route does; a test compares the
//! two when `threadteam` is there.

use crate::ParShape;
use core::num::NonZeroUsize;
use core::time::Duration;

#[cfg(feature = "threadteam")]
pub use threadteam::{PAYS, pays_at};

/// The multiple of the stack (mirrors `threadteam::PAYS`, for a build without the route).
#[cfg(not(feature = "threadteam"))]
pub const PAYS: f64 = 3.0;

/// `estimate >= PAYS * sections * dispatch_cost` (mirrors `threadteam::pays_at`).
#[cfg(not(feature = "threadteam"))]
pub fn pays_at(estimate: Duration, sections: u32, dispatch_cost: Duration) -> bool {
	if dispatch_cost == Duration::MAX {
		return false;
	}
	if sections == 0 {
		return true;
	}
	estimate.as_secs_f64() >= PAYS * f64::from(sections) * dispatch_cost.as_secs_f64()
}

/// The class of the machine, keyed by the kernel dispatch it will take, not by the CPU model.
/// The serial rate table has one row per class. The enum and the detection are kept in sync by
/// hand with `threadteam::HostClass`; a test compares the two when the parallel route is
/// compiled.
///
/// Rates and dispatch costs of one row must come from the same machine in the same clock state:
/// switching the boost on moves both by about 1.66 times with a stable ratio (within a few
/// percent), so a row that mixes boosted rates with unboosted costs, or the reverse, shifts every
/// floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostClass {
	/// `x86_64` with AVX-512F at run time (`x86_64-avx512`).
	X86_64Avx512,
	/// `x86_64` with AVX2 and FMA, without AVX-512F (`x86_64-avx2`).
	X86_64Avx2,
	/// Any other `x86_64` (`x86_64-scalar`).
	X86_64Scalar,
	/// `aarch64` on macOS (`aarch64-apple`).
	Aarch64Apple,
	/// Any other `aarch64` (`aarch64`).
	Aarch64,
	/// Everything else (`generic`).
	Generic,
}

impl HostClass {
	/// Every class, in declaration order.
	pub const ALL: [HostClass; 6] = [
		HostClass::X86_64Avx512,
		HostClass::X86_64Avx2,
		HostClass::X86_64Scalar,
		HostClass::Aarch64Apple,
		HostClass::Aarch64,
		HostClass::Generic,
	];

	/// The class of this host, detected at the first call and cached. With `std`, `x86_64` uses
	/// run time feature detection; without it the target features of the build decide.
	pub fn detect() -> Self {
		use core::sync::atomic::{AtomicU8, Ordering};
		static CLASS: AtomicU8 = AtomicU8::new(0);
		let c = CLASS.load(Ordering::Relaxed);
		if c != 0 {
			return Self::ALL[c as usize - 1];
		}
		let class = Self::detect_uncached();
		let idx = Self::ALL.iter().position(|&k| k == class).unwrap();
		CLASS.store(idx as u8 + 1, Ordering::Relaxed);
		class
	}

	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	fn detect_uncached() -> Self {
		if std::arch::is_x86_feature_detected!("avx512f") {
			HostClass::X86_64Avx512
		} else if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
			HostClass::X86_64Avx2
		} else {
			HostClass::X86_64Scalar
		}
	}

	#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
	fn detect_uncached() -> Self {
		if cfg!(target_feature = "avx512f") {
			HostClass::X86_64Avx512
		} else if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
			HostClass::X86_64Avx2
		} else {
			HostClass::X86_64Scalar
		}
	}

	#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
	fn detect_uncached() -> Self {
		HostClass::Aarch64Apple
	}

	#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
	fn detect_uncached() -> Self {
		HostClass::Aarch64
	}

	#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
	fn detect_uncached() -> Self {
		HostClass::Generic
	}

	/// The class name used in the docs and the tables.
	pub const fn name(self) -> &'static str {
		match self {
			HostClass::X86_64Avx512 => "x86_64-avx512",
			HostClass::X86_64Avx2 => "x86_64-avx2",
			HostClass::X86_64Scalar => "x86_64-scalar",
			HostClass::Aarch64Apple => "aarch64-apple",
			HostClass::Aarch64 => "aarch64",
			HostClass::Generic => "generic",
		}
	}
}

/// Serial throughput of the GEMM kernel, nanoseconds per multiply-add of the dtype (a complex
/// one for `c32`/`c64`), of the host class ([`rates`]).
pub mod ns_per_mac {
	use super::rates;
	/// `f64`.
	pub fn f64() -> f64 {
		rates().gemm_mac.real
	}
	/// `c64`.
	pub fn c64() -> f64 {
		rates().gemm_mac.cplx
	}
	/// `f32`: measured on the classes that have a measurement, half of `f64` on the others.
	pub fn f32() -> f64 {
		rates().gemm_single.real
	}
	/// `c32`: measured on the classes that have a measurement, half of `c64` on the others.
	pub fn c32() -> f64 {
		rates().gemm_single.cplx
	}
}

/// Serial time of one row swap of one column in the LU swap stage on the Apple M5 Max,
/// nanoseconds (the `aarch64-apple` value of [`Rates::lu_swap`]; the other classes have their own): random pivot
/// rows, columns of 64 to 1024 `f64`, cache resident, median of 0.31 to 1.0 over
/// `m` 128 to 1024 and 16 to 128 swaps per column (Apple M5 Max, one thread; memory bound, so
/// the same for every dtype of the same element width).
pub const NS_PER_SWAP: f64 = 0.55;

/// Floor of the parallel regions of a factorization family: the one type of every
/// `par_threshold` field (LU, full-pivoting LU, Bunch-Kaufman, QR with column pivoting,
/// bidiagonalization, Hessenberg and tridiagonal reductions).
///
/// The default is [`Derived`](Self::Derived): each region is decided by [`ParShape::pays`], its
/// serial time (units of work times the family's serial throughput, [`rate`]) against the
/// dispatch cost of the parallelism it would use and the number of sections it opens.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ParThreshold {
	/// Derive the floor of every region from its serial time and the dispatch cost.
	#[default]
	Derived,
	/// A fixed floor in the work unit of the family (documented where the family declares its
	/// gate: elements of the updated trailing block for the reductions, `f64` elements for LU), a
	/// region runs in parallel from that many units. `Fixed(0)` and `Fixed(1)` make every region
	/// parallel that has any work.
	Fixed(usize),
	/// Never run a region in parallel.
	Never,
}

/// The integer form used by the C API: `0` is [`Derived`](ParThreshold::Derived),
/// `usize::MAX` is [`Never`](ParThreshold::Never), any other value is
/// [`Fixed`](ParThreshold::Fixed).
impl From<usize> for ParThreshold {
	fn from(value: usize) -> Self {
		match value {
			0 => ParThreshold::Derived,
			usize::MAX => ParThreshold::Never,
			n => ParThreshold::Fixed(n),
		}
	}
}

/// See [`From<usize>`](ParThreshold#impl-From<usize>-for-ParThreshold); `Fixed(0)` is
/// written as `1`, the same behaviour, because `0` means derived.
impl From<ParThreshold> for usize {
	fn from(value: ParThreshold) -> Self {
		match value {
			ParThreshold::Derived => 0,
			ParThreshold::Fixed(n) => n.max(1),
			ParThreshold::Never => usize::MAX,
		}
	}
}

impl ParThreshold {
	/// The width a region of `units` of work in the unit of `rate` that opens `sections`
	/// sections runs at on `par` (1: sequential); the caller runs it on a sub-team of that width
	/// ([`with_width`](crate::utils::thread::with_width)).
	///
	/// [`Derived`](Self::Derived) is [`par_width`] of `units * rate`, the widest width that pays;
	/// an explicit [`Fixed`](Self::Fixed) floor keeps its rule, the whole degree when
	/// `units >= floor`; [`Never`](Self::Never) is 1.
	#[cfg_attr(feature = "par-trace", track_caller)]
	pub(crate) fn region_width<T: crate::ComplexField>(
		self,
		par: &crate::Par,
		units: usize,
		rate: Rate,
		sections: u32,
	) -> usize {
		match self {
			Self::Never => 1,
			Self::Fixed(floor) => {
				if units >= floor {
					par.degree()
				} else {
					1
				}
			},
			Self::Derived => par_width::<T>(par, units, rate, sections),
		}
	}
}

/// Serial throughput of a kernel family, nanoseconds per unit of work for the real and the
/// complex native scalars, per host class ([`Rates`]). Measured with one thread, L2 resident
/// shapes; `f32` and `c32` take half of the `f64` and `c64` value (twice the lanes per register
/// and half the bytes; checked for the GEMM only, where the Zen 5 measurement agrees to 0.2
/// percent), a scalar that is not native takes the `f64` or `c64` value.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rate {
	/// `f64` (and, halved, `f32`).
	pub real: f64,
	/// `c64` (and, halved, `c32`).
	pub cplx: f64,
}

impl Rate {
	/// Nanoseconds per unit for the scalar `T`.
	pub fn of<T: crate::ComplexField>(self) -> f64 {
		if const { T::IS_NATIVE_F32 } {
			self.real / 2.0
		} else if const { T::IS_NATIVE_C32 } {
			self.cplx / 2.0
		} else if const { T::IS_REAL } {
			self.real
		} else {
			self.cplx
		}
	}
}

/// The serial throughput constants of the kernel families of one host class, one per family
/// (design record: `threadteam` `docs/design/thresholds.md`, section 2.3). The unit is what the
/// gate of the family counts. A step of a factorization is timed inside the whole serial
/// factorization, so the constant includes the serial parts of a step (pivot search, Householder
/// generation) and slightly overestimates the time of the part that is split.
///
/// The row of a class is [`Rates::for_class`]; [`rates`] is the row of the host. Sources:
///
/// * `x86_64-avx512`: one thread on an AMD Ryzen AI 9 HX 470 (Zen 5) at 2.0 GHz, boost off,
///   median over shapes and rounds (`faer-benchmark`
///   `data/probes/20261026-awake-team-linux/tables-consts-exec3.md`); the divide and conquer and
///   sparse product from the region constants of the same file (`raw-exec3/rconsts-*.csv`).
/// * `aarch64-apple`: one thread on an Apple M5 Max, QoS `UserInitiated`, L2 resident shapes,
///   median over the shapes 96 to 768 and two processes
///   (`data/probes/20261026-threshold-recalibration`).
/// * every other class is unmeasured and takes the `generic` row: per entry the smaller of the
///   measured rows, so that an unknown host keeps work serial a little longer than it must
///   rather than dispatching too early.
///
/// A row's rates and the dispatch costs of the same class must come from the same machine and
/// clock state ([`HostClass`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rates {
	/// GEMM and every kernel counted in multiply-adds (triangular products and solves), per
	/// multiply-add.
	pub gemm_mac: Rate,
	/// GEMM in single precision, per multiply-add: `real` is `f32`, `cplx` is `c32`.
	pub gemm_single: Rate,
	/// The multiply-adds of the block Householder apply, per multiply-add.
	pub block_apply_mac: Rate,
	/// Full-pivoting LU, one step: rank-one update fused with the pivot search, per element of
	/// the trailing block `(m-k-1)*(n-k-1)`.
	pub lu_full_pivot: Rate,
	/// Bunch-Kaufman with full pivoting, one step: rank-1 or rank-2 update fused with the pivot
	/// search, per element of the trailing block `rem*rem`.
	pub bunch_kaufman: Rate,
	/// QR with column pivoting, one step: fused update of the trailing block and the norms, per
	/// element `(m-k-1)*(n-k-1)`.
	pub qr_col_pivot: Rate,
	/// Bidiagonalization, one step: the fused two-sided update and the matrix-vector product, per
	/// element `(m-k-1)*(n-k-1)`.
	pub bidiag: Rate,
	/// Hessenberg reduction (unblocked), one step: two matrix-vector products and a rank-one
	/// update, per element `n*n` of the trailing block.
	pub hessenberg: Rate,
	/// Tridiagonalization, one step: the symmetric fused update, per element `rem*rem/2` of the
	/// triangle.
	pub tridiag: Rate,
	/// Matrix-vector product, per matrix element (the mean of the column major and row major
	/// products).
	pub matvec: Rate,
	/// Divide and conquer bidiagonal SVD (`bidiag_svd`, both singular vectors, non deflating
	/// data), the whole serial solve of size `m` per `m^2`; the join of size `n` has the two
	/// halves as work, `2 (n/2)^2`. The `m^2` term (secular equation, deflation, permutations)
	/// is larger than the merge products below `m` about 1900, so the unit is `m^2`; the
	/// constant is the median over `m` 64 to 512 (it is size dependent: 48 to 62 on the M5 fit,
	/// 145 to 178 on the Zen 5). Real types only (the complex value repeats the real one).
	pub bidiag_svd_dc: Rate,
	/// Divide and conquer symmetric tridiagonal EVD (`tridiag_evd`, eigenvectors on), the whole
	/// serial solve of size `m` per `m^2`, used as `bidiag_svd_dc` (88 to 102 on the Zen 5 over
	/// `m` 64 to 512). Real types only.
	pub tridiag_evd_dc: Rate,
	/// Triangular inverse (`invert_*_triangular`), the whole serial inverse of size `m` per
	/// `m^3`; the join of size `n` has the two halves as work, `2 (n/2)^3`. The constant is the
	/// median over `m` 64 to 512 and falls with the size (on the Zen 5, `f64` 0.047 per `m^3`
	/// at 64 to 0.0145 at 512), so a join near the floor is a little overestimated.
	pub triangular_inverse: Rate,
	/// Sparse by sparse numeric product, per multiply-add of the symbolic flop count
	/// (`SparseMatMulInfo`, free to read); the median over the measured densities and sizes.
	pub sparse_matmul: Rate,
	/// Rank-one Householder apply (`N == 1`), per element of the target `M*K`.
	pub householder_rank_one: Rate,
	/// One row swap of one column in the LU swap stage (random pivot rows, columns of 64 to 1024
	/// elements, cache resident, one thread; memory bound, so the `c64` value is the larger
	/// element width, not four times the `f64` one).
	pub lu_swap: Rate,
}

const fn fmin(a: f64, b: f64) -> f64 {
	if b < a { b } else { a }
}

const fn rmin(a: Rate, b: Rate) -> Rate {
	Rate { real: fmin(a.real, b.real), cplx: fmin(a.cplx, b.cplx) }
}

const fn r(real: f64, cplx: f64) -> Rate {
	Rate { real, cplx }
}

impl Rates {
	/// Per entry the smaller value of the two rows.
	const fn min_of(a: &Rates, b: &Rates) -> Rates {
		Rates {
			gemm_mac: rmin(a.gemm_mac, b.gemm_mac),
			gemm_single: rmin(a.gemm_single, b.gemm_single),
			block_apply_mac: rmin(a.block_apply_mac, b.block_apply_mac),
			lu_full_pivot: rmin(a.lu_full_pivot, b.lu_full_pivot),
			bunch_kaufman: rmin(a.bunch_kaufman, b.bunch_kaufman),
			qr_col_pivot: rmin(a.qr_col_pivot, b.qr_col_pivot),
			bidiag: rmin(a.bidiag, b.bidiag),
			hessenberg: rmin(a.hessenberg, b.hessenberg),
			tridiag: rmin(a.tridiag, b.tridiag),
			matvec: rmin(a.matvec, b.matvec),
			bidiag_svd_dc: rmin(a.bidiag_svd_dc, b.bidiag_svd_dc),
			tridiag_evd_dc: rmin(a.tridiag_evd_dc, b.tridiag_evd_dc),
			triangular_inverse: rmin(a.triangular_inverse, b.triangular_inverse),
			sparse_matmul: rmin(a.sparse_matmul, b.sparse_matmul),
			householder_rank_one: rmin(a.householder_rank_one, b.householder_rank_one),
			lu_swap: rmin(a.lu_swap, b.lu_swap),
		}
	}

	/// `x86_64-avx512`: Zen 5 at 2.0 GHz, boost off (see the type docs).
	pub const X86_64_AVX512: Rates = Rates {
		gemm_mac: r(0.0641, 0.2565),
		gemm_single: r(0.0320, 0.1283),
		block_apply_mac: r(0.0676, 0.2685),
		lu_full_pivot: r(0.4011, 0.8679),
		bunch_kaufman: r(0.8295, 1.3904),
		qr_col_pivot: r(0.3645, 1.0320),
		bidiag: r(1.0427, 2.1997),
		hessenberg: r(0.8576, 2.2212),
		tridiag: r(0.7695, 1.9252),
		// mean of matvec_colmajor (0.1743, 0.3373) and matvec_rowmajor (0.1565, 0.5057)
		matvec: r(0.1654, 0.4215),
		bidiag_svd_dc: r(152.9, 152.9),
		tridiag_evd_dc: r(97.4, 97.4),
		triangular_inverse: r(0.01936, 0.07228),
		// median over spmm_{f64,c64}_n{2000,4000,8000,20000}_d{4,6,8,12,16,40}
		sparse_matmul: r(4.1315, 5.2342),
		householder_rank_one: r(0.4783, 1.0303),
		// Zen 5 at 2.0 GHz, one thread, a scratch probe of the swap loop: swaps of 64 to 256 rows over 768 to 960 columns, one
		// thread: 0.98 (`f64`) and 1.30 (`c64`) ns, 1.0 at 256 rows; the 0.55 of the M5 was
		// used for every class and made the swap stage of a 64 x 1024 LU 1.7 times too cheap
		lu_swap: r(0.98, 1.30),
	};

	/// `aarch64-apple`: Apple M5 Max (the constants of the first calibration).
	pub const AARCH64_APPLE: Rates = Rates {
		gemm_mac: r(0.037, 0.165),
		gemm_single: r(0.037 / 2.0, 0.165 / 2.0),
		block_apply_mac: r(0.037, 0.165),
		lu_full_pivot: r(0.216, 0.6),
		bunch_kaufman: r(0.238, 0.38),
		qr_col_pivot: r(0.202, 0.713),
		bidiag: r(0.48, 1.36),
		hessenberg: r(0.451, 1.57),
		tridiag: r(0.339, 1.16),
		matvec: r(0.1, 0.34),
		bidiag_svd_dc: r(53.0, 53.0),
		tridiag_evd_dc: r(31.0, 31.0),
		triangular_inverse: r(0.0096, 0.0344),
		sparse_matmul: r(1.5, 1.8),
		householder_rank_one: r(0.212, 0.725),
		lu_swap: r(NS_PER_SWAP, NS_PER_SWAP),
	};

	/// `generic` (every unmeasured class): per entry the smaller of the measured rows.
	pub const GENERIC: Rates = Rates::min_of(&Rates::X86_64_AVX512, &Rates::AARCH64_APPLE);

	/// The row of a host class.
	pub fn for_class(class: HostClass) -> &'static Rates {
		static AVX512: Rates = Rates::X86_64_AVX512;
		static APPLE: Rates = Rates::AARCH64_APPLE;
		static GENERIC: Rates = Rates::GENERIC;
		match class {
			HostClass::X86_64Avx512 => &AVX512,
			HostClass::Aarch64Apple => &APPLE,
			HostClass::X86_64Avx2 | HostClass::X86_64Scalar | HostClass::Aarch64 | HostClass::Generic => &GENERIC,
		}
	}
}

/// The rates of the host class of this process ([`HostClass::detect`]).
pub fn rates() -> &'static Rates {
	Rates::for_class(HostClass::detect())
}

/// The serial throughput constants of the host, one function per family ([`Rates`] has the
/// units and the sources).
pub mod rate {
	use super::{Rate, rates};
	macro_rules! families {
		($($name:ident),* $(,)?) => {$(
			#[doc = core::concat!("`Rates::", core::stringify!($name), "` of the host class.")]
			pub fn $name() -> Rate { rates().$name }
		)*};
	}
	families!(
		gemm_mac,
		block_apply_mac,
		lu_full_pivot,
		bunch_kaufman,
		qr_col_pivot,
		bidiag,
		hessenberg,
		tridiag,
		matvec,
		bidiag_svd_dc,
		tridiag_evd_dc,
		triangular_inverse,
		sparse_matmul,
		householder_rank_one,
		lu_swap,
	);
}

/// Nanoseconds per multiply-add of `T` for a GEMM, by the table above. Non-native scalars use
/// the `f64` rate for real types and the `c64` rate for complex ones, a lower bound of their
/// time, so such a region stays serial a little longer rather than dispatching too early.
pub(crate) fn gemm_ns_per_mac<T: crate::ComplexField>() -> f64 {
	if const { T::IS_NATIVE_F32 } {
		ns_per_mac::f32()
	} else if const { T::IS_NATIVE_C32 } {
		ns_per_mac::c32()
	} else if const { T::IS_REAL } {
		ns_per_mac::f64()
	} else {
		ns_per_mac::c64()
	}
}


/// Decision trace of the gates, for probes (feature `par-trace`). A gate calls
/// [`trace::shape`] with the dimensions it decides on, and the next decision on the thread
/// records them together with the units, the section count, the estimate and the answer.
#[cfg(feature = "par-trace")]
pub mod trace {
	use std::sync::Mutex;
	use std::sync::atomic::{AtomicBool, Ordering};
	/// One decision.
	#[derive(Clone, Debug)]
	pub struct Record {
		/// source line of the gate that asked (`#[track_caller]`)
		pub site: &'static str,
		/// the dimensions the gate named with [`shape`]
		pub dims: [usize; 3],
		/// units of work passed to the rule
		pub units: usize,
		/// sections passed to the rule
		pub sections: u32,
		/// estimated serial time in ns
		pub estimate_ns: f64,
		/// dispatch cost of one section at `width`, in ns
		pub dispatch_ns: f64,
		/// the width the gate chose, or the width it asked about (1: sequential)
		pub width: usize,
		/// the decision
		pub parallel: bool,
	}
	/// Override of the block apply column split gate, for probes: 0 the rule, 1 the previous
	/// rule (`M * K * 4 * M >= gemm threshold`), 2 always parallel, 3 never.
	pub static N_TASKS_MODE: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
	/// Probe (faer#160): nanoseconds and calls of the stages of the top level of the blocked QR,
	/// indexed by [`PHASE_PANEL`] and the constants after it. Temporary; not for merging.
	pub static PHASE_NS: [core::sync::atomic::AtomicU64; 5] = [const { core::sync::atomic::AtomicU64::new(0) }; 5];
	/// calls per stage, as [`PHASE_NS`]
	pub static PHASE_CALLS: [core::sync::atomic::AtomicU64; 5] = [const { core::sync::atomic::AtomicU64::new(0) }; 5];
	/// the recursive panel call at depth 0 (includes the leaves and the panel's own applies)
	pub const PHASE_PANEL: usize = 0;
	/// the unblocked leaves, at every depth
	pub const PHASE_LEAF: usize = 1;
	/// `upgrade_householder_factor` and the factor merge at depth 0
	pub const PHASE_FACTOR: usize = 2;
	/// the trailing block apply at depth 0
	pub const PHASE_APPLY: usize = 3;
	/// the block applies inside the panel (depth > 0)
	pub const PHASE_PANEL_APPLY: usize = 4;
	std::thread_local! {
		/// recursion depth of the blocked QR on this thread
		pub static QR_DEPTH: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
	}
	/// Add `ns` to stage `phase`.
	pub fn phase_add(phase: usize, ns: u64) {
		PHASE_NS[phase].fetch_add(ns, Ordering::Relaxed);
		PHASE_CALLS[phase].fetch_add(1, Ordering::Relaxed);
	}
	/// Zero the stage counters.
	pub fn phase_reset() {
		for i in 0..5 {
			PHASE_NS[i].store(0, Ordering::Relaxed);
			PHASE_CALLS[i].store(0, Ordering::Relaxed);
		}
	}
	static ON: AtomicBool = AtomicBool::new(false);
	static LOG: Mutex<Vec<Record>> = Mutex::new(Vec::new());
	std::thread_local! {
		static DIMS: core::cell::Cell<(&'static str, [usize; 3])> = const { core::cell::Cell::new(("", [0; 3])) };
	}
	/// Start or stop recording; starting clears the log.
	pub fn enable(on: bool) {
		if on {
			LOG.lock().unwrap().clear();
		}
		ON.store(on, Ordering::SeqCst);
	}
	/// Take the log.
	pub fn take() -> Vec<Record> {
		core::mem::take(&mut *LOG.lock().unwrap())
	}
	/// Name the dimensions of the next decision of this thread.
	pub fn shape(label: &'static str, a: usize, b: usize, c: usize) {
		DIMS.with(|d| d.set((label, [a, b, c])));
	}
	pub(crate) fn record(
		loc: &'static core::panic::Location<'static>,
		units: usize,
		sections: u32,
		estimate_ns: f64,
		dispatch_ns: f64,
		width: usize,
		parallel: bool,
	) {
		if !ON.load(Ordering::Relaxed) {
			return;
		}
		let (label, dims) = DIMS.with(|d| d.replace(("", [0; 3])));
		let site: &'static str = if label.is_empty() {
			Box::leak(format!("{}:{}", loc.file().rsplit('/').next().unwrap(), loc.line()).into_boxed_str())
		} else {
			label
		};
		LOG.lock().unwrap().push(Record { site, dims, units, sections, estimate_ns, dispatch_ns, width, parallel });
	}
}

/// [`ParShape::pays`] of `par` at `width` participants, without building the shape. A gate is asked
/// hundreds of times by a small factorization, and the shape of a team token copies its whole cost
/// table on every call (`Par::shape`: 15 ns, the decision itself 40 ns, against 12 ns here, where the
/// token answers from its own table). `width` is clamped to the degree, below 2 never pays.
pub(crate) fn par_pays_time(par: &crate::Par, estimate: Duration, sections: u32, width: usize) -> bool {
	let width = width.min(par.degree());
	let Some(width) = NonZeroUsize::new(width).filter(|w| w.get() >= 2) else {
		return false;
	};
	match par {
		crate::Par::Seq => false,
		#[cfg(feature = "threadteam")]
		crate::Par::Team(team) => {
			// the same cost the shape reports (threadteam's table, extrapolated by threadteam above
			// its largest anchor)
			pays_at(estimate, sections, team.dispatch_cost(width))
		},
		// a build without the route has no team token to ask
		#[cfg(not(feature = "threadteam"))]
		_ => false,
	}
}

/// Whether a region of serial time `estimate` that opens `sections` sections can pay on `par` at
/// *some* width: `estimate >= PAYS * sections * min over widths of the cost` ([`ParShape::min_dispatch_cost`]).
/// "Does not pay at the cheapest width" is the rejection that is right for any shape of table (a
/// rising one included); it says nothing about the width a region then chooses.
pub(crate) fn par_can_pay_time(par: &crate::Par, estimate: Duration, sections: u32) -> bool {
	if par.degree() < 2 {
		return false;
	}
	match par {
		crate::Par::Seq => false,
		#[cfg(feature = "threadteam")]
		crate::Par::Team(team) => par_can_pay_time_team(team, estimate, sections),
		#[cfg(not(feature = "threadteam"))]
		_ => false,
	}
}

/// [`par_can_pay_time`] for a team token.
#[cfg(feature = "threadteam")]
pub(crate) fn par_can_pay_time_team(team: &threadteam::TeamExec<'_>, estimate: Duration, sections: u32) -> bool {
	team.size().get() >= 2
		&& pays_at(estimate, sections, team.dispatch_costs().min_cost(threadteam::DispatchKind::Team))
}

/// The widest width in `2..=degree` at which a region of serial time `estimate` that opens
/// `sections` sections pays on `par`, or 1 (sequential) when none does.
///
/// Every width is considered: the search runs from the full width down and stops at the first
/// that pays, which is the widest. A narrower width that does not pay says nothing about a wider
/// one, because a measured row can fall with the width (`aarch64-apple`'s team row is 3.0, 2.3,
/// 1.4 us at widths 2, 4, 6).
pub(crate) fn par_width_time(par: &crate::Par, estimate: Duration, sections: u32) -> usize {
	par_width_time_upto(par, estimate, sections, usize::MAX)
}

/// [`par_width_time`] among the widths up to `max` (a region that cannot use more participants,
/// such as a split into at most `max` tasks).
pub(crate) fn par_width_time_upto(par: &crate::Par, estimate: Duration, sections: u32, max: usize) -> usize {
	match par {
		#[cfg(feature = "threadteam")]
		crate::Par::Team(team) => par_width_team(team, estimate, sections, max),
		_ => {
			let _ = (estimate, sections, max);
			1
		},
	}
}

/// [`par_width_time_upto`] for a team token.
#[cfg(feature = "threadteam")]
pub(crate) fn par_width_team(team: &threadteam::TeamExec<'_>, estimate: Duration, sections: u32, max: usize) -> usize {
	// the cheap rejection first: a region that does not pay at the cheapest width of the table
	// pays at none (a QR 128 asks hundreds of times)
	if max < 2 || !par_can_pay_time_team(team, estimate, sections) {
		return 1;
	}
	(2..=team.size().get().min(max))
		.rev()
		.find(|&w| pays_at(estimate, sections, team.dispatch_cost(NonZeroUsize::new(w).unwrap())))
		.unwrap_or(1)
}

/// [`par_width_time`] of `units` of work at `rate`: the width the region runs at (1:
/// sequential).
#[cfg_attr(feature = "par-trace", track_caller)]
pub(crate) fn par_width<T: crate::ComplexField>(
	par: &crate::Par,
	units: usize,
	rate: Rate,
	sections: u32,
) -> usize {
	par_width_upto::<T>(par, units, rate, sections, usize::MAX)
}

/// [`par_width`] among the widths up to `max`.
#[cfg_attr(feature = "par-trace", track_caller)]
pub(crate) fn par_width_upto<T: crate::ComplexField>(
	par: &crate::Par,
	units: usize,
	rate: Rate,
	sections: u32,
	max: usize,
) -> usize {
	if par.degree() < 2 || max < 2 {
		return 1;
	}
	let w = par_width_time_upto(par, serial_time(units, rate.of::<T>()), sections, max);
	#[cfg(feature = "par-trace")]
	trace_decision(par.shape(), w, units, rate.of::<T>(), sections, w > 1, core::panic::Location::caller());
	w
}

#[cfg(feature = "par-trace")]
fn trace_decision(
	shape: ParShape,
	width: usize,
	units: usize,
	ns: f64,
	sections: u32,
	r: bool,
	loc: &'static core::panic::Location<'static>,
) {
	let d = NonZeroUsize::new(width)
		.map(|w| shape.dispatch_cost(w).as_secs_f64() * 1e9)
		.unwrap_or(f64::INFINITY);
	trace::record(loc, units, sections, units as f64 * ns, d, width, r);
}

/// Whether a join of two halves pays: one section of width 2 (the two halves are the
/// participants; the budget is split between them, it is not a flat section over all of it), so
/// the dispatch cost is that of the shape at width 2 (`Team` 3.0 us, `Rayon` from outside 14 us
/// on the M5 table, against 1.4 and 19 at width 6). `units` of work at `rate`, `sections` joins.
#[cfg_attr(feature = "par-trace", track_caller)]
pub(crate) fn join_pays<T: crate::ComplexField>(
	par: &crate::Par,
	units: usize,
	rate: Rate,
	sections: u32,
) -> bool {
	if par.degree() < 2 {
		return false;
	}
	let r = par_pays_time(par, serial_time(units, rate.of::<T>()), sections, 2);
	#[cfg(feature = "par-trace")]
	trace_decision(par.shape(), 2, units, rate.of::<T>(), sections, r, core::panic::Location::caller());
	r
}

/// Whether the join of a recursion that halves a problem of size `n` pays: its work is the two
/// halves, `2 (n/2)^power` units of `rate`, one join ([`join_pays`]). The rate is that of a whole serial
/// solve of size `m` per `m^power`, so no recurrence is assumed beyond the power: the work of
/// the subproblems at this level is what the two calls below it cost, and the merge after the
/// join is not counted (it is not parallel through this join). `power` is 3 for the triangular
/// inverse and 2 for the divide and conquer solves, whose measured time is `a m^2 + b m^3`
/// with the first term dominating up to `m` about 1900.
pub(crate) fn halves_pays<T: crate::ComplexField>(
	par: &crate::Par,
	n: usize,
	rate: Rate,
	power: u32,
) -> bool {
	let h = n / 2;
	join_pays::<T>(par, h.saturating_pow(power).saturating_mul(2), rate, 1)
}

/// Estimated serial time of a block Householder apply with an `m x n` basis on an `m x k`
/// target: the two products of the update, `2 * m * n * k` multiply-adds at the GEMM rate, plus a
/// streaming term per target element (the temporaries and the triangular steps, `m * k` elements at
/// the rank one apply rate). Fitted over `m` 24 to 512, `n` 2 to 32 with one thread: the median
/// error is 7 percent (f64), 4 percent (c64); a single rate per multiply-add was 1.0 to 2.7 times
/// too low for the narrow bases (`n` 2 to 8).
pub(crate) fn block_apply_estimate<T: crate::ComplexField>(
	m: usize,
	n: usize,
	k: usize,
) -> Duration {
	block_apply_estimate_with::<T>(rates(), m, n, k)
}

/// [`block_apply_estimate`] on the row `rt`.
pub(crate) fn block_apply_estimate_with<T: crate::ComplexField>(
	rt: &Rates,
	m: usize,
	n: usize,
	k: usize,
) -> Duration {
	let macs = m.saturating_mul(n).saturating_mul(k).saturating_mul(2);
	// summed in nanoseconds: one `Duration`, not two and an addition (a gate asked per block apply)
	let ns = macs as f64 * rt.block_apply_mac.of::<T>() + m.saturating_mul(k) as f64 * rt.householder_rank_one.of::<T>();
	Duration::from_secs_f64(ns * 1e-9)
}

/// Serial time of `count` units at `ns` nanoseconds each.
pub(crate) fn serial_time(count: usize, ns: f64) -> Duration {
	Duration::from_secs_f64(count as f64 * ns * 1e-9)
}

impl ParShape {
	/// The cost of one parallel section of `width` participants on this shape, in the steady
	/// state (sections back to back); [`Duration::MAX`] for a sequential shape or `width < 2`.
	/// `width` is clamped to the degree of the shape.
	///
	/// `Team` is the cost of the team token the shape was taken from (threadteam's
	/// `DispatchKind::Team`, with the costs of that token's pool). Above the largest measured
	/// anchor of the row threadteam grows the cost linearly with the width (threadteam 0.1.1);
	/// faer keeps no second model.
	pub fn dispatch_cost(self, width: NonZeroUsize) -> Duration {
		let width = width.min(NonZeroUsize::new(self.degree().max(1)).unwrap());
		if width.get() < 2 {
			return Duration::MAX;
		}
		match self {
			Self::Seq => Duration::MAX,
			#[cfg(feature = "threadteam")]
			Self::Team(_, costs) => costs.cost(threadteam::DispatchKind::Team, width),
		}
	}

	/// The cheapest cost of one section over all widths of this shape (a lower bound of
	/// [`dispatch_cost`](Self::dispatch_cost) at every width), [`Duration::MAX`] for a shape of
	/// degree below 2.
	pub fn min_dispatch_cost(self) -> Duration {
		if self.degree() < 2 {
			return Duration::MAX;
		}
		match self {
			Self::Seq => Duration::MAX,
			#[cfg(feature = "threadteam")]
			Self::Team(_, costs) => costs.min_cost(threadteam::DispatchKind::Team),
		}
	}

	/// Whether a region of serial time `estimate` that opens `sections` sections of `width`
	/// participants pays on this shape: `estimate >= PAYS * sections * dispatch_cost(width)`.
	/// `sections` is the number of dispatches the region really performs.
	pub fn pays(self, estimate: Duration, sections: u32, width: NonZeroUsize) -> bool {
		pays_at(estimate, sections, self.dispatch_cost(width))
	}

	/// [`pays`](Self::pays) at the full degree of the shape.
	pub fn pays_full(self, estimate: Duration, sections: u32) -> bool {
		match NonZeroUsize::new(self.degree()) {
			Some(width) => self.pays(estimate, sections, width),
			None => false,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::c64;

	fn nz(n: usize) -> NonZeroUsize {
		NonZeroUsize::new(n).unwrap()
	}

	#[cfg(feature = "rayon")]
	#[test]
	fn seq_never_pays() {
		assert_eq!(ParShape::Seq.dispatch_cost(nz(4)), Duration::MAX);
		assert!(!ParShape::Seq.pays(Duration::from_secs(10), 1, nz(4)));
		assert!(!ParShape::Seq.pays_full(Duration::from_secs(10), 1));
	}

	#[cfg(feature = "rayon")]
	#[test]
	fn min_dispatch_cost_is_the_cheapest_width() {
		use threadteam::{DispatchCosts, DispatchKind, WidthCosts};
		let us = Duration::from_micros;
		// falling, flat and rising tables
		for t in [[us(9), us(5), us(3)], [us(4), us(4), us(4)], [us(2), us(7), us(12)]] {
			let costs = DispatchCosts::for_class(threadteam::HostClass::Generic).with(DispatchKind::Team, WidthCosts::new(t));
			let s = ParShape::Team(nz(6), costs);
			let want = (2..=6).map(|w| s.dispatch_cost(nz(w))).min().unwrap();
			assert_eq!(s.min_dispatch_cost(), want, "{t:?}");
		}
		assert_eq!(ParShape::Seq.min_dispatch_cost(), Duration::MAX);
		assert_eq!(ParShape::Team(nz(1), DispatchCosts::for_class(threadteam::HostClass::Generic)).min_dispatch_cost(), Duration::MAX);
	}

	#[cfg(feature = "rayon")]
	#[test]
	fn pays_is_the_multiple_of_the_cost() {
		let d = Duration::from_micros(2);
		assert!(pays_at(Duration::from_micros(6), 1, d));
		assert!(!pays_at(Duration::from_nanos(5999), 1, d));
		assert!(!pays_at(Duration::from_micros(11), 2, d));
		assert!(pays_at(Duration::from_micros(12), 2, d));
		assert!(!pays_at(Duration::from_secs(1), 1, Duration::MAX));
	}

	#[test]
	fn team_shape_carries_the_costs_of_its_token() {
		use threadteam::{DispatchCosts, DispatchKind, WidthCosts};
		let costs = DispatchCosts::for_class(threadteam::HostClass::Generic)
			.with(DispatchKind::Team, WidthCosts::flat(Duration::from_micros(5)));
		let s = ParShape::Team(nz(4), costs);
		assert_eq!(s.dispatch_cost(nz(4)), Duration::from_micros(5));
		assert_eq!(s.dispatch_cost(nz(1)), Duration::MAX);
		// away from the exact boundary: `pays` compares `Duration`s as floats, and
		// `3 * 5 us` and `15 us` are not the same double
		assert!(s.pays(Duration::from_micros(16), 1, nz(4)));
		assert!(!s.pays(Duration::from_micros(14), 1, nz(4)));
	}

	#[test]
	fn gemm_rates_are_per_dtype() {
		assert_eq!(gemm_ns_per_mac::<f64>(), ns_per_mac::f64());
		assert_eq!(gemm_ns_per_mac::<crate::c64>(), ns_per_mac::c64());
		assert_eq!(gemm_ns_per_mac::<f32>(), ns_per_mac::f32());
		assert_eq!(gemm_ns_per_mac::<crate::c32>(), ns_per_mac::c32());
		// the complex rate is 4.5 times the real one, not 1: the floor bug of the team route
		assert!(ns_per_mac::c64() > 4.0 * ns_per_mac::f64());
	}

	/// Smallest `units` (doubling, then bisection) for which the region pays on `shape`.
	fn floor_units<T: crate::ComplexField>(
		shape: ParShape,
		r: Rate,
		sections: u32,
	) -> usize {
		let pays = |u: usize| {
			shape.pays_full(serial_time(u, r.of::<T>()), sections)
		};
		let mut hi = 1usize;
		while !pays(hi) {
			hi *= 2;
			assert!(hi < 1 << 40, "never pays");
		}
		let mut lo = hi / 2;
		while hi - lo > 1 {
			let mid = (lo + hi) / 2;
			if pays(mid) {
				hi = mid;
			} else {
				lo = mid;
			}
		}
		hi
	}

	/// (name, rate, sections as counted in the code of the gate), on the row `rt`
	fn gates(rt: &Rates) -> [(&'static str, Rate, u32); 10] {
		[
			("lu_full_pivot", rt.lu_full_pivot, 1),
			("bunch_kaufman", rt.bunch_kaufman, 1),
			("qr_col_pivot", rt.qr_col_pivot, 1),
			("bidiag", rt.bidiag, 2),
			("hessenberg", rt.hessenberg, 2),
			("tridiag", rt.tridiag, 1),
			("matvec", rt.matvec, 1),
			("householder_rank_one", rt.householder_rank_one, 1),
			("gemm_mac", rt.gemm_mac, 1),
			("sparse_matmul", rt.sparse_matmul, 1),
		]
	}

	/// The M5 row: the tests that state numbers of the Apple team table use it with that table
	/// (rates and costs of one row).
	const M5: &Rates = &Rates::AARCH64_APPLE;

	/// Every entry of a row, as `(name, Rate)`.
	fn entries(rt: &Rates) -> [(&'static str, Rate); 16] {
		[
			("gemm_mac", rt.gemm_mac),
			("gemm_single", rt.gemm_single),
			("block_apply_mac", rt.block_apply_mac),
			("lu_full_pivot", rt.lu_full_pivot),
			("bunch_kaufman", rt.bunch_kaufman),
			("qr_col_pivot", rt.qr_col_pivot),
			("bidiag", rt.bidiag),
			("hessenberg", rt.hessenberg),
			("tridiag", rt.tridiag),
			("matvec", rt.matvec),
			("bidiag_svd_dc", rt.bidiag_svd_dc),
			("tridiag_evd_dc", rt.tridiag_evd_dc),
			("triangular_inverse", rt.triangular_inverse),
			("sparse_matmul", rt.sparse_matmul),
			("householder_rank_one", rt.householder_rank_one),
			("lu_swap", rt.lu_swap),
		]
	}

	#[test]
	fn every_class_resolves_to_its_table() {
		assert_eq!(Rates::for_class(HostClass::X86_64Avx512), &Rates::X86_64_AVX512);
		assert_eq!(Rates::for_class(HostClass::Aarch64Apple), &Rates::AARCH64_APPLE);
		for c in [HostClass::X86_64Avx2, HostClass::X86_64Scalar, HostClass::Aarch64, HostClass::Generic] {
			assert_eq!(Rates::for_class(c), &Rates::GENERIC, "{}", c.name());
		}
		// the rows are the measured ones
		assert_eq!(Rates::X86_64_AVX512.gemm_mac, Rate { real: 0.0641, cplx: 0.2565 });
		assert_eq!(Rates::X86_64_AVX512.gemm_single, Rate { real: 0.0320, cplx: 0.1283 });
		assert_eq!(Rates::AARCH64_APPLE.gemm_mac, Rate { real: 0.037, cplx: 0.165 });
		assert_eq!(Rates::AARCH64_APPLE.bidiag_svd_dc, Rate { real: 53.0, cplx: 53.0 });
		// the single precision GEMM entry of the first row agrees with half of the double one
		let g = Rates::X86_64_AVX512;
		assert!((g.gemm_single.real / (g.gemm_mac.real / 2.0) - 1.0).abs() < 0.01);
		assert!((g.gemm_single.cplx / (g.gemm_mac.cplx / 2.0) - 1.0).abs() < 0.01);
	}

	#[test]
	fn generic_is_the_per_entry_minimum_of_the_measured_rows() {
		let (a, b, g) = (entries(&Rates::X86_64_AVX512), entries(&Rates::AARCH64_APPLE), entries(&Rates::GENERIC));
		for i in 0..a.len() {
			let (name, ga) = (a[i].0, a[i].1);
			assert_eq!(g[i].1.real, ga.real.min(b[i].1.real), "{name} real");
			assert_eq!(g[i].1.cplx, ga.cplx.min(b[i].1.cplx), "{name} cplx");
			assert!(g[i].1.real <= ga.real && g[i].1.real <= b[i].1.real, "{name}");
		}
	}

	#[test]
	fn rows_are_positive() {
		for c in HostClass::ALL {
			for (name, r) in entries(Rates::for_class(c)) {
				assert!(r.real > 0.0 && r.cplx > 0.0, "{} {name}", c.name());
			}
		}
	}

	#[test]
	fn detection_is_stable_and_the_host_row_is_its_class() {
		let c = HostClass::detect();
		for _ in 0..4 {
			assert_eq!(HostClass::detect(), c);
		}
		assert_eq!(HostClass::detect_uncached(), c);
		assert_eq!(rates(), Rates::for_class(c));
		assert_eq!(rate::bidiag(), rates().bidiag);
		#[cfg(all(target_arch = "x86_64", feature = "std"))]
		if std::arch::is_x86_feature_detected!("avx512f") {
			assert_eq!(c, HostClass::X86_64Avx512);
		}
		#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
		assert_eq!(c, HostClass::Aarch64Apple);
	}

	/// faer's `HostClass` is kept in sync by hand with `threadteam`'s; they must agree.
	#[test]
	fn host_class_agrees_with_threadteam() {
		let tt = |c: HostClass| match c {
			HostClass::X86_64Avx512 => threadteam::HostClass::X86_64Avx512,
			HostClass::X86_64Avx2 => threadteam::HostClass::X86_64Avx2,
			HostClass::X86_64Scalar => threadteam::HostClass::X86_64Scalar,
			HostClass::Aarch64Apple => threadteam::HostClass::Aarch64Apple,
			HostClass::Aarch64 => threadteam::HostClass::Aarch64,
			HostClass::Generic => threadteam::HostClass::Generic,
		};
		assert_eq!(tt(HostClass::detect()), threadteam::HostClass::detect());
		for c in HostClass::ALL {
			assert_eq!(c.name(), tt(c).name());
		}
	}

	/// The team cost is threadteam's table, nothing else: faer keeps no second extrapolation
	/// (threadteam-rs#6). Above a row's largest anchor threadteam grows the cost in proportion to
	/// the width, so a region that does not pay at eight members does not pay at 64.
	#[test]
	fn the_team_cost_is_threadteams_table_at_every_width() {
		use threadteam::{DispatchCosts, DispatchKind, HostClass, WidthCosts};
		let us = Duration::from_micros;
		let rows = [
			DispatchCosts::for_class(HostClass::Generic),
			DispatchCosts::for_class(HostClass::X86_64Avx512),
			DispatchCosts::for_class(HostClass::X86_64Avx2),
			DispatchCosts::for_class(HostClass::Aarch64Apple),
			DispatchCosts::for_class(HostClass::Generic).with(DispatchKind::Team, WidthCosts::flat(us(1))),
		];
		for costs in rows {
			let s = ParShape::Team(nz(128), costs);
			for w in 2..=128 {
				assert_eq!(s.dispatch_cost(nz(w)), costs.cost(DispatchKind::Team, nz(w)), "{costs:?} width {w}");
			}
		}
		// above the largest anchor: `cost(w_m) * w / w_m` in integer nanoseconds. The flat row ends
		// at 6, the avx512 team row at 8, the Zen2 avx2 row at 64.
		let flat = ParShape::Team(nz(128), rows[4]);
		for (w, ns) in [(8, 1333), (12, 2000), (16, 2666), (32, 5333), (64, 10666), (128, 21333)] {
			assert_eq!(flat.dispatch_cost(nz(w)), Duration::from_nanos(ns), "flat width {w}");
		}
		let avx512 = ParShape::Team(nz(128), rows[1]);
		assert_eq!(avx512.dispatch_cost(nz(8)), Duration::from_nanos(980));
		assert_eq!(avx512.dispatch_cost(nz(64)), Duration::from_nanos(980 * 8));
		let zen2 = ParShape::Team(nz(128), rows[2]);
		assert_eq!(zen2.dispatch_cost(nz(64)), Duration::from_nanos(9333));
		assert_eq!(zen2.dispatch_cost(nz(128)), Duration::from_nanos(18666));
		// a region that does not pay at 8 members does not pay at 64 either
		let at_8 = flat.dispatch_cost(nz(8)) * 3;
		assert!(flat.pays(at_8, 1, nz(8)));
		assert!(!flat.pays(at_8, 1, nz(64)));
	}

	#[test]
	fn fixed_and_never_keep_their_meaning() {
		let r = rate::lu_full_pivot();
		// a fixed floor takes the whole degree at the floor, nothing below it; never is serial
		assert_eq!(ParThreshold::Fixed(100).region_width::<f64>(&crate::Par::Seq, 100, r, 1), 1);
		crate::team::with_team(6, |par| {
			assert_eq!(ParThreshold::Never.region_width::<f64>(&par, usize::MAX, r, 1), 1);
			assert_eq!(ParThreshold::Fixed(100).region_width::<f64>(&par, 99, r, 1), 1);
			assert_eq!(ParThreshold::Fixed(100).region_width::<f64>(&par, 100, r, 1), 6);
		});
		// the integer form of the C API
		assert_eq!(ParThreshold::from(0), ParThreshold::Derived);
		assert_eq!(ParThreshold::from(usize::MAX), ParThreshold::Never);
		assert_eq!(ParThreshold::from(7), ParThreshold::Fixed(7));
		assert_eq!(usize::from(ParThreshold::Fixed(0)), 1);
	}

	#[test]
	fn derived_gates_never_pay_on_a_sequential_shape() {
		let seq = crate::Par::Seq;
		for (_, r, sections) in gates(rates()) {
			assert_eq!(ParThreshold::Derived.region_width::<f64>(&seq, usize::MAX / 2, r, sections), 1);
			assert_eq!(ParThreshold::Derived.region_width::<c64>(&seq, usize::MAX / 2, r, sections), 1);
			assert!(!ParShape::Seq.pays_full(serial_time(usize::MAX / 2, r.of::<f64>()), sections));
		}
	}

	#[test]
	fn derived_floors_on_a_team_follow_the_rule() {
		use threadteam::DispatchCosts;
		let team = ParShape::Team(nz(6), DispatchCosts::for_class(threadteam::HostClass::Aarch64Apple));
		let d = team.dispatch_cost(nz(6)).as_secs_f64() * 1e9;
		for (name, r, sections) in gates(M5) {
			for (ty, ns) in [("f64", r.real), ("c64", r.cplx)] {
				let floor = if ty == "f64" {
					floor_units::<f64>(team, r, sections)
				} else {
					floor_units::<c64>(team, r, sections)
				};
				// the floor is PAYS * sections * D / rate, within the rounding of a `Duration` (nanoseconds)
				let want = PAYS * f64::from(sections) * d / ns;
				assert!(
					(floor as f64 - want).abs() <= 4.0 + want * 2e-3,
					"{name} {ty}: floor {floor}, rule {want}"
				);
			}
		}
		// representative sizes at width 6 on the M5 table (D = 1.4 us, 4.2 us per section)
		let pays = |u: usize, r: Rate, f: u32| {
			team.pays_full(serial_time(u, r.of::<f64>()), f)
		};
		// full pivoting LU: a 100 x 100 trailing block stays serial, 200 x 200 is parallel
		assert!(!pays(100 * 100, M5.lu_full_pivot, 1));
		assert!(pays(200 * 200, M5.lu_full_pivot, 1));
		// bidiagonalization counts two sections: it needs twice the work of a one-section step
		assert!(floor_units::<f64>(team, M5.bidiag, 2) >= 2 * floor_units::<f64>(team, M5.bidiag, 1) - 2);
		// a 256 x 256 matrix-vector product (65536 elements) pays on a team, not below 100 x 100
		assert!(pays(256 * 256, M5.matvec, 1));
		assert!(!pays(64 * 64, M5.matvec, 1));
		// the old fixed floors were 192*256 = 49152 elements for every one of these steps; the
		// derived floors differ per family (the faster the kernel, the larger the floor)
		let lu = floor_units::<f64>(team, M5.lu_full_pivot, 1);
		let bidiag = floor_units::<f64>(team, M5.bidiag, 1);
		assert!(lu > bidiag);
	}

	#[test]
	fn par_width_reads_the_degree_of_the_token() {
		let seq = crate::Par::Seq;
		assert_eq!(par_width::<f64>(&seq, usize::MAX / 2, rate::matvec(), 1), 1);
		crate::team::with_team(6, |par| {
			assert_eq!(par_width::<f64>(&par, 10_000_000, rate::matvec(), 1), 6);
			assert_eq!(par_width::<f64>(&par, 1_000, rate::matvec(), 1), 1);
			// a cap below the degree bounds the search
			assert_eq!(par_width_upto::<f64>(&par, 10_000_000, rate::matvec(), 1, 3), 3);
		});
		crate::team::with_team(1, |par| {
			assert_eq!(par_width::<f64>(&par, 10_000_000, rate::matvec(), 1), 1);
		});
	}

	#[test]
	fn rates_are_per_dtype() {
		let r = rate::bidiag();
		assert_eq!(r.of::<f64>(), r.real);
		assert_eq!(r.of::<c64>(), r.cplx);
		assert_eq!(r.of::<f32>(), r.real / 2.0);
		assert_eq!(r.of::<crate::c32>(), r.cplx / 2.0);
		assert_eq!(rate::gemm_mac().of::<f64>(), ns_per_mac::f64());
		assert_eq!(rate::gemm_mac().of::<c64>(), ns_per_mac::c64());
	}

	/// `Fixed(1)` makes every step of every reduction parallel, and the factors must equal the
	/// serial ones: the `Derived` default keeps the sizes of the other tests below the floors, so
	/// the parallel branches of the gated steps are exercised here.
	#[cfg(feature = "rayon")]
	mod forced_parallel {
		use super::*;
		use crate::dyn_stack::{MemBuffer, MemStack};
		use crate::internal_prelude::*;
		use crate::linalg::cholesky::lblt::factor as bk;
		use crate::linalg::evd::{hessenberg, tridiag};
		use crate::linalg::lu::full_pivoting::factor as fplu;
		use crate::linalg::qr::col_pivoting::factor as cpqr;
		use crate::linalg::svd::bidiag;
		use crate::{Auto, Mat, MatRef, Par};

		const N: usize = 96;

		fn fill<T: crate::ComplexField>(n: usize, seed: usize) -> Mat<T> {
			Mat::from_fn(n, n, |i, j| {
				let h = (i * 2654435761usize)
					.wrapping_add(j * 40503)
					.wrapping_add(seed * 977)
					% 1009;
				from_f64::<T>(h as f64 / 504.0 - 1.0)
			})
		}
		fn close<T: crate::ComplexField<Real = f64>>(
			what: &str,
			a: MatRef<'_, T>,
			b: MatRef<'_, T>,
		) {
			let mut err = 0.0f64;
			for j in 0..a.ncols() {
				for i in 0..a.nrows() {
					err = err.max((a[(i, j)].clone() - b[(i, j)].clone()).abs());
				}
			}
			core::assert!(err <= 1e-9, "{what}: error {err}");
		}

		fn check<T: crate::ComplexField<Real = f64>>(compare_complex_factors: bool) {
			let a0 = fill::<T>(N, 1);
			let sym = &a0 + a0.adjoint();
			// results with a given par and `Fixed(1)`
			let lu = |par: Par| {
				let mut a = a0.clone();
				let mut p = vec![0usize; 4 * N];
				let (r, rest) = p.split_at_mut(N);
				let (ri, rest) = rest.split_at_mut(N);
				let (c, ci) = rest.split_at_mut(N);
				let mut params: fplu::FullPivLuParams = Auto::<T>::auto();
				params.par_threshold = ParThreshold::Fixed(1);
				let mut mem = MemBuffer::new(fplu::lu_in_place_scratch::<usize, T>(
					N,
					N,
					par.shape(),
					params.into(),
				));
				fplu::lu_in_place(
					a.as_mut(), r, ri, c, ci, par, MemStack::new(&mut mem), params.into(),
				);
				a
			};
			crate::team::with_team(4, |par| close("lu", lu(Par::Seq).as_ref(), lu(par).as_ref()));
			let bunch = |par: Par| {
				let mut a = sym.clone();
				let mut perm = vec![0usize; N];
				let mut perm_inv = vec![0usize; N];
				let mut sub = Mat::<T>::zeros(N, 1);
				let mut params: bk::LbltParams = Auto::<T>::auto();
				params.pivoting = bk::PivotingStrategy::Full;
				params.par_threshold = ParThreshold::Fixed(1);
				let mut mem = MemBuffer::new(bk::cholesky_in_place_scratch::<usize, T>(
					N,
					par.shape(),
					params.into(),
				));
				bk::cholesky_in_place(
					a.as_mut(),
					sub.as_mut().col_mut(0).as_diagonal_mut(),
					&mut perm,
					&mut perm_inv,
					par,
					MemStack::new(&mut mem),
					params.into(),
				);
				a
			};
			crate::team::with_team(4, |par| close("bunch", bunch(Par::Seq).as_ref(), bunch(par).as_ref()));
			let qr = |par: Par| {
				let mut a = a0.clone();
				let mut h = Mat::<T>::zeros(16, N);
				let mut p = vec![0usize; N];
				let mut pi = vec![0usize; N];
				let mut params: cpqr::ColPivQrParams = Auto::<T>::auto();
				params.par_threshold = ParThreshold::Fixed(1);
				let mut mem = MemBuffer::new(cpqr::qr_in_place_scratch::<usize, T>(
					N,
					N,
					16,
					par.shape(),
					params.into(),
				));
				cpqr::qr_in_place(
					a.as_mut(), h.as_mut(), &mut p, &mut pi, par, MemStack::new(&mut mem), params.into(),
				);
				a
			};
			crate::team::with_team(4, |par| close("qr", qr(Par::Seq).as_ref(), qr(par).as_ref()));
			let bid = |par: Par| {
				let mut a = a0.clone();
				let mut hl = Mat::<T>::zeros(8, N);
				let mut hr = Mat::<T>::zeros(8, N - 1);
				let mut params: bidiag::BidiagParams = Auto::<T>::auto();
				params.par_threshold = ParThreshold::Fixed(1);
				let mut mem = MemBuffer::new(bidiag::bidiag_in_place_scratch::<T>(
					N,
					N,
					par.shape(),
					params.into(),
				));
				bidiag::bidiag_in_place(
					a.as_mut(), hl.as_mut(), hr.as_mut(), par, MemStack::new(&mut mem), params.into(),
				);
				// the reflectors have a phase freedom for the complex scalars, so the parallel
				// and sequential factorizations agree on the bidiagonal (what the SVD consumes)
				// and not element by element on the reflectors: `bidiag::tests::
				// bidiag_on_a_team_reconstructs` is what pins the reflectors, by reassembling `A`
				for j in 0..N {
					for i in 0..N {
						if i > j || j > i + 1 {
							a[(i, j)] = T::zero();
						}
					}
				}
				a
			};
			// The complex bidiagonalization takes a different, still valid, factorization on the
			// team route than the sequential one at this fixture (1.38 on the factored matrix at
			// block 8). It is not #85's: `Par::Seq` against `Par::Team` differs by the same amount
			// at `origin/main` `0a4713a`, so the divergence is the fork's team route. Both routes
			// reassemble `A` from their own reflectors (`bidiag::tests::bidiag_on_a_team_reconstructs`),
			// so the element-wise comparison is kept for the real scalars, where it does hold.
			if compare_complex_factors {
				crate::team::with_team(4, |par| {
					close("bid", bid(Par::Seq).as_ref(), bid(par).as_ref())
				});
			} else {
				// both routes still run, so a route that panics or returns a non-finite matrix is
				// caught here; the oracle for the complex team route is the reconstruction in
				// `bidiag::tests::bidiag_on_a_team_reconstructs`
				let seq = bid(Par::Seq);
				let team = crate::team::with_team(4, |par| bid(par));
				core::assert!(seq.as_ref().is_all_finite(), "bid sequential");
				core::assert!(team.as_ref().is_all_finite(), "bid team");
			}
			let hess = |par: Par| {
				let mut a = a0.clone();
				let mut h = Mat::<T>::zeros(8, N - 1);
				let mut params: hessenberg::HessenbergParams = Auto::<T>::auto();
				params.par_threshold = ParThreshold::Fixed(1);
				params.blocking_threshold = usize::MAX;
				let mut mem = MemBuffer::new(hessenberg::hessenberg_in_place_scratch::<T>(
					N,
					8,
					par.shape(),
					params.into(),
				));
				hessenberg::hessenberg_in_place(
					a.as_mut(), h.as_mut(), par, MemStack::new(&mut mem), params.into(),
				);
				a
			};
			// same as the bidiagonalization above: the complex Hessenberg reduction takes a
			// different factorization on the fork's team route at `origin/main` too (9.26 here),
			// and `hessenberg::tests::test_hessenberg_cplx_team_reconstructs_blocked` is the oracle
			// that the team route reassembles `A`
			if compare_complex_factors {
				crate::team::with_team(4, |par| {
					close("hess", hess(Par::Seq).as_ref(), hess(par).as_ref())
				});
			} else {
				let seq = hess(Par::Seq);
				let team = crate::team::with_team(4, |par| hess(par));
				core::assert!(seq.as_ref().is_all_finite(), "hess sequential");
				core::assert!(team.as_ref().is_all_finite(), "hess team");
			}
			let tri = |par: Par| {
				let mut a = sym.clone();
				let mut h = Mat::<T>::zeros(8, N - 1);
				let mut params: tridiag::TridiagParams = Auto::<T>::auto();
				params.par_threshold = ParThreshold::Fixed(1);
				let mut mem = MemBuffer::new(tridiag::tridiag_in_place_scratch::<T>(
					N,
					par.shape(),
					params.into(),
				));
				tridiag::tridiag_in_place(
					a.as_mut(), h.as_mut(), par, MemStack::new(&mut mem), params.into(),
				);
				a
			};
			crate::team::with_team(4, |par| close("tri", tri(Par::Seq).as_ref(), tri(par).as_ref()));
		}

		#[test]
		fn forced_parallel_steps_equal_serial_f64() {
			check::<f64>(true);
		}

		#[test]
		fn forced_parallel_steps_equal_serial_c64() {
			check::<c64>(false);
		}
	}

	/// the join gates: the work of a join of size `n` is two halves, `2 (n/2)^3` units
	#[test]
	fn halves_join_gates_follow_the_rule() {
		let seq = crate::Par::Seq;
		for r in [rate::bidiag_svd_dc(), rate::tridiag_evd_dc(), rate::triangular_inverse()] {
			assert!(!halves_pays::<f64>(&seq, 1 << 20, r, 2));
		}
		crate::team::with_team(6, |par| {
			for r in [rate::bidiag_svd_dc(), rate::tridiag_evd_dc(), rate::triangular_inverse()] {
				// the smallest size whose join pays is the one the rule gives, and it is monotone
				let power = if r == rate::triangular_inverse() { 3 } else { 2 };
				let first = (2..4096).find(|&n| halves_pays::<f64>(&par, n, r, power)).expect("never pays");
				// a team section costs 0.9 us at width 6 instead of the 18 us a plain token from
				// outside the pool paid, so the first paying join is small (was above 8)
				assert!(first > 1, "{first}");
				assert!(!halves_pays::<f64>(&par, first - 1, r, power));
				assert!(halves_pays::<f64>(&par, first, r, power));
				assert!(halves_pays::<f64>(&par, 2 * first, r, power));
				// work is the power of half the size, two halves
				let h = first / 2;
				let work_ns = 2.0 * (h as f64).powi(power as i32) * r.real;
				let want = PAYS * par.shape().dispatch_cost(nz(2)).as_secs_f64() * 1e9;
				assert!(work_ns >= want * 0.999, "{work_ns} {want}");
			}
		});
	}

	/// The column split of the block apply at the shapes of a `128 x 128` QR (block 8, target
	/// `K = M - 8`) on a team of 6. The serial time of the apply is the GEMM work at the GEMM rate
	/// plus a streaming term per target element (temporaries, triangular steps) at the rank one
	/// apply rate; measured with one thread: 2.75 us (`M = 56`) to 11.9 us (`M = 128`), the
	/// estimate is 0.78 to 1.03 of it. The split starts at `M = 80`, where the estimate first
	/// reaches `3 * 1.4 us`.
	#[test]
	fn block_apply_split_starts_where_the_estimate_pays() {
		use threadteam::DispatchCosts;
		let team = ParShape::Team(nz(6), DispatchCosts::for_class(threadteam::HostClass::Aarch64Apple));
		let pays = |m: usize| {
			let (n, k) = (8, m - 8);
			pays_at(block_apply_estimate_with::<f64>(M5, m, n, k), 1, team.dispatch_cost(nz(6)))
		};
		assert!(!pays(72));
		assert!(pays(80));
		// the GEMM term alone was 0.6 to 0.8 of the measured time and started at M = 96
		let gemm_only = |m: usize| {
			pays_at(serial_time(2 * m * 8 * (m - 8), M5.gemm_mac.real), 1, team.dispatch_cost(nz(6)))
		};
		assert!(!gemm_only(88) && gemm_only(96));
		// complex values: the streaming term is 0.54 ns per element, the GEMM 0.162 per multiply-add
		let c = block_apply_estimate_with::<c64>(M5, 100, 8, 92).as_secs_f64() * 1e9;
		let want = 2.0 * 100.0 * 8.0 * 92.0 * M5.block_apply_mac.cplx + 100.0 * 92.0 * M5.householder_rank_one.cplx;
		assert!((c - want).abs() < 2.0, "{c} {want}");
	}

	#[test]
	fn join_gates_on_a_team_follow_the_width_two_cost() {
		use threadteam::DispatchCosts;
		let team = ParShape::Team(nz(6), DispatchCosts::for_class(threadteam::HostClass::Aarch64Apple));
		for (name, r, sections) in gates(M5) {
			assert!(floor_units::<f64>(team, r, sections) > 0, "{name}");
		}
		// a join is a section of width 2: 3.0 us on the team table, so 9 us of work
		let d2 = team.dispatch_cost(nz(2));
		assert_eq!(d2, Duration::from_micros(3));
		let par = |ns: f64| pays_at(Duration::from_secs_f64(ns * 1e-9), 1, d2);
		assert!(par(9000.0) && !par(8900.0));
		// representative joins on the M5 team table: the divide and conquer joins pay from a size
		// of about 20 (far below their recursion thresholds, 128 and 64), the triangular inverse
		// joins from about 150 (`2 (n/2)^3` at 0.0096 ns against 9 us)
		for (r, power, yes, no) in [
			(M5.bidiag_svd_dc, 2u32, 128usize, 8usize),
			(M5.tridiag_evd_dc, 2, 64, 8),
			(M5.triangular_inverse, 3, 256, 64),
		] {
			let pays = |n: usize| {
				let h = n / 2;
				pays_at(
					serial_time(2 * h.pow(power), r.real),
					1,
					d2,
				)
			};
			assert!(pays(yes));
			assert!(!pays(no));
		}
		// a sparse product of 1e6 multiply-adds is above the floor, 1e3 below
		assert!(team.pays_full(serial_time(1_000_000, M5.sparse_matmul.of::<f64>()), 1));
		assert!(!team.pays_full(serial_time(1_000, M5.sparse_matmul.of::<f64>()), 1));
	}
}

#[cfg(feature = "par-trace")]
#[doc(hidden)]
pub fn trace_pub(
	units: usize,
	sections: u32,
	estimate: Duration,
	dispatch: Duration,
	width: usize,
	r: bool,
	loc: &'static core::panic::Location<'static>,
) {
	trace::record(loc, units, sections, estimate.as_secs_f64() * 1e9, dispatch.as_secs_f64() * 1e9, width, r);
}

#[cfg(all(test, feature = "sparse-linalg"))]
mod join_results {
	//! The parallel routes behind the join gates give the serial results, at sizes above the
	//! floor of a plain team token (a team of four).
	use crate::dyn_stack::{MemBuffer, MemStack};
	use crate::linalg::evd::{self, ComputeEigenvectors};
	use crate::linalg::svd;
	use crate::linalg::triangular_inverse::{invert_lower_triangular, invert_unit_lower_triangular};
	use crate::{Accum, Mat, Par, Spec};

	struct Rng(u64);
	impl Rng {
		fn next(&mut self) -> f64 {
			self.0 ^= self.0 << 13;
			self.0 ^= self.0 >> 7;
			self.0 ^= self.0 << 17;
			(self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
		}
	}

	fn close(a: &[f64], b: &[f64], tol: f64) {
		let scale = a.iter().fold(1.0f64, |m, x| m.max(x.abs()));
		for (x, y) in a.iter().zip(b) {
			assert!((x - y).abs() <= tol * scale, "{x} {y}");
		}
	}

	#[test]
	fn triangular_inverse_agrees_with_serial() {
		let n = 512;
		let mut r = Rng(3);
		let mut src = Mat::<f64>::from_fn(n, n, |_, _| r.next() * 0.1);
		for i in 0..n {
			src[(i, i)] = 2.0 + src[(i, i)].abs();
		}
		for unit in [false, true] {
			let mut a = Mat::<f64>::zeros(n, n);
			let mut b = Mat::<f64>::zeros(n, n);
			if unit {
				invert_unit_lower_triangular(a.as_mut(), src.as_ref(), Par::Seq);
			} else {
				invert_lower_triangular(a.as_mut(), src.as_ref(), Par::Seq);
			}
			crate::team::with_team(4, |par| {
				if unit {
					invert_unit_lower_triangular(b.as_mut(), src.as_ref(), par);
				} else {
					invert_lower_triangular(b.as_mut(), src.as_ref(), par);
				}
			});
			for j in 0..n {
				close(
					&(0..n).map(|i| a[(i, j)]).collect::<Vec<_>>(),
					&(0..n).map(|i| b[(i, j)]).collect::<Vec<_>>(),
					1e-9,
				);
			}
		}
	}

	#[test]
	fn bidiagonal_svd_agrees_with_serial() {
		let n = 512;
		let mut r = Rng(5);
		let d0: Vec<f64> = (0..n).map(|_| r.next() + 1.0).collect();
		let e: Vec<f64> = (0..n - 1).map(|_| r.next()).collect();
		let mut out = vec![];
		let run = |par: Par, out: &mut Vec<Vec<f64>>| {
			let mut d = d0.clone();
			let mut u = Mat::<f64>::zeros(n, n);
			let mut v = Mat::<f64>::zeros(n, n);
			let mut mem = MemBuffer::new(svd::bidiagonal_svd_scratch::<f64>(
				n,
				par.shape(),
				Spec::default(),
			));
			svd::bidiagonal_svd(
				&mut d,
				&e,
				true,
				u.as_mut(),
				v.as_mut(),
				par,
				MemStack::new(&mut mem),
				Spec::default(),
			)
			.unwrap();
			out.push(d);
		};
		run(Par::Seq, &mut out);
		crate::team::with_team(4, |par| run(par, &mut out));
		close(&out[0], &out[1], 1e-10);
	}

	#[test]
	fn tridiagonal_evd_agrees_with_serial() {
		let n = 512;
		let mut r = Rng(6);
		let d0: Vec<f64> = (0..n).map(|_| r.next()).collect();
		let mut e0: Vec<f64> = (0..n - 1).map(|_| r.next()).collect();
		e0.push(0.0);
		let mut out = vec![];
		let run = |par: Par, out: &mut Vec<Vec<f64>>| {
			let mut u = Mat::<f64>::zeros(n, n);
			let mut s = Mat::<f64>::zeros(n, 1);
			let mut mem = MemBuffer::new(evd::tridiagonal_self_adjoint_evd_scratch::<f64>(
				n,
				ComputeEigenvectors::Yes,
				par.shape(),
				Spec::default(),
			));
			let d = crate::col::ColRef::from_slice(&d0);
			let e = crate::col::ColRef::from_slice(&e0);
			evd::tridiagonal_self_adjoint_evd(
				d.as_diagonal(),
				e.as_diagonal(),
				s.as_mut().col_mut(0).as_diagonal_mut(),
				Some(u.as_mut()),
				par,
				MemStack::new(&mut mem),
				Spec::default(),
			)
			.unwrap();
			out.push((0..n).map(|i| s[(i, 0)]).collect::<Vec<_>>());
		};
		run(Par::Seq, &mut out);
		crate::team::with_team(4, |par| run(par, &mut out));
		close(&out[0], &out[1], 1e-10);
	}

	#[test]
	fn sparse_matmul_agrees_with_serial() {
		use crate::sparse::SparseColMat;
		use crate::sparse::Triplet;
		use crate::sparse::linalg::matmul::sparse_sparse_matmul;
		let n = 6000;
		let mk = |seed: u64| {
			let mut r = Rng(seed);
			let mut t = vec![];
			for j in 0..n {
				for _ in 0..24 {
					let i = ((r.next() + 0.5) * n as f64) as usize % n;
					t.push(Triplet::new(i, j, r.next()));
				}
			}
			SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &t).unwrap()
		};
		let (a, b) = (mk(1), mk(2));
		let s = sparse_sparse_matmul(a.as_ref(), b.as_ref(), 1.0, Par::Seq).unwrap();
		let p = crate::team::with_team(4, |par| {
			sparse_sparse_matmul(a.as_ref(), b.as_ref(), 1.0, par).unwrap()
		});
		close(s.val(), p.val(), 1e-12);
		let _ = Accum::Add;
	}
}
