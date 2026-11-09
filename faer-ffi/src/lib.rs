#![allow(non_snake_case)]
use core::ffi::c_void;
use faer::dyn_stack::StackReq;
use faer::{c32, c64, cx128, fx128};
trait RealField: faer::traits::RealField + Copy + 'static {}
impl<T: faer::traits::RealField + Copy + 'static> RealField for T {}
trait ComplexField: faer::traits::ComplexField + Copy + 'static {}
impl<T: faer::traits::ComplexField + Copy + 'static> ComplexField for T {}
type AsIs<T> = T;
#[derive(Copy, Clone)]
pub enum Never {}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct MatRef {
	pub ptr: *const c_void,
	pub nrows: usize,
	pub ncols: usize,
	pub row_stride: isize,
	pub col_stride: isize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct MatMut {
	pub ptr: *mut c_void,
	pub nrows: usize,
	pub ncols: usize,
	pub row_stride: isize,
	pub col_stride: isize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VecRef {
	pub ptr: *const c_void,
	pub len: usize,
	pub stride: isize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VecMut {
	pub ptr: *mut c_void,
	pub len: usize,
	pub stride: isize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SliceRef {
	pub ptr: *const c_void,
	pub len: usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SliceMut {
	pub ptr: *mut c_void,
	pub len: usize,
}
pub enum Scalar {}
pub enum Index {}
pub enum Real {}
#[repr(C)]
#[derive(Copy, Clone)]
pub enum Accum {
	Replace,
	Add,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub enum Conj {
	No,
	Yes,
}
/// A handle to a team: the ownership of the participants of a parallel region.
///
/// The handle owns one [`threadteam::Pool`] of its width, built by
/// [`libfaer_v0_23_team_create`]. Every operation borrows it for the duration of one call: the pool
/// becomes a team, the operation runs inside it, and the pool comes back out before the call
/// returns (see `Team::with_par`). No team token is carried, because C cannot express the scope
/// one is valid for, so a team is never held across a C frame and the calling thread is never inside
/// a team between calls. `NULL` stands for "no team", which runs the operation sequentially.
pub struct Team {
	/// participants, the calling thread included (`1` is sequential)
	threads: usize,
	/// the pool this handle owns, `None` when the width is one (sequential). A call moves it into a
	/// team and back, so two calls on one handle take turns instead of running at once.
	pool: std::sync::Mutex<Option<threadteam::Pool>>,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub enum Block {
	Rectangular,
	TriangularLower,
	TriangularUpper,
	StrictTriangularLower,
	StrictTriangularUpper,
	UnitTriangularLower,
	UnitTriangularUpper,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Layout {
	pub len_bytes: usize,
	pub align_bytes: usize,
}
impl From<StackReq> for Layout {
	fn from(value: StackReq) -> Self {
		Self {
			len_bytes: value.size_bytes(),
			align_bytes: value.align_bytes(),
		}
	}
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct MemAlloc {
	pub ptr: *mut c_void,
	pub len_bytes: usize,
}
impl MemAlloc {
	fn faer(self) -> &'static mut faer::dyn_stack::MemStack {
		faer::dyn_stack::MemStack::new(
			SliceMut {
				ptr: self.ptr,
				len: self.len_bytes,
			}
			.slice(),
		)
	}
}
impl Conj {
	fn faer(self) -> faer::Conj {
		match self {
			Conj::No => faer::Conj::No,
			Conj::Yes => faer::Conj::Yes,
		}
	}
}
impl Block {
	fn faer(self) -> faer::linalg::matmul::triangular::BlockStructure {
		match self {
			Block::Rectangular => faer::linalg::matmul::triangular::BlockStructure::Rectangular,
			Block::TriangularLower => faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
			Block::TriangularUpper => faer::linalg::matmul::triangular::BlockStructure::TriangularUpper,
			Block::StrictTriangularLower => faer::linalg::matmul::triangular::BlockStructure::StrictTriangularLower,
			Block::StrictTriangularUpper => faer::linalg::matmul::triangular::BlockStructure::StrictTriangularUpper,
			Block::UnitTriangularLower => faer::linalg::matmul::triangular::BlockStructure::UnitTriangularLower,
			Block::UnitTriangularUpper => faer::linalg::matmul::triangular::BlockStructure::UnitTriangularUpper,
		}
	}
}
fn scalar<T: ComplexField>(x: *const Scalar) -> T {
	unsafe { (x as *const T).read() }
}
fn real<T: RealField>(x: *const Real) -> T {
	unsafe { (x as *const T).read() }
}
impl MatRef {
	fn faer<'a, T>(self) -> faer::MatRef<'a, T> {
		unsafe {
			faer::MatRef::from_raw_parts(
				self.ptr as *const T,
				self.nrows,
				self.ncols,
				self.row_stride,
				self.col_stride,
			)
		}
	}
}
impl MatMut {
	fn faer<'a, T>(self) -> faer::MatMut<'a, T> {
		unsafe {
			faer::MatMut::from_raw_parts_mut(
				self.ptr as *mut T,
				self.nrows,
				self.ncols,
				self.row_stride,
				self.col_stride,
			)
		}
	}
}
#[allow(dead_code)]
impl VecRef {
	fn diag<'a, T>(self) -> faer::diag::DiagRef<'a, T> {
		unsafe {
			faer::diag::DiagRef::from_raw_parts(
				self.ptr as *const T,
				self.len,
				self.stride,
			)
		}
	}

	fn col<'a, T>(self) -> faer::col::ColRef<'a, T> {
		unsafe {
			faer::col::ColRef::from_raw_parts(
				self.ptr as *const T,
				self.len,
				self.stride,
			)
		}
	}

	fn row<'a, T>(self) -> faer::row::RowRef<'a, T> {
		unsafe {
			faer::row::RowRef::from_raw_parts(
				self.ptr as *const T,
				self.len,
				self.stride,
			)
		}
	}
}
#[allow(dead_code)]
impl VecMut {
	fn diag<'a, T>(self) -> faer::diag::DiagMut<'a, T> {
		unsafe {
			faer::diag::DiagMut::from_raw_parts_mut(
				self.ptr as *mut T,
				self.len,
				self.stride,
			)
		}
	}

	fn col<'a, T>(self) -> faer::col::ColMut<'a, T> {
		unsafe {
			faer::col::ColMut::from_raw_parts_mut(
				self.ptr as *mut T,
				self.len,
				self.stride,
			)
		}
	}

	fn row<'a, T>(self) -> faer::row::RowMut<'a, T> {
		unsafe {
			faer::row::RowMut::from_raw_parts_mut(
				self.ptr as *mut T,
				self.len,
				self.stride,
			)
		}
	}
}
impl SliceRef {
	fn slice<'a, T>(self) -> &'a [T] {
		if self.ptr.is_null() && self.len == 0 {
			&[]
		} else {
			unsafe { core::slice::from_raw_parts(self.ptr as _, self.len) }
		}
	}
}
impl SliceMut {
	fn slice<'a, T>(self) -> &'a mut [T] {
		if self.ptr.is_null() && self.len == 0 {
			&mut []
		} else {
			unsafe { core::slice::from_raw_parts_mut(self.ptr as _, self.len) }
		}
	}
}
impl Accum {
	fn faer(self) -> faer::Accum {
		match self {
			Accum::Replace => faer::Accum::Replace,
			Accum::Add => faer::Accum::Add,
		}
	}
}
/// Set while this thread is inside a call that holds a team, so a callback that calls back into this
/// API on any handle runs its own operation sequentially instead of waiting for a pool the same thread
/// holds. `faer-ffi` passes C callbacks into faer (`SchurParams::recommended_shift_count` and
/// friends), and those call back into the same entry points.
struct InCall;
thread_local! {
	static IN_CALL: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}
impl InCall {
	/// `None` when this thread already holds a team: the caller then runs sequentially.
	fn enter() -> Option<InCall> {
		// `if`, not `then_some`: `then_some` builds its value eagerly, so the untaken branch would drop
		// an `InCall` at once and clear the flag its caller set
		if IN_CALL.with(|inside| inside.replace(true)) {
			return None;
		}
		Some(InCall)
	}
}
impl Drop for InCall {
	fn drop(&mut self) {
		IN_CALL.with(|inside| inside.set(false));
	}
}
impl Team {
	/// the shape of the team, for the workspace queries (`*_scratch`) that never execute work
	fn shape(team: *const Team) -> faer::ParShape {
		match unsafe { team.as_ref() } {
			None => faer::ParShape::Seq,
			Some(team) => faer::ParShape::team(team.threads),
		}
	}

	/// the pool of a `threads`-wide handle: one thread per participant, the width of the team it hands
	/// out, and `None` for a sequential handle, which reserves nothing
	fn pool_for(threads: usize) -> Option<threadteam::Pool> {
		(threads > 1).then(|| {
			threadteam::Pool::builder()
				.threads(threads)
				.acquire_timeout(core::time::Duration::from_secs(30))
				.build()
				.expect("a thread pool for the team")
		})
	}

	/// Resets the width, rebuilding the owned pool when it changes. The new pool is a new width, so
	/// threadteam's cache does not hand it the old pool's workers: this is a reconfiguration, and it
	/// pays for its threads.
	fn set_threads(&mut self, threads: usize) {
		if threads == self.threads {
			return;
		}
		self.threads = threads;
		*self.pool.get_mut().unwrap_or_else(|e| e.into_inner()) =
			Team::pool_for(threads);
	}

	/// Runs `f` on the team this handle owns, and leaves it again before returning: the pool goes into
	/// a team for the duration of this call and comes back out, which is what keeps the calling thread
	/// out of the team between calls. The mutex is what makes the pool that travels safe: a call on a
	/// handle another thread is using waits for it to hand the pool back. A call nested in another call
	/// (see [`InCall`]) never reaches the pool, so it takes no lock.
	fn with_par<R>(&self, f: impl FnOnce(faer::Par<'_>) -> R) -> R {
		// a call made from inside another call (a C callback calling back into this API) cannot have the
		// pool: it runs sequentially, where waiting for the mutex this thread holds would never end
		let Some(_in_call) = InCall::enter() else {
			return f(faer::Par::Seq);
		};
		let mut held = self.pool.lock().unwrap_or_else(|e| e.into_inner());
		let Some(pool) = held.take() else {
			return f(faer::Par::Seq);
		};
		let team = match pool.into_team(self.threads) {
			Ok(team) => team,
			Err(err) => {
				let reason = err.reason();
				*held = Some(err.into_pool());
				panic!(
					"cannot turn the handle's own pool into a team: {reason}"
				)
			},
		};
		let r = {
			let mut exec = team.exec(self.threads);
			exec.team(|token| f(faer::Par::Team(token)))
				.ok()
				.expect("a team token for the handle's own team")
		};
		*held = Some(team.into_pool());
		r
	}
}
/// Runs `f` inside the team `team` names, and leaves it again before returning: the scoped token
/// `faer::Par` never outlives this call, so a handle stays the only thing C holds.
fn with_team_par<R>(
	team: *const Team,
	f: impl FnOnce(faer::Par<'_>) -> R,
) -> R {
	match unsafe { team.as_ref() } {
		None => f(faer::Par::Seq),
		Some(team) => team.with_par(f),
	}
}
macro_rules! funcs {
	({$(
		pub fn $func:ident<
			$($ty:ident),* $(,)?
		>(
			$($input:ident : $input_ty:ty),* $(,)?
		) $(-> $output_ty: ty)?
			$body:block
	)*}) => {
		$($crate::funcs!(
			impl 0 @ fn $func<$($ty,)*>(
				$($input: $input_ty,)*
			) $(-> $output_ty)? $body
		);)*
	};

	(impl 0 @
		fn $func:ident<
			$($ty:ident,)*
		>$inputs: tt -> $output_ty: ty
			$body:block
	) => {
		$crate::funcs!(
			impl 1 @ fn $func<$($ty,)*>$inputs -> $output_ty $body
		);
	};
	(impl 0 @
		fn $func:ident<
			$($ty:ident,)*
		>$inputs: tt
			$body:block
	) => {
		$crate::funcs!(
			impl 1 @ fn $func<$($ty,)*>$inputs -> () $body
		);
	};

	(impl 1 @
		fn $func:ident<
			$ty0:ident,
		>$inputs: tt -> $output_ty: ty
			$body:block
	) => {
		$crate::funcs!(
			impl 2 @ fn $func<$ty0 in (f32, f64, fx128, c32, c64, cx128,),>$inputs -> $output_ty $body
		);
	};

	(impl 1 @
		fn $func:ident<
			$ty1:ident,
			$ty0:ident,
		>$inputs: tt -> $output_ty: ty
			$body:block
	) => {
		$crate::funcs!(
			impl 2 @ fn $func<$ty1, $ty0 in (f32, f64, fx128, c32, c64, cx128,),>$inputs -> $output_ty $body
		);
	};

	(impl 2 @
		fn $func:ident<
			$ty0:ident in ($($ty0_list: ident,)*),
		>$inputs: tt -> $output_ty: ty
			$body:block
	) => {
		pastey::paste! {
			$(
				#[unsafe(no_mangle)]
				pub unsafe extern "C" fn [<libfaer_v0_23_ $func _ $ty0_list>] $inputs -> $output_ty { type $ty0 = $ty0_list;#[allow(unused_unsafe)] unsafe {$body} }
			)*
		}
	};

	(impl 2 @
		fn $func:ident<
			$ty1:ident,
			$ty0:ident in ($($ty0_list: ident,)*),
		>$inputs: tt -> $output_ty: ty
			$body:block
	) => {
		pastey::paste! {
			$(
				#[unsafe(no_mangle)]
				pub unsafe extern "C" fn [<libfaer_v0_23_ $func _u32_ $ty0_list >] $inputs -> $output_ty { type $ty0 = $ty0_list; type $ty1 = u32;#[allow(unused_unsafe)] unsafe {$body} }
			)*
		}
		pastey::paste! {
			$(
				#[unsafe(no_mangle)]
				pub unsafe extern "C" fn [<libfaer_v0_23_ $func _u64_ $ty0_list >] $inputs -> $output_ty { type $ty0 = $ty0_list; type $ty1 = u64;#[allow(unused_unsafe)] unsafe {$body} }
			)*
		}
	};
}
pub(crate) use funcs;
macro_rules! cerr {
	({$(

		#[ok($ok_path:path)]
		#[err($err_path:path)]
		pub enum $name:ident {
			Ok {$($field:ident: $field_ty:ty),* $(,)?},
			$($err:ident {$($err_field:ident: $err_field_ty:ty),* $(,)?}),*$(,)?
		}

	)*}) => {$(
		#[repr(C)]
		#[derive(Copy, Clone)]
		pub enum $name {
			Ok {$($field: $field_ty,)*},
			$($err {$($err_field: $err_field_ty,)*},)*
			Unknown,
		}

		impl Into<$name> for Result<$ok_path, $err_path> {
			fn into(self) -> $name {
				match self {
					#[allow(unused_variables)]
					Ok(this) => $name::Ok {
						$($field: this.$field,)*
					},
					$(Err(AsIs::<$err_path>::$err { $($err_field,)* }) => $name::$err {
						$($err_field,)*
					},)*
					#[allow(unreachable_patterns)]
					Err(_) => $name::Unknown,
				}
			}
		}
	)*};
}
macro_rules! cparams {
	({$(

		#[repr($path:path)]
		pub struct $name:ident {
			$($field:ident: $field_ty:ty),* $(,)?
		}

	)*}) => {$(
		#[repr(C)]
		#[derive(Copy, Clone)]
		pub struct $name {
			$($field: $field_ty,)*
		}

		funcs!({
			pub fn $name<T>() -> $name {
				let default: $path = faer::auto!(T);
				$name {
					$($field: default.$field.into(),)*
				}
			}
		});

		impl $name {
			#[allow(dead_code)]
			fn faer<T: ComplexField>(self) -> faer::Spec<$path, T> {
				<$path>::from(self).into()
			}
		}

		impl From<$name> for $path {
			#[allow(unused_variables)]
			fn from(value: $name) -> $path {
				$path {
					$($field: value.$field.into(),)*
					..faer::auto!(f32)
				}
			}
		}
		impl From<$path> for $name {
			#[allow(unused_variables)]
			fn from(value: $path) -> $name {
				$name {
					$($field: value.$field.into(),)*
				}
			}
		}
	)*};
}
#[cfg(feature = "linalg")]
pub mod linalg {
	use super::*;
	use faer::linalg as la;
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum ComputeSvdVectors {
		No,
		Thin,
		Full,
	}
	impl ComputeSvdVectors {
		fn faer(self) -> la::svd::ComputeSvdVectors {
			match self {
				ComputeSvdVectors::No => la::svd::ComputeSvdVectors::No,
				ComputeSvdVectors::Thin => la::svd::ComputeSvdVectors::Thin,
				ComputeSvdVectors::Full => la::svd::ComputeSvdVectors::Full,
			}
		}
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum ComputeEigenvectors {
		No,
		Yes,
	}
	impl ComputeEigenvectors {
		fn faer(self) -> la::evd::ComputeEigenvectors {
			match self {
				ComputeEigenvectors::No => la::evd::ComputeEigenvectors::No,
				ComputeEigenvectors::Yes => la::evd::ComputeEigenvectors::Yes,
			}
		}
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum PivotingStrategy {
		/// searches for the k-th pivot in the k-th column
		Partial,
		/// searches for the k-th pivot in the k-th column, as well as the tail
		/// of the diagonal of the matrix
		PartialDiag,
		/// searches for pivots that are locally optimal
		Rook,
		/// searches for pivots that are locally optimal, as well as the tail of
		/// the diagonal of the matrix
		RookDiag,

		/// searches for pivots that are globally optimal
		Full,
	}
	impl From<PivotingStrategy> for la::cholesky::lblt::factor::PivotingStrategy {
		fn from(value: PivotingStrategy) -> Self {
			match value {
				PivotingStrategy::Partial => {
					la::cholesky::lblt::factor::PivotingStrategy::Partial
				},
				PivotingStrategy::PartialDiag => {
					la::cholesky::lblt::factor::PivotingStrategy::PartialDiag
				},
				PivotingStrategy::Rook => {
					la::cholesky::lblt::factor::PivotingStrategy::Rook
				},
				PivotingStrategy::RookDiag => {
					la::cholesky::lblt::factor::PivotingStrategy::RookDiag
				},
				PivotingStrategy::Full => {
					la::cholesky::lblt::factor::PivotingStrategy::Full
				},
			}
		}
	}
	impl From<la::cholesky::lblt::factor::PivotingStrategy> for PivotingStrategy {
		fn from(value: la::cholesky::lblt::factor::PivotingStrategy) -> Self {
			match value {
				la::cholesky::lblt::factor::PivotingStrategy::Partial => {
					PivotingStrategy::Partial
				},
				la::cholesky::lblt::factor::PivotingStrategy::PartialDiag => {
					PivotingStrategy::PartialDiag
				},
				la::cholesky::lblt::factor::PivotingStrategy::Rook => {
					PivotingStrategy::Rook
				},
				la::cholesky::lblt::factor::PivotingStrategy::RookDiag => {
					PivotingStrategy::RookDiag
				},
				la::cholesky::lblt::factor::PivotingStrategy::Full => {
					PivotingStrategy::Full
				},
				_ => PivotingStrategy::Partial,
			}
		}
	}
	cerr!({
		#[ok(faer::linalg::cholesky::llt::factor::LltInfo)]
		#[err(faer::linalg::cholesky::llt::factor::LltError)]
		pub enum LltStatus {
			Ok { dynamic_regularization_count: usize },
			NonPositivePivot { index: usize },
		}
		#[ok(faer::linalg::cholesky::llt_pivoting::factor::PivLltInfo)]
		#[err(faer::linalg::cholesky::llt::factor::LltError)]
		pub enum PivLltStatus {
			Ok {
				rank: usize,
				transposition_count: usize,
			},
			NonPositivePivot {
				index: usize,
			},
		}
		#[ok(faer::linalg::cholesky::ldlt::factor::LdltInfo)]
		#[err(faer::linalg::cholesky::ldlt::factor::LdltError)]
		pub enum LdltStatus {
			Ok { dynamic_regularization_count: usize },
			ZeroPivot { index: usize },
		}
		#[ok(faer::linalg::cholesky::lblt::factor::LbltInfo)]
		#[err(Never)]
		pub enum LbltStatus {
			Ok { transposition_count: usize },
		}
		#[ok(faer::linalg::qr::no_pivoting::factor::QrInfo)]
		#[err(Never)]
		pub enum QrStatus {
			Ok { rank: usize },
		}
		#[ok(faer::linalg::qr::col_pivoting::factor::ColPivQrInfo)]
		#[err(Never)]
		pub enum ColPivQrStatus {
			Ok { transposition_count: usize },
		}
		#[ok(faer::linalg::lu::partial_pivoting::factor::PartialPivLuInfo)]
		#[err(Never)]
		pub enum PartialPivLuStatus {
			Ok { transposition_count: usize },
		}
		#[ok(faer::linalg::lu::full_pivoting::factor::FullPivLuInfo)]
		#[err(Never)]
		pub enum FullPivLuStatus {
			Ok { transposition_count: usize },
		}
	});
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum SvdStatus {
		Ok { padding: usize },
		NoConvergence { padding: usize },
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum EvdStatus {
		Ok { padding: usize },
		NoConvergence { padding: usize },
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum GevdStatus {
		Ok { padding: usize },
		NoConvergence { padding: usize },
	}
	impl Into<SvdStatus> for Result<(), la::svd::SvdError> {
		fn into(self) -> SvdStatus {
			match self {
				Ok(()) => SvdStatus::Ok { padding: 0 },
				Err(la::svd::SvdError::NoConvergence) => {
					SvdStatus::NoConvergence { padding: 0 }
				},
			}
		}
	}
	impl Into<EvdStatus> for Result<(), la::evd::EvdError> {
		fn into(self) -> EvdStatus {
			match self {
				Ok(()) => EvdStatus::Ok { padding: 0 },
				Err(la::evd::EvdError::NoConvergence) => {
					EvdStatus::NoConvergence { padding: 0 }
				},
			}
		}
	}
	impl Into<GevdStatus> for Result<(), la::gevd::GevdError> {
		fn into(self) -> GevdStatus {
			match self {
				Ok(()) => GevdStatus::Ok { padding: 0 },
				Err(la::gevd::GevdError::NoConvergence) => {
					GevdStatus::NoConvergence { padding: 0 }
				},
			}
		}
	}
	cparams!({
		#[repr(faer::linalg::cholesky::llt::factor::LltParams)]
		pub struct LltParams {
			recursion_threshold: usize,
			block_size: usize,
		}
		#[repr(faer::linalg::cholesky::llt_pivoting::factor::PivLltParams)]
		pub struct PivLltParams {
			block_size: usize,
		}
		#[repr(faer::linalg::cholesky::ldlt::factor::LdltParams)]
		pub struct LdltParams {
			recursion_threshold: usize,
			block_size: usize,
		}
		#[repr(faer::linalg::cholesky::lblt::factor::LbltParams)]
		pub struct LbltParams {
			pivoting: PivotingStrategy,
			par_threshold: usize,
			block_size: usize,
		}
		#[repr(faer::linalg::qr::no_pivoting::factor::QrParams)]
		pub struct QrParams {
			blocking_threshold: usize,
		}
		#[repr(faer::linalg::qr::col_pivoting::factor::ColPivQrParams)]
		pub struct ColPivQrParams {
			blocking_threshold: usize,
			par_threshold: usize,
		}
		#[repr(faer::linalg::lu::partial_pivoting::factor::PartialPivLuParams)]
		pub struct PartialPivLuParams {
			recursion_threshold: usize,
			block_size: usize,
			par_threshold: usize,
		}
		#[repr(faer::linalg::lu::full_pivoting::factor::FullPivLuParams)]
		pub struct FullPivLuParams {
			par_threshold: usize,
		}
		#[repr(faer::linalg::svd::bidiag::BidiagParams)]
		pub struct BidiagParams {
			par_threshold: usize,
		}
		#[repr(faer::linalg::evd::tridiag::TridiagParams)]
		pub struct TridiagParams {
			par_threshold: usize,
		}
		#[repr(faer::linalg::evd::hessenberg::HessenbergParams)]
		pub struct HessenbergParams {
			par_threshold: usize,
			blocking_threshold: usize,
		}
		#[repr(faer::linalg::gevd::gen_hessenberg::GeneralizedHessenbergParams)]
		pub struct GeneralizedHessenbergParams {
			block_size: usize,
			blocking_threshold: usize,
		}
		#[repr(faer::linalg::evd::EvdFromSchurParams)]
		pub struct EvdFromSchurParams {
			recursion_threshold: usize,
		}
		#[repr(faer::linalg::evd::schur::SchurParams)]
		pub struct SchurParams {
			recommended_shift_count: extern "C" fn(
				matrix_dimension: usize,
				active_block_dimension: usize,
			) -> usize,
			recommended_deflation_window: extern "C" fn(
				matrix_dimension: usize,
				active_block_dimension: usize,
			) -> usize,
			blocking_threshold: usize,
			nibble_threshold: usize,
		}
		#[repr(faer::linalg::gevd::GeneralizedSchurParams)]
		pub struct GeneralizedSchurParams {
			relative_cost_estimate_of_shift_chase_to_matmul:
				extern "C" fn(
					matrix_dimension: usize,
					active_block_dimension: usize,
				) -> usize,
			recommended_shift_count: extern "C" fn(
				matrix_dimension: usize,
				active_block_dimension: usize,
			) -> usize,
			recommended_deflation_window: extern "C" fn(
				matrix_dimension: usize,
				active_block_dimension: usize,
			) -> usize,
			blocking_threshold: usize,
			nibble_threshold: usize,
		}
		#[repr(faer::linalg::svd::SvdParams)]
		pub struct SvdParams {
			bidiag: BidiagParams,
			qr: QrParams,
			recursion_threshold: usize,
			qr_ratio_threshold: f64,
		}
		#[repr(faer::linalg::evd::SelfAdjointEvdParams)]
		pub struct SelfAdjointEvdParams {
			tridiag: TridiagParams,
			recursion_threshold: usize,
		}
		#[repr(faer::linalg::evd::EvdParams)]
		pub struct EvdParams {
			hessenberg: HessenbergParams,
			schur: SchurParams,
			evd_from_schur: EvdFromSchurParams,
		}
		#[repr(faer::linalg::gevd::GevdParams)]
		pub struct GevdParams {
			hessenberg: GeneralizedHessenbergParams,
			schur: GeneralizedSchurParams,
			evd_from_schur: GevdFromSchurParams,
		}
	});
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub struct GevdFromSchurParams {
		pub padding: usize,
	}
	impl GevdFromSchurParams {
		#[allow(dead_code)]
		fn faer<T: ComplexField>(
			self,
		) -> faer::Spec<la::gevd::GevdFromSchurParams, T> {
			<la::gevd::GevdFromSchurParams>::from(self).into()
		}
	}
	impl From<GevdFromSchurParams> for la::gevd::GevdFromSchurParams {
		#[allow(unused_variables)]
		fn from(value: GevdFromSchurParams) -> la::gevd::GevdFromSchurParams {
			la::gevd::GevdFromSchurParams { ..faer::auto!(f32) }
		}
	}
	impl From<la::gevd::GevdFromSchurParams> for GevdFromSchurParams {
		#[allow(unused_variables)]
		fn from(_: la::gevd::GevdFromSchurParams) -> GevdFromSchurParams {
			GevdFromSchurParams { padding: 0 }
		}
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub struct LltRegularization {
		/// regularized value
		pub dynamic_regularization_delta: *const Real,
		/// regularization threshold
		pub dynamic_regularization_epsilon: *const Real,
	}
	impl LltRegularization {
		fn faer<T: RealField>(
			self,
		) -> faer::linalg::cholesky::llt::factor::LltRegularization<T> {
			{
				faer::linalg::cholesky::llt::factor::LltRegularization {
					dynamic_regularization_delta: real(
						self.dynamic_regularization_delta,
					),
					dynamic_regularization_epsilon: real(
						self.dynamic_regularization_epsilon,
					),
				}
			}
		}
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub struct LdltRegularization {
		/// regularized value
		pub dynamic_regularization_delta: *const Real,
		/// regularization threshold
		pub dynamic_regularization_epsilon: *const Real,

		/// i8
		pub dynamic_regularization_signs: SliceMut,
	}
	impl LdltRegularization {
		fn faer<'a, T: RealField>(
			self,
		) -> faer::linalg::cholesky::ldlt::factor::LdltRegularization<'a, T> {
			{
				faer::linalg::cholesky::ldlt::factor::LdltRegularization {
					dynamic_regularization_delta: real(
						self.dynamic_regularization_delta,
					),
					dynamic_regularization_epsilon: real(
						self.dynamic_regularization_epsilon,
					),
					dynamic_regularization_signs: if self
						.dynamic_regularization_signs
						.ptr
						.is_null()
					{
						None
					} else {
						Some(self.dynamic_regularization_signs.slice())
					},
				}
			}
		}
	}
	funcs!({
		pub fn matmul<T>(
			C: MatMut,
			accum: Accum,
			A: MatRef,
			B: MatRef,
			alpha: *const Scalar,
			team: *const Team,
		) {
			with_team_par(team, |mut par| {
				la::matmul::matmul(
					C.faer::<T>(),
					accum.faer(),
					A.faer::<T>(),
					B.faer::<T>(),
					scalar(alpha),
					par.reborrow(),
				);
			})
		}
		pub fn matmul_triangular<T>(
			C: MatMut,
			C_block: Block,
			accum: Accum,
			A: MatRef,
			A_block: Block,
			B: MatRef,
			B_block: Block,
			alpha: *const Scalar,
			team: *const Team,
		) {
			with_team_par(team, |mut par| {
				la::matmul::triangular::matmul(
					C.faer::<T>(),
					C_block.faer(),
					accum.faer(),
					A.faer::<T>(),
					A_block.faer(),
					B.faer::<T>(),
					B_block.faer(),
					scalar(alpha),
					par.reborrow(),
				);
			})
		}
		// SOLVE
		pub fn solve_triangular_lower_in_place<T>(
			L: MatRef,
			L_conj: Conj,
			rhs: MatMut,
			team: *const Team,
		) {
			with_team_par(team, |mut par| {
				la::triangular_solve::solve_lower_triangular_in_place_with_conj(
					L.faer::<T>(),
					L_conj.faer(),
					rhs.faer(),
					par.reborrow(),
				);
			})
		}
		pub fn solve_triangular_upper_in_place<T>(
			U: MatRef,
			U_conj: Conj,
			rhs: MatMut,
			team: *const Team,
		) {
			with_team_par(team, |mut par| {
				la::triangular_solve::solve_upper_triangular_in_place_with_conj(
					U.faer::<T>(),
					U_conj.faer(),
					rhs.faer(),
					par.reborrow(),
				);
			})
		}
		pub fn solve_unit_triangular_lower_in_place<T>(
			L: MatRef,
			L_conj: Conj,
			rhs: MatMut,
			team: *const Team,
		) {
			with_team_par(team, |mut par| {
				la::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(L.faer::<T>(), L_conj.faer(), rhs.faer(), par.reborrow());
			})
		}
		pub fn solve_unit_triangular_upper_in_place<T>(
			U: MatRef,
			U_conj: Conj,
			rhs: MatMut,
			team: *const Team,
		) {
			with_team_par(team, |mut par| {
				la::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(U.faer::<T>(), U_conj.faer(), rhs.faer(), par.reborrow());
			})
		}
		// INVERSE
		pub fn inverse_triangular_lower_in_place<T>(
			L_inv: MatMut,
			L: MatRef,
			team: *const Team,
		) {
			with_team_par(team, |mut par| {
				la::triangular_inverse::invert_lower_triangular(
					L_inv.faer::<T>(),
					L.faer(),
					par.reborrow(),
				);
			})
		}
		pub fn inverse_triangular_upper_in_place<T>(
			L_inv: MatMut,
			L: MatRef,
			team: *const Team,
		) {
			with_team_par(team, |mut par| {
				la::triangular_inverse::invert_upper_triangular(
					L_inv.faer::<T>(),
					L.faer(),
					par.reborrow(),
				);
			})
		}
		pub fn inverse_unit_triangular_lower_in_place<T>(
			L_inv: MatMut,
			L: MatRef,
			team: *const Team,
		) {
			with_team_par(team, |mut par| {
				la::triangular_inverse::invert_unit_lower_triangular(
					L_inv.faer::<T>(),
					L.faer(),
					par.reborrow(),
				);
			})
		}
		pub fn inverse_unit_triangular_upper_in_place<T>(
			L_inv: MatMut,
			L: MatRef,
			team: *const Team,
		) {
			with_team_par(team, |mut par| {
				la::triangular_inverse::invert_unit_upper_triangular(
					L_inv.faer::<T>(),
					L.faer(),
					par.reborrow(),
				);
			})
		}
		// LLT
		pub fn llt_factor_in_place_scratch<T>(
			dim: usize,
			team: *const Team,
			params: LltParams,
		) -> Layout {
			la::cholesky::llt::factor::cholesky_in_place_scratch::<T>(
				dim,
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn llt_factor_in_place<T>(
			A: MatMut,
			regularization: LltRegularization,
			team: *const Team,
			mem: MemAlloc,
			params: LltParams,
		) -> LltStatus {
			with_team_par(team, |mut par| {
				la::cholesky::llt::factor::cholesky_in_place::<T>(
					A.faer(),
					regularization.faer(),
					par.reborrow(),
					mem.faer(),
					params.faer(),
				)
				.into()
			})
		}
		pub fn llt_solve_in_place_scratch<T>(
			dim: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::cholesky::llt::solve::solve_in_place_scratch::<T>(
				dim,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn llt_solve_in_place<T>(
			L: MatRef,
			A_conj: Conj,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::llt::solve::solve_in_place_with_conj(
					L.faer::<T>(),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn llt_reconstruct_scratch<T>(dim: usize, team: *const Team) -> Layout {
			la::cholesky::llt::reconstruct::reconstruct_scratch::<T>(
				dim,
				Team::shape(team),
			)
			.into()
		}
		pub fn llt_reconstruct<T>(
			A: MatMut,
			L: MatRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::llt::reconstruct::reconstruct(
					A.faer::<T>(),
					L.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn llt_inverse_scratch<T>(dim: usize, team: *const Team) -> Layout {
			la::cholesky::llt::inverse::inverse_scratch::<T>(dim, Team::shape(team))
				.into()
		}
		pub fn llt_inverse<T>(
			A_inv: MatMut,
			L: MatRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::llt::inverse::inverse(
					A_inv.faer::<T>(),
					L.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn piv_llt_factor_in_place_scratch<I, T>(
			dim: usize,
			team: *const Team,
			params: PivLltParams,
		) -> Layout {
			la::cholesky::llt_pivoting::factor::cholesky_in_place_scratch::<I, T>(dim, Team::shape(team), params.faer()).into()
		}
		pub fn piv_llt_factor_in_place<I, T>(
			A: MatMut,
			perm_fwd: SliceMut,
			perm_bwd: SliceMut,
			team: *const Team,
			mem: MemAlloc,
			params: PivLltParams,
		) -> PivLltStatus {
			with_team_par(team, |mut par| {
				la::cholesky::llt_pivoting::factor::cholesky_in_place::<I, T>(
					A.faer(),
					perm_fwd.slice(),
					perm_bwd.slice(),
					par.reborrow(),
					mem.faer(),
					params.faer(),
				)
				.map(|x| x.0)
				.into()
			})
		}
		pub fn piv_llt_solve_in_place_scratch<I, T>(
			dim: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::cholesky::llt_pivoting::solve::solve_in_place_scratch::<I, T>(
				dim,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn piv_llt_solve_in_place<I, T>(
			L: MatRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			A_conj: Conj,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::llt_pivoting::solve::solve_in_place_with_conj(
					L.faer::<T>(),
					faer::perm::PermRef::<I>::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn piv_llt_reconstruct_scratch<I, T>(
			dim: usize,
			team: *const Team,
		) -> Layout {
			la::cholesky::llt_pivoting::reconstruct::reconstruct_scratch::<I, T>(dim, Team::shape(team)).into()
		}
		pub fn piv_llt_reconstruct<I, T>(
			A: MatMut,
			L: MatRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::llt_pivoting::reconstruct::reconstruct(
					A.faer::<T>(),
					L.faer(),
					faer::perm::PermRef::<I>::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn piv_llt_inverse_scratch<I, T>(dim: usize, team: *const Team) -> Layout {
			la::cholesky::llt_pivoting::inverse::inverse_scratch::<I, T>(
				dim,
				Team::shape(team),
			)
			.into()
		}
		pub fn piv_llt_inverse<I, T>(
			A_inv: MatMut,
			L: MatRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::llt_pivoting::inverse::inverse(
					A_inv.faer::<T>(),
					L.faer(),
					faer::perm::PermRef::<I>::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		// LDLT
		pub fn ldlt_factor_in_place_scratch<T>(
			dim: usize,
			team: *const Team,
			params: LdltParams,
		) -> Layout {
			la::cholesky::ldlt::factor::cholesky_in_place_scratch::<T>(
				dim,
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn ldlt_factor_in_place<T>(
			A: MatMut,
			regularization: LdltRegularization,
			team: *const Team,
			mem: MemAlloc,
			params: LdltParams,
		) -> LdltStatus {
			with_team_par(team, |mut par| {
				la::cholesky::ldlt::factor::cholesky_in_place::<T>(
					A.faer(),
					regularization.faer(),
					par.reborrow(),
					mem.faer(),
					params.faer(),
				)
				.into()
			})
		}
		pub fn ldlt_solve_in_place_scratch<T>(
			dim: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::cholesky::ldlt::solve::solve_in_place_scratch::<T>(
				dim,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn ldlt_solve_in_place<T>(
			L: MatRef,
			D: VecRef,
			A_conj: Conj,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::ldlt::solve::solve_in_place_with_conj(
					L.faer::<T>(),
					D.diag(),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn ldlt_reconstruct_scratch<T>(dim: usize, team: *const Team) -> Layout {
			la::cholesky::ldlt::reconstruct::reconstruct_scratch::<T>(
				dim,
				Team::shape(team),
			)
			.into()
		}
		pub fn ldlt_reconstruct<T>(
			A: MatMut,
			L: MatRef,
			D: VecRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::ldlt::reconstruct::reconstruct(
					A.faer::<T>(),
					L.faer(),
					D.diag(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn ldlt_inverse_scratch<T>(dim: usize, team: *const Team) -> Layout {
			la::cholesky::ldlt::inverse::inverse_scratch::<T>(dim, Team::shape(team))
				.into()
		}
		pub fn ldlt_inverse<T>(
			A_inv: MatMut,
			L: MatRef,
			D: VecRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::ldlt::inverse::inverse(
					A_inv.faer::<T>(),
					L.faer(),
					D.diag(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		// LBLT
		pub fn lblt_factor_in_place_scratch<I, T>(
			dim: usize,
			team: *const Team,
			params: LbltParams,
		) -> Layout {
			la::cholesky::lblt::factor::cholesky_in_place_scratch::<I, T>(
				dim,
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn lblt_factor_in_place<I, T>(
			A: MatMut,
			subdiag: VecMut,
			perm_fwd: SliceMut,
			perm_bwd: SliceMut,
			team: *const Team,
			mem: MemAlloc,
			params: LbltParams,
		) -> LbltStatus {
			with_team_par(team, |mut par| {
				Ok(la::cholesky::lblt::factor::cholesky_in_place::<I, T>(
					A.faer(),
					subdiag.diag(),
					perm_fwd.slice(),
					perm_bwd.slice(),
					par.reborrow(),
					mem.faer(),
					params.faer(),
				)
				.0)
				.into()
			})
		}
		pub fn lblt_solve_in_place_scratch<I, T>(
			dim: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::cholesky::lblt::solve::solve_in_place_scratch::<I, T>(
				dim,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn lblt_solve_in_place<I, T>(
			L: MatRef,
			diag: VecRef,
			subdiag: VecRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::lblt::solve::solve_in_place_with_conj::<I, T>(
					L.faer(),
					diag.diag(),
					subdiag.diag(),
					A_conj.faer(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn lblt_reconstruct_scratch<I, T>(dim: usize, team: *const Team) -> Layout {
			la::cholesky::lblt::reconstruct::reconstruct_scratch::<I, T>(
				dim,
				Team::shape(team),
			)
			.into()
		}
		pub fn lblt_reconstruct<I, T>(
			A: MatMut,
			L: MatRef,
			diag: VecRef,
			subdiag: VecRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::lblt::reconstruct::reconstruct::<I, T>(
					A.faer(),
					L.faer(),
					diag.diag(),
					subdiag.diag(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn lblt_inverse_scratch<I, T>(dim: usize, team: *const Team) -> Layout {
			la::cholesky::lblt::inverse::inverse_scratch::<I, T>(
				dim,
				Team::shape(team),
			)
			.into()
		}
		pub fn lblt_inverse<I, T>(
			A_inv: MatMut,
			L: MatRef,
			diag: VecRef,
			subdiag: VecRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::cholesky::lblt::inverse::inverse::<I, T>(
					A_inv.faer(),
					L.faer(),
					diag.diag(),
					subdiag.diag(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		// HOUSEHOLDER
		pub fn apply_householder_on_the_left_scratch<T>(
			dim: usize,
			block_size: usize,
			rhs_ncols: usize,
		) -> Layout {
			la::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(dim, block_size, rhs_ncols).into()
		}
		pub fn apply_householder_on_the_left<T>(
			householder_basis: MatRef,
			householder_factor: MatRef,
			householder_conj: Conj,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj::<T>(
					householder_basis.faer(),
					householder_factor.faer(),
					householder_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn apply_householder_transpose_on_the_left_scratch<T>(
			dim: usize,
			block_size: usize,
			rhs_ncols: usize,
		) -> Layout {
			la::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(dim, block_size, rhs_ncols).into()
		}
		pub fn apply_householder_transpose_on_the_left<T>(
			householder_basis: MatRef,
			householder_factor: MatRef,
			householder_conj: Conj,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj::<T>(
					householder_basis.faer(),
					householder_factor.faer(),
					householder_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn apply_householder_on_the_right_scratch<T>(
			dim: usize,
			block_size: usize,
			rhs_ncols: usize,
		) -> Layout {
			la::householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(dim, block_size, rhs_ncols).into()
		}
		pub fn apply_householder_on_the_right<T>(
			householder_basis: MatRef,
			householder_factor: MatRef,
			householder_conj: Conj,
			lhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::householder::apply_block_householder_sequence_on_the_right_in_place_with_conj::<T>(
					householder_basis.faer(),
					householder_factor.faer(),
					householder_conj.faer(),
					lhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn apply_householder_transpose_on_the_right_scratch<T>(
			dim: usize,
			block_size: usize,
			rhs_ncols: usize,
		) -> Layout {
			la::householder::apply_block_householder_sequence_transpose_on_the_right_in_place_scratch::<T>(dim, block_size, rhs_ncols).into()
		}
		pub fn apply_householder_transpose_on_the_right<T>(
			householder_basis: MatRef,
			householder_factor: MatRef,
			householder_conj: Conj,
			lhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::householder::apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj::<T>(
					householder_basis.faer(),
					householder_factor.faer(),
					householder_conj.faer(),
					lhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		// QR
		pub fn qr_recommended_block_size<T>(
			nrows: usize,
			ncols: usize,
		) -> usize {
			la::qr::no_pivoting::factor::recommended_block_size::<T>(
				nrows, ncols,
			)
		}
		pub fn qr_factor_in_place_scratch<T>(
			nrows: usize,
			ncols: usize,
			block_size: usize,
			team: *const Team,
			params: QrParams,
		) -> Layout {
			la::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
				nrows,
				ncols,
				block_size,
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn qr_factor_in_place<T>(
			A: MatMut,
			Q_coeff: MatMut,
			team: *const Team,
			mem: MemAlloc,
			params: QrParams,
		) -> QrStatus {
			with_team_par(team, |mut par| {
				Ok(la::qr::no_pivoting::factor::qr_in_place::<T>(
					A.faer(),
					Q_coeff.faer(),
					par.reborrow(),
					mem.faer(),
					params.faer(),
				))
				.into()
			})
		}
		pub fn qr_solve_in_place_scratch<T>(
			dim: usize,
			block_size: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::qr::no_pivoting::solve::solve_in_place_scratch::<T>(
				dim,
				block_size,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn qr_solve_in_place<T>(
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			A_conj: Conj,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::qr::no_pivoting::solve::solve_in_place_with_conj::<T>(
					Q_basis.faer(),
					Q_coeff.faer(),
					R.faer(),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn qr_solve_transpose_in_place_scratch<T>(
			dim: usize,
			block_size: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::qr::no_pivoting::solve::solve_transpose_in_place_scratch::<T>(
				dim,
				block_size,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn qr_solve_transpose_in_place<T>(
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			A_conj: Conj,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::qr::no_pivoting::solve::solve_transpose_in_place_with_conj::<T>(
					Q_basis.faer(),
					Q_coeff.faer(),
					R.faer(),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn qr_solve_lstsq_in_place_scratch<T>(
			nrows: usize,
			ncols: usize,
			block_size: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::qr::no_pivoting::solve::solve_lstsq_in_place_scratch::<T>(
				nrows,
				ncols,
				block_size,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn qr_solve_lstsq_in_place<T>(
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			A_conj: Conj,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::qr::no_pivoting::solve::solve_lstsq_in_place_with_conj::<T>(
					Q_basis.faer(),
					Q_coeff.faer(),
					R.faer(),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn qr_reconstruct_scratch<T>(
			nrows: usize,
			ncols: usize,
			block_size: usize,
			team: *const Team,
		) -> Layout {
			la::qr::no_pivoting::reconstruct::reconstruct_scratch::<T>(
				nrows,
				ncols,
				block_size,
				Team::shape(team),
			)
			.into()
		}
		pub fn qr_reconstruct<T>(
			A: MatMut,
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::qr::no_pivoting::reconstruct::reconstruct::<T>(
					A.faer(),
					Q_basis.faer(),
					Q_coeff.faer(),
					R.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn qr_inverse_scratch<T>(
			dim: usize,
			block_size: usize,
			team: *const Team,
		) -> Layout {
			la::qr::no_pivoting::inverse::inverse_scratch::<T>(
				dim,
				block_size,
				Team::shape(team),
			)
			.into()
		}
		pub fn qr_inverse<T>(
			A: MatMut,
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::qr::no_pivoting::inverse::inverse::<T>(
					A.faer(),
					Q_basis.faer(),
					Q_coeff.faer(),
					R.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn colpiv_qr_factor_in_place_scratch<I, T>(
			nrows: usize,
			ncols: usize,
			block_size: usize,
			team: *const Team,
			params: ColPivQrParams,
		) -> Layout {
			la::qr::col_pivoting::factor::qr_in_place_scratch::<I, T>(
				nrows,
				ncols,
				block_size,
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn colpiv_qr_factor_in_place<I, T>(
			A: MatMut,
			Q_coeff: MatMut,
			perm_fwd: SliceMut,
			perm_bwd: SliceMut,
			team: *const Team,
			mem: MemAlloc,
			params: ColPivQrParams,
		) -> ColPivQrStatus {
			with_team_par(team, |mut par| {
				Ok(la::qr::col_pivoting::factor::qr_in_place::<I, T>(
					A.faer(),
					Q_coeff.faer(),
					perm_fwd.slice(),
					perm_bwd.slice(),
					par.reborrow(),
					mem.faer(),
					params.faer(),
				)
				.0)
				.into()
			})
		}
		pub fn colpiv_qr_solve_in_place_scratch<I, T>(
			dim: usize,
			block_size: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::qr::col_pivoting::solve::solve_in_place_scratch::<I, T>(
				dim,
				block_size,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn colpiv_qr_solve_in_place<I, T>(
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::qr::col_pivoting::solve::solve_in_place_with_conj::<I, T>(
					Q_basis.faer(),
					Q_coeff.faer(),
					R.faer(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn colpiv_qr_solve_transpose_in_place_scratch<I, T>(
			dim: usize,
			block_size: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::qr::col_pivoting::solve::solve_transpose_in_place_scratch::<I, T>(dim, block_size, rhs_ncols, Team::shape(team)).into()
		}
		pub fn colpiv_qr_solve_transpose_in_place<I, T>(
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::qr::col_pivoting::solve::solve_transpose_in_place_with_conj::<
					I,
					T,
				>(
					Q_basis.faer(),
					Q_coeff.faer(),
					R.faer(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn colpiv_qr_solve_lstsq_in_place_scratch<I, T>(
			nrows: usize,
			ncols: usize,
			block_size: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::qr::col_pivoting::solve::solve_lstsq_in_place_scratch::<I, T>(
				nrows,
				ncols,
				block_size,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn colpiv_qr_solve_lstsq_in_place<I, T>(
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::qr::col_pivoting::solve::solve_lstsq_in_place_with_conj::<I, T>(
					Q_basis.faer(),
					Q_coeff.faer(),
					R.faer(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn colpiv_qr_reconstruct_scratch<I, T>(
			nrows: usize,
			ncols: usize,
			block_size: usize,
			team: *const Team,
		) -> Layout {
			la::qr::col_pivoting::reconstruct::reconstruct_scratch::<I, T>(
				nrows,
				ncols,
				block_size,
				Team::shape(team),
			)
			.into()
		}
		pub fn colpiv_qr_reconstruct<I, T>(
			A: MatMut,
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::qr::col_pivoting::reconstruct::reconstruct::<I, T>(
					A.faer(),
					Q_basis.faer(),
					Q_coeff.faer(),
					R.faer(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn colpiv_qr_inverse_scratch<I, T>(
			dim: usize,
			block_size: usize,
			team: *const Team,
		) -> Layout {
			la::qr::col_pivoting::inverse::inverse_scratch::<I, T>(
				dim,
				block_size,
				Team::shape(team),
			)
			.into()
		}
		pub fn colpiv_qr_inverse<I, T>(
			A: MatMut,
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::qr::col_pivoting::inverse::inverse::<I, T>(
					A.faer(),
					Q_basis.faer(),
					Q_coeff.faer(),
					R.faer(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		// LU
		pub fn partial_piv_lu_factor_in_place_scratch<I, T>(
			dim: usize,
			block_size: usize,
			team: *const Team,
			params: PartialPivLuParams,
		) -> Layout {
			la::lu::partial_pivoting::factor::lu_in_place_scratch::<I, T>(
				dim,
				block_size,
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn partial_piv_lu_factor_in_place<I, T>(
			A: MatMut,
			perm_fwd: SliceMut,
			perm_bwd: SliceMut,
			team: *const Team,
			mem: MemAlloc,
			params: PartialPivLuParams,
		) -> PartialPivLuStatus {
			with_team_par(team, |mut par| {
				Ok(la::lu::partial_pivoting::factor::lu_in_place::<I, T>(
					A.faer(),
					perm_fwd.slice(),
					perm_bwd.slice(),
					par.reborrow(),
					mem.faer(),
					params.faer(),
				)
				.0)
				.into()
			})
		}
		pub fn partial_piv_lu_solve_in_place_scratch<I, T>(
			dim: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::lu::partial_pivoting::solve::solve_in_place_scratch::<I, T>(
				dim,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn partial_piv_lu_solve_in_place<I, T>(
			L: MatRef,
			U: MatRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::lu::partial_pivoting::solve::solve_in_place_with_conj::<I, T>(
					L.faer(),
					U.faer(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn partial_piv_lu_solve_transpose_in_place_scratch<I, T>(
			dim: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<
				I,
				T,
			>(dim, rhs_ncols, Team::shape(team))
			.into()
		}
		pub fn partial_piv_lu_solve_transpose_in_place<I, T>(
			L: MatRef,
			U: MatRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::lu::partial_pivoting::solve::solve_transpose_in_place_with_conj::<
					I,
					T,
				>(
					L.faer(),
					U.faer(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn partial_piv_lu_reconstruct_scratch<I, T>(
			nrows: usize,
			ncols: usize,
			team: *const Team,
		) -> Layout {
			la::lu::partial_pivoting::reconstruct::reconstruct_scratch::<I, T>(
				nrows,
				ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn partial_piv_lu_reconstruct<I, T>(
			A: MatMut,
			L: MatRef,
			U: MatRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::lu::partial_pivoting::reconstruct::reconstruct::<I, T>(
					A.faer(),
					L.faer(),
					U.faer(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn partial_piv_lu_inverse_scratch<I, T>(
			dim: usize,
			team: *const Team,
		) -> Layout {
			la::lu::partial_pivoting::inverse::inverse_scratch::<I, T>(
				dim,
				Team::shape(team),
			)
			.into()
		}
		pub fn partial_piv_lu_inverse<I, T>(
			A: MatMut,
			L: MatRef,
			U: MatRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::lu::partial_pivoting::inverse::inverse::<I, T>(
					A.faer(),
					L.faer(),
					U.faer(),
					faer::perm::PermRef::new_unchecked(
						perm_fwd.slice(),
						perm_bwd.slice(),
						Ord::min(perm_fwd.len, perm_bwd.len),
					),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn full_piv_lu_factor_in_place_scratch<I, T>(
			dim: usize,
			block_size: usize,
			team: *const Team,
			params: FullPivLuParams,
		) -> Layout {
			la::lu::full_pivoting::factor::lu_in_place_scratch::<I, T>(
				dim,
				block_size,
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn full_piv_lu_factor_in_place<I, T>(
			A: MatMut,
			row_perm_fwd: SliceMut,
			row_perm_bwd: SliceMut,
			col_perm_fwd: SliceMut,
			col_perm_bwd: SliceMut,
			team: *const Team,
			mem: MemAlloc,
			params: FullPivLuParams,
		) -> FullPivLuStatus {
			with_team_par(team, |mut par| {
				Ok(la::lu::full_pivoting::factor::lu_in_place::<I, T>(
					A.faer(),
					row_perm_fwd.slice(),
					row_perm_bwd.slice(),
					col_perm_fwd.slice(),
					col_perm_bwd.slice(),
					par.reborrow(),
					mem.faer(),
					params.faer(),
				)
				.0)
				.into()
			})
		}
		pub fn full_piv_lu_solve_in_place_scratch<I, T>(
			dim: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::lu::full_pivoting::solve::solve_in_place_scratch::<I, T>(
				dim,
				rhs_ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn full_piv_lu_solve_in_place<I, T>(
			L: MatRef,
			U: MatRef,
			A_conj: Conj,
			row_perm_fwd: SliceRef,
			row_perm_bwd: SliceRef,
			col_perm_fwd: SliceRef,
			col_perm_bwd: SliceRef,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::lu::full_pivoting::solve::solve_in_place_with_conj::<I, T>(
					L.faer(),
					U.faer(),
					faer::perm::PermRef::new_unchecked(
						row_perm_fwd.slice(),
						row_perm_bwd.slice(),
						Ord::min(row_perm_fwd.len, row_perm_bwd.len),
					),
					faer::perm::PermRef::new_unchecked(
						col_perm_fwd.slice(),
						col_perm_bwd.slice(),
						Ord::min(col_perm_fwd.len, col_perm_bwd.len),
					),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn full_piv_lu_solve_transpose_in_place_scratch<I, T>(
			dim: usize,
			rhs_ncols: usize,
			team: *const Team,
		) -> Layout {
			la::lu::full_pivoting::solve::solve_transpose_in_place_scratch::<I, T>(dim, rhs_ncols, Team::shape(team)).into()
		}
		pub fn full_piv_lu_solve_transpose_in_place<I, T>(
			L: MatRef,
			U: MatRef,
			A_conj: Conj,
			row_perm_fwd: SliceRef,
			row_perm_bwd: SliceRef,
			col_perm_fwd: SliceRef,
			col_perm_bwd: SliceRef,
			rhs: MatMut,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::lu::full_pivoting::solve::solve_transpose_in_place_with_conj::<
					I,
					T,
				>(
					L.faer(),
					U.faer(),
					faer::perm::PermRef::new_unchecked(
						row_perm_fwd.slice(),
						row_perm_bwd.slice(),
						Ord::min(row_perm_fwd.len, row_perm_bwd.len),
					),
					faer::perm::PermRef::new_unchecked(
						col_perm_fwd.slice(),
						col_perm_bwd.slice(),
						Ord::min(col_perm_fwd.len, col_perm_bwd.len),
					),
					A_conj.faer(),
					rhs.faer(),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn full_piv_lu_reconstruct_scratch<I, T>(
			nrows: usize,
			ncols: usize,
			team: *const Team,
		) -> Layout {
			la::lu::full_pivoting::reconstruct::reconstruct_scratch::<I, T>(
				nrows,
				ncols,
				Team::shape(team),
			)
			.into()
		}
		pub fn full_piv_lu_reconstruct<I, T>(
			A: MatMut,
			L: MatRef,
			U: MatRef,
			row_perm_fwd: SliceRef,
			row_perm_bwd: SliceRef,
			col_perm_fwd: SliceRef,
			col_perm_bwd: SliceRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::lu::full_pivoting::reconstruct::reconstruct::<I, T>(
					A.faer(),
					L.faer(),
					U.faer(),
					faer::perm::PermRef::new_unchecked(
						row_perm_fwd.slice(),
						row_perm_bwd.slice(),
						Ord::min(row_perm_fwd.len, row_perm_bwd.len),
					),
					faer::perm::PermRef::new_unchecked(
						col_perm_fwd.slice(),
						col_perm_bwd.slice(),
						Ord::min(col_perm_fwd.len, col_perm_bwd.len),
					),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		pub fn full_piv_lu_inverse_scratch<I, T>(
			dim: usize,
			team: *const Team,
		) -> Layout {
			la::lu::full_pivoting::inverse::inverse_scratch::<I, T>(
				dim,
				Team::shape(team),
			)
			.into()
		}
		pub fn full_piv_lu_inverse<I, T>(
			A: MatMut,
			L: MatRef,
			U: MatRef,
			row_perm_fwd: SliceRef,
			row_perm_bwd: SliceRef,
			col_perm_fwd: SliceRef,
			col_perm_bwd: SliceRef,
			team: *const Team,
			mem: MemAlloc,
		) {
			with_team_par(team, |mut par| {
				la::lu::full_pivoting::inverse::inverse::<I, T>(
					A.faer(),
					L.faer(),
					U.faer(),
					faer::perm::PermRef::new_unchecked(
						row_perm_fwd.slice(),
						row_perm_bwd.slice(),
						Ord::min(row_perm_fwd.len, row_perm_bwd.len),
					),
					faer::perm::PermRef::new_unchecked(
						col_perm_fwd.slice(),
						col_perm_bwd.slice(),
						Ord::min(col_perm_fwd.len, col_perm_bwd.len),
					),
					par.reborrow(),
					mem.faer(),
				)
			})
		}
		// SVD
		pub fn svd_scratch<T>(
			nrows: usize,
			ncols: usize,
			compute_U: ComputeSvdVectors,
			compute_V: ComputeSvdVectors,
			team: *const Team,
			params: SvdParams,
		) -> Layout {
			la::svd::svd_scratch::<T>(
				nrows,
				ncols,
				compute_U.faer(),
				compute_V.faer(),
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn svd<T>(
			A: MatRef,
			U: MatMut,
			S: VecMut,
			V: MatMut,
			team: *const Team,
			mem: MemAlloc,
			params: SvdParams,
		) -> SvdStatus {
			with_team_par(team, |mut par| {
				let U = if U.ncols == 0 { None } else { Some(U.faer()) };
				let V = if V.ncols == 0 { None } else { Some(V.faer()) };
				la::svd::svd::<T>(
					A.faer(),
					S.diag(),
					U,
					V,
					par.reborrow(),
					mem.faer(),
					params.faer(),
				)
				.into()
			})
		}
		// EVD (self adjoint)
		pub fn self_adjoint_evd_scratch<T>(
			dim: usize,
			compute_U: ComputeEigenvectors,
			team: *const Team,
			params: SelfAdjointEvdParams,
		) -> Layout {
			la::evd::self_adjoint_evd_scratch::<T>(
				dim,
				compute_U.faer(),
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn self_adjoint_evd<T>(
			A: MatRef,
			U: MatMut,
			S: VecMut,
			team: *const Team,
			mem: MemAlloc,
			params: SelfAdjointEvdParams,
		) -> EvdStatus {
			with_team_par(team, |mut par| {
				let U = if U.ncols == 0 { None } else { Some(U.faer()) };
				la::evd::self_adjoint_evd::<T>(
					A.faer(),
					S.diag(),
					U,
					par.reborrow(),
					mem.faer(),
					params.faer(),
				)
				.into()
			})
		}
		// EVD (general)
		pub fn evd_scratch<T>(
			dim: usize,
			compute_left: ComputeEigenvectors,
			compute_right: ComputeEigenvectors,
			team: *const Team,
			params: EvdParams,
		) -> Layout {
			la::evd::evd_scratch::<T>(
				dim,
				compute_left.faer(),
				compute_right.faer(),
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn evd<T>(
			A: MatRef,
			UL: MatMut,
			UR: MatMut,
			S: VecMut,
			S_im: VecMut,
			team: *const Team,
			mem: MemAlloc,
			params: EvdParams,
		) -> EvdStatus {
			with_team_par(team, |mut par| {
				type R = <T as faer::traits::ComplexField>::Real;
				if const { <T as faer::traits::ComplexField>::IS_REAL } {
					let UL = if UL.ncols == 0 { None } else { Some(UL.faer()) };
					let UR = if UR.ncols == 0 { None } else { Some(UR.faer()) };
					la::evd::evd_real::<R>(
						A.faer(),
						S.diag(),
						S_im.diag(),
						UL,
						UR,
						par.reborrow(),
						mem.faer(),
						params.faer(),
					)
					.into()
				} else {
					let UL = if UL.ncols == 0 { None } else { Some(UL.faer()) };
					let UR = if UR.ncols == 0 { None } else { Some(UR.faer()) };
					la::evd::evd_cplx::<R>(
						A.faer(),
						S.diag(),
						UL,
						UR,
						par.reborrow(),
						mem.faer(),
						params.faer(),
					)
					.into()
				}
			})
		}
		// EVD (general)
		pub fn generalized_evd_scratch<T>(
			dim: usize,
			compute_left: ComputeEigenvectors,
			compute_right: ComputeEigenvectors,
			team: *const Team,
			params: GevdParams,
		) -> Layout {
			la::gevd::gevd_scratch::<T>(
				dim,
				compute_left.faer(),
				compute_right.faer(),
				Team::shape(team),
				params.faer(),
			)
			.into()
		}
		pub fn generalized_evd<T>(
			A: MatMut,
			B: MatMut,
			UL: MatMut,
			UR: MatMut,
			alpha: VecMut,
			alpha_im: VecMut,
			beta: VecMut,
			team: *const Team,
			mem: MemAlloc,
			params: GevdParams,
		) -> GevdStatus {
			with_team_par(team, |mut par| {
				type R = <T as faer::traits::ComplexField>::Real;
				if const { <T as faer::traits::ComplexField>::IS_REAL } {
					let UL = if UL.ncols == 0 { None } else { Some(UL.faer()) };
					let UR = if UR.ncols == 0 { None } else { Some(UR.faer()) };
					la::gevd::gevd_real::<R>(
						A.faer(),
						B.faer(),
						alpha.diag(),
						alpha_im.diag(),
						beta.diag(),
						UL,
						UR,
						par.reborrow(),
						mem.faer(),
						params.faer(),
					)
					.into()
				} else {
					let UL = if UL.ncols == 0 { None } else { Some(UL.faer()) };
					let UR = if UR.ncols == 0 { None } else { Some(UR.faer()) };
					la::gevd::gevd_cplx::<R>(
						A.faer(),
						B.faer(),
						alpha.diag(),
						beta.diag(),
						UL,
						UR,
						par.reborrow(),
						mem.faer(),
						params.faer(),
					)
					.into()
				}
			})
		}
	});
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libfaer_v0_23_alloc(
	size: usize,
	align: usize,
) -> *mut c_void {
	unsafe {
		std::alloc::alloc(
			match core::alloc::Layout::from_size_align(size, align) {
				Ok(layout) => layout,
				Err(_) => return core::ptr::null_mut(),
			},
		) as *mut c_void
	}
}
/// Creates a team handle of `threads` participants: the calling thread and `threads - 1` reserved
/// workers, the same convention as `faer::team::with_team`. Zero and one both mean sequential, and a
/// sequential handle reserves nothing.
///
/// The handle **owns its pool**: the workers are built here, once, and every operation handed the
/// handle borrows them for the duration of that one call (see `Team::with_par`) and hands them
/// back before returning. A handle reused for N calls builds one pool, not N. The threads exist from
/// here until [`libfaer_v0_23_team_destroy`]; if the host cannot provide them, this function panics
/// inside the pool build, and a panic crossing this `extern "C"` boundary aborts the process. A
/// handle is a promise that the threads exist, and the operations return no status that could carry
/// an acquisition error.
///
/// The rules the ownership brings, in full:
///
/// * **Two threads, one handle.** An operation may be called on one handle from two threads at once
///   as long as they write different elements. They do not run at once: the pool is a single object
///   that one call holds for its duration, so the second waits for the first to hand it back. A
///   caller that wants its own threads to be parallel wants a handle per thread.
/// * **A callback must not nest a parallel call.** An operation called from inside another operation
///   on the same thread (the `SchurParams` callbacks, for one) runs sequentially: it gets no team,
///   where waiting for the pool its caller holds would never end. It is correct, just not parallel.
///   Such a call takes no lock, so the turn-taking above does not order it against another thread
///   using the same handle: a nested call runs whenever it is made.
/// * **[`libfaer_v0_23_team_set_threads`] and [`libfaer_v0_23_team_destroy`] are exclusive**: no
///   operation may be running on the handle, on any thread, while they are called.
/// * **After `destroy` the handle is gone.** Any later operation, `set_threads` or second `destroy`
///   on that pointer is a use after free: the C caller owns the pointer and nothing can detect it,
///   so invalidate your copy where you free it. `destroy(NULL)` is accepted and ignored.
/// * **`fork` without `exec` is not supported.** A forked child inherits threadteam's process-wide
///   cache of its workers (and its mutex) without the threads, so any handle there — the inherited
///   one and a fresh one alike — can block on a lock the parent held or reuse workers it does not
///   have. Create handles in the process that uses them, and `exec` before using a team in a child
///   whose parent already ran one. Whether `create` should record the creator's pid and refuse to
///   work in a process it did not create the handle in is **undecided**, and nothing checks it today.
///
/// The handle's contents are not the caller's to write: it is opaque and must be created and
/// released by these three functions only.
///
/// ```c
/// #include "faer.h"
///
/// // line comments only: this block sits inside a C comment
/// double a[4] = {1, 2, 3, 4};   // 2 x 2, column major
/// double c[4] = {0, 0, 0, 0};
/// double one = 1.0;
/// FaerV0_24_Team *team = libfaer_v0_23_team_create(4);   // NULL would run sequentially
/// libfaer_v0_23_matmul_f64(
///     (FaerV0_24_MatMut){c, 2, 2, 1, 2}, FaerV0_24_Accum_Replace,
///     (FaerV0_24_MatRef){a, 2, 2, 1, 2}, (FaerV0_24_MatRef){a, 2, 2, 1, 2},
///     (const FaerV0_24_Scalar *)&one, team);    // c = a * a on 4 participants
/// libfaer_v0_23_team_set_threads(team, 2);     // not while an operation uses the handle
/// libfaer_v0_23_team_destroy(team);
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libfaer_v0_23_team_create(threads: usize) -> *mut Team {
	let threads = threads.max(1);
	Box::into_raw(Box::new(Team {
		threads,
		pool: std::sync::Mutex::new(Team::pool_for(threads)),
	}))
}
/// Releases a handle from [`libfaer_v0_23_team_create`] and the pool it owns; `NULL` is accepted and
/// ignored.
///
/// The handle is destroyed exactly once, and only while no operation that was handed it is still
/// running. Dropping the pool returns its workers to threadteam's process-wide cache rather than
/// ending the threads, which is threadteam's own behaviour for a pool with default thread settings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libfaer_v0_23_team_destroy(team: *mut Team) {
	if !team.is_null() {
		unsafe { drop(Box::from_raw(team)) };
	}
}
/// Resets the width of an existing handle, so a caller changes its thread count without rebuilding
/// it. Zero and one both mean sequential, and a sequential handle releases its pool; widening it
/// again builds a new one. Needs exclusive access to the handle: no operation that was handed it may
/// be running, and no other call may be setting it.
///
/// Resizing is a reconfiguration, not a per-call cost: the pool is rebuilt here.
///
/// The width is read once, when an operation takes the pool, so a workspace query
/// (`*_scratch`) made before the change no longer describes an operation of the new width: some
/// sizes depend on the number of participants (full-pivot LU sizes one pivot candidate per
/// participant), so query the scratch again after setting the width.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libfaer_v0_23_team_set_threads(team: *mut Team, threads: usize) {
	if let Some(team) = unsafe { team.as_mut() } {
		team.set_threads(threads.max(1));
	}
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libfaer_v0_23_dealloc(
	ptr: *mut c_void,
	size: usize,
	align: usize,
) {
	unsafe {
		std::alloc::dealloc(
			ptr as *mut u8,
			core::alloc::Layout::from_size_align_unchecked(size, align),
		)
	}
}
/// Counts the allocations the measuring thread makes, so a test can state what one more call on a
/// handle costs (tensor4all/faer#333). Only the calling thread counts: the tests of this crate run
/// in parallel, and a member of a pool allocates on its own thread.
#[cfg(test)]
mod alloc_probe {
	use core::alloc::{GlobalAlloc, Layout};
	use core::cell::Cell;
	use std::alloc::System;

	pub struct Counting;
	thread_local! {
		static ON: Cell<bool> = const { Cell::new(false) };
		static COUNT: Cell<usize> = const { Cell::new(0) };
	}
	impl Counting {
		fn tick() {
			let _ = ON.try_with(|on| {
				if on.get() {
					let _ = COUNT.try_with(|c| c.set(c.get() + 1));
				}
			});
		}
	}
	// SAFETY: every method forwards to `System`, which is the allocator this replaces.
	unsafe impl GlobalAlloc for Counting {
		unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
			Self::tick();
			unsafe { System.alloc(layout) }
		}
		unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
			Self::tick();
			unsafe { System.alloc_zeroed(layout) }
		}
		unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
			Self::tick();
			unsafe { System.realloc(ptr, layout, new_size) }
		}
		unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
			unsafe { System.dealloc(ptr, layout) }
		}
	}
	/// Starts counting the allocations of the calling thread at zero.
	pub fn reset() {
		COUNT.with(|c| c.set(0));
		ON.with(|on| on.set(true));
	}
	/// Stops counting and returns the count since [`reset`].
	#[must_use]
	pub fn take() -> usize {
		ON.with(|on| on.set(false));
		COUNT.with(|c| c.get())
	}
}
#[cfg(test)]
#[global_allocator]
static ALLOCATIONS: alloc_probe::Counting = alloc_probe::Counting;
#[cfg(test)]
mod tests {
	use super::*;
	use crate::linalg as la;

	/// The affinity mask of the calling thread. A team built by `faer::team::with_team` places its
	/// leader on the one CPU no member takes, and the mask comes back when the team is dropped
	/// (threadteam 0.5, Linux), so a mask that still names a single CPU after a call is the
	/// observable form of "the call did not leave the team".
	fn mask() -> Option<String> {
		let status = std::fs::read_to_string("/proc/thread-self/status").ok()?;
		Some(
			status
				.lines()
				.find(|l| l.starts_with("Cpus_allowed_list:"))?
				.to_owned(),
		)
	}

	/// `(a * b)` on `n` x `n` f64 matrices, through the C entry point.
	fn matmul(c: &mut faer::Mat<f64>, a: &faer::Mat<f64>, b: &faer::Mat<f64>, team: *const Team) {
		let alpha = 1.0f64;
		let c = MatMut {
			ptr: c.as_ptr_mut() as *mut c_void,
			nrows: c.nrows(),
			ncols: c.ncols(),
			row_stride: c.row_stride(),
			col_stride: c.col_stride(),
		};
		let (a, b) = (MatRef::of(a), MatRef::of(b));
		unsafe {
			la::libfaer_v0_23_matmul_f64(
				c,
				Accum::Replace,
				a,
				b,
				&raw const alpha as *const Scalar,
				team,
			)
		};
	}

	impl MatRef {
		fn of<T>(m: &faer::Mat<T>) -> MatRef {
			MatRef {
				ptr: m.as_ptr() as *const c_void,
				nrows: m.nrows(),
				ncols: m.ncols(),
				row_stride: m.row_stride(),
				col_stride: m.col_stride(),
			}
		}
	}

	/// The sequential product, from the Rust API.
	fn want(n: usize, a: &faer::Mat<f64>, b: &faer::Mat<f64>) -> faer::Mat<f64> {
		let mut c = faer::Mat::<f64>::zeros(n, n);
		faer::linalg::matmul::matmul(
			c.as_mut(),
			faer::Accum::Replace,
			a.as_ref(),
			b.as_ref(),
			1.0,
			faer::Par::Seq,
		);
		c
	}

	fn close(x: &faer::Mat<f64>, y: &faer::Mat<f64>) -> bool {
		let (n, m) = (x.nrows(), x.ncols());
		(0..n).all(|i| (0..m).all(|j| (x[(i, j)] - y[(i, j)]).abs() <= 1e-10 * (1.0 + x[(i, j)].abs())))
	}

	/// faer's dispatch counters are process wide, and both tests dispatch (one reads the counters,
	/// the other runs a factorization): one test at a time keeps them apart.
	static TEAM_TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

	/// (a) a null handle runs sequentially, (b) a handle runs the same product through a team,
	/// (c) the calling thread is outside the team again once the call returns, and the handle can be
	/// reused, re-widened and destroyed. The mask is checked after a control team of the same width,
	/// because only a host where `with_team` places its leader can show the difference.
	#[test]
	fn the_team_handle_is_entered_per_call_and_left_again() {
		let _turn = TEAM_TURN.lock().unwrap_or_else(|e| e.into_inner());
		let n = 512;
		let a = faer::Mat::<f64>::from_fn(n, n, |i, j| ((i * 31 + j * 7) % 13) as f64 - 6.0);
		let b = faer::Mat::<f64>::from_fn(n, n, |i, j| ((i * 5 + j * 11) % 17) as f64 - 8.0);
		let expected = want(n, &a, &b);

		// (a) no team: the sequential product, and no region on a team
		let mut got = faer::Mat::<f64>::zeros(n, n);
		let before = mask();
		faer::team::stats::enable(true);
		let _ = faer::team::stats::take();
		matmul(&mut got, &a, &b, core::ptr::null());
		let serial = faer::team::stats::take();
		faer::team::stats::enable(false);
		assert!(close(&expected, &got), "a null handle must give the sequential product");
		assert_eq!(mask(), before, "a null handle must not touch the calling thread");
		let teamless = serial.run.iter().skip(2).sum::<usize>()
			+ serial.split.iter().skip(2).sum::<usize>()
			+ serial.gemm_tiled
			+ serial.gemm_team_kernel;
		assert_eq!(teamless, 0, "a null handle must not run on a team: {serial:?}");

		// the control: a held team narrows the mask, and hands it back when it goes. threadteam
		// places its leader only where the caller's mask is at least as wide as the team, and that
		// placement is what the check below observes, so it is required whenever the mask allows it.
		let cpus = cpus();
		let width = cpus.clamp(2, 4);
		let placed = {
			let before = mask();
			let inside = faer::team::with_team(width, |_| mask());
			assert_eq!(mask(), before, "a dropped team must hand the mask back");
			inside.is_some() && inside != before
		};
		// Deliberately a failure, not a skip: the placement is the observable the check below needs,
		// so a host where threadteam cannot pin (a mask wider than its `cpu_set_t`, or affinity
		// syscalls denied) must say so instead of reporting coverage it does not have.
		if cpus >= width {
			assert!(placed, "threadteam placed no leader on a mask of {cpus} CPUs");
		} else {
			eprintln!("a mask of {cpus} CPU(s) cannot hold a team of {width}: placement is not checked");
		}

		// (b) and (c): the same product on the handle, no team left behind
		let team = unsafe { libfaer_v0_23_team_create(width) };
		assert!(!team.is_null());
		let before = mask();
		let mut got = faer::Mat::<f64>::zeros(n, n);
		faer::team::stats::enable(true);
		let _ = faer::team::stats::take();
		matmul(&mut got, &a, &b, team);
		let parallel = faer::team::stats::take();
		faer::team::stats::enable(false);
		assert!(close(&expected, &got), "a team handle must give the sequential product");
		let on_team = parallel.run.iter().skip(2).sum::<usize>()
			+ parallel.split.iter().skip(2).sum::<usize>()
			+ parallel.gemm_tiled
			+ parallel.gemm_team_kernel;
		assert!(on_team > 0, "the handle ran no region on its team: {parallel:?}");
		if placed {
			assert_eq!(
				mask(),
				before,
				"the calling thread is still inside the team after the call"
			);
		}

		// a narrower width on the same handle, and a destroy of a null handle
		unsafe { libfaer_v0_23_team_set_threads(team, 1) };
		let mut got = faer::Mat::<f64>::zeros(n, n);
		let before = mask();
		faer::team::stats::enable(true);
		let _ = faer::team::stats::take();
		matmul(&mut got, &a, &b, team);
		let one = faer::team::stats::take();
		faer::team::stats::enable(false);
		assert!(close(&expected, &got), "a one-participant team must give the same product");
		assert_eq!(
			one.run.iter().skip(2).sum::<usize>() + one.split.iter().skip(2).sum::<usize>() + one.gemm_tiled + one.gemm_team_kernel,
			0,
			"a one-participant team must be sequential: {one:?}"
		);
		assert_eq!(mask(), before, "a one-participant team must not place the caller");
		unsafe {
			libfaer_v0_23_team_destroy(team);
			libfaer_v0_23_team_destroy(core::ptr::null_mut());
		}
	}

	/// The handles this process may use: the count of CPUs in the calling thread's affinity mask,
	/// which is what threadteam requires a team to fit into before it places its leader.
	fn cpus() -> usize {
		let Some(list) = mask() else { return 1 };
		let Some(list) = list.split(':').nth(1) else { return 1 };
		list.split(',').map(|item| match item.split_once('-') {
			Some((lo, hi)) => hi.trim().parse::<usize>().unwrap_or(0) - lo.trim().parse::<usize>().unwrap_or(0) + 1,
			None => 1,
		}).sum()
	}

	fn perm_of(n: usize) -> Vec<u32> {
		(0..n as u32).collect()
	}

	/// The workspace queries read the handle's width without entering a team, and the size they
	/// return is enough for the operation they describe: full-pivot LU sizes one pivot candidate per
	/// participant, so a query with a team is larger than the sequential one, and a factorization run
	/// on exactly that buffer with that team succeeds. A query that ignored the handle, or one made
	/// at another width, would under-size the buffer the operation then carves its candidate list
	/// out of.
	#[test]
	fn scratch_sizes_follow_the_handle_without_entering_the_team() {
		let _turn = TEAM_TURN.lock().unwrap_or_else(|e| e.into_inner());
		let n = 64usize;
		let params = unsafe { la::libfaer_v0_23_FullPivLuParams_f64() };
		let width = cpus().clamp(2, 4);
		let team = unsafe { libfaer_v0_23_team_create(width) };
		let seq = unsafe {
			la::libfaer_v0_23_full_piv_lu_factor_in_place_scratch_u32_f64(n, n, core::ptr::null(), params)
		};
		let par = unsafe { la::libfaer_v0_23_full_piv_lu_factor_in_place_scratch_u32_f64(n, n, team, params) };
		assert!(
			par.len_bytes > seq.len_bytes,
			"a query with a team must size more than the sequential one: {} vs {}",
			seq.len_bytes,
			par.len_bytes
		);

		// the query leaves no team behind: the calling thread is still on its own CPUs. That it
		// never enters one is a property of the code path (`Team::shape` is the only reader of the
		// handle there), not something a mask that is restored after a drop could show.
		let before = mask();
		unsafe {
			la::libfaer_v0_23_full_piv_lu_reconstruct_scratch_u32_f64(n, n, team);
		};
		assert_eq!(mask(), before, "a scratch query must not enter a team");

		// a factorization on exactly the queried buffer, with the same team
		let mut a = faer::Mat::<f64>::from_fn(n, n, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
		let mut row_fwd = vec![0u32; n];
		let mut row_bwd = vec![0u32; n];
		let (mut col_fwd, mut col_bwd) = (vec![0u32; n], vec![0u32; n]);
		let mem = unsafe {
			MemAlloc {
				ptr: std::alloc::alloc(core::alloc::Layout::from_size_align(par.len_bytes, par.align_bytes).unwrap()) as *mut c_void,
				len_bytes: par.len_bytes,
			}
		};
		assert!(!mem.ptr.is_null());
		let status = unsafe {
			la::libfaer_v0_23_full_piv_lu_factor_in_place_u32_f64(
				MatMut { ptr: a.as_ptr_mut() as *mut c_void, nrows: n, ncols: n, row_stride: a.row_stride(), col_stride: a.col_stride() },
				SliceMut { ptr: row_fwd.as_mut_ptr() as *mut c_void, len: n },
				SliceMut { ptr: row_bwd.as_mut_ptr() as *mut c_void, len: n },
				SliceMut { ptr: col_fwd.as_mut_ptr() as *mut c_void, len: n },
				SliceMut { ptr: col_bwd.as_mut_ptr() as *mut c_void, len: n },
				team,
				mem,
				params,
			)
		};
		assert!(matches!(status, la::FullPivLuStatus::Ok { .. }), "the factorization on the queried buffer failed");
		for perm in [&row_fwd, &row_bwd, &col_fwd, &col_bwd] {
			let mut sorted = perm.clone();
				sorted.sort_unstable();
			assert_eq!(sorted, perm_of(n), "not a permutation");
		}
		unsafe {
			std::alloc::dealloc(mem.ptr as *mut u8, core::alloc::Layout::from_size_align(par.len_bytes, par.align_bytes).unwrap());
			libfaer_v0_23_team_destroy(team);
		}
	}
	/// The handle is shared by the two threads the C contract allows to call it at once, so it must be
	/// `Send` and `Sync`: the mutex is what makes a shared call take turns (tensor4all/faer#333).
	#[test]
	fn the_handle_is_shareable() {
		fn assert<T: Send + Sync>() {}
		assert::<Team>();
	}

	/// Two threads may call operations on one handle at once: they take turns on the pool the handle
	/// owns, so no two calls are ever inside the team together. The counter is the claim the docs make,
	/// and it is checked where the team is entered rather than inferred from the products (which a
	/// per-call pool would also produce). A handle per thread is what a caller wants for its own threads
	/// to be parallel; sharing one is the safe shape, not the fast one.
	#[test]
	fn two_threads_share_one_handle_and_take_turns() {
		let _turn = TEAM_TURN.lock().unwrap_or_else(|e| e.into_inner());
		let width = cpus().clamp(2, 4);
		let team = unsafe { libfaer_v0_23_team_create(width) };
		let addr = team as usize;
		let inside = std::sync::atomic::AtomicUsize::new(0);
		let most = std::sync::atomic::AtomicUsize::new(0);
		std::thread::scope(|s| {
			for _ in 0..3 {
				let (inside, most) = (&inside, &most);
				s.spawn(move || {
					let team = addr as *const Team;
					for _ in 0..4 {
						with_team_par(team, |par| {
							use std::sync::atomic::Ordering::AcqRel;
							most.fetch_max(inside.fetch_add(1, AcqRel) + 1, AcqRel);
							assert_eq!(par.degree(), width, "a call got another width");
							std::thread::sleep(core::time::Duration::from_millis(2));
							inside.fetch_sub(1, AcqRel);
						});
					}
				});
			}
		});
		unsafe { libfaer_v0_23_team_destroy(team) };
		assert_eq!(
			most.load(std::sync::atomic::Ordering::Acquire),
			1,
			"two calls on one handle were inside the team at once"
		);
	}

	/// An operation called from inside another operation on the same thread gets no team: faer calls
	/// back into C (`SchurParams::recommended_shift_count` and its siblings), and such a callback waits
	/// for the pool its caller holds if it is handed one, which never ends. It runs sequentially
	/// instead, so the result is the same and the call returns. A regression here hangs rather than
	/// fails, which is what a deadlock looks like from a test runner.
	#[test]
	fn a_nested_call_runs_sequentially() {
		let _turn = TEAM_TURN.lock().unwrap_or_else(|e| e.into_inner());
		let width = cpus().clamp(2, 4);
		let n = 96;
		let a = faer::Mat::<f64>::from_fn(n, n, |i, j| ((i * 31 + j * 7) % 13) as f64 - 6.0);
		let b = faer::Mat::<f64>::from_fn(n, n, |i, j| ((i * 5 + j * 11) % 17) as f64 - 8.0);
		let expected = want(n, &a, &b);
		let team = unsafe { libfaer_v0_23_team_create(width) };
		let mut inner_c = faer::Mat::<f64>::zeros(n, n);
		let mut outer_c = faer::Mat::<f64>::zeros(n, n);
		with_team_par(team, |_| {
			// the callback: the same handle, from the thread its caller is already using
			matmul(&mut inner_c, &a, &b, team);
			assert_eq!(
				with_team_par(team, |par| par.degree()),
				1,
				"a nested call was handed a team its caller holds"
			);
		});
		matmul(&mut outer_c, &a, &b, team);
		unsafe { libfaer_v0_23_team_destroy(team) };
		assert!(close(&expected, &inner_c), "the nested call gave another product");
		assert!(close(&expected, &outer_c), "the handle was unusable after a nested call");
	}

	/// faer#333: a handle reused for N calls pays for its pool once, so the calls after the first cost
	/// what one reservation costs; a handle built per call pays that pool again on every call. Both
	/// shapes are measured in one run, so they see the same threadteam thread-cache state: the first
	/// call of a pool is threadteam's first acquisition of it (cold on the members' own state, thousands
	/// of allocations here) and the pool-building shape pays that on every call, so the reused handle's
	/// first call is excluded and its tail is compared with the whole of the pool-per-call shape. The
	/// comparison is a sum over the same number of calls, so no single call being warm or cold decides
	/// it. Before the pool moved into the handle both shapes cost the same, per call.
	#[test]
	fn a_reused_handle_builds_its_pool_once() {
		let _turn = TEAM_TURN.lock().unwrap_or_else(|e| e.into_inner());
		let width = cpus().clamp(2, 4);
		let n = 64;
		const CALLS: usize = 8;
		let a = faer::Mat::<f64>::from_fn(n, n, |i, j| ((i * 31 + j * 7) % 13) as f64 - 6.0);
		let b = faer::Mat::<f64>::from_fn(n, n, |i, j| ((i * 5 + j * 11) % 17) as f64 - 8.0);
		let mut c = faer::Mat::<f64>::zeros(n, n);

		// one handle, CALLS calls: the first is the pool's first acquisition, the rest are its tail
		let team = unsafe { libfaer_v0_23_team_create(width) };
		let mut per_call = [0usize; CALLS];
		for slot in &mut per_call {
			alloc_probe::reset();
			matmul(&mut c, &a, &b, team);
			*slot = alloc_probe::take();
		}

		// CALLS handles of one call each (the same number of calls, one pool built for each)
		alloc_probe::reset();
		for _ in 0..CALLS {
			let team = unsafe { libfaer_v0_23_team_create(width) };
			matmul(&mut c, &a, &b, team);
			unsafe { libfaer_v0_23_team_destroy(team) };
		}
		let rebuilt = alloc_probe::take();

		unsafe { libfaer_v0_23_team_destroy(team) };
		let reused: usize = per_call[1..].iter().sum();
		println!(
			"width {width}: per call on one handle {per_call:?} (tail {reused}), {CALLS} handles of one call {rebuilt}"
		);
		assert!(
			2 * reused < rebuilt,
			"the tail of one handle cost {reused} allocations against {rebuilt} for the same calls on a pool per call: the handle builds its pool per call"
		);
	}

	/// The token an operation runs under carries the handle width, in both directions: a
	/// `with_team_par` that asked for a fixed width still passes the result checks above, because
	/// those observe that *some* parallel region ran, not how many participants it had.
	#[test]
	fn the_token_carries_the_handle_width() {
		let _turn = TEAM_TURN.lock().unwrap_or_else(|e| e.into_inner());
		let width = cpus().clamp(3, 4);
		let team = unsafe { libfaer_v0_23_team_create(width) };
		assert_eq!(
			with_team_par(team, |par| par.degree()),
			width,
			"a {width}-participant handle ran on another width"
		);
		unsafe { libfaer_v0_23_team_set_threads(team, 3) };
		assert_eq!(
			with_team_par(team, |par| par.degree()),
			3,
			"set_threads(3) did not reach the token"
		);
		unsafe { libfaer_v0_23_team_set_threads(team, 1) };
		assert_eq!(
			with_team_par(team, |par| par.degree()),
			1,
			"one participant must be sequential"
		);
		// widening again rebuilds the pool the sequential handle released
		unsafe { libfaer_v0_23_team_set_threads(team, width) };
		assert_eq!(
			with_team_par(team, |par| par.degree()),
			width,
			"a handle widened back to {width} ran on another width"
		);
		assert_eq!(
			with_team_par(core::ptr::null(), |par| par.degree()),
			1,
			"a null handle must be sequential"
		);
		unsafe { libfaer_v0_23_team_destroy(team) };
	}
}
