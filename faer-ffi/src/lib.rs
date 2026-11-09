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
	/// out, and `None` for a sequential handle, which reserves nothing.
	///
	/// The workers are **dedicated** (a `thread_name` makes them so, threadteam 0.5.1
	/// `Pool` docs): dropping the pool joins them, and neither `build` nor the drop touches
	/// threadteam's process-wide cache of shared workers. That is what makes `destroy` stop the
	/// threads, and what keeps a child forked after a `destroy` from inheriting cached Rayon workers
	/// whose threads it does not have (its `create` would wait for them forever).
	fn pool_for(threads: usize) -> Option<threadteam::Pool> {
		(threads > 1).then(|| {
			let builder = threadteam::Pool::builder()
				.threads(threads)
				.thread_name(|i| {
					#[cfg(test)]
					tests::live_workers::spawned();
					format!("faer-ffi-{i}")
				})
				.acquire_timeout(core::time::Duration::from_secs(30));
			#[cfg(test)]
			let builder = builder.on_start(|_| tests::live_workers::started());
			builder.build().expect("a thread pool for the team")
		})
	}

	/// Resets the width, rebuilding the owned pool when it changes: the old pool's workers are joined
	/// and the new width starts its own. This is a reconfiguration, and it pays for its threads.
	///
	/// The old pool is dropped (its workers joined) *before* the new one is built, so the handle never
	/// asks for the old and the new workers at once: assigning `pool_for(threads)` in one statement
	/// would evaluate the new pool first and hold both, which on a host at its thread limit turns a
	/// shrink into a failed build.
	fn set_threads(&mut self, threads: usize) {
		if threads == self.threads {
			return;
		}
		self.threads = threads;
		let pool = self.pool.get_mut().unwrap_or_else(|e| e.into_inner());
		drop(pool.take());
		*pool = Team::pool_for(threads);
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
	/// For a test that hands EVD a C callback: the fields are C's to write, but private in Rust.
	#[cfg(test)]
	impl EvdParams {
		pub(crate) fn with_shift_callback(
			mut self,
			shifts: extern "C" fn(usize, usize) -> usize,
			blocking_threshold: usize,
		) -> Self {
			self.schur.recommended_shift_count = shifts;
			self.schur.blocking_threshold = blocking_threshold;
			self
		}
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
/// * **After `fork`, a child uses only handles it creates itself.** A handle created before the
///   fork names worker threads that do not exist in the child: an operation on it there can wait for
///   them forever, and `destroy` there would join threads of another process, so the child must
///   neither use nor destroy it. A handle the child creates starts its own workers, which are
///   dedicated to it and never pass through threadteam's process-wide cache, so it works whether the
///   parent's handles were destroyed before the fork or are still alive at it. The usual limits of
///   forking a multithreaded process still apply to everything else (a lock another parent thread
///   held at the fork stays held in the child), so `exec` remains the safe shape for a child that
///   does more than this. Whether `create` should record the creator's pid and refuse an inherited
///   handle is **undecided**, and nothing checks it today.
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
/// running. Dropping the pool ends its workers: they are dedicated to this handle (not threadteam's
/// process-wide cache of shared workers), so when `destroy` returns every worker thread of the
/// handle has been joined.
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
		unsafe fn realloc(
			&self,
			ptr: *mut u8,
			layout: Layout,
			new_size: usize,
		) -> *mut u8 {
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

	/// The handle's own workers alive at once, and the most there were. A worker is counted when
	/// `build` spawns it (the `thread_name` hook runs on the building thread, before `build` returns)
	/// and uncounted when its thread exits (a thread-local set by the `on_start` hook is dropped there,
	/// before the pool's drop has joined it). Only the handle's dedicated workers are counted, not
	/// threadteam's shared ones, and every test that builds a handle holds [`TEAM_TURN`].
	pub(super) mod live_workers {
		use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
		static LIVE: AtomicUsize = AtomicUsize::new(0);
		static MOST: AtomicUsize = AtomicUsize::new(0);
		struct Exit;
		impl Drop for Exit {
			fn drop(&mut self) {
				LIVE.fetch_sub(1, SeqCst);
			}
		}
		thread_local!(static EXIT: core::cell::RefCell<Option<Exit>> = const { core::cell::RefCell::new(None) });
		pub fn spawned() {
			MOST.fetch_max(LIVE.fetch_add(1, SeqCst) + 1, SeqCst);
		}
		pub fn started() {
			EXIT.with(|exit| *exit.borrow_mut() = Some(Exit));
		}
		/// starts a new high-water mark at the current count, and returns that count
		pub fn reset() -> usize {
			let live = LIVE.load(SeqCst);
			MOST.store(live, SeqCst);
			live
		}
		pub fn most() -> usize {
			MOST.load(SeqCst)
		}
		pub fn live() -> usize {
			LIVE.load(SeqCst)
		}
	}

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
							most.fetch_max(
								inside.fetch_add(1, AcqRel) + 1,
								AcqRel,
							);
							assert_eq!(
								par.degree(),
								width,
								"a call got another width"
							);
							std::thread::sleep(
								core::time::Duration::from_millis(2),
							);
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

	/// `set_threads` lets the old pool's workers go before the new width starts its own, so a
	/// reconfiguration never asks the host for both at once: shrinking 4 to 2 needs no more threads
	/// than the handle already has, which is what a host at its thread limit (cgroup `pids.max`) can
	/// still grant. Built the other way round, the new pool exists before the old one is dropped, and
	/// a failed build is the `expect` in [`Team::pool_for`], which aborts the caller.
	#[test]
	fn set_threads_drops_the_old_pool_before_it_builds_the_new_one() {
		let _turn = TEAM_TURN.lock().unwrap_or_else(|e| e.into_inner());
		let before = live_workers::live();
		let team = unsafe { libfaer_v0_23_team_create(4) };
		assert_eq!(
			live_workers::reset(),
			before + 4,
			"a 4-wide handle did not start 4 workers of its own"
		);
		unsafe { libfaer_v0_23_team_set_threads(team, 2) };
		assert_eq!(
			live_workers::most(),
			before + 4,
			"shrinking 4 to 2 had the old and the new pool alive at once"
		);
		assert_eq!(
			live_workers::live(),
			before + 2,
			"the old pool's workers were not joined"
		);
		live_workers::reset();
		unsafe { libfaer_v0_23_team_set_threads(team, 3) };
		assert_eq!(
			live_workers::most(),
			before + 3,
			"growing 2 to 3 had the old and the new pool alive at once"
		);
		unsafe { libfaer_v0_23_team_destroy(team) };
		assert_eq!(live_workers::live(), before, "destroy left workers alive");
	}

	/// An operation called from inside another operation on the same thread gets no team: faer calls
	/// back into C (`SchurParams::recommended_shift_count` and its siblings), and such a callback waits
	/// for the pool its caller holds if it is handed one, which never ends. It runs sequentially
	/// instead, so the result is the same and the call returns. A regression here is a deadlock (or
	/// a panic on relocking the handle's mutex), so the outer call runs on a helper thread and is
	/// given a deadline: the test fails instead of hanging the runner and the tests queued behind
	/// [`TEAM_TURN`].
	#[test]
	fn a_nested_call_runs_sequentially() {
		let _turn = TEAM_TURN.lock().unwrap_or_else(|e| e.into_inner());
		let width = cpus().clamp(2, 4);
		let n = 96;
		let a = faer::Mat::<f64>::from_fn(n, n, |i, j| {
			((i * 31 + j * 7) % 13) as f64 - 6.0
		});
		let b = faer::Mat::<f64>::from_fn(n, n, |i, j| {
			((i * 5 + j * 11) % 17) as f64 - 8.0
		});
		let expected = want(n, &a, &b);
		let team = unsafe { libfaer_v0_23_team_create(width) };
		let addr = team as usize;
		let (done, wait) = std::sync::mpsc::channel();
		let (a2, b2) = (a.clone(), b.clone());
		std::thread::spawn(move || {
			let team = addr as *const Team;
			let mut inner_c = faer::Mat::<f64>::zeros(n, n);
			let inner_degree = with_team_par(team, |_| {
				// the callback: the same handle, from the thread its caller is already using
				matmul(&mut inner_c, &a2, &b2, team);
				with_team_par(team, |par| par.degree())
			});
			let _ = done.send((inner_c, inner_degree));
		});
		let (inner_c, inner_degree) = wait
			.recv_timeout(core::time::Duration::from_secs(30))
			.unwrap_or_else(|e| {
				panic!("a call nested in another call on the same handle never returned ({e:?}): it waited for the pool its caller holds")
			});
		let mut outer_c = faer::Mat::<f64>::zeros(n, n);
		matmul(&mut outer_c, &a, &b, team);
		unsafe { libfaer_v0_23_team_destroy(team) };
		assert_eq!(
			inner_degree, 1,
			"a nested call was handed a team its caller holds"
		);
		assert!(
			close(&expected, &inner_c),
			"the nested call gave another product"
		);
		assert!(
			close(&expected, &outer_c),
			"the handle was unusable after a nested call"
		);
	}

	/// The review's reproducer through the real callback path rather than a closure: an EVD whose
	/// `SchurParams::recommended_shift_count` callback multiplies two other matrices on the *same*
	/// handle, for a sequential handle and a parallel one. Before the reentrancy guard this hung: EVD
	/// holds the handle's lock while faer calls the callback, and the nested multiplication waited for
	/// it. The EVD runs on a helper thread and is given a deadline, so a regression fails instead of
	/// hanging the runner. The callback also records whether it ran on the thread that made the outer
	/// call, which is the thread the guard marks.
	#[test]
	fn an_evd_callback_may_call_back_on_the_same_handle() {
		use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
		static HANDLE: AtomicUsize = AtomicUsize::new(0);
		static CALLER: AtomicUsize = AtomicUsize::new(0);
		static CALLED: AtomicUsize = AtomicUsize::new(0);
		static OFF_THREAD: AtomicBool = AtomicBool::new(false);
		static WRONG: AtomicBool = AtomicBool::new(false);
		fn tid() -> usize {
			thread_local!(static ID: u8 = const { 0 });
			ID.with(|id| id as *const u8 as usize)
		}
		extern "C" fn shifts(dim: usize, active: usize) -> usize {
			CALLED.fetch_add(1, SeqCst);
			if tid() != CALLER.load(SeqCst) {
				OFF_THREAD.store(true, SeqCst);
			}
			let n = 24;
			let a = faer::Mat::<f64>::from_fn(n, n, |i, j| {
				((i * 3 + j) % 7) as f64 - 3.0
			});
			let b = faer::Mat::<f64>::from_fn(n, n, |i, j| {
				((i + 2 * j) % 5) as f64 - 2.0
			});
			let mut c = faer::Mat::<f64>::zeros(n, n);
			matmul(&mut c, &a, &b, HANDLE.load(SeqCst) as *const Team);
			if !close(&want(n, &a, &b), &c) {
				WRONG.store(true, SeqCst);
			}
			let _ = (dim, active);
			12
		}

		let _turn = TEAM_TURN.lock().unwrap_or_else(|e| e.into_inner());
		for width in [1, cpus().clamp(2, 4)] {
			CALLED.store(0, SeqCst);
			let team = unsafe { libfaer_v0_23_team_create(width) };
			HANDLE.store(team as usize, SeqCst);
			let (done, wait) = std::sync::mpsc::channel();
			std::thread::spawn(move || {
				CALLER.store(tid(), SeqCst);
				let n = 160usize;
				let a = faer::Mat::<f64>::from_fn(n, n, |i, j| {
					(((i * 37 + j * 11) % 23) as f64 - 11.0) / 7.0
				});
				let params = unsafe { la::libfaer_v0_23_EvdParams_f64() }
					.with_shift_callback(shifts, 32);
				let team = HANDLE.load(SeqCst) as *const Team;
				let no = la::ComputeEigenvectors::No;
				let layout = unsafe {
					la::libfaer_v0_23_evd_scratch_f64(n, no, no, team, params)
				};
				let lay = core::alloc::Layout::from_size_align(
					layout.len_bytes.max(1),
					layout.align_bytes,
				)
				.unwrap();
				let mem = MemAlloc {
					ptr: unsafe { std::alloc::alloc(lay) } as *mut c_void,
					len_bytes: layout.len_bytes,
				};
				let (mut re, mut im) = (vec![0.0f64; n], vec![0.0f64; n]);
				let empty = MatMut {
					ptr: core::ptr::null_mut(),
					nrows: n,
					ncols: 0,
					row_stride: 1,
					col_stride: n as isize,
				};
				let status = unsafe {
					la::libfaer_v0_23_evd_f64(
						MatRef::of(&a),
						empty,
						empty,
						VecMut {
							ptr: re.as_mut_ptr() as *mut c_void,
							len: n,
							stride: 1,
						},
						VecMut {
							ptr: im.as_mut_ptr() as *mut c_void,
							len: n,
							stride: 1,
						},
						team,
						mem,
						params,
					)
				};
				unsafe { std::alloc::dealloc(mem.ptr as *mut u8, lay) };
				let trace: f64 = (0..n).map(|i| a[(i, i)]).sum();
				let sum: f64 = re.iter().sum();
				let _ = done.send((
					matches!(status, la::EvdStatus::Ok { .. }),
					(trace - sum).abs(),
				));
			});
			let (ok, err) = wait
				.recv_timeout(core::time::Duration::from_secs(60))
				.unwrap_or_else(|_| panic!("width {width}: an EVD whose callback calls the same handle never returned (deadlock)"));
			unsafe { libfaer_v0_23_team_destroy(team) };
			assert!(ok, "width {width}: the EVD did not converge");
			assert!(
				err < 1e-8,
				"width {width}: the eigenvalues do not sum to the trace ({err})"
			);
			assert!(
				CALLED.load(SeqCst) > 0,
				"width {width}: the callback was never called, so nothing was tested"
			);
			assert!(
				!WRONG.load(SeqCst),
				"width {width}: the nested product was wrong"
			);
			assert!(
				!OFF_THREAD.load(SeqCst),
				"width {width}: the callback ran off the calling thread, where the guard does not reach"
			);
		}
	}

	/// faer#333: a handle reused for N calls pays for its pool once, so the calls after the first cost
	/// what one reservation costs. The reference is what the handle replaced, measured in the same
	/// run on the same matrices: `faer::team::with_team` per call, which is what every C call ran on
	/// `origin/main` (it builds a pool of shared workers from threadteam's cache, reserves it and
	/// drops it again). The first call of each shape is excluded: for the handle it is threadteam's
	/// first acquisition of the pool (thousands of allocations, the members' own state), for the
	/// reference the first build of its cached workers. The tails are summed over the same number of
	/// calls, so no single call being warm (1 allocation) or cold (6, after the members parked) decides
	/// it, and the bound asks for a third less than the reference: with the implementation reverted to
	/// a pool per call the two tails are the same code and cost the same.
	#[test]
	fn a_reused_handle_builds_its_pool_once() {
		let _turn = TEAM_TURN.lock().unwrap_or_else(|e| e.into_inner());
		let width = cpus().clamp(2, 4);
		let n = 64;
		const CALLS: usize = 8;
		let a = faer::Mat::<f64>::from_fn(n, n, |i, j| {
			((i * 31 + j * 7) % 13) as f64 - 6.0
		});
		let b = faer::Mat::<f64>::from_fn(n, n, |i, j| {
			((i * 5 + j * 11) % 17) as f64 - 8.0
		});
		let mut c = faer::Mat::<f64>::zeros(n, n);

		// one handle, CALLS calls through the C entry point
		let team = unsafe { libfaer_v0_23_team_create(width) };
		let mut reused = [0usize; CALLS];
		for slot in &mut reused {
			alloc_probe::reset();
			matmul(&mut c, &a, &b, team);
			*slot = alloc_probe::take();
		}
		unsafe { libfaer_v0_23_team_destroy(team) };

		// the same CALLS products on a pool per call, the shape the handle replaced
		let mut per_call = [0usize; CALLS];
		for slot in &mut per_call {
			alloc_probe::reset();
			faer::team::with_team(width, |par| {
				faer::linalg::matmul::matmul(
					c.as_mut(),
					faer::Accum::Replace,
					a.as_ref(),
					b.as_ref(),
					1.0,
					par,
				)
			});
			*slot = alloc_probe::take();
		}

		let tail = |calls: &[usize]| -> usize { calls[1..].iter().sum() };
		let (reused_tail, per_call_tail) = (tail(&reused), tail(&per_call));
		println!(
			"width {width}: one handle {reused:?} (tail {reused_tail}), a pool per call \
			 {per_call:?} (tail {per_call_tail})"
		);
		assert!(
			3 * reused_tail < 2 * per_call_tail,
			"the tail of one handle cost {reused_tail} allocations against \
			 {per_call_tail} for the same calls on a pool per call: the handle \
			 builds its pool per call"
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
