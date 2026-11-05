#![no_std]
#![no_main]
#![feature(alloc_error_handler)]
#![feature(lang_items)]

use core::alloc::{GlobalAlloc, Layout};

/// The global allocator type.
#[derive(Default)]
pub struct Allocator;

#[cfg(windows)]
extern "C" {
	fn _aligned_malloc(size: libc::size_t, align: libc::size_t) -> *mut libc::c_void;
	fn _aligned_free(memblock: *mut libc::c_void);
}

unsafe impl GlobalAlloc for Allocator {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		#[cfg(not(windows))]
		{
			let mut mem: *mut libc::c_void = core::ptr::null_mut();
			libc::posix_memalign(&mut mem, layout.align(), layout.size());
			mem as *mut u8
		}

		#[cfg(windows)]
		{
			_aligned_malloc(layout.size(), layout.align()) as *mut u8
		}
	}

	unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
		#[cfg(not(windows))]
		libc::free(ptr as *mut libc::c_void);

		#[cfg(windows)]
		_aligned_free(ptr as *mut libc::c_void)
	}
}

#[panic_handler]
fn panic(panic_info: &core::panic::PanicInfo) -> ! {
	let message = panic_info.message();
	libc_print::libc_eprintln!("{message}");
	unsafe { libc::exit(1) };
}

/// If there is an out of memory error, just panic.
#[alloc_error_handler]
fn my_allocator_error(_layout: Layout) -> ! {
	panic!("out of memory");
}

// The personality routine is a language item: rustc mangles it itself, and from the 1.89
// toolchain on it rejects a `#[no_mangle]` on an internal language item.
#[lang = "eh_personality"]
extern "C" fn rust_eh_personality() {}

/// The static global allocator.
#[global_allocator]
static GLOBAL_ALLOCATOR: Allocator = Allocator;

#[no_mangle]
pub extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
	let _ = (&argc, &argv);

	let m = faer::mat![[2.0, 1.0]];
	libc_print::libc_println!("{m:?}");
	libc_print::libc_println!("{:?}", m.transpose() * &m);
	// Exercise the provided-stack upgrade in a real no_std executable.
	let basis = faer::mat![[1.0_f64, 0.0], [0.2, 1.0], [0.3, 0.4]];
	let mut factor = faer::Mat::<f64>::zeros(2, 2);
	factor[(0, 0)] = 0.565;
	factor[(1, 1)] = 0.58;
	let req = faer::linalg::householder::upgrade_householder_factor_scratch::<
		f64,
	>(3, 2, 2, 1, faer::ParShape::Seq);
	let mut memory = faer::dyn_stack::MemBuffer::new(req);
	faer::linalg::householder::upgrade_householder_factor(
		factor.as_mut(),
		basis.as_ref(),
		2,
		1,
		faer::Par::Seq,
		faer::dyn_stack::MemStack::new(&mut memory),
	);
	assert!((factor[(0, 1)] - 0.32).abs() < 1e-12);
	0
}
