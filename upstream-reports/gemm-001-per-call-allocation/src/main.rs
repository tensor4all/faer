use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
struct Count;
static N: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { N.fetch_add(1, Relaxed); System.alloc(l) }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { System.dealloc(p, l) }
}
#[global_allocator]
static A: Count = Count;
fn main() {
    for n in [8usize, 16, 32, 64, 128, 256] {
        let a = vec![1.0f64; n * n];
        let b = vec![1.0f64; n * n];
        let mut c = vec![0.0f64; n * n];
        let mut run = || unsafe {
            gemm::gemm(n, n, n, c.as_mut_ptr(), n as isize, 1, false, a.as_ptr(), n as isize, 1,
                b.as_ptr(), n as isize, 1, 0.0, 1.0, false, false, false, gemm::Parallelism::None)
        };
        run();
        let before = N.load(Relaxed);
        run();
        println!("gemm 0.19.0 f64 n={n}: warm-call allocations = {}", N.load(Relaxed) - before);
    }
}
