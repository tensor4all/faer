// Single-threaded square products C = A B, column major, with the published `gemm` 0.19.0
// and with `t4a-gemm` 0.20.1 (the same crate plus Apple SME kernels), timed in one process.
// The two implementations alternate sample by sample, so drift on the host hits both alike.
use std::hint::black_box;
use std::time::Instant;

const SAMPLES: usize = 11;
const SAMPLE_SECONDS: f64 = 0.05;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

trait Scalar: Copy + Default + 'static {
    const NAME: &'static str;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn zero() -> Self;
    fn one() -> Self;
    fn eps() -> f64;
}
impl Scalar for f64 {
    const NAME: &'static str = "f64";
    fn from_f64(x: f64) -> Self { x }
    fn to_f64(self) -> f64 { self }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn eps() -> f64 { f64::EPSILON }
}
impl Scalar for f32 {
    const NAME: &'static str = "f32";
    fn from_f64(x: f64) -> Self { x as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn eps() -> f64 { f32::EPSILON as f64 }
}

#[derive(Clone, Copy, PartialEq)]
enum Imp {
    Upstream,
    Sme,
}

fn product<T: Scalar>(imp: Imp, n: usize, a: &[T], b: &[T], c: &mut [T]) {
    let ld = n as isize;
    // the whole call is timed, including the crate's own dispatch and packing
    unsafe {
        match imp {
            Imp::Upstream => gemm::gemm(
                n, n, n, c.as_mut_ptr(), ld, 1, false, a.as_ptr(), ld, 1, b.as_ptr(), ld, 1,
                T::zero(), T::one(), false, false, false, gemm::Parallelism::None,
            ),
            Imp::Sme => gemm_sme::gemm(
                n, n, n, c.as_mut_ptr(), ld, 1, false, a.as_ptr(), ld, 1, b.as_ptr(), ld, 1,
                T::zero(), T::one(), false, false, false, gemm_sme::Parallelism::None,
            ),
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// median seconds per product for each implementation, their max/min spreads, and whether the
/// two results are bit-identical
fn measure<T: Scalar>(n: usize) -> (f64, f64, f64, f64, bool) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ n as u64);
    let a: Vec<T> = (0..n * n).map(|_| T::from_f64(rng.next())).collect();
    let b: Vec<T> = (0..n * n).map(|_| T::from_f64(rng.next())).collect();
    let mut cu = vec![T::default(); n * n];
    let mut cs = vec![T::default(); n * n];
    // prime: page in the buffers, run each kernel's dispatch and probes once, and size a sample
    for _ in 0..2 {
        product(Imp::Upstream, n, &a, &b, &mut cu);
        product(Imp::Sme, n, &a, &b, &mut cs);
    }
    let t = Instant::now();
    product(Imp::Upstream, n, &a, &b, &mut cu);
    let calls = ((SAMPLE_SECONDS / t.elapsed().as_secs_f64()).ceil() as usize).max(1);
    let (mut tu, mut ts) = (vec![], vec![]);
    for _ in 0..SAMPLES {
        for imp in [Imp::Upstream, Imp::Sme] {
            let c = if imp == Imp::Upstream { &mut cu } else { &mut cs };
            let t = Instant::now();
            for _ in 0..calls {
                product(imp, n, black_box(&a), black_box(&b), c);
                black_box(&mut *c);
            }
            let s = t.elapsed().as_secs_f64() / calls as f64;
            if imp == Imp::Upstream { tu.push(s) } else { ts.push(s) }
        }
    }
    let spread = |v: &Vec<f64>| {
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(0.0, f64::max);
        hi / lo
    };
    // a timing means nothing for a wrong product: four entries against a sum formed here,
    // and the two results compared bit for bit
    for (i, j) in [(0, 0), (n - 1, n - 1), (n / 3, n / 2), (7, n - 2)] {
        let exact: f64 = (0..n).map(|k| a[i + k * n].to_f64() * b[k + j * n].to_f64()).sum();
        for c in [&cu, &cs] {
            let err = (c[i + j * n].to_f64() - exact).abs();
            assert!(err <= n as f64 * T::eps(), "{} n={n}: entry ({i}, {j}) is off by {err:e}", T::NAME);
        }
    }
    let identical = cu.iter().zip(&cs).all(|(x, y)| x.to_f64().to_bits() == y.to_f64().to_bits());
    (median(tu.clone()), median(ts.clone()), spread(&tu), spread(&ts), identical)
}

fn main() {
    // macOS has no thread affinity; ask for the performance cores
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
    let amx = cfg!(feature = "upstream-amx");
    println!(
        "upstream gemm 0.19.0{} vs t4a-gemm 0.20.1 (apple-sme), 1 thread, median of {SAMPLES} samples",
        if amx { " (experimental-apple-amx)" } else { "" }
    );
    println!("| type | n | upstream ms | upstream GFLOP/s | SME ms | SME GFLOP/s | upstream / SME | max/min of the samples (upstream, SME) | results bit-identical |");
    println!("|---|---:|---:|---:|---:|---:|---:|---|---|");
    let mut worst = f64::INFINITY;
    fn row<T: Scalar>(n: usize, worst: &mut f64) {
        let (tu, ts, su, ss, identical) = measure::<T>(n);
        let gf = |t: f64| 2.0 * (n as f64).powi(3) / t * 1e-9;
        let ratio = tu / ts;
        if n >= 512 {
            *worst = worst.min(ratio);
        }
        println!(
            "| {} | {n} | {:.3} | {:.1} | {:.3} | {:.1} | **{ratio:.2}** | {su:.2}, {ss:.2} | {} |",
            T::NAME, tu * 1e3, gf(tu), ts * 1e3, gf(ts), if identical { "yes" } else { "no" }
        );
    }
    for n in [256, 512, 1024, 2048] {
        row::<f64>(n, &mut worst);
    }
    for n in [256, 512, 1024, 2048] {
        row::<f32>(n, &mut worst);
    }
    // the defect: on a host with SME, upstream runs at least two times slower than the same
    // product on SME from n = 512 up
    if worst >= 2.0 {
        println!("SLOW: upstream gemm 0.19.0 is {worst:.2}x or more behind the SME kernels on this host");
        std::process::exit(1);
    }
    println!("not shown on this host: smallest upstream / SME ratio from n = 512 is {worst:.2}");
}
