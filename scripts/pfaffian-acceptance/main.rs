use faer::{Mat, Par, c64};
use faer::traits::{ComplexField, ext::*};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::pfaffian::{pfaffian, pfaffian_scratch};
use std::{hint::black_box, time::{Duration, Instant}};
struct Lane;
impl pulp::WithSimd for Lane {
    type Output = (&'static str, usize, usize);
    fn with_simd<S: pulp::Simd>(self, _: S) -> Self::Output {
        let real = std::mem::size_of::<S::f64s>() / 8;
        (std::any::type_name::<S>(), real, if S::IS_SCALAR {1} else {real/2})
    }
}
fn run<T: ComplexField<Real=f64>>(a: Mat<T>, dtype: &str, width: usize, smoke: bool) {
    let n = a.nrows();
    let mut mem = MemBuffer::new(pfaffian_scratch::<T>(n));
    let mut times = [0.0; 10];
    let mut iterations = [0u64; 10];
    let mut result = None;
    let mut execute = |mut par: Par<'_>| {
        if smoke {
            result = Some(pfaffian(a.as_ref(), par.reborrow(), MemStack::new(&mut mem)).unwrap());
        } else {
            let priming = Instant::now();
            let mut warmups = 0;
            while warmups < 3 || priming.elapsed() < Duration::from_millis(500) {
                black_box(pfaffian(black_box(a.as_ref()), par.reborrow(), MemStack::new(&mut mem))).unwrap();
                warmups += 1;
            }
            for i in 0..10 {
                let start = Instant::now();
                loop {
                    let value = black_box(pfaffian(black_box(a.as_ref()), par.reborrow(), MemStack::new(&mut mem)));
                    result = Some(value.unwrap());
                    iterations[i] += 1;
                    if start.elapsed() >= Duration::from_millis(500) { break; }
                }
                times[i] = start.elapsed().as_secs_f64() / iterations[i] as f64;
            }
        }
    };
    if width == 0 { execute(Par::Seq); } else { faer::team::with_team(width, execute); }
    let value = result.unwrap();
    assert!(value.real().is_finite() && value.imag().is_finite());
    let mut sorted = times;
    sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let median = (sorted[4]+sorted[5])/2.0;
    let (lane, real, complex) = pulp::Arch::new().dispatch(Lane);
    println!("{{\"implementation\":\"faer\",\"n\":{n},\"dtype\":\"{dtype}\",\"threads\":{width},\"median_seconds\":{median:.17e},\"samples_seconds\":{times:?},\"batch_iterations\":{iterations:?},\"pf_re\":{:.17e},\"pf_im\":{:.17e},\"allocation_note\":\"matrix and scratch allocated before timing; all internal public-call work timed; team held for team modes\",\"runtime_lane\":\"{lane}\",\"f64_lanes\":{real},\"c64_lanes\":{complex}}}", value.real(), value.imag());
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args[2].parse().unwrap();
    let dtype = &args[3];
    let width: usize = args[4].parse().unwrap();
    assert!([0,1,4,8].contains(&width));
    assert!(args.len() == 5 || (args.len() == 6 && args[5] == "--smoke"));
    let smoke = args.iter().any(|a| a == "--smoke");
    let bytes = std::fs::read(&args[1]).unwrap();
    assert_eq!(bytes.len() % 8, 0);
    let words: Vec<_> = bytes.chunks_exact(8).map(|a| f64::from_le_bytes(a.try_into().unwrap())).collect();
    if dtype == "f64" {
        assert_eq!(words.len(), n*n);
        run(Mat::<f64>::from_fn(n,n,|i,j| words[j*n+i]), dtype, width, smoke);
    } else {
        assert_eq!(dtype, "c64"); assert_eq!(words.len(), n*n*2);
        run(Mat::<c64>::from_fn(n,n,|i,j| c64::new(words[(j*n+i)*2],words[(j*n+i)*2+1])), dtype, width, smoke);
    }
}
