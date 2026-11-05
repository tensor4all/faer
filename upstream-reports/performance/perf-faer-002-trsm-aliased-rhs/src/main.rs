// A unit lower triangular solve L X = B, in place, one thread, f64. The same B is stored
// with a column stride that is a power of two (so its columns fall into few L1 sets) and with
// that stride plus 8 elements. A third arm keeps the power-of-two storage but copies B into a
// padded buffer, solves there and copies back, which is how the fork fixes it.
// All three must give the same bits; only the time may differ.
use faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place;
use faer::{Mat, MatMut, MatRef, Par};
use std::hint::black_box;
use std::time::Instant;

const SAMPLES: usize = 11;
const SAMPLE_SECONDS: f64 = 0.02;
const PAD: usize = 8;

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    InPlace,
    Padded,
    Staged,
}

struct Case {
    b: usize,
    stride: usize,
    l: Mat<f64>,
    rhs: Vec<f64>,
    aliased: Vec<f64>,
    padded: Vec<f64>,
    stage: Vec<f64>,
}

impl Case {
    fn new(b: usize, stride: usize) -> Self {
        let mut s = 0x2545_f491_4f6c_dd1d ^ (b * stride) as u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        // small off-diagonal entries keep the solution of order one
        let l = Mat::from_fn(b, b, |i, j| match i.cmp(&j) {
            core::cmp::Ordering::Equal => 1.0,
            core::cmp::Ordering::Greater => r() / b as f64,
            core::cmp::Ordering::Less => 0.0,
        });
        let rhs = (0..b * b).map(|_| r()).collect();
        Case {
            b,
            stride,
            l,
            rhs,
            aliased: vec![0.0; stride * b],
            padded: vec![0.0; (stride + PAD) * b],
            stage: vec![0.0; (stride + PAD) * b],
        }
    }

    fn load(&mut self, arm: Arm) {
        let (b, rhs) = (self.b, &self.rhs);
        let (buf, ld) = match arm {
            Arm::Padded => (&mut self.padded, self.stride + PAD),
            _ => (&mut self.aliased, self.stride),
        };
        for j in 0..b {
            buf[j * ld..j * ld + b].copy_from_slice(&rhs[j * b..j * b + b]);
        }
    }

    /// one solve, timed from the call to its return (the staged arm includes both copies)
    fn solve(&mut self, arm: Arm) -> f64 {
        let (b, s) = (self.b, self.stride);
        let l = self.l.as_ref();
        let t = Instant::now();
        match arm {
            Arm::InPlace => solve(l, view(&mut self.aliased, b, s)),
            Arm::Padded => solve(l, view(&mut self.padded, b, s + PAD)),
            Arm::Staged => {
                let mut stage = view(&mut self.stage, b, s + PAD);
                let mut x = view(&mut self.aliased, b, s);
                stage.copy_from(&x);
                solve(l, stage.as_mut());
                x.copy_from(&stage);
            }
        }
        black_box(&mut self.aliased);
        t.elapsed().as_secs_f64()
    }

    fn result(&self, arm: Arm) -> Vec<u64> {
        let (b, ld, buf) = match arm {
            Arm::Padded => (self.b, self.stride + PAD, &self.padded),
            _ => (self.b, self.stride, &self.aliased),
        };
        (0..b).flat_map(|j| buf[j * ld..j * ld + b].iter().map(|x| x.to_bits())).collect()
    }
}

fn view(buf: &mut [f64], b: usize, ld: usize) -> MatMut<'_, f64> {
    MatMut::from_column_major_slice_with_stride_mut(&mut buf[..ld * (b - 1) + b], b, b, ld)
}

fn solve(l: MatRef<'_, f64>, x: MatMut<'_, f64>) {
    solve_unit_lower_triangular_in_place(l, x, Par::Seq);
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    // macOS has no thread affinity; ask for the performance cores
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
    println!("faer 0.24.4 solve_unit_lower_triangular_in_place, f64, b x b triangle and b x b right hand side, 1 thread, median of {SAMPLES} samples");
    println!("| b | stride (bytes) | in place, us | stride + {PAD}, us | staged, us | in place / staged | in place / stride + {PAD} | three results bit-identical |");
    println!("|---:|---|---:|---:|---:|---:|---:|---|");
    let mut worst: f64 = 0.0;
    for b in [64, 128, 256] {
        for stride in [512, 1024, 2048] {
            let mut c = Case::new(b, stride);
            let arms = [Arm::InPlace, Arm::Padded, Arm::Staged];
            // prime, and check the bits before timing anything
            let mut bits = vec![];
            for arm in arms {
                c.load(arm);
                c.solve(arm);
                bits.push(c.result(arm));
            }
            let identical = bits[0] == bits[1] && bits[1] == bits[2];
            assert!(identical, "b={b} stride={stride}: the three arms disagree");
            c.load(Arm::InPlace);
            let calls = ((SAMPLE_SECONDS / c.solve(Arm::InPlace)).ceil() as usize).max(1);
            let mut times = [vec![], vec![], vec![]];
            for _ in 0..SAMPLES {
                for (k, arm) in arms.into_iter().enumerate() {
                    // the right hand side is reloaded before every call, outside the timed region
                    let mut total = 0.0;
                    for _ in 0..calls {
                        c.load(arm);
                        total += c.solve(arm);
                    }
                    times[k].push(total / calls as f64);
                }
            }
            let [t0, t1, t2] = times.map(median);
            worst = worst.max(t0 / t2);
            println!(
                "| {b} | {stride} ({} KiB) | {:.1} | {:.1} | {:.1} | **{:.2}** | {:.2} | {} |",
                stride * 8 / 1024,
                t0 * 1e6,
                t1 * 1e6,
                t2 * 1e6,
                t0 / t2,
                t0 / t1,
                if identical { "yes" } else { "no" }
            );
        }
    }
    if worst >= 1.5 {
        println!("SLOW: the in-place solve at a power-of-two stride is up to {worst:.2}x the staged one");
        std::process::exit(1);
    }
    println!("not shown on this host: the largest in place / staged ratio is {worst:.2}");
}
