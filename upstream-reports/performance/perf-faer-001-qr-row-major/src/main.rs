// QR (unpivoted and column pivoted) of the same matrix stored row major and column major, one
// thread, f64. Arm "row major": faer 0.24.4's qr_in_place on the row-major storage, as a caller
// would pass it.
// Arm "adapted": copy the row-major storage into a column-major buffer, factor there, copy the
// factor back, which is how the fork handles a row-major input. Arm "column major": the
// factorization of an already column-major copy, with no copies, for scale.
// The adapted result (factor, block coefficients and, when pivoting, the permutation) must equal
// the column-major one bit for bit.
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::qr::col_pivoting::factor as col_piv;
use faer::linalg::qr::no_pivoting::factor::{qr_in_place, qr_in_place_scratch, recommended_block_size};
use faer::{Mat, MatMut, MatRef, Par};
use std::hint::black_box;
use std::time::Instant;

const SAMPLES: usize = 7;
const SAMPLE_SECONDS: f64 = 0.1;

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    RowMajor,
    Adapted,
    ColMajor,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    NoPivoting,
    ColPivoting,
}

struct Case {
    kind: Kind,
    m: usize,
    n: usize,
    a: Mat<f64>,
    // row-major storage: entry (i, j) at i * n + j
    row: Vec<f64>,
    col: Mat<f64>,
    q_coeff: Mat<f64>,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    mem: MemBuffer,
}

fn row_view(buf: &mut [f64], m: usize, n: usize) -> MatMut<'_, f64> {
    MatMut::from_row_major_slice_mut(buf, m, n)
}

impl Case {
    fn new(kind: Kind, m: usize, n: usize) -> Self {
        let mut s = 0x9e37_79b9_7f4a_7c15 ^ (m * 31 + n) as u64;
        let a = Mat::from_fn(m, n, |_, _| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        });
        let (bs, req) = match kind {
            Kind::NoPivoting => {
                let bs = recommended_block_size::<f64>(m, n);
                (bs, qr_in_place_scratch::<f64>(m, n, bs, Par::Seq, Default::default()))
            }
            // the column-pivoted factorization is unblocked: one row of coefficients
            Kind::ColPivoting => (1, col_piv::qr_in_place_scratch::<usize, f64>(m, n, 1, Par::Seq, Default::default())),
        };
        let mem = MemBuffer::new(StackReq::any_of(&[req]));
        Case {
            kind,
            m,
            n,
            a,
            row: vec![0.0; m * n],
            col: Mat::zeros(m, n),
            q_coeff: Mat::zeros(bs, Ord::min(m, n)),
            perm: vec![0; n],
            perm_inv: vec![0; n],
            mem,
        }
    }

    fn load(&mut self, arm: Arm) {
        let (m, n) = (self.m, self.n);
        match arm {
            Arm::ColMajor => self.col.copy_from(&self.a),
            _ => row_view(&mut self.row, m, n).copy_from(&self.a),
        }
    }

    fn factor(&mut self, arm: Arm) -> f64 {
        let (m, n) = (self.m, self.n);
        let stack = MemStack::new(&mut self.mem);
        let (kind, q, perm, perm_inv) = (self.kind, self.q_coeff.as_mut(), &mut self.perm[..], &mut self.perm_inv[..]);
        let qr = move |a: MatMut<'_, f64>| match kind {
            Kind::NoPivoting => {
                qr_in_place(a, q, Par::Seq, stack, Default::default());
            }
            Kind::ColPivoting => {
                col_piv::qr_in_place(a, q, perm, perm_inv, Par::Seq, stack, Default::default());
            }
        };
        let t = Instant::now();
        match arm {
            Arm::RowMajor => qr(row_view(&mut self.row, m, n)),
            Arm::ColMajor => qr(self.col.as_mut()),
            Arm::Adapted => {
                // the copies are inside the timed region: they are the price of the adaptation
                self.col.copy_from(row_view(&mut self.row, m, n).as_ref());
                qr(self.col.as_mut());
                row_view(&mut self.row, m, n).copy_from(&self.col);
            }
        }
        black_box(&mut self.row);
        t.elapsed().as_secs_f64()
    }

    /// the factor (R and the reflectors), the block coefficients and the permutation, as bits,
    /// entry by entry
    fn result(&mut self, arm: Arm) -> Vec<u64> {
        let (m, n) = (self.m, self.n);
        let f: MatRef<'_, f64> = match arm {
            Arm::ColMajor => self.col.as_ref(),
            _ => MatRef::from_row_major_slice(&self.row, m, n),
        };
        let q = self.q_coeff.as_ref();
        let mut bits: Vec<u64> = (0..n).flat_map(|j| (0..m).map(move |i| f[(i, j)].to_bits())).collect();
        bits.extend((0..q.ncols()).flat_map(|j| (0..q.nrows()).map(move |i| q[(i, j)].to_bits())));
        if self.kind == Kind::ColPivoting {
            bits.extend(self.perm.iter().map(|&p| p as u64));
        }
        bits
    }
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
    println!("faer 0.24.4 qr_in_place, f64, 1 thread, median of {SAMPLES} samples");
    println!("| QR | m x n | row major, ms | adapted (copy, factor, copy back), ms | column major, ms | row major / adapted | adapted = column major, bit for bit | max |row major - column major| over R |");
    println!("|---|---|---:|---:|---:|---:|---|---:|");
    let mut worst: f64 = 0.0;
    let cases = [
        (Kind::NoPivoting, 256, 256),
        (Kind::NoPivoting, 512, 512),
        (Kind::NoPivoting, 1024, 1024),
        (Kind::NoPivoting, 1024, 256),
        (Kind::NoPivoting, 4096, 256),
        (Kind::ColPivoting, 256, 256),
        (Kind::ColPivoting, 512, 512),
        (Kind::ColPivoting, 1024, 64),
        (Kind::ColPivoting, 4096, 256),
    ];
    for (kind, m, n) in cases {
        let mut c = Case::new(kind, m, n);
        let name = if kind == Kind::NoPivoting { "no pivoting" } else { "column pivoting" };
        let arms = [Arm::RowMajor, Arm::Adapted, Arm::ColMajor];
        let mut bits = vec![];
        for arm in arms {
            c.load(arm);
            c.factor(arm);
            bits.push(c.result(arm));
        }
        let exact = bits[1] == bits[2];
        assert!(exact, "{name} {m}x{n}: the adapted factorization is not the column-major one");
        // the direct row-major factorization is a different rounding of the same QR: report it
        let mut diff: f64 = 0.0;
        for j in 0..n {
            for i in 0..=Ord::min(j, m - 1) {
                let x = f64::from_bits(bits[0][i + j * m]);
                let y = f64::from_bits(bits[2][i + j * m]);
                diff = diff.max((x - y).abs());
            }
        }
        c.load(Arm::RowMajor);
        let calls = ((SAMPLE_SECONDS / c.factor(Arm::RowMajor)).ceil() as usize).max(1);
        let mut times = [vec![], vec![], vec![]];
        for _ in 0..SAMPLES {
            for (k, arm) in arms.into_iter().enumerate() {
                // the input is reloaded before every call, outside the timed region
                let mut total = 0.0;
                for _ in 0..calls {
                    c.load(arm);
                    total += c.factor(arm);
                }
                times[k].push(total / calls as f64);
            }
        }
        let [t0, t1, t2] = times.map(median);
        worst = worst.max(t0 / t1);
        println!(
            "| {name} | {m} x {n} | {:.2} | {:.2} | {:.2} | **{:.2}** | {} | {diff:.1e} |",
            t0 * 1e3,
            t1 * 1e3,
            t2 * 1e3,
            t0 / t1,
            if exact { "yes" } else { "no" }
        );
    }
    if worst >= 1.25 {
        println!("SLOW: QR of a row-major matrix is up to {worst:.2}x the copy-and-factor route");
        std::process::exit(1);
    }
    println!("not shown on this host: the largest row major / adapted ratio is {worst:.2}");
}
