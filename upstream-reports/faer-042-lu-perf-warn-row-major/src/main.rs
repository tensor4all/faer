// Column-pivoted LU of a contiguous 64 x 64 row-major matrix, with `perf-warn` enabled. The
// predicate fires when `col_stride == 1`, which is what a contiguous row-major matrix has, so the
// call warns "Found matrix with generic strides" although its own message names row-major as one of
// the two layouts it prefers. A contiguous column-major input does not warn (`row_stride == 1`
// leaves both clauses false), which is the asymmetry.
//
// `Mat` is column-major, so its transpose view is the contiguous row-major input:
// `row_stride = n`, `col_stride = 1`. Pass `column-major` as the first argument for the control
// run, which is the same matrix in the other preferred layout and does not warn in either version.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::factor::{lu_in_place, lu_in_place_scratch};
use faer::{Mat, Par};
use std::sync::atomic::{AtomicUsize, Ordering};

static WARNINGS: AtomicUsize = AtomicUsize::new(0);

struct Counter;

impl log::Log for Counter {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        if record.target() == "faer_perf" {
            eprintln!("warning: {}", record.args());
            WARNINGS.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn flush(&self) {}
}

static LOGGER: Counter = Counter;

fn main() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Warn);

    let n = 64;
    let mut a = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            4.0
        } else {
            1.0 / (1 + i + j) as f64
        }
    });
    let column_major = std::env::args().any(|arg| arg == "column-major");
    let mut a_view = if column_major {
        a.as_mut()
    } else {
        a.transpose_mut()
    };
    let (row_stride, col_stride) = (a_view.row_stride(), a_view.col_stride());

    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    let mut mem = MemBuffer::new(lu_in_place_scratch::<usize, f64>(
        n,
        n,
        Par::Seq,
        Default::default(),
    ));
    let (info, _) = lu_in_place(
        a_view.as_mut(),
        &mut perm,
        &mut perm_inv,
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    );

    let warnings = WARNINGS.load(Ordering::SeqCst);
    let layout = if column_major { "column-major" } else { "row-major" };
    println!("input: {n} x {n} contiguous {layout} (row_stride = {row_stride}, col_stride = {col_stride})");
    println!("factorisation: partial pivoting, {} transpositions", info.transposition_count);
    println!("expected `faer_perf` warnings: 0");
    println!("actual `faer_perf` warnings: {warnings}");
    if warnings != 0 {
        std::process::exit(1);
    }
}
