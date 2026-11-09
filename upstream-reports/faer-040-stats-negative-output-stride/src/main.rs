// `stats::col_mean` / `col_varm` (and so `row_*`) normalise a negative stride
// along the *output* axis by reversing the input only, so the results are
// written in reverse order. A view and an owned copy of the same matrix must
// give the same answer; here they do not, and for `varm` the reversed means
// are applied to the reversed columns, so the view's values are not even a
// permutation of the expected ones.
//
// The output length follows the crate's own assertions: `row_*` writes one
// value per column (`out.ncols() == mat.ncols()`), `col_*` one per row.
use faer::stats::{col_mean, col_varm, row_mean, row_varm, NanHandling};
use faer::{Col, Mat, Row};

fn main() {
    let mut failures = 0usize;

    // Column j holds [0, 2(j+1)]: means 1, 2, 3, 4 and variances 2, 8, 18, 32.
    let a = Mat::<f64>::from_fn(2, 4, |i, j| (j + 1) as f64 * if i == 0 { 0.0 } else { 2.0 });

    // A view with a negative column stride: the output axis is reversed, so the
    // expected means are 4, 3, 2, 1.
    let reversed_cols = a.reverse_cols();
    let owned_cols = reversed_cols.to_owned();
    for nan in [NanHandling::Propagate, NanHandling::Ignore] {
        let (mut mv, mut mo) = (Row::<f64>::zeros(4), Row::<f64>::zeros(4));
        row_mean(mv.as_mut(), reversed_cols, nan);
        row_mean(mo.as_mut(), owned_cols.as_ref(), nan);
        let (mut vv, mut vo) = (Row::<f64>::zeros(4), Row::<f64>::zeros(4));
        row_varm(vv.as_mut(), reversed_cols, mo.as_ref(), nan);
        row_varm(vo.as_mut(), owned_cols.as_ref(), mo.as_ref(), nan);
        println!("row_mean {nan:?}: view {:?}  owned {:?}", mv, mo);
        println!("row_varm {nan:?}: view {:?}  owned {:?}", vv, vo);
        if mv != mo {
            failures += 1;
        }
        if vv != vo {
            failures += 1;
        }
    }

    // The same defect on the other axis, with a negative row stride.
    let reversed_rows = a.reverse_rows();
    let owned_rows = reversed_rows.to_owned();
    for nan in [NanHandling::Propagate, NanHandling::Ignore] {
        let (mut mv, mut mo) = (Col::<f64>::zeros(2), Col::<f64>::zeros(2));
        col_mean(mv.as_mut(), reversed_rows, nan);
        col_mean(mo.as_mut(), owned_rows.as_ref(), nan);
        let (mut vv, mut vo) = (Col::<f64>::zeros(2), Col::<f64>::zeros(2));
        col_varm(vv.as_mut(), reversed_rows, mo.as_ref(), nan);
        col_varm(vo.as_mut(), owned_rows.as_ref(), mo.as_ref(), nan);
        println!("col_mean {nan:?}: view {:?}  owned {:?}", mv, mo);
        println!("col_varm {nan:?}: view {:?}  owned {:?}", vv, vo);
        if mv != mo {
            failures += 1;
        }
        if vv != vo {
            failures += 1;
        }
    }

    if failures > 0 {
        println!("\nWRONG: {failures} reduction(s) disagree between a negative-stride view and its owned copy");
        std::process::exit(1);
    }
}
