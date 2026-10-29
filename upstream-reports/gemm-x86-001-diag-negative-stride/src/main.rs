// dst = a * diag(d) * b for a 1 x 3 row `a`, a 3 x 1 column `b` and d = [1, 2, 3].
// a = [1, 0, 0] and b = [1, 1, 1], so the product picks out d[0] and must be 1.
// The same product is computed twice: once with `a` stored forwards, once with `a`
// stored backwards (negative column stride). Both describe the same matrix.
use private_gemm_x86::{gemm, Accum, DType, DstKind, IType, InstrSet};

fn product(a_ptr: *const f64, a_col_stride: isize, d: &[f64; 3], b: &[f64; 3]) -> f64 {
    let alpha = 1.0f64;
    let mut dst = 0.0f64;
    unsafe {
        gemm(
            DType::F64, IType::U64, InstrSet::Avx256,
            1, 1, 3, // rows, columns, depth
            &mut dst as *mut f64 as *mut (), 1, 1,
            core::ptr::null(), core::ptr::null(), DstKind::Full, Accum::Replace,
            a_ptr as *const (), 1, a_col_stride, false,
            d.as_ptr() as *const (), 1, // the diagonal, stride 1
            b.as_ptr() as *const (), 1, 3, false,
            &alpha as *const f64 as *const (),
            1, // one thread
        );
    }
    dst
}

fn main() {
    assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"), "needs AVX2 and FMA");
    let d = [1.0, 2.0, 3.0];
    let b = [1.0, 1.0, 1.0];

    // `a` stored forwards: memory [1, 0, 0], column stride +1.
    let forwards = [1.0, 0.0, 0.0];
    let got_forwards = product(forwards.as_ptr(), 1, &d, &b);

    // The same `a` stored backwards: memory [0, 0, 1], pointer at the last element,
    // column stride -1. Column t of `a` is still a[t].
    let backwards = [0.0, 0.0, 1.0];
    let got_backwards = product(backwards[2..].as_ptr(), -1, &d, &b);

    println!("expected:                 1");
    println!("a stored forwards:        {got_forwards}");
    println!("a stored backwards:       {got_backwards}");
    if got_backwards != 1.0 {
        println!("WRONG: the backwards case used d[2] instead of d[0]");
        std::process::exit(1);
    }
}
