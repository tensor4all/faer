// Sparse LU of the singular matrix A = [[1, 1], [1, 1]], with each backend forced.
// The numeric factorization must report the zero pivot instead of returning Ok.
use dyn_stack::{MemBuffer, MemStack};
use faer::prelude::*;
use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuSymbolicParams, NumericLu};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, Triplet};
use faer::Conj;

fn main() {
    let entries = [
        Triplet::new(0, 0, 1.0f64),
        Triplet::new(0, 1, 1.0),
        Triplet::new(1, 0, 1.0),
        Triplet::new(1, 1, 1.0),
    ];
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(2, 2, &entries).unwrap();
    let mut wrong = false;
    for (name, threshold) in [
        ("simplicial", SupernodalThreshold::FORCE_SIMPLICIAL),
        ("supernodal", SupernodalThreshold::FORCE_SUPERNODAL),
    ] {
        let symbolic = factorize_symbolic_lu(
            a.symbolic(),
            LuSymbolicParams {
                supernodal_flop_ratio_threshold: threshold,
                ..Default::default()
            },
        )
        .unwrap();
        let mut numeric = NumericLu::<usize, f64>::new();
        let result = symbolic.factorize_numeric_lu(
            &mut numeric,
            a.as_ref(),
            Par::Seq,
            MemStack::new(&mut MemBuffer::new(
                symbolic.factorize_numeric_lu_scratch::<f64>(Par::Seq, Default::default()),
            )),
            Default::default(),
        );
        match result {
            Err(e) => println!("{name}: expected an error, got {e:?}"),
            Ok(lu) => {
                let mut x = Mat::<f64>::ones(2, 1);
                lu.solve_in_place_with_conj(
                    Conj::No,
                    x.as_mut(),
                    Par::Seq,
                    MemStack::new(&mut MemBuffer::new(
                        symbolic.solve_in_place_scratch::<f64>(1, Par::Seq),
                    )),
                );
                println!(
                    "{name}: expected an error, got Ok; solve of A x = [1, 1] gives x = [{}, {}]",
                    x[(0, 0)],
                    x[(1, 0)]
                );
                wrong = true;
            }
        }
    }
    if wrong {
        println!("WRONG: a singular matrix factors without an error");
        std::process::exit(1);
    }
}
