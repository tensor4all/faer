// `Mat::<T>` is generic over `T` with no numeric bound, and `with_capacity` is a
// safe reserving constructor, so `Mat::<()>::with_capacity(1, 1)` must behave
// like every other empty reservation: an empty matrix with reserved capacity.
// Instead `RawMatUnit::try_with_capacity` computes `align / size` when
// `align > size`, and for a zero-sized `T` `size_of::<T>() == 0` while the
// alignment is nonzero, so the call panics with a division by zero.
use faer::Mat;

fn main() {
    // Controls: both reservations that do not have a zero element size succeed.
    assert_eq!(Mat::<()>::with_capacity(0, 0).shape(), (0, 0));
    assert_eq!(Mat::<f64>::with_capacity(1, 1).shape(), (0, 0));
    println!("controls: empty ZST reservation and f64 reservation both succeed");

    let reserved = std::panic::catch_unwind(|| Mat::<()>::with_capacity(1, 1).shape());
    match reserved {
        Ok(shape) => {
            println!("Mat::<()>::with_capacity(1, 1) expected shape (0, 0) actual {shape:?}");
            if shape != (0, 0) {
                println!("WRONG: unexpected shape for a zero-sized element");
                std::process::exit(1);
            }
        }
        Err(_) => {
            println!("Mat::<()>::with_capacity(1, 1) expected shape (0, 0) actual panic (see the message above)");
            println!("WRONG: a safe reserving constructor panicked for a zero-sized element");
            std::process::exit(1);
        }
    }
}
