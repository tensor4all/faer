# faer-041: `Mat::<()>::with_capacity(1, 1)` panics with a division by zero

**What happens:** the safe reserving constructor `Mat::<T>::with_capacity` panics with `attempt to divide by zero` when `T` is zero-sized (`T = ()`). The empty reservation `Mat::<()>::with_capacity(0, 0)` and an ordinary `Mat::<f64>::with_capacity(1, 1)` both succeed, so the panic depends only on the element size and the requested capacity.
**Which input:** `Mat::<()>::with_capacity(1, 1)`; any nonempty reservation for a zero-sized element reaches the same line.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-09.

## Reproduce

```sh
cargo run
```

The program runs the two controls, then the failing reservation inside `catch_unwind`, and exits with status 1 when the call panics.

| call | expected | `faer` 0.24.4 |
|---|---|---|
| `Mat::<()>::with_capacity(0, 0)` | shape `(0, 0)` | `(0, 0)` |
| `Mat::<f64>::with_capacity(1, 1)` | shape `(0, 0)` | `(0, 0)` |
| `Mat::<()>::with_capacity(1, 1)` | shape `(0, 0)` | **panic** at `mat/matown.rs:76`: `attempt to divide by zero` |

## Cause

`Mat<T>` has no numeric bound, and `with_capacity` documents a resizable container like a two-dimensional `Vec`; nothing restricts `T` to a nonzero size. `RawMatUnit::try_with_capacity` (`src/mat/matown.rs:66-84` in 0.24.4) pads the row capacity so that the byte row stride `row_capacity * size` is a multiple of the allocation's alignment:

```rust
let size = core::mem::size_of::<T>();
let prev_align = core::mem::align_of::<T>();
let align = align_for(size, prev_align, core::mem::needs_drop::<T>());
if align > size {
    row_capacity = row_capacity
        .msrv_checked_next_multiple_of(align / size)   // :76, align / 0
        .ok_or(TryReserveError::CapacityOverflow)?;
}
```

For a zero-sized `T`, `size == 0` while `align >= 1`, so `align > size` holds and `align / size` divides by zero. An empty reservation escapes it because the capacity is already sufficient and the branch is not reached.

## How the fork fixed it

The padding is skipped when `size == 0` (`if size > 0 && align > size`, `faer/src/mat/matown.rs`): a zero-sized element has a byte stride of 0 for every capacity, so there is nothing to pad, and the size arithmetic below already yields an empty layout with no allocation. Making a zero-sized matrix constructible exposed two `DropMat` sites that had passed an element count where a byte stride was expected (`init_with` and `truncate`); both now multiply by `size_of::<T>()`.

Fork tests: `zero_sized_elements_reserve_without_panicking` and `cleanup_of_aligned_and_zero_sized_elements_uses_a_byte_stride` in `faer/tests/owned_capacity.rs`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same `align / size` is in upstream `main` at `0ecd846` (2026-10-22), `faer/src/mat/matown.rs:76`.

## References

* Fork fix: tensor4all/faer#288 (merge commit `a977480`).
* Where it was found: tensor4all/faer#286, downstream AtelierArith/Colmat.jl.
