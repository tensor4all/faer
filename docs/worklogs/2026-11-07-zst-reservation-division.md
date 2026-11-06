# Zero-sized elements in the owned reservation and cleanup paths (#286)

## Decisions

- `RawMatUnit::try_with_capacity` pads `row_capacity` up to a multiple of
  `align / size` so that the row stride in bytes (`row_capacity * size`) is a
  multiple of the allocation's alignment. For a zero-sized element that stride
  is already 0 for every `row_capacity`, and `align / size` divides by zero, so
  the division is now skipped when `size == 0` and nothing else changes.
- Two alternatives were rejected. Rounding the capacity up to one element
  reserves nothing (any `rows * cols` of a ZST already has size 0, so the
  allocation branch is never taken) and would report a capacity the caller did
  not ask for. Skipping the reservation entirely in `Mat::with_capacity` would
  special-case the ZST path at the wrong layer: `try_reserve` after a resize
  goes through the same function, and the size arithmetic below the alignment
  block already handles size 0 (`layout` stays empty, no `alloc`/`dealloc`).
- No bound on `T`, no public API change, and no behaviour change for
  non-zero-sized elements: the guard is the same predicate the division already
  assumed.
- Making a ZST matrix constructible exposed a second, independent defect on the
  same invariant: `DropMat` advances its column pointer in bytes
  (`wrapping_byte_add`) but `init_with` and `truncate`'s column branch handed it
  the element count `row_capacity` instead of `row_capacity * size_of::<T>()`.
  Both sites now pass bytes, which is what the other two `DropMat` sites
  (`Own::drop` and `truncate`'s row branch) already computed. For a ZST the
  correct byte stride is 0, which puts every column at the same address; for an
  aligned drop-carrying non-ZST it is the stride the padding above was
  installed to make a multiple of the alignment.

## Verification conclusions and constraints

- Both guarded lines are upstream faer (`sarah-quiñones`; the division is
  `c01bf0b0`/`e95ba701`, the two `byte_col_stride` sites are older), so upstream
  has both defects; not reported there, since upstream feedback is deferred.
  Measured revision: this fork at `a055727`, which carries that code unmodified.
- Reservation: `zero_sized_elements_reserve_without_panicking` fails on the
  pre-change tree with `attempt to divide by zero` at
  `faer/src/mat/matown.rs:76:36` and covers `Mat::<()>::with_capacity(1, 1)`,
  `Mat::<()>::with_capacity(0, 0)`, growth to a larger ZST matrix, and an
  ordinary `Mat::<f64>::with_capacity(1, 1)`.
- Cleanup: `cleanup_of_aligned_and_zero_sized_elements_uses_a_byte_stride` uses
  a `#[repr(align(8))]` drop-counting element and a `#[repr(align(64))]` ZST.
  Without the byte stride it aborts in `DropCol::drop`'s `from_raw_parts_mut`
  and Miri reports `encountered an unaligned reference (required 8 byte
  alignment but found 1)` at `faer/src/mat/matown.rs:274`; with it the dropped
  ids are exactly the three columns' values.
- Miri (`cargo +nightly miri test -p t4a-faer --test owned_capacity`, miri
  0.1.0, nightly 2026-07-13) is clean on the whole file. The aligned-drop defect
  is not ZST-specific: on the unmodified `a055727` tree,
  `Mat::<#[repr(align(8))] struct S(u32)>::from_fn(1, 3, ..)` followed by
  `truncate(1, 0)` is already the same Miri error, so the byte stride is a
  pre-existing upstream unsoundness that the ZST fix would otherwise put behind
  a constructible matrix.
- Not verified: cleanup unwinding when a user closure panics mid-column is
  covered by Miri probes but not by a committed test; `undo`/`rebuild` and the
  borrowed-view conversions were not exercised with a ZST beyond the committed
  resize, truncate and drop paths. No timing claim: one branch outside any
  inner loop, and one multiply at cleanup.
