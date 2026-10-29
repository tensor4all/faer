# No heap allocation inside a parallel region (tensor4all/faer#111)

## Decisions
- **What allocated, per region, on `origin/main` `ad23987`:**
  - `utils::team::for_each_child`, behind every `for_each` and `for_each_with_par`, collected every item of the
    iterator into a `Vec` (one entry per column for `par_col_iter_mut`), then built a second `Vec` of slots.
  - The full-pivoting LU step built a `best` vector of `nthreads` pivot candidates, and so did both
    Bunch-Kaufman full-pivoting updates (rank one and rank two).
  - `utils::thread::join_sized` boxed its two closures, which runs for each LU swap stage.
- **Why.** The monitor ruled that these allocations are a mechanism defect, not a host constant.
  - They were also suspected of causing #109's floor-level loss: `full_piv_lu f64 128` at 8T on evo was +6.6 to
    +10.4% after #109 ran its per-step regions down to the rule's floor.
  - **That suspicion was wrong.** The evo re-measurement of this change (2026-10-29 07:54Z) puts the cell at
    1.0025 of `main` with the allocations gone, and finds no regression elsewhere.
  - The change stands on its own as a cleanup of the mechanism.
  - The remaining candidates for the floor-level loss are the fixed per-section costs of the 19 steps that
    narrow: the reduction merge, and the trailing block's columns moving between cores as the width changes.
    Whether to chase them is decided later.
- **`for_each_child`.**
  - It takes the producer through `with_producer`, splits it on the leader into the blocks with `split_at`,
    and stores the pieces. The splits are relative to what remains; the block boundaries are those of
    `par_split_indices`.
  - Each block's leader takes its piece and iterates it there, so `P::IntoIter` never has to be `Send`.
  - Everything runs inside the producer callback, so producers that borrow from its frame stay valid:
    threadteam joins every share before it returns.
  - **Storage.** An inline array of 64 `Option<P>` slots is used when there are at most 64 blocks and a slot
    is at most 256 bytes, so the array is at most 16 KiB. A producer can carry state of any size; rayon's
    `map_with` stores its state by value. Otherwise the producer is halved recursively with `team.split`
    and two slots per level, which costs one section per level and allocates nothing either.
- **Pivot candidates.** They come from the call's scratch:
  - `full_pivoting::factor::lu_in_place_scratch` and `lblt::factor::cholesky_in_place_scratch` grow by
    `par.degree()` entries of `(usize, usize, Real)`;
  - the buffer is passed down to the step loop and reused by every step.

  In `cholesky_in_place_scratch` it sits in an `or` with the blocked variant's workspace, since a call uses
  one or the other.
- **`join_sized`.** Two typed slots and a match on the child index, instead of `Box<dyn FnOnce>`.
- **Not changed.**
  - The householder upgrade's `partials` vector. It is allocated per call, not per step.
  - A fixed-cost term in the cost model. Whether one is still needed is decided after the re-measurement.

## Verification conclusions and constraints
- **Allocation test:** `faer/tests/no_step_alloc.rs` is its own test binary, so it can own a `#[global_allocator]`.
  It counts every allocation of the process, on any thread, while one call runs. Setup: teams of 4, 8 and 70
  (70 takes the recursive path), regions forced parallel with `ParThreshold::Fixed(0)`, a warm-up call first.
  - It finds 0 allocations for full-pivoting LU 96, Bunch-Kaufman with full pivoting 96, and `join_sized`.
  - It checks afterwards, outside the counted interval, that each call dispatched a parallel section.
  - On `origin/main` the same test counts 190 allocations in one full-pivoting LU 96 call at width 4.
- **Scope of the promise.** It covers faer's own dispatch and workspaces. An iterator adapter that allocates in
  its own `split_at` (rayon's `map_with` clones its state) still does.
- **Performance**, measured by the monitor on evo: no regression. `full_piv_lu f64 128` at 8T is at 1.0025 of
  `main`, so #109's +7 to 10% on that cell remains.
