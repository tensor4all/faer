# Performance tips for this fork

Read this before writing or reviewing performance-sensitive code here (R-performance-51). Each entry
is a lesson that cost a measurement to learn; the worklog it cites has the numbers and the host.
`docs/worklogs/` holds the history of each change, `PATCHSET.md` what the fork carries today; this
file holds what carries over to the next change.

## Measure the breakdown before designing

Two design reviews of the blocked bidiagonalisation did not predict either of the two effects that
decided it (strided rows, worker wake-ups). One per-stage timing pass did
(`docs/worklogs/2026-10-23-blocked-bidiag.md`). Put temporary timers around the stages of the loop
you are changing, at 1T and 8T, before choosing where to work; remove them before committing.

## Rows of a column-major matrix are strided operands

A row of `A` has stride `m`. `matmul` reaches a `matvec` kernel only for a one-column or one-row
destination (`faer/src/linalg/matmul/mod.rs:1236`), and the two kernels accept different layouts:
`matvec_colmajor` takes a contiguous destination (`ColMut<'_, T, Dim<'M>, ContiguousFwd>`, `:891-899`)
and a `rhs` of any stride, while `matvec_rowmajor` takes a contiguous `rhs` and accepts a strided
destination (`:762-770`). Which one a call reaches depends on the leading transposition in
`matmul_imp` (`:1226-1310`), so the layout that costs a gather is the *rhs* in one kernel and the
*destination* in the other; nothing here says a strided operand alone reaches the general kernel.
The matrix has its own requirement: the column-major kernel needs a column-major `lhs`
(`:1238-1239`) and the row-major one a row-major `lhs` (`:1253-1254`), so an `lhs` strided in both
directions runs neither, and gathering a vector does not change that. A reflector's norm becomes a
strided reduction in any case. Gather the row into a contiguous buffer
once, update it, reflect it and use it as the products' vector there, scatter once
(`bidiag_blocked_panel`'s `w`). The row-side steps of the panel went from 15 ms to 6 ms at n = 512.

## A leading dimension that is a multiple of 1 KiB aliases L1 sets

The L1 here is 48 KiB, 12-way, 64 sets: addresses 4 KiB apart share a set, and a 512-row `f64`
copy has exactly that column stride, so every entry of a row lands in one set and twelve of them fit.
The bidiagonalisation touches a row per step and ran a fifth slower at 512 than at 500 or 520. The
SVD driver pads its work copy by 64 bytes per column when the stride would be a multiple of 1 KiB
(`svd/mod.rs` `work_rows`): 18.6 -> 15.3 ms at eight threads. Benchmark shapes are powers of two,
so this is where it shows; a vendor's `?gebrd` at 512 matches our 500.

## The team is the caller's, and it is held across the work

`Par::Team` is the exclusive authority over the reserved members of a `threadteam` team. A reserved
member spins for a bounded budget of polls and then parks. The budget is the pool's, not faer's
(threadteam 0.5.1 as locked in `Cargo.lock`; `faer/Cargo.toml` requires `^0.5.1`):

* by default 300,000 polls on x86-64 (about 3-7 ms depending on the core's `PAUSE`) and 2^21 on
  aarch64 (`SPIN_POLLS_DEFAULT`, threadteam `src/pool.rs`), unless the caller sets
  `PoolBuilder::spin_polls`;
* at most 100 polls when the scope is oversubscribed, that is when it has more participants than
  CPUs in the caller's set (`THROTTLED_POLLS`, threadteam `src/park.rs`), whatever the pool's setting.

With the default budget on a scope that is not oversubscribed, a sequence of regions closer together
than the budget pays no wake and a longer gap parks the members once. On an oversubscribed scope the
window is 100 polls, so only regions issued almost back to back avoid a wake. Each member keeps the
range it had in the previous region, so a core's L2 keeps its slice of the trailing block across
them: the blocked bidiagonalisation's two products per step went 10.3 and 10.6 ms to 2.7 and 3.7 ms
at 512, 8T, with nothing but the reservation (`docs/worklogs/2026-10-23-blocked-bidiag.md`; the
reservation was a spindle group there, the mechanism threadteam kept when it replaced it).

The library never builds a team; the width, the lifetime and the thread count are the caller's
(`colmat::team` is the route, not the policy). A caller that runs parallel work repeatedly owns the
`threadteam::Pool` and `Team` once and lends a `Par` from the team to each call:

```rust
// once: 8 participants, the calling thread and 7 reserved workers
let pool = threadteam::Pool::builder().threads(8).build()?;
let team = pool.into_team(8).ok().expect("an idle pool");
// per call: the reservation is already held, the call gets the authority
for (a, h) in work {
    let mut exec = team.exec(8);
    exec.team(|t| qr_in_place(a.as_mut(), h.as_mut(), Par::Team(t), stack, params))
        .ok().expect("a team token for an idle team");
}
```

`colmat::team::with_team(width, ..)` is the one-off form: it builds a pool and acquires its members
inside the closure, measured at 50 us to 5 ms per call on the M5 depending on whether the threads
came back from threadteam's cache of idle pools, against 0.6-1 us for a token taken from a team the
caller already owns. On a call of tens of milliseconds that is lost in the work; in a loop of short
calls it is the difference, and no measurement here shows `with_team` buying anything back.

Corollary: a region runs on the members of the token it was handed, never on a runtime outside it.
The entry points of `utils::thread` (`for_each`, `for_each_with_par`, `join_raw`, `join_sized`,
`split`, `with_width`) divide the token and deal the work to its members. A rayon iterator
(`par_chunks_mut`, `par_iter`) is only a splitter there: `for_each` consumes its producer and deals
the blocks to the team, so no rayon job is started and the machine does not run more threads than the
caller granted. Rayon itself is a private pool in this fork (`threadteam` builds its own
`rayon::ThreadPool` and never starts the process-global one), so driving the iterator with rayon
(`.for_each`, `rayon::join`) instead of handing it to `utils::thread` is what leaves the token's
members for the calling thread's registry.

## A region costs about 1.5 us on a team, and a threaded gemv of 512 is 6 us

Measured with `probes/gemv-threads` (faer-benchmark, diagnostic): one 8-job region on a held team
costs 1.75 us, and `threadteam`'s `aarch64-apple` row puts a section at 1.4 us at width 6 and 3.0 us
at width 2. The matvec path is taken only for a one-row or one-column destination (`m == 1` or
`n == 1`; the one-column case is direct, the one-row case transposes), and its width is
`cost::par_width` over the left operand's element count — `lhs.nrows() * lhs.ncols()`, so `m*k`, or
`n*k` after the one-row transpose (`faer/src/linalg/matmul/mod.rs:775`, `:904`) — at `rate::matvec`
with one section, not a fixed element count: below the width that pays it runs sequentially. A step
that issues two regions pays about 3-4 us of synchronisation; 384 steps are 1.5 ms. Design the loop
so that the regions are the big products, not the short ones.

## Short products are bandwidth-bound in L2, not call-bound

A 512x16 vector product costs 1.2 us at 50 GB/s with `Par::Seq`. Fusing eight of them per step buys
little; halving the panel width halves them (8-16 flat, 32 slower).

## dqds: the sweep is latency-bound, so only the sweep count matters

One division sits on the dependency chain per entry (`delta -> ahat -> a / ahat -> delta`), about
10 ns at this clock, the same for everyone. Speed comes from fewer sweeps: a shift that is rarely
rejected (the trailing 2x2's eigenvalue backed off by twice its eigenvector-weighted second-order
coupling), a release test in the eigenvalue's units (`(100 eps)^2 (sigma + a)`, not the shifted
factors, which go to zero as the shift converges), ping-pong buffers so a rejection costs no copy
(`docs/worklogs/2026-10-23-dqds-shift.md`: 13 accepted and 9 rejected sweeps per value at 64 became
3.7 and 0.3). Count sweeps before touching the shift.

## Thresholds are measured, with the table beside the constant

Block width and floor (`bidiag.rs`), the dqds back-off factor, the padding threshold: each has its
scan in the worklog and a one-line summary next to the constant. A threshold without its table is a
guess.

## A panel made of short regions is hidden, not split further

The blocked QR's panel at 4096 is 121 ms at 16T and barely shrinks with the thread count or the block
width (block 128 → 32 cuts its arithmetic 4x and its time 104 → 86 ms). It is a fixed cost per
recursion node, at serial `dgeqr2`'s speed. Splitting its applies by rows or packing their basis
changed it by 1 ms; overlapping it with the previous block's trailing apply on a split team
(look-ahead) is the lever (`docs/worklogs/2026-10-30-qr-lookahead.md`, faer#160). Before threading a
stage further, time it at block widths that differ by 4x: if it does not follow the arithmetic, hide it.

## An overwrite-only entry does not keep the replace entry's dispatch for free

`matmul_with_conj_uninit` writes a fresh destination, and the routes it takes are not the replace
entry's: it fills and returns for an empty `k` (`:1675-1679`), fills and re-enters for a one-row
product (`:1680-1691`), takes `t4a-private-gemm-x86` with `Accum::Replace` for a native scalar above
one row when AVX-512F, or AVX2 plus FMA, is detected (`:1692-1709`); a native scalar without that
kernel — another target, a missing instruction set, or `std` off — fills and re-enters
(`:1786-1793`); and other scalars compute each element in place (`:1795-1810`). The original claim — that the replace entry's dispatch is kept — was wrong in both
directions, so an overwrite-only entry has to restore the dispatch by hand, and only where the fill
it was avoiding is smaller than the kernel it lands on (tensor4all/colmat-rs#200). That last condition is
guidance, not a rule the dispatch applies: the worklog measured it for `m = 1` on a few shapes and
found `m = 2/8/16` unchanged (`docs/worklogs/2026-11-03-matmul-uninit-thin.md`), so measure before
relying on it elsewhere.
