# `faer-ffi`: the team handle owns its pool (tensor4all/faer#333)

Issue: tensor4all/faer#333. Branch `fix/333-ffi-team-pool`, base `origin/main` `f3873492`.

## Why

#332 gave the C API a caller-owned handle, `FaerV0_24_Team`, but the handle carried only the width:
every call ran `faer::team::with_team`, which builds a `threadteam::Pool`, reserves its members and
drops the pool again at the end of that call. The parallel boundary was therefore paid per call
rather than per handle, which is the wrong shape for a caller that makes many calls.

## Decisions

- The handle owns one `threadteam::Pool` of its width, built by `libfaer_v0_23_team_create` and
  dropped by `libfaer_v0_23_team_destroy`. A width of one is sequential and owns no pool.
- A call moves the pool into a `threadteam::Team` and moves it back out before returning
  (`Pool::into_team` / `Team::into_pool`, faer-ffi's `Team::with_par`). The caller is outside the team
  between calls, as in #332.
- Rejected: holding `threadteam::Team` in the handle. It would make a call allocate nothing at all,
  but threadteam 0.5 pins the thread that *creates* a team to `S[0]` for the team's whole life
  (`pin::TeamPin`), so a C caller's thread would sit pinned to one CPU between calls. The crate's
  affinity test observes that placement directly, so the cost is visible, not theoretical.
- What a call still pays is threadteam's own reservation: one allocation (the `Reservation` id list)
  and the affinity syscalls of placing the leader for the call. Removing that needs a pool-level
  lending API in threadteam, which is an upstream change and not taken here.
- The pool lives in a `Mutex<Option<_>>`, so the pool a call borrows is owned by exactly one call.
  The consequence is the behavioural change of this issue: two threads may call operations on one
  handle, but their calls take turns on the pool instead of running at once (#332 documented the
  handle as shareable for reading).
- `set_threads` is a setting on the owned object: it rebuilds the pool for the new width, and
  narrowing to one releases it. Resizing is a reconfiguration, so a pool build there is intended;
  threadteam's process-wide thread cache hands the new pool the threads of the old one.
- `faer-ffi` names `threadteam` directly for those four calls rather than gaining a pool-lending
  helper in `faer::team`: that would be a new public Rust API for one consumer, and the pattern
  `with_team`'s own documentation recommends to a caller that owns a pool.
- No C API surface change: the same three handle functions with the same signatures, and the same
  entry points taking `const FaerV0_24_Team *`. `NULL` still means serial, and a caller that never
  creates a handle still gets correct results.

## Verification conclusions and constraints

- Acceptance is the allocation count, measured on the real C entry points:
  `tests::a_reused_handle_builds_its_pool_once` counts the allocations the calling thread makes,
  5 calls on a fresh handle and then 5 more on the same live handle. Before this change the second
  batch cost 65 allocations (13 per call, the issue's earlier reference measurement: 140 for 10
  calls); after it, 5 (1 per call: threadteam's reservation). The first batch costs ~14000 either
  way, because it also starts the pool's threads.
- The probe counts only the calling thread. The crate's tests run in parallel and a pool member
  allocates on its own thread, so a global counter would mix tests; a thread-local one is also what
  makes the number stable across runs.
- The existing affinity test `the_team_handle_is_entered_per_call_and_left_again` still passes: the
  calling thread's mask is unchanged after a call, and `set_threads(.., 1)` places nothing. That is
  the observable that would fail if the handle held a team instead of a pool.
- The serialisation the mutex buys is tested, not only documented: `tests::two_threads_share_one_handle`
  puts two threads on one handle with their own outputs and both get the sequential product, and
  `tests::the_handle_is_shareable` fails to compile if the handle stops being `Send + Sync`.
- `tests::the_token_carries_the_handle_width` now also widens a handle that was narrowed to one, so
  the rebuilt pool is on the path too (the width the token reports is the width the caller set).
- The lifetime rules are written on `libfaer_v0_23_team_create` (and in the generated `faer.h`), since
  that is where a C caller reads them: two threads on one handle, `set_threads`/`destroy` exclusive,
  use after `destroy`, and `fork` without `exec`.
- **Undecided, left open deliberately:** whether a handle inherited across `fork` should be detected
  (a stored creator pid compared in `create` and each call) instead of blocking on workers the child
  does not have. Today the rule is "create the handle in the process that uses it", and nothing
  checks it.
- Not verified: `fork()` without `exec` (documented, not tested); Windows and macOS, where threadteam
  does not pin and the placement changes. `faer-ffi/faer.hpp` remains stale for the reason recorded in
  the #332 worklog.
