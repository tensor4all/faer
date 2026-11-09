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
- The pool's workers are **dedicated** (`PoolBuilder::thread_name`), not threadteam's default shared
  workers. A shared pool's drop parks its workers in threadteam's process-wide cache, keyed by width,
  and the next default pool of that width takes them. That made two of the handle's promises false
  (review r1, finding 3): `destroy` did not stop the threads, and a child forked after a `destroy`
  inherited the cached Rayon pool without its threads, so the child's own `create` waited forever in
  the affinity broadcast to them. Dedicated workers are joined by the drop and never enter the cache.
  The cost is that a `create` (or a resize) always starts threads instead of reusing cached ones,
  which is a per-handle cost, not a per-call one.
- A call moves the pool into a `threadteam::Team` and moves it back out before returning
  (`Pool::into_team` / `Team::into_pool`, faer-ffi's `Team::with_par`). The caller is outside the team
  between calls, as in #332.
- Rejected: holding `threadteam::Team` in the handle. It would make a call allocate nothing at all,
  but threadteam 0.5 pins the thread that *creates* a team to `S[0]` for the team's whole life
  (`pin::TeamPin`), so a C caller's thread would sit pinned to one CPU between calls. The crate's
  affinity test observes that placement directly, so the cost is visible, not theoretical.
- What a call still pays is threadteam's own reservation: warm, one allocation (the `Reservation` id
  list); after its members park, six. Removing that needs a pool-level lending API in threadteam,
  which is an upstream change and not taken here.
- The pool lives in a `Mutex<Option<_>>`, so the pool a call borrows is owned by exactly one call.
  The consequence is the behavioural change of this issue: two threads may call operations on one
  handle, but their calls take turns on the pool instead of running at once (#332 documented the
  handle as shareable for reading).
- A call made from inside another call on the same thread — a C callback calling back into this API,
  which `SchurParams::recommended_shift_count` and its siblings do — gets no team and runs
  sequentially. Waiting for the pool its caller holds would never end, and the mutex is not
  reentrant, so the thread-local `InCall` guard is what keeps that path alive. Nesting is not
  parallel; it is correct. The lock is not moved out of the call instead: the pool *is* the thing
  the lock protects and the call holds it for its duration (it is inside the team), so a second
  caller that did not wait would have to run without it anyway. The guard gives a nested call the
  same thing — sequential — without changing what an unrelated thread sees.
- `set_threads` is a setting on the owned object: it rebuilds the pool for the new width, and
  narrowing to one releases it. Resizing is a reconfiguration, so a pool build there is intended.
- `faer-ffi` names `threadteam` directly for those calls rather than gaining a pool-lending helper in
  `faer::team`: that would be a new public Rust API for one consumer, and the pattern `with_team`'s
  own documentation recommends to a caller that owns a pool.
- No C API surface change: the same three handle functions with the same signatures, and the same
  entry points taking `const FaerV0_24_Team *`. `NULL` still means serial, and a caller that never
  creates a handle still gets correct results.

## Verification conclusions and constraints

- Acceptance is the allocation count, measured on the real C entry point:
  `tests::a_reused_handle_builds_its_pool_once` counts the allocations the calling thread makes for
  each of eight calls on one handle, and for the same eight products run the way `origin/main` ran
  every C call, `faer::team::with_team` per call (a pool built from threadteam's cache, reserved and
  dropped). The first call of each shape is excluded (the handle's first acquisition of its pool,
  thousands of allocations; the reference's first build of its cached workers) and the tails are
  summed, requiring the handle's to be under two thirds of the reference's. Measured, three runs each:
  tail 10 against 91 (calls `[14019, 4, 1, 1, 1, 1, 1, 1]` against `13` each); with a 20 ms sleep
  before every call, so the members park and every reservation takes threadteam's cold path, tail 42
  (`6` each) against 91; with `Team::with_par` reverted to `faer::team::with_team` per call, tail 98
  against 91, which fails. An earlier version compared against eight handles of one call each, which
  stopped discriminating once the workers became dedicated: a `create` then starts threads, so that
  shape cost 544–616 and the reverted implementation passed against it.
- The pool itself is built by `team_create`, not by the first call, and `create` is outside the
  measured region: what the first call pays is the acquisition.
- The probe counts only the calling thread. The crate's tests run in parallel and a pool member
  allocates on its own thread, so a global counter would mix tests.
- The existing affinity test `the_team_handle_is_entered_per_call_and_left_again` still passes: the
  calling thread's mask is unchanged after a call, and `set_threads(.., 1)` places nothing. That is
  the observable that would fail if the handle held a team instead of a pool.
- `tests::two_threads_share_one_handle_and_take_turns` counts the calls inside the team from three
  threads and requires the maximum to be one, which is the serialisation the docs promise; the
  products alone would not show it, since a per-call pool produces them too. It forces no overlap
  (the calls sleep to make one likely), so a fully serialised schedule would also pass it: it can
  show that two calls were never inside the team at once, not that they were ever offered the chance.
  A call nested in another call is the exception the docs state: it never takes the lock, so it is not
  ordered against another thread using the same handle.
  `tests::the_handle_is_shareable` fails to compile if the handle stops being `Send + Sync`.
- `tests::a_nested_call_runs_sequentially` is the reentrancy guard's test. Without the guard the
  nested call waits for a mutex the same thread holds, so the regression is a hang (measured: the test
  binary was killed at 45 s) rather than a wrong answer; the guard is also what a `then_some` instead
  of an `if` silently breaks, which is worth pinning.
- `tests::an_evd_callback_may_call_back_on_the_same_handle` is the review's reproducer through the
  real path: `libfaer_v0_23_evd_f64` with a `recommended_shift_count` callback that multiplies on the
  same handle, for a width-1 handle and a parallel one. It runs the EVD on a helper thread with a 60 s
  deadline, so a regression fails instead of hanging CI. With the guard disabled it fails at width 1
  (the review's case: the sequential handle also holds the lock through the call); with it, both
  widths pass, the callback is called, the nested product is right, and the callback ran on the
  calling thread, which is where the thread-local guard reaches.
- `faer-ffi/tests/fork.rs` measures the `fork` rule: `destroy` returns the process to the thread
  count it had before `create`; a child forked while a parent handle is alive creates, uses and
  destroys its own; and a child forked after `create, use, destroy` does the same. Each child runs
  under `alarm(20)`. With threadteam's shared workers (the `thread_name` removed) the thread count
  after `destroy` is 6 against 2 and the destroy-then-fork child is killed by `SIGALRM`; with dedicated
  workers all three pass.
- `tests::the_token_carries_the_handle_width` also widens a handle that was narrowed to one, so the
  rebuilt pool is on the path too (the width the token reports is the width the caller set).
- The lifetime rules are written on `libfaer_v0_23_team_create` (and in the regenerated `faer.h`),
  since that is where a C caller reads them.
- **Undecided, left open deliberately:** whether an *inherited* handle should be refused after a
  `fork`. A handle the child creates works (measured above); one it inherited names threads the child
  does not have. Detecting that (a creator pid compared in each call) would add a user-visible
  failure mode to a boundary that has no error channel, so it wants an explicit decision rather than
  a rider on this change.
- Not verified: an operation on an inherited handle in a forked child (documented as unsupported, not
  run); Windows and macOS, where threadteam does not pin and the placement changes.
  `faer-ffi/faer.hpp` remains stale for the reason recorded in
  the #332 worklog.
