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
  parallel; it is correct.
- `set_threads` is a setting on the owned object: it rebuilds the pool for the new width, and
  narrowing to one releases it. Resizing is a reconfiguration, so a pool build there is intended.
- `faer-ffi` names `threadteam` directly for those calls rather than gaining a pool-lending helper in
  `faer::team`: that would be a new public Rust API for one consumer, and the pattern `with_team`'s
  own documentation recommends to a caller that owns a pool.
- No C API surface change: the same three handle functions with the same signatures, and the same
  entry points taking `const FaerV0_24_Team *`. `NULL` still means serial, and a caller that never
  creates a handle still gets correct results.

## Verification conclusions and constraints

- Acceptance is the allocation count, measured on the real C entry points:
  `tests::a_reused_handle_builds_its_pool_once` counts the allocations the calling thread makes, once
  per call, on one live handle. Before this change every call cost 13 allocations (the old handle
  built and dropped a pool per call, the issue's own reference measurement: 140 for 10 calls); after
  it, the first call costs the pool's construction and every later call 1 (threadteam's warm
  reservation). The test asserts on the cheapest of eight calls, so a pool rebuild on every call
  cannot pass, and a member that parks and makes the next reservation take the cold six-allocation
  path cannot fail.
- The probe counts only the calling thread. The crate's tests run in parallel and a pool member
  allocates on its own thread, so a global counter would mix tests.
- The existing affinity test `the_team_handle_is_entered_per_call_and_left_again` still passes: the
  calling thread's mask is unchanged after a call, and `set_threads(.., 1)` places nothing. That is
  the observable that would fail if the handle held a team instead of a pool.
- `tests::two_threads_share_one_handle_and_take_turns` counts the calls inside the team from three
  threads and requires the maximum to be one, which is the serialisation the docs promise; the
  products alone would not show it, since a per-call pool produces them too.
  `tests::the_handle_is_shareable` fails to compile if the handle stops being `Send + Sync`.
- `tests::a_nested_call_runs_sequentially` is the reentrancy guard's test. Without the guard the
  nested call waits for a mutex the same thread holds, so the regression is a hang (measured: the test
  binary was killed at 45 s) rather than a wrong answer; the guard is also what a `then_some` instead
  of an `if` silently breaks, which is worth pinning.
- `tests::the_token_carries_the_handle_width` also widens a handle that was narrowed to one, so the
  rebuilt pool is on the path too (the width the token reports is the width the caller set).
- The lifetime rules are written on `libfaer_v0_23_team_create` (and in the regenerated `faer.h`),
  since that is where a C caller reads them.
- **Undecided, left open deliberately:** whether a handle should be refused after a `fork`. Both an
  inherited handle and a fresh one inherit threadteam's process-wide worker cache and its mutex, so
  "create the handle in the child" is not a fix by itself: `exec` before using a team in a child whose
  parent already ran one. Detecting the case (a creator pid compared in `create` and in each call)
  would add a user-visible failure mode to a boundary that has no error channel, so it wants an
  explicit decision rather than a rider on this change.
- Not verified: `fork()` without `exec` (documented, not tested); Windows and macOS, where threadteam
  does not pin and the placement changes. `faer-ffi/faer.hpp` remains stale for the reason recorded in
  the #332 worklog.
