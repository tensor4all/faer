# The `faer-ffi` parallel boundary: a team handle instead of `Par` by value (tensor4all/faer#98)

Issue: tensor4all/faer#98. Branch `fix/98-ffi-team-boundary`, base `origin/main` `e4c6d57e`.

## Why

`#85` removed `Par::rayon`, which the C API translated `ParTag::Rayon` into, so `faer-ffi` stopped
compiling and was excluded from the workspace. The break is not a rename: `Par<'team>` is the
exclusive authority over a team, and its `Team`/`TeamExec` token is `!Copy` and scoped. C has no
scope to express that borrow, so a C caller cannot own the token. Returning one by value (what
`ParTag` did) was always wrong for the new design.

## Decisions

- The boundary type is an opaque handle, `FaerV0_24_Team`, created by `libfaer_v0_23_team_create`
  and released by `libfaer_v0_23_team_destroy`; `libfaer_v0_23_team_set_threads` resets the width
  of an existing handle. Every operation takes `const FaerV0_24_Team *team`, and `NULL` runs
  sequentially. It carries the width and nothing else: no token, no pool, no process-wide state.
- The Rust side enters the team for exactly one call and leaves it before returning
  (`with_team_par` → `faer::team::with_team(width, |par| ..)`), so no team is ever held across a C
  frame. A caller that keeps a handle across calls or threads therefore cannot corrupt the
  ownership; the only cost is that each call reserves a team again. `threadteam` returns the
  threads of a dropped pool to a process-wide cache, so repeated calls at one width reuse running
  threads instead of spawning them (`utils::team::tests::with_team_calls_of_one_width_reuse_the_threads`).
- Workspace queries (`*_scratch`) take the same handle but never enter a team: they ask
  `Team::shape` for a `ParShape`, the `Copy` shape that carries no right to execute. Sizing a
  buffer must not reserve threads, and `ParShape::team(width)` is exactly the query type faer
  documents for it. A shape from the handle differs from one taken off a token only in the
  measured dispatch costs, and no workspace query reads them: `dispatch_cost` appears in the
  runtime gates (`utils/cost`, `householder`, the LU swap stage, `gemm`'s floor) and in tests
  only, so the sizes agree.
- `ParTag` and `Par` are removed from the C API (with the `impl Par -> faer::Par` conversion). The
  C API is not consumed by tensor4all and has no released ABI, so no compatibility shim is kept.
- A team that cannot be acquired panics inside `with_team` (threadteam's acquisition timeout), and
  that panic aborts across the `extern "C"` boundary. This is documented on
  `libfaer_v0_23_team_create` rather than plumbed into a status: the operations return `void` or a
  computation status, so there is no channel for an acquisition error, and adding one would change
  every signature for a condition that means the host could not start threads at all.
- The handle contract says what concurrency it allows: shared for reading by operations, exclusive
  for `set_threads` and `destroy`, destroyed exactly once, and no operation running while either of
  those two is called. `set_threads` also invalidates a workspace query made earlier, because the
  size can depend on the width (full-pivot LU sizes one pivot candidate per participant).
- `faer-ffi` returns to the workspace (`members`), and `cargo build -p faer-ffi` /
  `cargo test -p faer-ffi` are part of the local gates.

## Verification conclusions and constraints

- `faer-ffi/src/lib.rs` has two tests, both on the real C entry points:
  - `the_team_handle_is_entered_per_call_and_left_again`: a `NULL` handle gives the sequential
    product and fills no team dispatch counter; a handle of 4 runs regions on the team (faer's
    counters) and gives the same product; the calling thread's affinity mask is unchanged after the
    call, checked against a control `with_team` that narrows it (the mask is the observable form of
    "the call did not leave the team", per threadteam's leader placement); `set_threads(.., 1)`
    then runs sequentially and places nothing; `destroy(NULL)` is a no-op.
  - `scratch_sizes_follow_the_handle_without_entering_the_team`: on full-pivot LU, whose scratch
    sizes one pivot candidate per participant, the width-4 query is larger than the sequential one
    (1120 vs 1048 bytes on this host) and a factorization on exactly that buffer with a team of 4
    succeeds with valid permutations — a query that ignored the handle would under-size it. A
    reconstruct query leaves the calling thread's mask alone, so it leaves no team behind; that the
    queries never *enter* one is a property of the code path (`Team::shape` is the only reader of
    the handle there, `with_team_par` is not called), which a restored mask cannot show.
- The committed `faer.h` is regenerated (`FAER_FFI=1`, nightly, because cbindgen expands the
  `funcs!` macro) and its usage example compiles and runs as a C11 program against the built
  `cdylib` (`c11 -Wall -Wextra`, no warnings). The example uses `//` comments only: a `/* */` inside
  a cbindgen doc comment ends the surrounding `/** */` block.
- The mask check needs room to observe placement: the test sizes its team from the caller's mask and
  requires threadteam to place its leader when the mask is at least that wide. A narrower mask is
  reported on stderr; a host where threadteam cannot pin at all (a mask wider than its `cpu_set_t`,
  affinity syscalls denied) fails the test deliberately rather than passing without the observable. Measured here: `Cpus_allowed_list:
  60-63`, and the mask comes back to it after each call.
- Not verified: `faer-ffi/faer.hpp`, the C++ wrapper, is stale for reasons that predate this change
  (it calls `libfaer_v0_24_*` entry points, which this crate never exported) and nothing builds it;
  it is left alone rather than half-migrated. Windows and macOS were not run. `fork()` without
  `exec` after a team has run is not supported: threadteam keeps its pooled workers in a
  process-wide cache and a child inherits the cache without the threads, as with any thread pool.
  That was not tested here.
- Removed C API surface: `ParTag` (`FaerV0_24_ParTag`), `Par` (`FaerV0_24_Par`) and its `tag`/
  `nthreads` fields; every `par: FaerV0_24_Par` parameter became `const FaerV0_24_Team *team`.
