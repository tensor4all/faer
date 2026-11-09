# Block-aligned split in the multiple-block Householder factor upgrade (#265)

## Decisions
- `upgrade_householder_factor` splits a factor that spans several target blocks at
  `(block_count / 2) * block_size` instead of `block_count / 2`. The old offset counted
  blocks but was passed to `split_at_mut`/`split_at_col`/`split_at_row`, which take columns
  and rows, so a half could begin inside a target block. Neither half then saw that block's
  pairs whole: pairs that straddled the split were never written, and the second half's own
  block grid was shifted, so it wrote between blocks.
- The new offset follows the rule of the previous-block branch in the same function
  (`(prev_block_count / 2) * prev_block_size`). The halves still run through `join_raw`
  with its existing team split. Thread widths and work floors are unchanged.
- No scratch query. The function allocates its own row-split partials (see #93), and no
  `*_scratch` function mirrors this recursion, so there is no second partition to align.
  The scratch-taking API in the closed PR #267 is out of scope here.
- The call still panics (asserts) when `block_size` is not a multiple of `prev_block_size`
  and several target blocks are present. The new test pins this, so a future change has to
  choose a partition on purpose.

## Verification conclusions and constraints
- `faer/tests/public_householder_upgrade.rs` checks every factor entry on its own. Each
  within-block coefficient is compared with `V^H V` formed in `c64` from the stored basis,
  within `4 m eps sum_k |V_ki| |V_kj|`. The denominators, the lower triangle and the
  between-block entries must be exactly what the caller passed. The basis's diagonal and
  strict upper part hold garbage, so any read of them fails the test.
- Before the fix, every multiple-block case failed at f32/f64/c32/c64, both sequentially
  and on a team. The single-block and `prev_block_size == block_size` cases passed both
  before and after; they are controls. The wide team cases (`n = 37`, `70` at 4096 rows)
  are required to open team regions, so the team path is exercised there. The narrower team
  cases may stay under the work floor; that is the cost model's decision.
- Upstream Codeberg `0ecd846` (`faer/src/linalg/householder.rs:158`) has the same
  `mid = block_count / 2`. This was established by reading the source only. The reproduction
  was not run against upstream, and nothing was reported upstream (deferred-feedback rule).
- In-tree callers (blocked QR, Hessenberg, tridiagonal, bidiagonal) pass one target block
  per call, which takes the single-block branch. The issue reports the same for Colmat's
  wrapper. No change in their results is expected, and none was measured beyond their
  existing tests.
