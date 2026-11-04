# Leaf width of the partial pivoting LU for `f64`

No upstream revision was measured for this change. Host: evo2, AMD Ryzen AI MAX+ 395 (Zen 5), boost on.

## Decisions

- **Cause, from the breakdown.** With the `U` panel solve no longer aliasing
  (`2026-11-05-trsm-aliased-rhs.md`), temporary stage timers in `lu_in_place_recursion` put the
  unblocked leaves at 0.65 of 1.26 ms at n = 512 on 8 threads (52 %) and 1.87 of 4.24 ms at 1024 (44 %).
  The leaves are sequential, so their time does not fall with the team; at 1T they are 29 % and 15 %.
  Inside a leaf no step dominates (pivot search, row swap, scaling and the rank-one update are each
  well under a microsecond per column): the leaf is per-column overhead over a 16-column panel whose
  columns share L1 sets at a power-of-two stride.
- **Chosen: a narrower leaf for `f64`.** `recursion_threshold` 16 -> 8, which moves work from the leaf
  loop into the recursion's solves and products. Scan (median of 31 interleaved calls, team of T):
  16 -> 8 is 6-7 % at n = 256, 9-16 % at 512, 4-11 % at 1024, 1-7 % at 2048-4096; 4 is within 2 % of 8
  (ahead at large n, behind at 256). The table sits next to the constant.
- **Scope.** Only native `f64`: `Auto<T>` is per type and the other types were not scanned, so they keep
  16. The parameter is public; a caller that sets it explicitly is unaffected.
- **Not chosen here:** a faster leaf kernel (staging the panel, a fused SIMD column loop) and overlapping
  the leaves with the trailing update (look-ahead, `cost::lookahead_width` of #168). Both are larger
  changes; the narrower leaf is the measured step that costs one constant.

## Verification conclusions and constraints

- `f64_default_leaf_factors`: the f64 default is 8 and f32/c64 keep 16; `P A = L U` to rounding at
  n = 7, 8, 9, 17, 100, 257 (leaf tails), sequential and on a 4-member team. The existing LU tests
  (`threaded_lu_matches_the_sequential_one`, `test_plu`) run with the new default; the full lib suite passes.
- Campaign numbers: `~/work/lu-chol-opt/` on evo2.
- **Unverified.** f32/c32/c64 (not scanned, unchanged); other hosts.
