# Small-size f64 residuals after 0.25.0: what they are made of (tensor4all/faer#70)

## Goal

The 0.25.0 merge gate (#61, pair 3) left a handful of small f64 cells 1-6% slower at 4T/8T. The
issue recorded the numbers and one suspect: "the cost of entering parallel work from a Rayon pool
worker (`InPool`) under the 0.25 `Par<'team>` path", and noted that `solve 64x64` does not fit it
because its 1T rows were also slower.

This record attributes the residuals before anything is changed. **No threshold, rate or region
rule is touched here**: the numbers below are evidence for a later decision, not that decision.

## Identities

| | value |
| --- | --- |
| baseline | `5a27bcece2f9f0d1c1b94c9b42dd7bfc4a94dc39` (`t4a-faer` 0.24.10) |
| candidate | `0e15729ad011e3813a41c168167ec61ed11a9dbb` (`t4a-faer` 0.25.0) |
| upstream baseline | `faer-v0.24.4` = `0539947ffb757a739d7e703a7d2fa0c792a909c1` |
| harness | `faer-benchmark` branch `harness/faer-0.25` at `1e53e9c4e9bf6483b27577f439291d2b85acf4cc` (harness code identical to the pair-3 `777f3d5`: `git diff --stat 777f3d5 1e53e9c -- fbench benchmarks scripts Cargo.toml Cargo.lock` is empty) |
| toolkit | `campaign-toolkit` `77bc6e7328a2f78860d7ae7e3525b5a1b5ed1f3b` (the revision the harness branch pins) |
| spindle (probe) | `0.2.6`, the version both revisions lock, patched in a throwaway copy |

## Host

`lscpu | head -20`, `nproc`, `git -C <checkout> rev-parse HEAD`:

```
Model name:  AMD Ryzen AI 9 HX 470 w/ Radeon 890M
CPU(s):      24          Thread(s) per core: 2      Core(s) per socket: 12
Frequency boost: disabled
L3:          0-3 share 16 MiB; 4-11 share 8 MiB (SMT siblings of 4-11 are 16-23)
nproc:       8   (inside the agent's affinity mask 0-3,12-15; the host has 24)
```

Measured cores 4-11, observer 0-3,12-15. The agent was pinned there for the whole session
(`taskset -cp <pid>` → `0-3,12-15`), and every run below is a `taskset -c` pin of the measured
process to `4` (1T), `4-7` (4T) or `4-11` (8T). The harness's own idle guard checks the measured
cores *and their SMT siblings* three seconds before and after every cell; all twelve runs of the
pair passed it on attempt 1 (`pinned.sh: cpus 4-11 idle before and after (attempt 1)`), so no run
was discarded.

## Reproduction of the issue's numbers

```
scripts/record_run.py zen5-cpu faer-kernels --checkout <checkout> --upstream extern/upstream-faer \
  --engines faer-fork,faer-upstream,lapack-mkl --aa 3 --reps 5 --prime-ms 250 --vendor mkl \
  --no-publish --regimes linalg-small --families qr_factor,solve --shapes 64 \
  --dtypes f64,c64 --threads 1,4,8 --label i70-base     # and --label i70-cand
```

Runs (harness checkout `../faer-benchmark-70`, branch `harness/faer-0.25`, kept locally and not
committed):

* baseline `data/results/zen5-cpu/faer-kernels/20261110T171647Z-i70-base`
* candidate `data/results/zen5-cpu/faer-kernels/20261110T172110Z-i70-cand`
* control (`faer-upstream` row, unchanged by the diff): median ratio **1.0028** over 24 cells
  (min 0.9895, max 1.0245), so the machine reproduced between the two runs.

Statistic as in the issue: median of the three process-set medians of `total_ms`, `faer-fork` row,
candidate/baseline.

| family | order | T | 0.24.10 ms | 0.25.0 ms | ratio | issue ratio | reproduced |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qr_factor | F | 4 | 0.0735 | 0.0778 | **1.0590** | 1.057 | yes |
| qr_factor | F | 8 | 0.0736 | 0.0777 | **1.0557** | 1.042 | yes, slightly larger |
| qr_factor | F | 1 | 0.0748 | 0.0761 | 1.0180 | "within 0.99-1.01" | no, small serial drift |
| qr_factor | C | 4 | 0.0900 | 0.0926 | 1.0294 | 1.021 | yes, slightly larger |
| qr_factor | C | 8 | 0.0936 | 0.0961 | 1.0261 | 1.024 | yes |
| solve | C | 1 | 0.0316 | 0.0318 | 1.0051 | 1.030 | **no** |
| solve | C | 4 | 0.0316 | 0.0325 | 1.0295 | 1.015 | yes, larger |
| solve | C | 8 | 0.0317 | 0.0323 | 1.0193 | 1.038 | yes, smaller |
| solve | F | 1 | 0.0257 | 0.0259 | 1.0066 | 1.039 | **no** |
| solve | F | 4 | 0.0261 | 0.0269 | 1.0300 | 1.015 | yes, larger |
| solve | F | 8 | 0.0262 | 0.0269 | 1.0256 | 1.031 | yes |

Two findings from the reproduction alone:

* `qr_factor` f64 F 64x64 reproduces at both widths (1.059 vs 1.057 at 4T).
* `solve`'s **1T rows do not reproduce**: 1.005 and 1.007 here against 1.030 and 1.039 in the issue.
  The inference drawn from them — that `solve` "has at least one serial cause" — is therefore not
  supported on this host, and the phase breakdown below replaces it.

## What the candidate actually does: no parallel region at all

The probe is a small crate outside every repository that links one of the two checkouts, primes
for 250 ms and then clocks five calls inside `spindle::with_lock`, exactly as `fbench` does
(`fbench/src/main.rs` `measure_row`, `fbench/src/timing.rs` `measure`). The `spindle` 0.2.6 that
both revisions lock was copied to `/tmp` and given one counter: `for_each` records
`(n_jobs, elapsed)` when enabled. Nothing in either faer checkout was edited — `git status
--porcelain` is empty in both, before and after.

```
taskset -c 4-7 ./probe qr f 4
qr  f 4t  median_ms=0.071614  regions=10  [w4:10x3903ns]      # 5a27bce
qr  f 4t  median_ms=0.078416  regions=0                        # 0e15729
```

Four interleaved repetitions per cell (base, candidate, base, candidate, ...), medians of the five
clocked samples; `regions` is per clocked call, from the counter:

| cell | 0.24.10 regions / call | 0.24.10 ms | 0.25.0 regions / call | 0.25.0 ms |
| --- | --- | --- | --- | --- |
| qr_factor f64 F 64x64 1T | 0 | 0.0748 | 0 | 0.0759 |
| qr_factor f64 F 64x64 4T | **2** (w4, 3.9 us each) | 0.0718 | **0** | 0.0784 |
| qr_factor f64 F 64x64 8T | **2** (w8, 4.1-4.3 us each) | 0.0721-0.0731 | **0** | 0.0771-0.0774 |
| qr_factor f64 C 64x64 4T | 2 (w4, 4.1 us each) | 0.0896 | 0 | 0.0938 |
| qr_factor f64 C 64x64 8T | 2 (w8, 4.2 us each) | 0.0934 | 0 | — |
| solve f64 F 64x64 1/4/8T | 0 | — | 0 | — |
| solve f64 C 64x64 1/4/8T | 0 | — | 0 | — |

So the `InPool`-entry hypothesis is **not** what the measured cells show: at 64x64 the 0.25 build
opens **zero** regions, at 4T and at 8T, for QR and for solve. There is no per-region entry cost to
pay because no region is entered. (The harness builds the fork row with features `std, rayon`, so
the `threadteam` route — `Par::Team`, the reservation of team members — is not compiled into the
measured binary at all. `InPool` still matters, but only as the number the *decision* compares
against, below.)

The empty-section cost of one region from inside the pool, measured in the same pool under the same
pins (`probe empty 8`, 50 clocked sections after 200 warm-up, two runs):

```
empty 8t w2 median_us=2.02 / 2.15   min_us=1.57 / 1.62
empty 8t w4 median_us=2.23 / 2.11   min_us=1.82 / 1.65
empty 8t w6 median_us=2.41 / 2.30   min_us=1.64 / 1.68
empty 8t w8 median_us=2.53 / 2.33   min_us=1.93 / 1.67
```

## Where the difference comes from

The decision rule changed between the revisions, and that is the whole difference:

* **0.24.10** has no cost gate in `matmul`: `match par { Par::Seq => serial loop, Par::Rayon(n) =>
  spindle::for_each(...) }` (`faer/src/linalg/matmul/mod.rs`). A caller that passes
  `Par::Rayon(4)` parallelises every job grid it reaches.
* **0.25.0** gates every region: `pays_at(estimate, sections, dispatch_cost)` accepts only when
  `estimate >= PAYS * sections * dispatch_cost` with `PAYS = 3.0`
  (`faer/src/utils/cost.rs`), and for a caller inside the Rayon pool `dispatch_cost` is the
  `InPool` row — 3.0, 3.6, 3.5 us at widths 2, 4, 6 on `x86_64-avx512`
  (`in_pool_anchors`, and the doc comment above it). A region whose serial estimate is a couple of
  microseconds cannot clear 3 x 3.6 us, so it is declined and run inline.

Both halves of the residual are inferred by comparing each revision with its own 1T run, where the
gates are never asked ("inferred" in the sense set out under the closing paragraph below):

| quantity | 0.24.10 | 0.25.0 |
| --- | --- | --- |
| qr_factor f64 F 64x64, 1T | 0.0748 ms | 0.0759 ms |
| qr_factor f64 F 64x64, 4T | 0.0718 ms (**3.1 us faster than its own 1T**) | 0.0784 ms (**2.5 us slower than its own 1T**) |
| qr_factor f64 F 64x64, 8T | 0.0726 ms (**2.2 us faster than its own 1T**) | 0.0773 ms (**1.3 us slower than its own 1T**) |

* **2.2-3.1 us: the two regions 0.24.10 opened and that paid are declined.** 0.24.10's 4T run is
  3.1 us faster than its own 1T run, with 2 regions per call whose dispatch alone costs
  2 x 2.2 us = 4.4 us at width 4 — so those regions did about 7.5 us of work for a 4.4 us entry.
  0.25.0 runs the same work inline.
* **1.3-2.5 us: the gates.** 0.25.0's 4T run is 2.5 us *slower* than its own 1T run, and the
  only thing 4T does that 1T does not is ask the gates (`Par::Seq` short-circuits on
  `par.degree() < 2`, `Par::Rayon(4)` evaluates them). Enabling the candidate's own `par-trace`
  feature counts the traced decisions in one clocked call:
  `decisions=26 parallel_yes=0` at 4T and `decisions=12 parallel_yes=0` at 8T, all from the
  `upgrade_above_floor` site, against `decisions=0` at 1T. That counter covers the `par_pays` and
  `join_pays` sites only, so 26 is a lower bound on the decisions a 64x64 QR asks. The bypass below
  puts the gate at 1.24 us at 4T and 0.52 us at 8T, so this differential overstates it.
* **1.1 us: a serial drift that is not attributed here.** 0.25.0's 1T run is 1.1 us slower than
  0.24.10's in the probe and 1.3 us slower in the harness (1.5% and 1.8%). At 1T the gates are not
  asked at all, so this is a change in the serial path or code layout, not a region cost. It is the
  one open question this record leaves.

This is a **differential attribution, not a measured decomposition**. Each term is a difference
taken along one axis between the two revisions, and the three are chained, so
`(c1 - b1) + (b1 - b4) + (c4 - c1)` equals `c4 - b4` for any numbers, whatever caused them. So
1.1 + 3.1 + 2.5 = 6.7 us against the probe's measured 0.0784 - 0.0718 = 6.6 us at 4T, and 1.1 +
2.2 + 1.3 = 4.6 us against 0.0773 - 0.0726 = 4.7 us at 8T, agree up to the rounding of each term,
and would agree for any attribution whatsoever. What the measurements do support separately is the
*axis*: the 4T-minus-1T difference lies on the code path the revisions differ in (the gate), and its
cost is read off the timings, not from the decision counter.

### The gate, measured at the same width

The comparison that turns the gate term from an inference into a measurement is a bypass at the
same width, and it was run afterwards. `faer/src/utils/cost.rs` in the candidate was given a
temporary env-gated override at its three gate entry points (`par_pays`, `join_pays`,
`par_can_pay_time`): `FAER_GATE=0` answers "no" without computing the estimate, the shape or the
comparison, `FAER_GATE=1` answers "yes", unset runs the rule. At these sizes the rule already
answers "no" everywhere, so `unset` and `FAER_GATE=0` produce the same decision and the rest of the
execution is identical: the difference between them is the gate and nothing else. One binary, one
width, interleaved:

```
# qr_factor f64 F 64x64, four interleaved repetitions each, taskset -c 4-7 / 4-11
GATE=unset 4t  0.076352  0.076632  0.076362  0.076262   median 0.076357
GATE=0     4t  0.075160  0.075681  0.075070  0.074799   median 0.075115   -> 1.24 us
GATE=unset 8t  0.075671  0.075360  0.075260  0.075090   median 0.075315
GATE=0     8t  0.075270  0.074859  0.074529  0.074729   median 0.074794   -> 0.52 us
```

**The gate costs 1.24 us at 4T and 0.52 us at 8T** for this cell, against a recorded residual of
4.3 us and 4.1 us. It is a real component and a minor one, and it is smaller than the 1.3-2.5 us
the chained differential put on it. The override was compiled into a separate `-dirty` target
directory; the clean binary the recorder built was left untouched, and the override was reverted
afterwards.

Forcing every gate to "yes" instead (`FAER_GATE=1`) takes the same cell to 0.1019 ms at 4T and
0.1021 ms at 8T, +26 us: the rule's rejection is not obviously wrong, and the residual is not
simply "the gate is too conservative". That arm forces *every* region open, which is not 0.24.10's
set, so it bounds the wrong side and is not a stand-in for the baseline.

What the bypass does not do is separate the estimate from the shape and the comparison; it removes
all three. And it leaves the remainder of the residual unattributed: 4.3 - 1.24 = 3.1 us at 4T and
4.1 - 0.52 = 3.6 us at 8T are not accounted for by any measurement here. The region counts and
0.24.10's own 4T-versus-1T contrast (0.07490 against 0.07318 ms in the same direct runs, 1.72 us)
point at the parallelism the candidate declines, but
that is an inference, and 0.24.10's own contrast is not the same quantity as what the candidate
would gain by taking those regions. The `InPool` row is on the path and is what makes the gate
decline — but the cost is the *declined parallelism plus the question*, not the entry, and only the
question has been measured.

## solve: the residual is one phase, and it is the same gate

The harness's timed region for `solve` is the composite — copy the input, allocate the factor and
its scratch, factor, split L/U, allocate the solve scratch, solve — so the probe clocks each step
(`probe solve f 4`, five clocked calls, medians in microseconds):

| phase | 0.24.10 1T | 0.24.10 4T | 0.24.10 8T | 0.25.0 1T | 0.25.0 4T | 0.25.0 8T |
| --- | --- | --- | --- | --- | --- | --- |
| copy (`to_owned`) | 1.25 | 1.24 | 1.25 | 1.27 | 1.27 | 1.25 |
| alloc (`fwd`,`bwd`,lu scratch) | 0.18 | 0.19 | 0.22 | 0.18 | 0.20 | 0.20 |
| factor (`lu_in_place`) | 19.27 | 19.45 | 19.37 | 19.37 | 19.61 | 19.32 |
| split L/U (harness code) | 1.66 | 1.64 | 1.59 | 1.62 | 1.58 | 1.58 |
| solve scratch alloc | 0.07 | 0.07 | 0.07 | 0.08 | 0.07 | 0.07 |
| **`solve_in_place_with_conj`** | 3.49 | 3.55 | 3.56 | 3.56 | **4.28** | **4.17** |
| total | 26.08 | 26.25 | 26.06 | 26.14 | 26.96 | 26.59 |

`solve` c64 C shows the same shape (`solve_in_place`: 3.76 / 3.81 / 3.82 against
3.92 / 4.63 / 4.49). Everything except the triangular solve is flat between the revisions, and the
triangular solve moves only at 4T/8T, while both revisions open **zero** regions for `solve` at
64x64 — so 0.24.10 was already serial here and the 0.25.0 call does strictly more only on the gate
path.

So the "at least one serial cause" the issue inferred for `solve` does not exist on this host: the
1T rows are flat (1.005, 1.007 in the harness; +0.2% in the probe). The `c4 - c1` / `c8 - c1`
difference of `solve_in_place` is 0.72 / 0.61 us for F and 0.71 / 0.57 us for C, and the gate is the
only code the two revisions differ by on that axis. The same-width bypass confirms it, on the
recorded cell (three interleaved repetitions each):

| cell | gate unset | `FAER_GATE=0` | gate cost | recorded residual |
| --- | --- | --- | --- | --- |
| solve f64 F 64x64 4T | 0.026529 | 0.025979 | **0.55 us** | 0.78 us |
| solve f64 F 64x64 8T | 0.026991 | 0.026800 | **0.19 us** | 0.67 us |
| solve f64 C 64x64 4T | 0.032270 | 0.032000 | **0.27 us** | 0.93 us |
| solve f64 C 64x64 8T | 0.031790 | 0.031649 | **0.14 us** | 0.61 us |

The gate is 0.14-0.55 us of a 0.61-0.93 us residual, so it accounts for 23-71% of it and is the
largest identified term; the rest is inside the process-set spread of cells this small (0.2-0.5 us).

## Limitations

* One host, one shape, one dtype per family. The 8T `qr_factor` F ratio here (1.056) is larger than
  the issue's 1.042, and the 4T one (1.059) matches.
* **The probe and the harness disagree on the *size* of the effect, and that is unexplained.** 6.6 us
  against 4.3 us at 4T and 4.7 us against 4.1 us at 8T; the 1T serial residual disagrees too, 1.1 us
  in the probe against 1.3 us in the harness. A heavier fixture that dilutes a relative overhead
  leaves the absolute difference where it is, so it does not explain the gap, and no other cause was
  isolated here. The two agree on the sign of the effect, on its phase (for `solve`) and on the
  ordering of the terms. The one absence they agree on is the absence of **parallel-region entry**:
  neither revision opens a region in these 64x64 cells (counter above), which says nothing about a
  flat 1T run — the 1T residual is real and still unattributed. The harness number is the one to
  quote as the residual and the probe's is the one that decomposes it, but the residual's size is
  known to two values, not one.
* **Build-to-build and session-to-session variation is 0.8-1.3 us for these cells**, the same order
  as the parts being discussed. The candidate built from the same tree with the (inert,
  `FAER_GATE`-unset) override is 1.30 us faster at 1T and 0.84 us at 4T than the clean candidate
  binary, interleaved three times each:
  `clean 1t 0.075621 0.075581 0.075370 / patched 1t 0.074088 0.074529 0.074278`, and the
  candidate's 4T reading moved by 1.1 us between two sessions of this record (0.07796 to 0.07684).
  This is a measured contributor to the probe-versus-harness gap above, not an explanation of it:
  no build of the probe was compared with itself, and the gap is 2.3 us. It is why a difference this
  small is only resolvable within one binary, as the bypass above is.
* `par-trace` counts the `par_pays`/`join_pays` decisions only, so the decision count is a lower
  bound; the gate cost quoted above comes from the bypass, not from the counter.
* The region counter is on `spindle::for_each`, which is where both revisions dispatch a Rayon-route
  region. It cannot see a region the 0.25 build would open through `threadteam`, because the
  measured binary does not enable that feature.
* Nothing here says the 0.25 floors are wrong. A 4 us region that pays 3.1 us net is exactly the
  case a `PAYS = 3` rule is meant to be conservative about; whether the `InPool` row overstates the
  cost for these sizes, or `PAYS` is too high for a region of this length, is the next question and
  is not answered here.

## Not changed

No rate, region rule, threshold or `cost.rs` table was modified, and no probe was left in any
checkout: the `FAER_GATE` override and the `spindle` counter were both reverted,
`git status --porcelain` is empty in both measured revisions, and the probe crate lives outside the
repository.
