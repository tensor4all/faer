# Cross-CCD team stall and the LU region gate (evo2): a finding, no code change

No upstream revision was measured. Host: evo2, AMD Ryzen AI MAX+ 395 (Zen 5, 16 cores in two 8-core L3
domains, one NUMA node), Linux 7.0 (EEVDF, `PREEMPT_LAZY`), boost on. Evidence: `~/work/lu-chol-opt/runs/`
on evo2 (`stall-diag`, `geometry-probe`, `fix-probe2`, `four-arm`, `rollout-chol`) and the probes under
`~/work/lu-chol-opt/probes/`.

## What was established

- **Symptom.** With a team spanning both L3 domains (9-14 threads), a fraction of processes run every faer
  call 40-260x slower (LU n = 128 at 11T: 11.99 ms against 0.060 ms sequential). Stalled times are
  multiples of about 3.0 ms. Within one domain (T <= 8) the effect is absent or isolated single calls.
  MKL in the same processes and placements: no stalls; OpenBLAS: fewer, larger ones.
- **Not memory.** Zero minor and major page faults inside stalled calls; every team member reading the
  whole matrix before each call changes nothing (25/30 stalled calls either way). One NUMA node.
- **Not acquisition or thread creation.** A team reserved once and held stalls more (20/20 rounds with a
  60 ms gap between rounds) than one re-acquired per round; `spawned_workers()` does not grow; the
  harness never acquires inside a timed call.
- **Mechanism.** Stalled calls carry involuntary context switches (about one per 3 ms) and almost no
  voluntary ones; the leader never sleeps in a join or barrier (`wait_counters`: 0). Stalled rounds
  follow member parks one to one (`member_parks` about 11 per gap, 0-38 with no gap). Sampling thread
  states every millisecond shows two runnable team threads on one CPU for seconds while CPUs of the
  other domain sit idle. Reading: members woken after a park are placed onto busy CPUs, and since every
  participant spins, the stack persists and each join waits a scheduler slice. Which kernel rule leaves
  the stack in place was not identified (no root, no tracing): preferred-core ranking favours the other
  domain, and the kernel has no LLC-aggregation scheduling.
- **Fix, measured (caller side).** One CPU per participant: members by `pin_participants(true)` and the
  leader pinned to `S[0]` inside the root scope (members read the leader's full mask at scope entry;
  pinning the leader first hands every member a one-CPU mask, which the first probe did by mistake).
  LU 11T: n = 128 11.99 -> 0.071 ms, n = 256 33.0 -> 0.310 ms, 0/4 processes stalled; 4T and 8T within
  1 %. Four arms at 11T (n = 128 / 256): nothing pinned 11.99 / 33.0; members and leader 0.071 / 0.310;
  members only (the shipped option) 3.02 (2 of 4 processes stalled) / 0.313; leader only 11.98 / 15.6.
  Both are needed. Cholesky shows the same stall and the same cure (11T n = 256: 5.92 -> 0.131 ms), so the
  mechanism is the team's, not LU's.
- **Section cost across domains.** Back-to-back team sections with 10 us shares: 0.48-0.84 us within one
  domain (the `x86_64-avx512` row says 0.60-0.98, measured on an HX 470 up to width 8), 1.5-3.2 us once a
  section includes a member of the other domain, against 1.1-1.7 us that the row extrapolates.

## Rejected: an argmin-time gate for the LU regions

Trailing update and swap stage gated by `argmin_w estimate / w + PAYS * sections * cost(w)` (sequential when
no width beats the serial time), instead of the widest width that affords its dispatches. Team time against
the current rule, f64, held team: 4T and 8T 0.99-1.03 at n = 128-4096; 11T and 14T 0.89-1.12, as often
worse as better. Not adopted. At small n the regions this gate controls are not the only ones opened (the
products and solves have their own gates), and above 8 threads the dispatch row underprices cross-domain
sections by about 2x, which no gate rule corrects.

## What remains unverified

- Whether the stack also forms on other two-domain hosts (Zen 4/5 desktops, EPYC CCDs) and on macOS.
- A library-side form of the fix (threadteam pinning the leader for the root scope, or `team::with_team`
  doing so) and its cost when the caller's thread later calls a vendor library inside the scope.
- A cross-domain dispatch row for this host class.
