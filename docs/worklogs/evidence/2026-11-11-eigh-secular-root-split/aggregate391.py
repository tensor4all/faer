#!/usr/bin/env python3
"""Aggregate run.py output. usage: aggregate391.py <pristine.csv> <instr.csv>

All timings in ms per call, median of the processes of a cell (one cell per process).
Critical path (as in docs/worklogs/2026-11-11-eigh-dc-merge-phases.md table B): for each
level, the per-node time of a phase (thread-time / nodes) times how many nodes of that level
run one after another (1 when every join above it ran its halves in parallel), summed.
"""
import csv, statistics, sys
from collections import defaultdict

GROUPS = {
    'a': ['secular'],
    'b': ['z_build', 'merge_perm', 'deflate', 'sort_perm'],
    'c': ['lowner', 'vectors', 'hh_apply'],
    'd': ['gemm'],
    "d'": ['copy_out'],
    'e': ['leaf'],
    'other': ['split', 'outer_pre', 'dc_wrap', 'outer_post'],
}
# phases timed around the whole solve on the calling thread (instrumentation.patch: `outer_pre` and
# `outer_post` once per call, `dc_wrap` twice: the set-up before the recursion and the rescale after
# it); every other phase is lapped once per recursion node
WRAPPER = {'outer_pre', 'dc_wrap', 'outer_post'}
CELLS = [(d, n, t) for d in ['f64', 'c64'] for n in [64, 96, 128] for t in [1, 8]]


DROPPED = defaultdict(int)
KEPT = defaultdict(int)  # (file, variant, dtype, n, T) -> processes the medians are taken over


def load(path):
    rows = defaultdict(list)  # (variant,dtype,n,T,key,level,phase) -> [values per rep]
    with open(path) as f:
        recs = [r for r in csv.reader(l for l in f if not l.startswith('#')) if r[0] != 'variant']
    # a process after which the measured CPUs 0-15 were >= 3% busy is dropped (whole process)
    bad = {tuple(r[:5]) for r in recs if r[5] == 'busy_after' and float(r[8]) >= 0.03}
    for r in recs:
        if tuple(r[:5]) in bad:
            if r[5] == 'calls':
                DROPPED[(path, r[0], r[1], int(r[2]), int(r[3]))] += 1
            continue
        v, d, n, t, rep, key, level, phase, value = r
        if key == 'calls':
            KEPT[(path, v, d, int(n), int(t))] += 1
        rows[(v, d, int(n), int(t), key, level, phase)].append(value)
    return rows


def med(rows, *k):
    xs = rows.get(k)
    if not xs:
        return None
    return statistics.median(float(x) for x in xs)


def kept_table():
    print()
    print('## processes per cell (the medians above are over these; 5 started per cell)')
    files = sorted({k[0] for k in KEPT})
    print('| dtype | n | T | ' + ' | '.join(f'{f} {v}' for f in files for v in ('main', 'branch')) + ' |')
    print('| --- | ---: | ---: |' + ' ---: |' * (2 * len(files)))
    for d, n, t in CELLS:
        print(f'| {d} | {n} | {t} | ' + ' | '.join(str(KEPT[(f, v, d, n, t)]) for f in files for v in ('base', 'head')) + ' |')


def main():
    pr = load(sys.argv[1])
    ins = load(sys.argv[2]) if len(sys.argv) > 2 else None
    print('## call time, pristine builds (ms, median of 5 processes; mean per call in each process)')
    print('| dtype | n | T | main | branch | branch / main | spread main (min-max) | spread branch |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |')
    for d, n, t in CELLS:
        b = med(pr, 'base', d, n, t, 'call_mean_ns', '', '') / 1e6
        h = med(pr, 'head', d, n, t, 'call_mean_ns', '', '') / 1e6
        sb = [float(x) / 1e6 for x in pr[('base', d, n, t, 'call_mean_ns', '', '')]]
        sh = [float(x) / 1e6 for x in pr[('head', d, n, t, 'call_mean_ns', '', '')]]
        print(f'| {d} | {n} | {t} | {b:.3f} | {h:.3f} | {h / b:.3f} | {min(sb):.3f}-{max(sb):.3f} | {min(sh):.3f}-{max(sh):.3f} |')
    print()
    print('## accuracy and output bits (last call of each process)')
    print('| dtype | n | T | residual / (n eps |T|) main | branch | orthogonality / (n eps) main | branch | digests main | digests branch | same bits |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |')
    for d, n, t in CELLS:
        def one(v, key):
            xs = set(pr[(v, d, n, t, key, '', '')])
            return xs
        rb, rh = one('base', 'residual_neps'), one('head', 'residual_neps')
        ob, oh = one('base', 'orth_neps'), one('head', 'orth_neps')
        gb, gh = one('base', 'digest'), one('head', 'digest')
        print(f"| {d} | {n} | {t} | {'/'.join(sorted(rb))} | {'/'.join(sorted(rh))} | {'/'.join(sorted(ob))} | {'/'.join(sorted(oh))} | {'/'.join(sorted(gb))} | {'/'.join(sorted(gh))} | {'yes' if gb == gh and len(gb) == 1 else 'NO'} |")
    print()
    print('allocations per call (max over processes):',
          max(float(x) for k, xs in pr.items() if k[4] == 'allocs_per_call' for x in xs))
    if not ins:
        print('dropped processes (busy after):', dict(DROPPED))
        kept_table()
        return
    print()
    print('## critical path by phase group, instrumented builds (ms)')
    print('| dtype | n | T | variant | ' + ' | '.join(f'({g})' for g in GROUPS) + ' | estimate | instr call | top-level (a) |')
    print('| --- | ---: | ---: | --- |' + ' ---: |' * (len(GROUPS) + 3))
    for d, n, t in CELLS:
        for v in ['base', 'head']:
            levels = sorted({int(k[5]) for k in ins if k[:4] == (v, d, n, t) and k[4] == 'ns'})
            # serial factor per level from the joins above it
            factor = {}
            f = 1.0
            for l in levels:
                factor[l] = f
                hp = med(ins, v, d, n, t, 'halves_par', str(l), '') or 0.0
                hs = med(ins, v, d, n, t, 'halves_seq', str(l), '') or 0.0
                if hp + hs > 0:
                    f *= 1.0 + hs / (hp + hs)
            tot = {}
            for g, phases in GROUPS.items():
                s = 0.0
                for l in levels:
                    for p in phases:
                        ns = med(ins, v, d, n, t, 'ns', str(l), p)
                        nodes = med(ins, v, d, n, t, 'nodes', str(l), p)
                        if ns is None or not nodes:
                            continue
                        # a wrapper phase's laps are sequential on the calling thread, never
                        # concurrent nodes: they are summed, not divided by their count
                        if t > 1 and p not in WRAPPER:
                            per = ns / nodes
                        else:
                            per = ns
                        s += per * (factor[l] if t > 1 else 1.0)
                tot[g] = s / 1e6
            est = sum(tot.values())
            call = med(ins, v, d, n, t, 'call_mean_ns', '', '') / 1e6
            top = (med(ins, v, d, n, t, 'ns', '0', 'secular') or 0) / 1e6
            name = 'main' if v == 'base' else 'branch'
            print(f'| {d} | {n} | {t} | {name} | ' + ' | '.join(f'{tot[g]:.3f}' for g in GROUPS) + f' | {est:.3f} | {call:.3f} | {top:.3f} |')
    print()
    print('## secular phase per level, per node (ms), instrumented')
    print('| dtype | n | T | level | nodes | main | branch | branch / main |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for d, n, t in CELLS:
        levels = sorted({int(k[5]) for k in ins if k[:4] == ('base', d, n, t) and k[6] == 'secular'})
        for l in levels:
            def per(v):
                ns = med(ins, v, d, n, t, 'ns', str(l), 'secular')
                nodes = med(ins, v, d, n, t, 'nodes', str(l), 'secular')
                return ns / nodes / 1e6, nodes
            b, nb = per('base')
            h, _ = per('head')
            print(f'| {d} | {n} | {t} | {l} | {nb:.0f} | {b:.4f} | {h:.4f} | {h / b:.3f} |')
    print()
    print('dropped processes (busy after):', dict(DROPPED))
    kept_table()


main()
