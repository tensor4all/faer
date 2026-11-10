#!/usr/bin/env python3
"""tensor4all/faer#391 increment 1: recompute every table of the worklog from the raw CSV.

usage: python3 aggregate.py phases-raw.csv [counts-raw.csv [revision-raw.csv]]
Every value is the median over the 5 processes of the per-call value one process reports.
"""
import csv, sys, statistics as st
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
v = defaultdict(list)  # (build,mode,dtype,n,threads,thr,key,level,phase) -> [value per rep]
per_rep = defaultdict(float)  # (build,mode,dtype,n,threads,thr,rep) -> sum of all ns rows
for r in rows:
    cell = (r['build'], r['mode'], r['dtype'], int(r['n']), int(r['threads']), r['thr'])
    v[cell + (r['key'], r['level'], r['phase'])].append(float(r['value']))
    if r['key'] == 'ns':
        per_rep[cell + (r['rep'],)] += float(r['value'])

def med(cell, key, level='', phase=''):
    x = v.get(cell + (key, str(level), phase))
    return st.median(x) if x else 0.0

GROUPS = [
    ('(a) secular', ['secular']),
    ('(b) deflation+perm', ['z_build', 'merge_perm', 'deflate', 'sort_perm']),
    ('(c) rotations', ['lowner', 'vectors', 'hh_apply']),
    ('(d) gemm', ['gemm']),
    ("(d') copy-out", ['copy_out']),
    ('(e) leaf QR', ['leaf']),
    ('other', ['split', 'outer_pre', 'dc_wrap', 'outer_post']),
]
MS = 1e-6

def levels(cell):
    return sorted({int(k[7]) for k in v if k[:6] == cell and k[6] == 'ns'})

def group_ms(cell, level, names):
    return sum(med(cell, 'ns', level, p) for p in names) * MS

print('## coverage (instrumented phase sum vs uninstrumented call), ms, median of 5 processes')
print('| dtype | n | T | thr | phase sum (instr) | call (instr) | call (pristine) | sum/pristine - 1 |')
print('| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |')
for d in ('f64', 'c64'):
    for n in (64, 96, 128):
        for t in (1, 8):
            for thr in ('16', 'default'):
                ci = ('instr', 'stage2', d, n, t, thr)
                cb = ('base', 'stage2', d, n, t, thr)
                sums = [per_rep[ci + (str(r),)] for r in range(1, 6) if ci + (str(r),) in per_rep]
                if not sums:
                    continue
                s = st.median(sums) * MS
                ci_call = med(ci, 'call_mean_ns') * MS
                cb_call = med(cb, 'call_mean_ns') * MS
                print(f'| {d} | {n} | {t} | {thr} | {s:.3f} | {ci_call:.3f} | {cb_call:.3f} | {100*(s/cb_call-1):+.1f}% |')

print('\n## phase table, ms per call (thread-time summed over the nodes of a level), median of 5 processes')
for d in ('f64', 'c64'):
    for n in (64, 96, 128):
        for t in (1, 8):
            for thr in ('16', 'default'):
                ci = ('instr', 'stage2', d, n, t, thr)
                if not levels(ci):
                    continue
                print(f'\n### {d} n={n} {t}T thr={thr}  (pristine call {med(("base",)+ci[1:], "call_mean_ns")*MS:.3f} ms)')
                print('| level | nodes | mean k | ' + ' | '.join(g for g, _ in GROUPS) + ' | level sum | halves par/seq | gemm par/seq |')
                print('| ---: | ---: | ---: | ' + ' | '.join('---:' for _ in GROUPS) + ' | ---: | --- | --- |')
                tot = [0.0] * len(GROUPS)
                for l in levels(ci):
                    nodes = max(med(ci, 'nodes', l, p) for p in ('split', 'leaf', 'secular', 'outer_pre'))
                    merges = med(ci, 'nodes', l, 'secular')
                    k = med(ci, 'k_sum', l) / merges if merges else 0
                    g = [group_ms(ci, l, names) for _, names in GROUPS]
                    tot = [a + b for a, b in zip(tot, g)]
                    hp = f"{med(ci,'halves_par',l):.0f}/{med(ci,'halves_seq',l):.0f}"
                    gp = f"{med(ci,'gemm_par',l):.0f}/{med(ci,'gemm_seq',l):.0f}"
                    print(f'| {l} | {nodes:.0f} | {k:.1f} | ' + ' | '.join(f'{x:.3f}' for x in g) + f' | {sum(g):.3f} | {hp} | {gp} |')
                s = sum(tot)
                print('| all | | | ' + ' | '.join(f'{x:.3f}' for x in tot) + f' | {s:.3f} | | |')
                print('| share | | | ' + ' | '.join(f'{100*x/s:.0f}%' for x in tot) + ' | 100% | | |')

print('\n## 8T critical path estimate: sum over levels of (level time / nodes) x serial multiplicity, ms')
print('| dtype | n | thr | (a) | (b) | (c) | (d) | (d\') | (e) | other | estimate | measured call |')
print('| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for d in ('f64', 'c64'):
    for n in (64, 96, 128):
        thr = '16'
        ci = ('instr', 'stage2', d, n, 8, thr)
        mult = 1.0
        acc = [0.0] * len(GROUPS)
        for l in levels(ci):
            for gi, (_, names) in enumerate(GROUPS):
                for p in names:
                    nd = med(ci, 'nodes', l, p)
                    if nd:
                        acc[gi] += mult * med(ci, 'ns', l, p) / (1 if p in ('outer_pre', 'outer_post', 'dc_wrap') else nd) * MS
            hp, hs = med(ci, 'halves_par', l), med(ci, 'halves_seq', l)
            if hp + hs:
                mult *= (hp + 2 * hs) / (hp + hs)
        print(f'| {d} | {n} | {thr} | ' + ' | '.join(f'{x:.3f}' for x in acc) + f' | {sum(acc):.3f} | {med(ci, "call_mean_ns")*MS:.3f} |')

print('\n## leaf: QR alone (recursion_threshold = usize::MAX), f64, 1T, ms per call')
print('| n | pristine call | instr leaf phase | per n^2 (ns) | per n^3 (ns) |')
print('| ---: | ---: | ---: | ---: | ---: |')
for n in (8, 12, 16, 24, 32, 48, 64, 96, 128):
    cb = ('base', 'leaf', 'f64', n, 1, 'default')
    ci = ('instr', 'leaf', 'f64', n, 1, 'default')
    c = med(cb, 'call_mean_ns')
    print(f'| {n} | {c*MS:.4f} | {med(ci, "ns", 0, "leaf")*MS:.4f} | {c/n**2:.1f} | {c/n**3:.2f} |')

print('\n## leaves inside the recursion (thr=16, 1T): count and mean time per leaf, ms')
for d in ('f64', 'c64'):
    for n in (64, 96, 128):
        ci = ('instr', 'stage2', d, n, 1, '16')
        sizes = sorted({int(k[7]) for k in v if k[:6] == ci and k[6] == 'leaf_size'})
        tot_leaf = sum(med(ci, 'ns', l, 'leaf') for l in levels(ci)) * MS
        cnt = sum(med(ci, 'leaf_size', s) for s in sizes)
        print(f'{d} n={n}: leaves {dict((s, med(ci, "leaf_size", s)) for s in sizes)}, leaf total {tot_leaf:.3f} ms, per leaf {tot_leaf/cnt*1e3:.1f} us')

print('\n## allocations per call (whole stage 2), median of 5 processes')
for b in ('base', 'instr'):
    for d in ('f64', 'c64'):
        line = []
        for n in (64, 96, 128):
            for t in (1, 8):
                for thr in ('16', 'default'):
                    line.append(f'{n}/{t}T/{thr}={med((b, "stage2", d, n, t, thr), "allocs_per_call"):.2f}')
        print(b, d, ' '.join(line))
print('per (level, phase) nonzero allocation counts (instr):')
for k, x in sorted(v.items()):
    if k[0] == 'instr' and k[6] == 'allocs' and st.median(x) != 0:
        print(k, st.median(x))

if len(sys.argv) > 2:
    crow = list(csv.DictReader(open(sys.argv[2])))
    cv = defaultdict(list)
    for r in crow:
        cv[(r['dtype'], int(r['n']), r['thr'], r['key'], int(r['level']))].append(float(r['value']))
    print('\n## secular-equation work per level (count-only build, 1T), per call')
    print('| dtype | n | thr | level | roots | O(k) passes | passes/root | bisection roots | bisection passes | terms (k x passes) |')
    print('| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for d in ('f64', 'c64'):
        for n in (64, 96, 128):
            for thr in ('16', 'default'):
                ls = sorted({k[4] for k in cv if k[:3] == (d, n, thr) and k[3] == 'sec_roots'})
                for l in ls:
                    g = lambda key: st.median(cv.get((d, n, thr, key, l), [0]))
                    roots, ev, ba = g('sec_roots'), g('sec_evals'), g('sec_batch')
                    passes = ev + ba  # a batch is one pass over k with 4 lanes
                    print(f'| {d} | {n} | {thr} | {l} | {roots:.0f} | {passes:.0f} | {passes/roots:.1f} | {g("sec_bisect_roots"):.0f} | {g("sec_bisect_evals"):.0f} | {g("sec_terms"):.0f} |')

if len(sys.argv) > 3:
    rv = defaultdict(list)
    for r in csv.DictReader(open(sys.argv[3])):
        if r['key'] == 'call_mean_ns':
            rv[(r['build'], r['dtype'], int(r['n']), int(r['threads']))].append(float(r['value']) * MS)
    print('\n## stage 2 at recursion_threshold = 16, unmodified: record revision 1ac38c6d vs main 859feef7, ms')
    print('| dtype | n | T | 1ac38c6d | 859feef7 | main / record rev |')
    print('| --- | ---: | ---: | ---: | ---: | ---: |')
    for d in ('f64', 'c64'):
        for n in (64, 96, 128):
            for t in (1, 8):
                a, b = st.median(rv[('rec', d, n, t)]), st.median(rv[('base', d, n, t)])
                print(f'| {d} | {n} | {t} | {a:.3f} | {b:.3f} | {b/a:.3f} |')

print('\n## worklog table A: recursion_threshold = 16, per level, ms per call (thread-time), median of 5 processes')
print("| dtype | n | T | level | nodes x k | (a) secular | (b) deflation+perm | (c) rotations | (d) GEMM | (d') copy-out | (e) leaf QR | other | level sum |")
print('| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for d in ('f64', 'c64'):
    for n in (64, 96, 128):
        for t in (1, 8):
            c = ('instr', 'stage2', d, n, t, '16')
            tot = [0.0] * len(GROUPS)
            for l in levels(c):
                g = [group_ms(c, l, names) for _, names in GROUPS]
                tot = [a + b for a, b in zip(tot, g)]
                mg = med(c, 'nodes', l, 'secular')
                nk = f"{mg:.0f} x {med(c, 'k_sum', l) / mg:.0f}" if mg else f"{med(c, 'nodes', l, 'leaf'):.0f} leaves of {n >> l}"
                print(f'| {d} | {n} | {t} | {l} | {nk} | ' + ' | '.join(f'{x:.3f}' for x in g) + f' | {sum(g):.3f} |')
            s = sum(tot)
            print(f'| {d} | {n} | {t} | **all** | | ' + ' | '.join(f'**{x:.3f}** ({100*x/s:.0f}%)' for x in tot) + f' | **{s:.3f}** |')
