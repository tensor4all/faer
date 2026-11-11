#!/usr/bin/env python3
"""tensor4all/faer#391 increment 3 step 1: before/after runner (one cell per process).

usage: run.py <out.csv> <variant=binary>... [--reps 5]

Every (dtype, n, T) cell runs each variant `reps` times, the variants interleaved per rep
(A B A B ...), each in its own process. 1T on CPU 0, 8T on CPUs 0-7 (two 4-core CCXs of the
EPYC 7702P; no SMT). Before every process the measured domain (CPUs 0-15, i.e. the measured
CCXs and their neighbours) must be idle: under 3% busy over 300 ms, otherwise the runner waits;
the waits are logged, never a result retry.
"""
import os, subprocess, sys, time, itertools

MEASURED = list(range(16))


def busy(cpus, dt=0.3):
    def snap():
        out = {}
        for line in open('/proc/stat'):
            if line.startswith('cpu') and line[3].isdigit():
                f = line.split()
                c = int(f[0][3:])
                v = list(map(int, f[1:]))
                idle = v[3] + v[4]
                out[c] = (sum(v), idle)
        return out
    a = snap(); time.sleep(dt); b = snap()
    worst = 0.0
    for c in cpus:
        tot = b[c][0] - a[c][0]; idle = b[c][1] - a[c][1]
        if tot > 0:
            worst = max(worst, 1 - idle / tot)
    return worst


def gate(log):
    waited = 0
    while True:
        w = busy(MEASURED)
        if w < 0.03:
            return waited
        waited += 1
        log.write(f'# gate: busy {w:.2f}, waiting\n'); log.flush()
        time.sleep(2)


def main():
    out = sys.argv[1]
    reps = 5
    variants = []
    for a in sys.argv[2:]:
        if a.startswith('--reps='):
            reps = int(a.split('=')[1])
        else:
            k, v = a.split('=', 1)
            variants.append((k, v))
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith('RAYON_') or k.startswith('OMP_') or k.startswith('MKL_'))}
    cells = list(itertools.product(['f64', 'c64'], [64, 96, 128], [1, 8]))
    with open(out, 'w') as log:
        log.write('variant,dtype,n,T,rep,key,level,phase,value\n')
        for (dtype, n, t) in cells:
            for rep in range(reps):
                order = variants if rep % 2 == 0 else variants[::-1]
                for (name, binary) in order:
                    gate(log)
                    cpus = '0' if t == 1 else '0-7'
                    r = subprocess.run(['taskset', '-c', cpus, binary, 'stage2', dtype, str(n), str(t), '16'],
                                       capture_output=True, text=True, env=env)
                    if r.returncode != 0:
                        raise SystemExit(f'{name} {dtype} {n} {t}: {r.stderr}')
                    after = busy(MEASURED)
                    log.write(f'{name},{dtype},{n},{t},{rep},busy_after,,,{after:.3f}\n')
                    for line in r.stdout.splitlines():
                        log.write(f'{name},{dtype},{n},{t},{rep},{line}\n')
                    log.flush()
            print(dtype, n, t, 'done', flush=True)


main()
