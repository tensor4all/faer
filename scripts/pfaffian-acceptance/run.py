#!/usr/bin/env python3
"""Bounded faer#176 experiment; all outputs and rejected guards are retained."""
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent
BENCH = Path.home() / 'projects/tensor4all/faer-benchmark'
BASE = 'e765e0a9702cb355e421d8e77ba5f20a82cf0ea1'
PFA = '5ad8a9b11020db5ac6acc9705408213ee67a1294'
SIZES = (2, 64, 128, 256, 512, 1024)
TYPES = ('f64', 'c64')


def command(args, **kw):
    return subprocess.run([str(a) for a in args], check=True, **kw)


def output(args, **kw):
    return subprocess.check_output([str(a) for a in args], text=True, **kw).strip()


def cpu_set(s):
    out = set()
    for piece in s.split(','):
        ends = [int(x) for x in piece.split('-')]
        if len(ends) == 1:
            out.add(ends[0])
        elif len(ends) == 2 and ends[0] <= ends[1]:
            out.update(range(ends[0], ends[1] + 1))
        else:
            raise ValueError('invalid CPU list: ' + s)
    return out


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def perf_preflight(out):
    with (out / 'perf-preflight.log').open('w') as log:
        if shutil.which('perf') is None:
            log.write('perf executable not found\n')
            available = False
        else:
            available = subprocess.run(['perf', 'stat', '-e', 'cycles', '--', 'true'], stdout=log, stderr=subprocess.STDOUT).returncode == 0
    if not available:
        try:
            paranoid = Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip()
        except OSError:
            paranoid = 'unknown'
        status = f'perf unavailable (perf_event_paranoid={paranoid})'
        print(status, flush=True)
        require(os.environ.get('T4A_ALLOW_NO_PERF') == '1', status + '; see perf-preflight.log')
        return status
    return 'perf available'


def finish(out, perf_status, guard, bins, env):
    # Sampling runs are separate from timings. Never treat cache misses as DRAM bytes.
    if perf_status == 'perf available':
        matrix = out / 'matrices/n1024-f64.bin'
        for arm in ('base', 'b16', 'b32'):
            tag = 'profile-' + arm
            with (out / (tag + '.stderr')).open('w') as err:
                run = subprocess.run(['bash', str(guard), '4-11', '--', 'perf', 'record', '-g', '-o', str(out / (tag + '.data')), '--', str(bins[arm]), str(matrix), '1024', 'f64', '8'], env=env, text=True, stdout=subprocess.PIPE, stderr=err)
            (out / (tag + '.stdout')).write_text(run.stdout)
            require(run.returncode == 0, f'{tag} failed or guard rejected; no profile-based conclusion')
            with (out / (tag + '.txt')).open('w') as report:
                command(['perf', 'report', '--stdio', '--no-children', '-i', out / (tag + '.data')], env=env, stdout=report, stderr=subprocess.STDOUT)
    profile = 'See profile-*.txt; inspect for permutation work.' if perf_status == 'perf available' else 'not measured'
    (out / 'REPORT.md').write_text(f'# Pfaffian #176 evidence\n\nSee summary.csv, samples.csv, manifest.json and correctness.json. Timing rows report threshold comparisons, not an adoption decision. The n64/n1024 Seq-versus-base ratios are regression guards requiring review.\n\n## Profile\n{profile}\n\n## Bandwidth\nnot measured\n')
    print('See', out / 'summary.csv', flush=True)
    if perf_status == 'perf available':
        raise RuntimeError('timing and profiles complete, but measured DRAM bandwidth and manual acceptance review remain outstanding')
    print('Wall-clock timing complete; profile and bandwidth not measured; manual acceptance review remains outstanding.', flush=True)


def main():
    require(len(sys.argv) == 2, 'usage: run_acceptance.sh NEW_OUTPUT_DIRECTORY')
    out = Path(sys.argv[1]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    print('output:', out, flush=True)
    # Never time on any other host or use two L3s for the eight participants.
    model = Path('/proc/cpuinfo').read_text()
    require('AMD Ryzen AI 9 HX 470' in model, 'not the declared zen5-cpu profile')
    measured = set(range(4, 12))
    affinity = set(os.sched_getaffinity(0))
    l3 = cpu_set(Path('/sys/devices/system/cpu/cpu4/cache/index3/shared_cpu_list').read_text().strip())
    require(measured <= l3 and not affinity & l3, f'observer affinity {sorted(affinity)} overlaps measured L3 {sorted(l3)}')
    require(all(cpu_set(Path(f'/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list').read_text().strip()) == l3 for c in measured), 'selected CPUs span L3 domains')
    cores = {Path(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read_text().strip() for c in measured}
    require(len(cores) == 8, 'not eight physical cores')
    boost = Path('/sys/devices/system/cpu/cpufreq/boost')
    require(boost.is_file() and boost.read_text().strip() == '0', 'CPU boost must be confirmed off before running; do not alter it here')
    require(BENCH.is_dir(), f'missing campaign repository: {BENCH}')
    perf_status = perf_preflight(out)
    guard = BENCH / 'benchmarks/scripts/pinned.sh'
    require(guard.is_file(), 'missing pinned.sh')
    dirty = output(['git', '-C', ROOT, 'status', '--porcelain', '--untracked-files=no'])
    require(not dirty, 'candidate checkout has tracked changes; commit or use a clean checkout')
    head = output(['git', '-C', ROOT, 'rev-parse', 'HEAD'])
    require(output(['git', '-C', ROOT, 'merge-base', '--is-ancestor', BASE, head], stderr=subprocess.STDOUT) == '', 'candidate does not descend from #177')
    # The campaign's lock serializes build, verification, profiling and timing.
    lockfile = BENCH / 'target/campaign.lock'
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lock = lockfile.open('a+')
    deadline = time.monotonic() + 600
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            require(time.monotonic() < deadline, 'campaign lock unavailable for 600 seconds')
            time.sleep(5)
    env = dict(os.environ)
    env.pop('CARGO_TARGET_DIR', None)
    env.update(TMPDIR=str(out), CARGO_BUILD_JOBS='4')
    meta = {'candidate_sha': head, 'base_sha': BASE, 'pfapack_sha': PFA,
            'benchmark_sha': output(['git', '-C', BENCH, 'rev-parse', 'HEAD']),
            'guard_sha256': digest(guard), 'idle_guard_sha256': digest(guard.parent / 'idle_cpus.py'),
            'measured_cpus': sorted(measured), 'measured_l3': sorted(l3), 'observer_affinity': sorted(affinity), 'cpufreq_boost': boost.read_text().strip(),
            'cpufreq_governor': output(['bash', '-c', 'cat /sys/devices/system/cpu/cpu{4,5,6,7,8,9,10,11}/cpufreq/scaling_governor']).split(),
            'rustc': output(['rustc', '-Vv']), 'cargo': output(['cargo', '-V']),
            'gfortran': output(['gfortran', '--version']).splitlines()[0],
            'env': {k: env.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'GOTO_NUM_THREADS', 'MKLROOT', 'RUSTFLAGS')},
            'perf_status': perf_status, 'allow_no_perf': os.environ.get('T4A_ALLOW_NO_PERF') == '1',
            'prime_seconds': .5, 'warmups': 3, 'samples': 10, 'sessions': 3,
            'protocol': 'custom Pfaffian public-call diagnostic; median of process medians; 1 guarded attempt'}
    (out / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('building isolated copies', flush=True)
    copies = {}
    for name, revision in (
        ('base', BASE),
        ('cand', head),
        ('b16', head),
        ('b32', head),
    ):
        dest = out / ('src-' + name)
        dest.mkdir()
        archive = subprocess.Popen(['git', '-C', str(ROOT), 'archive', revision], stdout=subprocess.PIPE)
        command(['tar', '-x', '-C', dest], stdin=archive.stdout)
        archive.stdout.close()
        require(archive.wait() == 0, 'git archive failed')
        mod = dest / 'faer/src/linalg/pfaffian/mod.rs'
        # The candidate's default width is 24; the two alternates are the same
        # source at 16 and 32, so the width scan does not need two branches.
        if name in ('b16', 'b32'):
            width = name[1:]
            s = mod.read_text()
            require(s.count('const PANEL: usize = 24;') == 1,
                    f'cannot derive {name}')
            mod.write_text(s.replace('const PANEL: usize = 24;',
                                     f'const PANEL: usize = {width};'))
        copies[name] = dest
        meta[name + '_source_sha256'] = digest(mod)
    pfadir = out / 'pfapack'
    command(['git', 'clone', '--quiet', '--no-checkout', 'https://github.com/basnijholt/pfapack.git', pfadir], env=env)
    command(['git', '-C', pfadir, 'checkout', '--quiet', '--detach', PFA], env=env)
    require(output(['git', '-C', pfadir, 'rev-parse', 'HEAD']) == PFA, 'unexpected PfaPack source')
    objdir = out / 'fortran-build'
    objdir.mkdir()
    srcs = sorted((pfadir / 'original_source/fortran').glob('*.f'))
    require(bool(srcs), 'missing PfaPack sources')
    command(['gfortran', '-O3', '-fPIC', '-c', *srcs], cwd=objdir, env=env, stdout=(out / 'fortran-build.log').open('w'), stderr=subprocess.STDOUT)
    fbin = out / 'fortran-runner'
    command(['gfortran', '-O3', '-ffree-line-length-none', ASSETS / 'runner.f90', *sorted(objdir.glob('*.o')), '-lopenblas', '-o', fbin], cwd=objdir, env=env)
    bins = {'fortran': fbin}
    for name, dest in copies.items():
        project = out / ('probe-' + name)
        (project / 'src').mkdir(parents=True)
        shutil.copyfile(ASSETS / 'main.rs', project / 'src/main.rs')
        shutil.copyfile(ASSETS / 'Cargo.lock', project / 'Cargo.lock')
        (project / 'Cargo.toml').write_text('[package]\nname="pfaffian-faer-probe"\nversion="0.0.0"\nedition="2021"\n[dependencies]\nfaer={package="t4a-faer",path="' + str(dest / 'faer') + '"}\npulp={version="0.22.2",default-features=false,features=["std","x86-v3"]}\n[profile.release]\ndebug=0\n')
        with (out / (name + '-build.log')).open('w') as log:
            command(['cargo', 'build', '--release', '--locked', '-j4', '--manifest-path', project / 'Cargo.toml'], env=env, stdout=log, stderr=subprocess.STDOUT)
        bins[name] = project / 'target/release/pfaffian-faer-probe'
    meta['binaries_sha256'] = {name: digest(path) for name, path in bins.items()}
    meta['fortran_ldd'] = output(['ldd', fbin])
    match = re.search(r'libopenblas\S*\s+=>\s+(\S+)', meta['fortran_ldd'])
    require(match is not None and Path(match.group(1)).is_file(), 'could not identify linked OpenBLAS')
    meta['openblas_path'] = str(Path(match.group(1)).resolve())
    meta['openblas_sha256'] = digest(meta['openblas_path'])
    meta['faer_benchmark_status'] = output(['git', '-C', BENCH, 'status', '--porcelain', '--untracked-files=no'])
    meta['asset_sha256'] = {name: digest(ASSETS / name) for name in ('run.py', 'main.rs', 'runner.f90', 'generate.py', 'Cargo.lock')}
    (out / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    command(['python3', ASSETS / 'generate.py', out / 'matrices'], env=env)
    print('48 correctness checks per candidate', flush=True)
    references = {}
    for n in SIZES:
        for dtype in TYPES:
            matrix = out / f'matrices/n{n}-{dtype}.bin'
            row = json.loads(output(['taskset', '-c', '4', fbin, matrix, n, dtype, 1, '--smoke'], env=env))
            references[n, dtype] = complex(row['pf_re'], row['pf_im'])
    checks = []
    for name in ('base', 'cand', 'b16', 'b32'):
        for n in SIZES:
            for dtype in TYPES:
                matrix = out / f'matrices/n{n}-{dtype}.bin'
                mode_values = []
                for width in (0, 1, 4, 8):
                    row = json.loads(output(['taskset', '-c', '4-11', bins[name], matrix, n, dtype, width, '--smoke'], env=env))
                    bits = (row['pf_re'], row['pf_im'])
                    actual = complex(*bits); ref = references[n, dtype]
                    require(abs(actual - ref) <= 5e-10 * max(abs(actual), abs(ref), 1e-300), f'{name} n={n} {dtype} width={width}: numerical mismatch {actual} != {ref}')
                    mode_values.append(struct.pack('<dd', *bits))
                    checks.append({'arm': name, 'n': n, 'dtype': dtype, 'width': width, 'actual': [actual.real, actual.imag], 'reference': [ref.real, ref.imag]})
                require(all(x == mode_values[0] for x in mode_values), f'{name} n={n} {dtype}: team result differs from Seq (bitwise)')
    (out / 'correctness.json').write_text(json.dumps(checks, indent=2) + '\n')
    require(len(checks) == 192, 'incomplete correctness checks')
    rows = []
    with (out / 'samples.csv').open('w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['session', 'arm', 'n', 'dtype', 'width', 'median_seconds', 'samples_seconds', 'batch_iterations'])
        writer.writeheader(); csvfile.flush()
        for session in range(1, 4):
            for n in (64, 256, 512, 1024):
                for dtype in TYPES:
                    matrix = out / f'matrices/n{n}-{dtype}.bin'
                    schedule = [('fortran', 1), ('base', 0), ('cand', 0), ('b16', 0), ('b32', 0)] + ([] if n == 64 else [('base', 8), ('cand', 8), ('b16', 8), ('b32', 8)])
                    shift = (session - 1) * 2 % len(schedule)
                    for arm, width in schedule[shift:] + schedule[:shift]:
                        tag = f's{session}-n{n}-{dtype}-{arm}-{width}'
                        cpus = '4-11' if width == 8 else '4'
                        print(tag, flush=True)
                        require(boost.read_text().strip() == '0', 'CPU boost changed during run')
                        require(digest(bins[arm]) == meta['binaries_sha256'][arm], f'{arm} binary changed')
                        with (out / (tag + '.stderr')).open('w') as err:
                            run = subprocess.run(['bash', str(guard), cpus, '--', str(bins[arm]), str(matrix), str(n), dtype, str(width)], env=env, text=True, stdout=subprocess.PIPE, stderr=err)
                        (out / (tag + '.stdout')).write_text(run.stdout)
                        require(run.returncode == 0, f'{tag} failed or idle-guard rejected ({run.returncode}); see retained evidence; no retry')
                        require(boost.read_text().strip() == '0', 'CPU boost changed during run')
                        require(digest(bins[arm]) == meta['binaries_sha256'][arm], f'{arm} binary changed')
                        result = json.loads(run.stdout)
                        value = complex(result['pf_re'], result['pf_im']); ref = references[n, dtype]
                        require(abs(value - ref) <= 5e-10 * max(abs(value), abs(ref), 1e-300), f'{tag}: timed result mismatch')
                        require(len(result['samples_seconds']) == 10 and all(math.isfinite(x) and x > 0 for x in result['samples_seconds']), f'{tag}: invalid samples')
                        require(abs(statistics.median(result['samples_seconds']) - result['median_seconds']) <= 1e-9 * result['median_seconds'], f'{tag}: median mismatch')
                        counts = result['batch_iterations']
                        require(len(counts) == 10 and all(isinstance(ncalls, int) and ncalls > 0 for ncalls in counts), f'{tag}: invalid batch counts')
                        require(all(seconds * count >= .499 for seconds, count in zip(result['samples_seconds'], counts)), f'{tag}: timed batch shorter than 0.5 seconds')
                        row = {'session': session, 'arm': arm, 'n': n, 'dtype': dtype, 'width': width, 'median_seconds': result['median_seconds'], 'samples_seconds': json.dumps(result['samples_seconds']), 'batch_iterations': json.dumps(counts)}
                        rows.append(row); writer.writerow(row); csvfile.flush()
    summary = []
    for n in (64, 256, 512, 1024):
        for dtype in TYPES:
            def median(arm, width):
                xs = [r['median_seconds'] for r in rows if r['n'] == n and r['dtype'] == dtype and r['arm'] == arm and r['width'] == width]
                require(len(xs) == 3, 'missing session')
                return statistics.median(xs)
            ref = median('fortran', 1); base = median('base', 0)
            for arm in ('cand', 'b16', 'b32'):
                seq = median(arm, 0); team = median(arm, 8) if n != 64 else None
                summary.append({'n': n, 'dtype': dtype, 'arm': arm, 'fortran_s': ref, 'base_seq_s': base, 'seq_s': seq, 'team8_s': team,
                    'seq_over_fortran': seq / ref, 'fortran_over_team8': ref / team if team else None,
                    'seq_over_base': seq / base, 'useful_gflops_seq': n**3 / (3e9 * seq) if dtype == 'f64' else None,
                    'useful_gflops_team8': n**3 / (3e9 * team) if dtype == 'f64' and team else None,
                    'formal_thresholds_met': (seq <= 1.05 * ref and team <= ref / 4) if team else None})
    with (out / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    finish(out, perf_status, guard, bins, env)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('INCOMPLETE:', exc, file=sys.stderr)
        sys.exit(1)
